"""
Advanced AI Trading Strategy for RealTradR

This module implements an advanced trading strategy that combines:
- Machine learning price predictions
- Sentiment analysis from social media
- Technical indicators and market regime detection
- Advanced risk management
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

# Try to import TensorFlow, but use fallback if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TensorFlow successfully imported")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"TensorFlow import failed: {e}. Will use fallback ML models.")
    TENSORFLOW_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Import our custom modules
from .technical_indicators import TechnicalIndicators, MarketRegime
from .sentiment_analyzer import get_symbol_sentiment
from .risk_management import RiskManager

# Import ML models based on availability
if TENSORFLOW_AVAILABLE:
    from .ml_models import LSTMModel, EnsembleModel, SentimentAwareModel, ModelManager
else:
    from .ml_models_fallback import ModelManager

# Load environment variables
load_dotenv()

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_API_BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"

# Load strategy configuration
def load_strategy_config():
    """Load strategy configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                              "strategy_config.json")
    
    default_config = {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "cash_limit": 100000,
        "short_window": 10,
        "long_window": 30,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
        "sentiment_weight": 0.3,
        "technical_weight": 0.4,
        "ml_weight": 0.3,
        "position_sizing": "kelly",  # "equal", "kelly", "volatility"
        "max_position_pct": 20.0,    # Maximum position size as % of portfolio
        "use_market_regime": True,   # Adjust strategy based on market regime
        "use_sentiment": True,       # Use sentiment analysis
        "use_ml_models": True,       # Use machine learning models
        "risk_free_rate": 0.02,      # Risk-free rate for Kelly criterion
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded strategy config from {config_path}")
                return config
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return default_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return default_config


class AdvancedAIStrategy:
    """Advanced AI Trading Strategy combining multiple signals"""
    
    def __init__(self, config=None, db=None):
        """
        Initialize the strategy
        
        Args:
            config: Strategy configuration (default: load from file)
            db: Optional database session
        """
        # Load configuration
        self.config = config or load_strategy_config()
        self.db = db
        
        # Extract configuration values
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
        self.cash_limit = self.config.get("cash_limit", 100000)
        self.sentiment_weight = self.config.get("sentiment_weight", 0.3)
        self.technical_weight = self.config.get("technical_weight", 0.4)
        self.ml_weight = self.config.get("ml_weight", 0.3)
        self.position_sizing = self.config.get("position_sizing", "kelly")
        self.max_position_pct = self.config.get("max_position_pct", 20.0)
        self.use_market_regime = self.config.get("use_market_regime", True)
        self.use_sentiment = self.config.get("use_sentiment", True)
        self.use_ml_models = self.config.get("use_ml_models", True)
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            ALPACA_API_BASE_URL,
            api_version="v2"
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            stop_loss_pct=self.config.get("stop_loss_pct", 2.0),
            take_profit_pct=self.config.get("take_profit_pct", 5.0),
            max_drawdown_pct=self.config.get("max_drawdown_pct", 25.0),
            risk_free_rate=self.config.get("risk_free_rate", 0.02),
            max_correlation=self.config.get("max_correlation", 0.7)
        )
        
        # Initialize ML models dictionary
        self.model_manager = ModelManager(
            model_dir=self.config.get("model_dir", "./models")
        )
        
        # Track signals and positions
        self.signals = {}
        self.positions = {}
        
        logger.info(f"Initialized AdvancedAIStrategy with symbols: {self.symbols}")
        logger.info(f"Cash limit: ${self.cash_limit}")
        logger.info(f"Paper trading: {ALPACA_PAPER_TRADING}")
        logger.info(f"Signal weights: Sentiment={self.sentiment_weight}, Technical={self.technical_weight}, ML={self.ml_weight}")
    
    def get_historical_data(self, symbol, days=60, timeframe="1D"):
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days of history to retrieve
            timeframe: Time frame for data ("1D", "1H", "15Min", etc.)
            
        Returns:
            DataFrame with historical data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
            
            # Map string timeframe to Alpaca TimeFrame
            tf_map = {
                "1D": TimeFrame.Day,
                "1H": TimeFrame.Hour,
                "15Min": TimeFrame.Minute,
                "5Min": TimeFrame.Minute,
                "1Min": TimeFrame.Minute
            }
            
            alpaca_tf = tf_map.get(timeframe, TimeFrame.Day)
            
            # Handle minute-level timeframes
            if timeframe.endswith("Min"):
                minutes = int(timeframe.split("Min")[0])
                bars = self.api.get_bars(
                    symbol,
                    alpaca_tf,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    adjustment='raw',
                    limit=10000
                ).df
                
                # Resample for specific minute intervals if needed
                if minutes > 1:
                    bars = bars.resample(f'{minutes}T').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
            else:
                # Get daily or hourly data
                bars = self.api.get_bars(
                    symbol,
                    alpaca_tf,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    adjustment='raw'
                ).df
            
            # Reset index to make timestamp a column
            bars = bars.reset_index()
            
            logger.info(f"Got {len(bars)} bars for {symbol}")
            
            return bars
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_technical_signals(self, df):
        """
        Get technical signals from price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with technical signals
        """
        try:
            # Calculate all technical indicators
            indicators_df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Get latest values
            latest = indicators_df.iloc[-1]
            
            # Detect market regime
            market_regime = MarketRegime(int(latest.get('market_regime', 0)))
            
            # Calculate technical signals
            signals = {}
            
            # Moving average signals
            signals['ma_signal'] = 1 if latest['sma_10'] > latest['sma_50'] else -1
            
            # MACD signal
            signals['macd_signal'] = 1 if latest['macd_histogram'] > 0 else -1
            
            # RSI signal (oversold/overbought)
            rsi = latest['rsi_14']
            if rsi < 30:
                signals['rsi_signal'] = 1  # Oversold, bullish
            elif rsi > 70:
                signals['rsi_signal'] = -1  # Overbought, bearish
            else:
                signals['rsi_signal'] = 0  # Neutral
            
            # Bollinger Bands signal
            price = latest['close']
            if price < latest['bb_lower']:
                signals['bb_signal'] = 1  # Below lower band, potential bounce
            elif price > latest['bb_upper']:
                signals['bb_signal'] = -1  # Above upper band, potential reversal
            else:
                signals['bb_signal'] = 0  # Within bands, neutral
            
            # ADX trend strength
            adx = latest['adx']
            if adx > 25:
                if latest['plus_di'] > latest['minus_di']:
                    signals['adx_signal'] = 1  # Strong uptrend
                else:
                    signals['adx_signal'] = -1  # Strong downtrend
            else:
                signals['adx_signal'] = 0  # Weak trend
            
            # Combine signals with equal weights
            weights = {
                'ma_signal': 0.2,
                'macd_signal': 0.2,
                'rsi_signal': 0.2,
                'bb_signal': 0.2,
                'adx_signal': 0.2
            }
            
            # Adjust weights based on market regime
            if market_regime == MarketRegime.BULL_STRONG:
                # In strong bull market, favor trend-following indicators
                weights['ma_signal'] = 0.3
                weights['macd_signal'] = 0.3
                weights['adx_signal'] = 0.2
                weights['rsi_signal'] = 0.1
                weights['bb_signal'] = 0.1
            elif market_regime == MarketRegime.BEAR_STRONG:
                # In strong bear market, favor mean-reversion and oscillators
                weights['rsi_signal'] = 0.3
                weights['bb_signal'] = 0.3
                weights['ma_signal'] = 0.1
                weights['macd_signal'] = 0.1
                weights['adx_signal'] = 0.2
            
            # Calculate combined signal
            combined_signal = sum(signals[k] * weights[k] for k in signals)
            
            return {
                'individual_signals': signals,
                'combined_signal': combined_signal,
                'market_regime': market_regime,
                'indicators': {
                    'rsi': latest['rsi_14'],
                    'macd': latest['macd_line'],
                    'macd_signal': latest['macd_signal'],
                    'macd_hist': latest['macd_histogram'],
                    'bb_upper': latest['bb_upper'],
                    'bb_middle': latest['bb_middle'],
                    'bb_lower': latest['bb_lower'],
                    'adx': latest['adx']
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical signals: {e}")
            return {
                'individual_signals': {},
                'combined_signal': 0,
                'market_regime': MarketRegime.UNKNOWN,
                'indicators': {}
            }
    
    def get_sentiment_signal(self, symbol, days=3):
        """
        Get sentiment signal for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment signal
        """
        if not self.use_sentiment:
            return {'sentiment_score': 0, 'sentiment_signal': 0}
        
        try:
            # Get sentiment from our sentiment analyzer
            sentiment_data = get_symbol_sentiment(symbol, days, self.db)
            
            if not sentiment_data:
                return {'sentiment_score': 0, 'sentiment_signal': 0}
            
            # Extract compound sentiment score
            sentiment_score = sentiment_data.get('compound', 0)
            
            # Convert to signal (-1 to 1)
            if sentiment_score > 0.2:
                sentiment_signal = 1  # Bullish
            elif sentiment_score < -0.2:
                sentiment_signal = -1  # Bearish
            else:
                sentiment_signal = 0  # Neutral
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_signal': sentiment_signal,
                'details': sentiment_data
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment signal for {symbol}: {e}")
            return {'sentiment_score': 0, 'sentiment_signal': 0}
    
    def get_ml_prediction(self, symbol, df):
        """
        Get machine learning prediction for a symbol
        
        Args:
            symbol: Stock symbol
            df: DataFrame with historical data
            
        Returns:
            Dictionary with ML prediction
        """
        if not self.use_ml_models:
            return {'predicted_price': None, 'predicted_change': 0, 'ml_signal': 0}
        
        try:
            # Make prediction using the model manager
            prediction = self.model_manager.predict(df, symbol)
            
            logger.info(f"ML prediction for {symbol}: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return {'predicted_price': None, 'predicted_change': 0, 'ml_signal': 0}
    
    def combine_signals(self, technical_signal, sentiment_signal, ml_signal):
        """
        Combine different signals into a final trading signal
        
        Args:
            technical_signal: Technical analysis signal
            sentiment_signal: Sentiment analysis signal
            ml_signal: Machine learning signal
            
        Returns:
            Combined signal (-1 to 1)
        """
        # Get signal values
        tech_value = technical_signal.get('combined_signal', 0)
        sent_value = sentiment_signal.get('sentiment_signal', 0)
        ml_value = ml_signal.get('ml_signal', 0)
        
        # Combine signals with configured weights
        combined = (
            tech_value * self.technical_weight +
            sent_value * self.sentiment_weight +
            ml_value * self.ml_weight
        )
        
        return combined
    
    def calculate_position_size(self, symbol, signal, portfolio_value):
        """
        Calculate position size based on signal strength and risk
        
        Args:
            symbol: Stock symbol
            signal: Combined signal (-1 to 1)
            portfolio_value: Current portfolio value
            
        Returns:
            Target position size in dollars
        """
        # Get current price
        try:
            current_price = float(self.api.get_latest_trade(symbol).price)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0
        
        # Absolute signal strength (0 to 1)
        signal_strength = abs(signal)
        
        # Base position size as percentage of portfolio
        if self.position_sizing == "equal":
            # Equal position sizing
            base_pct = self.max_position_pct / len(self.symbols)
            
        elif self.position_sizing == "volatility":
            # Volatility-based position sizing
            try:
                # Get historical data
                hist_data = self.get_historical_data(symbol, days=30)
                
                # Calculate daily returns
                returns = hist_data['close'].pct_change().dropna()
                
                # Calculate volatility (annualized)
                vol = returns.std() * np.sqrt(252)
                
                # Inverse volatility weighting
                base_pct = (self.max_position_pct / vol) / 10
                
                # Cap at maximum position size
                base_pct = min(base_pct, self.max_position_pct)
                
            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {e}")
                base_pct = self.max_position_pct / len(self.symbols)
            
        elif self.position_sizing == "kelly":
            # Kelly criterion position sizing
            try:
                # Get historical data
                hist_data = self.get_historical_data(symbol, days=60)
                
                # Calculate daily returns
                returns = hist_data['close'].pct_change().dropna()
                
                # Calculate win rate and average win/loss
                wins = returns[returns > 0]
                losses = returns[returns < 0]
                
                win_rate = len(wins) / len(returns)
                avg_win = wins.mean() if len(wins) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
                
                # Calculate Kelly percentage
                if avg_loss > 0:
                    kelly_pct = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                else:
                    kelly_pct = win_rate
                
                # Apply a fraction of Kelly (half-Kelly is common)
                base_pct = max(0, kelly_pct * 0.5 * self.max_position_pct)
                
                # Cap at maximum position size
                base_pct = min(base_pct, self.max_position_pct)
                
            except Exception as e:
                logger.error(f"Error calculating Kelly criterion for {symbol}: {e}")
                base_pct = self.max_position_pct / len(self.symbols)
        else:
            # Default to equal sizing
            base_pct = self.max_position_pct / len(self.symbols)
        
        # Scale by signal strength
        position_pct = base_pct * signal_strength
        
        # Calculate dollar amount
        position_size = portfolio_value * (position_pct / 100)
        
        # Adjust for signal direction
        if signal < 0:
            position_size = -position_size
        
        return position_size
    
    def execute_trades(self, symbol, target_position_size):
        """
        Execute trades to reach target position size
        
        Args:
            symbol: Stock symbol
            target_position_size: Target position size in dollars (negative for short)
            
        Returns:
            Dictionary with trade information
        """
        try:
            # Get current position
            current_position = 0
            try:
                position = self.api.get_position(symbol)
                current_position = float(position.market_value)
            except Exception:
                # No position
                pass
            
            # Calculate difference
            diff = target_position_size - current_position
            
            # Skip small adjustments
            if abs(diff) < 100:
                logger.info(f"Skipping small position adjustment for {symbol}: ${diff:.2f}")
                return {
                    "symbol": symbol,
                    "action": "hold",
                    "amount": 0,
                    "current_position": current_position,
                    "target_position": target_position_size
                }
            
            # Get current price
            current_price = float(self.api.get_latest_trade(symbol).price)
            
            # Calculate quantity
            quantity = int(abs(diff) / current_price)
            
            if quantity == 0:
                return {
                    "symbol": symbol,
                    "action": "hold",
                    "amount": 0,
                    "current_position": current_position,
                    "target_position": target_position_size
                }
            
            # Determine side
            if diff > 0:
                side = "buy"
            else:
                side = "sell"
            
            # Execute order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type="market",
                time_in_force="day"
            )
            
            logger.info(f"Executed {side} order for {quantity} shares of {symbol}")
            
            return {
                "symbol": symbol,
                "action": side,
                "quantity": quantity,
                "amount": diff,
                "current_position": current_position,
                "target_position": target_position_size,
                "order_id": order.id
            }
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {
                "symbol": symbol,
                "action": "error",
                "error": str(e),
                "current_position": 0,
                "target_position": target_position_size
            }
    
    def get_performance_metrics(self):
        """
        Get performance metrics for the strategy
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get account information
            account = self.api.get_account()
            
            # Get portfolio history
            portfolio_hist = self.api.get_portfolio_history(period="1M", timeframe="1D")
            
            # Calculate daily returns
            equity = portfolio_hist.equity
            
            # Handle empty equity array or all zeros
            if not equity or all(e == 0 for e in equity):
                return {
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "equity": float(account.equity) if hasattr(account, 'equity') else 0.0,
                    "buying_power": float(account.buying_power) if hasattr(account, 'buying_power') else 0.0,
                    "daily_returns": [0.0],
                    "note": "No trading activity detected"
                }
            
            daily_returns = [0]
            
            for i in range(1, len(equity)):
                if equity[i-1] > 0:
                    daily_return = (equity[i] - equity[i-1]) / equity[i-1]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0)
            
            # Calculate metrics
            last_equity = float(account.last_equity) if hasattr(account, 'last_equity') and float(account.last_equity) > 0 else float(account.equity)
            current_equity = float(account.equity)
            
            # Avoid division by zero
            if last_equity > 0:
                total_return = (current_equity - last_equity) / last_equity * 100
            else:
                total_return = 0
            
            # Annualized return
            trading_days = len(equity)
            if trading_days > 1 and total_return != 0:
                annualized_return = ((1 + total_return/100) ** (252/trading_days) - 1) * 100
            else:
                annualized_return = 0
            
            # Sharpe ratio
            risk_free_rate = self.config.get("risk_free_rate", 0.02) / 252  # Daily risk-free rate
            excess_returns = [r - risk_free_rate for r in daily_returns]
            sharpe_ratio = 0
            
            if len(excess_returns) > 1 and np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # Maximum drawdown
            max_drawdown = 0
            
            if len(equity) > 0:
                peak = equity[0]
                
                for value in equity:
                    if value > peak:
                        peak = value
                    
                    # Avoid division by zero
                    if peak > 0:
                        drawdown = (peak - value) / peak
                        max_drawdown = max(max_drawdown, drawdown)
            
            max_drawdown *= 100  # Convert to percentage
            
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "equity": current_equity,
                "buying_power": float(account.buying_power) if hasattr(account, 'buying_power') else 0.0,
                "daily_returns": daily_returns
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def run(self):
        """
        Run the strategy once
        
        Returns:
            Dictionary with strategy results
        """
        try:
            # Get account information
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            
            logger.info(f"Portfolio value: ${portfolio_value:.2f}, Cash: ${cash:.2f}")
            
            # Process each symbol
            results = {}
            
            for symbol in self.symbols:
                try:
                    # Get historical data
                    df = self.get_historical_data(symbol, days=60)
                    
                    if df.empty:
                        logger.error(f"No historical data available for {symbol}")
                        continue
                    
                    # Get technical signals
                    technical_signal = self.get_technical_signals(df)
                    
                    # Get sentiment signal
                    sentiment_signal = self.get_sentiment_signal(symbol)
                    
                    # Get ML prediction
                    ml_signal = self.get_ml_prediction(symbol, df)
                    
                    # Combine signals
                    combined_signal = self.combine_signals(
                        technical_signal, sentiment_signal, ml_signal
                    )
                    
                    # Calculate position size
                    target_position_size = self.calculate_position_size(
                        symbol, combined_signal, portfolio_value
                    )
                    
                    # Execute trades
                    trade_result = self.execute_trades(symbol, target_position_size)
                    
                    # Store results
                    results[symbol] = {
                        "technical_signal": technical_signal,
                        "sentiment_signal": sentiment_signal,
                        "ml_signal": ml_signal,
                        "combined_signal": combined_signal,
                        "target_position_size": target_position_size,
                        "trade_result": trade_result
                    }
                    
                    logger.info(f"Processed {symbol}: Signal={combined_signal:.2f}, Target=${target_position_size:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results[symbol] = {"error": str(e)}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "cash": cash,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return {"error": str(e)}


# Main function to run the strategy
def main():
    """Main function to run the strategy"""
    try:
        # Initialize strategy
        strategy = AdvancedAIStrategy()
        
        # Run strategy
        result = strategy.run()
        
        # Get performance metrics
        metrics = strategy.get_performance_metrics()
        
        # Print summary
        print("\n=== Strategy Execution Summary ===")
        print(f"Portfolio Value: ${metrics.get('equity', 0):.2f}")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print("=================================\n")
        
        # Print signal summary
        print("=== Signal Summary ===")
        for symbol, data in result.get("results", {}).items():
            if "error" in data:
                print(f"{symbol}: Error - {data['error']}")
                continue
                
            combined_signal = data.get("combined_signal", 0)
            target_position = data.get("target_position_size", 0)
            trade_result = data.get("trade_result", {})
            action = trade_result.get("action", "unknown")
            
            print(f"{symbol}: Signal={combined_signal:.2f}, Position=${target_position:.2f}, Action={action}")
        
        print("=====================\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    main()
