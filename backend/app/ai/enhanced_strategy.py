"""
Enhanced AI Trading Strategy for RealTradR

This module implements an advanced trading strategy that combines:
- Machine learning price predictions with ensemble models
- Adaptive market regime detection
- Dynamic position sizing based on volatility and Kelly criterion
- Advanced risk management with correlation-based portfolio optimization
- Multiple timeframe analysis for more robust signals
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import time
import random

# Import RealTradR modules
from backend.app.ai.risk_management import RiskManager
from backend.app.ai.technical_indicators import TechnicalIndicators, MarketRegime
from backend.app.ai.ml_models_fallback import ModelManager
from backend.app.utils.performance import PerformanceMonitor

# Load environment variables
load_dotenv()

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_API_BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import TensorFlow, fall back to scikit-learn models if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"TensorFlow import failed: {e}. Will use fallback ML models.")


def load_strategy_config(config_file="strategy_config.json"):
    """
    Load strategy configuration from JSON file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_file}: {e}")
        # Return default configuration
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "cash_limit": 100000.0,
            "sentiment_weight": 0.3,
            "technical_weight": 0.4,
            "ml_weight": 0.3,
            "position_sizing": "kelly",
            "max_position_pct": 20.0,
            "use_market_regime": True,
            "use_sentiment": True,
            "use_ml_models": True
        }


class EnhancedAIStrategy:
    """Enhanced AI Trading Strategy combining multiple signals with adaptive features"""
    
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
        self.setup_alpaca_api()
        
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
        
        # Initialize technical indicators
        self.ti = TechnicalIndicators()
        
        # Track signals and positions
        self.signals = {}
        self.positions = {}
        self.market_regimes = {}
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Initialized EnhancedAIStrategy with symbols: {self.symbols}")
        logger.info(f"Cash limit: ${self.cash_limit}")
        logger.info(f"Paper trading: {ALPACA_PAPER_TRADING}")
        logger.info(f"Signal weights: Sentiment={self.sentiment_weight}, Technical={self.technical_weight}, ML={self.ml_weight}")
    
    def setup_alpaca_api(self):
        """Set up Alpaca API"""
        try:
            # Get API credentials from environment or config
            api_credentials = self.config.get("api_credentials", {})
            self.api_key = api_credentials.get("api_key") or os.getenv("ALPACA_API_KEY")
            self.api_secret = api_credentials.get("api_secret") or os.getenv("ALPACA_API_SECRET")
            self.base_url = api_credentials.get("base_url") or os.getenv("ALPACA_API_BASE_URL")
            self.paper_trading = api_credentials.get("paper_trading", True)
            self.use_mock_data = self.config.get("market_data", {}).get("use_mock_data", False)
            self.use_mock_api = False
            
            # Check if we should use mock data
            if self.use_mock_data:
                logger.info("Using mock data for testing")
                self.use_mock_api = True
                # Initialize mock API objects to prevent attribute errors
                self.api = MockAlpacaAPI()
                self.data_api = MockAlpacaAPI()
                self.has_market_data_access = False
                self.has_historical_data_access = False
                return False
            
            # Validate credentials
            if not self.api_key or not self.api_secret:
                logger.warning("Alpaca API credentials not found. Using mock API.")
                self.use_mock_api = True
                # Initialize mock API objects to prevent attribute errors
                self.api = MockAlpacaAPI()
                self.data_api = MockAlpacaAPI()
                self.has_market_data_access = False
                self.has_historical_data_access = False
                return False
                
            if not self.base_url:
                self.base_url = "https://paper-api.alpaca.markets" if self.paper_trading else "https://api.alpaca.markets"
            
            # Initialize API
            self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
            
            # Test connection
            try:
                account = self.api.get_account()
                logger.info(f"Connected to Alpaca API. Account ID: {account.id}")
                logger.info(f"Account status: {account.status}")
                logger.info(f"Account cash: ${float(account.cash):.2f}")
                
                # Set up data API for market data
                data_url = "https://data.alpaca.markets"
                self.data_api = tradeapi.REST(self.api_key, self.api_secret, data_url, api_version='v2')
                
                # Test snapshot access (which we know works)
                test_symbol = "AAPL"
                try:
                    snapshot = self.data_api.get_snapshot(test_symbol)
                    logger.info(f"Market data API connection successful. {test_symbol} price: ${snapshot.latest_trade.price:.2f}")
                    self.has_market_data_access = True
                except Exception as e:
                    logger.warning(f"Market data snapshot access failed: {e}")
                    logger.warning("Will use mock data for market snapshots")
                    self.has_market_data_access = False
                
                # Test historical data access (which may not work with free tier)
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=1)
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    bars = self.data_api.get_bars(test_symbol, "1D", start=start_str, end=end_str).df
                    if not bars.empty:
                        logger.info(f"Historical data access successful. Got {len(bars)} bars.")
                        self.has_historical_data_access = True
                    else:
                        logger.warning("Historical data access returned empty result")
                        self.has_historical_data_access = False
                except Exception as e:
                    logger.warning(f"Historical data access failed: {e}")
                    logger.warning("Will use mock data for historical data")
                    self.has_historical_data_access = False
                
                self.use_mock_api = False
                return True
            except Exception as e:
                logger.error(f"Error connecting to Alpaca API: {e}")
                logger.error("API authentication failed. Please check your API credentials.")
                logger.error("For paper trading, make sure you're using paper trading credentials.")
                logger.error("For live trading, make sure you have proper permissions.")
                self.use_mock_api = True
                return False
                
        except Exception as e:
            logger.error(f"Error setting up Alpaca API: {e}")
            self.use_mock_api = True
            return False
    
    def get_current_market_data(self, symbol):
        """
        Get current market data for a symbol using the snapshot API
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Dictionary with current market data
        """
        if self.use_mock_api:
            return self.generate_mock_snapshot(symbol)
            
        try:
            # Get snapshot from Alpaca
            snapshot = self.data_api.get_snapshot(symbol)
            
            # Check if snapshot is valid
            if snapshot is None:
                logger.warning(f"No snapshot data available for {symbol}")
                return self.generate_mock_snapshot(symbol)
                
            # Extract relevant data from snapshot
            try:
                # Handle different snapshot object structures
                if hasattr(snapshot, 'latest_trade'):
                    # New Alpaca API structure
                    price = snapshot.latest_trade.p if snapshot.latest_trade else None
                    volume = snapshot.latest_trade.s if snapshot.latest_trade else 0
                    timestamp = snapshot.latest_trade.t if snapshot.latest_trade else datetime.now()
                elif hasattr(snapshot, 'daily'):
                    # Legacy Alpaca API structure
                    price = snapshot.daily.c if snapshot.daily else None
                    volume = snapshot.daily.v if snapshot.daily else 0
                    timestamp = snapshot.daily.t if snapshot.daily else datetime.now()
                else:
                    # SnapshotV2 structure
                    price = snapshot.latest_quote.ap if hasattr(snapshot, 'latest_quote') and snapshot.latest_quote else None
                    volume = 0  # Not available in this structure
                    timestamp = datetime.now()
                
                if price is None:
                    logger.warning(f"No price data in snapshot for {symbol}")
                    return self.generate_mock_snapshot(symbol)
                    
                return {
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "timestamp": timestamp
                }
                
            except Exception as e:
                logger.error(f"Error extracting data from snapshot for {symbol}: {e}")
                return self.generate_mock_snapshot(symbol)
                
        except Exception as e:
            logger.error(f"Error getting snapshot for {symbol}: {e}")
            return self.generate_mock_snapshot(symbol)
            
    def generate_mock_snapshot(self, symbol):
        """
        Generate mock snapshot data for testing
        
        Args:
            symbol: Symbol to generate data for
            
        Returns:
            Dictionary with mock snapshot data
        """
        # Generate realistic price based on symbol
        if symbol == "AAPL":
            price = random.uniform(180.0, 220.0)
        elif symbol == "MSFT":
            price = random.uniform(320.0, 380.0)
        elif symbol == "GOOGL":
            price = random.uniform(130.0, 160.0)
        elif symbol == "AMZN":
            price = random.uniform(160.0, 200.0)
        elif symbol == "META":
            price = random.uniform(450.0, 550.0)
        else:
            price = random.uniform(50.0, 500.0)
            
        # Generate volume
        volume = random.randint(1000000, 10000000)
        
        # Return snapshot data
        return {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now()
        }
        
    def generate_synthetic_data(self, symbol, days, snapshot):
        """
        Generate synthetic historical data based on current snapshot
        
        Args:
            symbol: Symbol to generate data for
            days: Number of days of data to generate
            snapshot: Current market data snapshot
            
        Returns:
            DataFrame with synthetic historical price data
        """
        # Get current price from snapshot
        current_price = snapshot.get("price", 100.0)
        
        # Generate dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Oldest first
        
        # Generate price data with realistic volatility and trend
        prices = []
        volatility = 0.02  # 2% daily volatility
        trend = 0.0001  # Slight upward trend
        
        # Start with current price and work backwards
        price = current_price
        for i in range(days):
            # Add random noise
            daily_return = np.random.normal(trend, volatility)
            price = price / (1 + daily_return)  # Work backwards
            prices.append(price)
            
        prices.reverse()  # Oldest first
        
        # Generate volume data
        volumes = [int(np.random.normal(10000000, 5000000)) for _ in range(days)]
        
        # Create DataFrame
        df = pd.DataFrame({
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            "close": prices,
            "volume": volumes
        }, index=dates)
        
        return df
    
    def get_historical_data(self, symbol, days=60, timeframe="1D"):
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Symbol to get data for
            days: Number of days of historical data to retrieve
            timeframe: Timeframe for the data (1D, 1H, 15Min, etc.)
            
        Returns:
            DataFrame with historical price data
        """
        # If we're using mock data or don't have historical data access, generate synthetic data
        if self.use_mock_api or not hasattr(self, 'has_historical_data_access') or not self.has_historical_data_access:
            logger.info(f"Using mock data for {symbol}")
            return self.generate_mock_data(symbol, timeframe, days)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Try to get data from Alpaca
            try:
                bars = self.data_api.get_bars(symbol, timeframe, start=start_str, end=end_str).df
                
                if bars is not None and not bars.empty:
                    logger.info(f"Retrieved {len(bars)} bars of historical data for {symbol}")
                    return bars
                else:
                    logger.warning(f"No historical data returned for {symbol}, using mock data")
                    return self.generate_mock_data(symbol, timeframe, days)
                    
            except Exception as e:
                # Check if this is a 403 Forbidden error (API plan limitation)
                if "403" in str(e) and "Forbidden" in str(e):
                    logger.warning(f"Historical data access forbidden for {symbol} (API plan limitation)")
                    # Update the flag to avoid future API calls that will fail
                    self.has_historical_data_access = False
                else:
                    logger.error(f"Error getting historical data for {symbol}: {e}")
                
                # Use mock data as fallback
                use_mock_data_on_failure = self.config.get("market_data", {}).get("use_mock_data_on_api_failure", True)
                if use_mock_data_on_failure:
                    logger.info(f"Falling back to mock data for {symbol}")
                    return self.generate_mock_data(symbol, timeframe, days)
                else:
                    # Return empty DataFrame
                    return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in get_historical_data for {symbol}: {e}")
            return self.generate_mock_data(symbol, timeframe, days)
    
    def get_multi_timeframe_data(self, symbol, days=60):
        """
        Get data for multiple timeframes for a symbol.
        
        Args:
            symbol: Symbol to get data for
            days: Number of days of data to get
            
        Returns:
            Dictionary with data for each timeframe
        """
        try:
            # Define timeframes
            timeframes = self.config.get("market_data", {}).get("timeframes", ["1D", "1H", "15Min"])
            
            # Get data for each timeframe
            data = {}
            for timeframe in timeframes:
                data[timeframe] = self.get_historical_data(symbol, timeframe, days)
            
            return data
        except Exception as e:
            logger.warning(f"Error getting multi-timeframe data for {symbol}: {e}")
            
            # Return mock data for all timeframes
            mock_data = {}
            for timeframe in self.config.get("market_data", {}).get("timeframes", ["1D", "1H", "15Min"]):
                # Calculate date range
                end_date_obj = datetime.now()
                start_date_obj = end_date_obj - timedelta(days=days)
                
                # Generate mock data
                mock_data[timeframe] = self.generate_synthetic_data(symbol, timeframe, start_date_obj, end_date_obj)
            
            return mock_data
    
    def calculate_technical_signals(self, data, symbol=None):
        """
        Calculate technical signals for a symbol
        
        Args:
            data: DataFrame with price data
            symbol: Symbol to calculate signals for
            
        Returns:
            Dictionary with technical signals
        """
        try:
            # Ensure we have a proper DataFrame with the right columns
            if isinstance(data, pd.Series):
                # Convert Series to DataFrame
                data = pd.DataFrame({data.name: data})
            
            # Make sure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Missing required column {col} in data for {symbol}")
                    # If close is missing, we can't proceed
                    if col == 'close':
                        return None
            
            # Add technical indicators
            try:
                indicators_df = TechnicalIndicators.add_all_indicators(data)
            except Exception as e:
                logger.error(f"Error adding technical indicators: {e}")
                return None
            
            # Detect market regime
            try:
                market_regime = TechnicalIndicators.detect_market_regime(
                    data['close'], data['high'], data['low'], 
                    data['volume'] if 'volume' in data.columns else None
                )
                self.market_regimes[symbol] = market_regime
            except Exception as e:
                logger.error(f"Error detecting market regime: {e}")
                market_regime = MarketRegime.NEUTRAL
                self.market_regimes[symbol] = market_regime
            
            # Calculate technical signal
            try:
                signal = TechnicalIndicators.calculate_technical_signal(indicators_df, market_regime)
            except Exception as e:
                logger.error(f"Error calculating technical signal: {e}")
                signal = 0
            
            # Return signals
            return {
                "combined": signal,
                "market_regime": market_regime
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical signals for {symbol}: {e}")
            return None
    
    def get_ml_prediction(self, symbol, data):
        """
        Get ML prediction for a symbol
        
        Args:
            symbol: Symbol to get prediction for
            data: DataFrame with price data
            
        Returns:
            Dictionary with ML prediction
        """
        try:
            if isinstance(data, str):
                logger.error(f"Invalid data format for ML prediction: {type(data)}")
                return {"signal": 0, "confidence": 0}
                
            # Check if we have enough data
            if len(data) < 30:
                logger.warning(f"Not enough data for ML prediction for {symbol}")
                return {"signal": 0, "confidence": 0}
            
            # Log training data
            logger.info(f"Training model for {data}...")
            
            # Get prediction from model manager
            try:
                prediction = self.model_manager.predict(data, symbol)
                
                # Calculate signal from prediction
                if prediction is not None:
                    predicted_return = prediction.get('predicted_return', 0)
                    confidence = prediction.get('confidence', 0)
                    
                    # Scale the signal based on the predicted return
                    if abs(predicted_return) < 0.001:
                        signal = 0  # Very small prediction, treat as neutral
                    else:
                        # Scale to -1 to 1 range
                        signal = max(min(predicted_return * 100, 1), -1)
                    
                    return {
                        "signal": signal,
                        "confidence": confidence,
                        "predicted_return": predicted_return
                    }
                else:
                    logger.warning(f"No prediction returned for {symbol}")
                    return {"signal": 0, "confidence": 0}
            except Exception as e:
                logger.error(f"Error getting ML prediction for {data}: {e}")
                return {"signal": 0, "confidence": 0}
                
        except Exception as e:
            logger.error(f"Error in get_ml_prediction for {symbol}: {e}")
            return {"signal": 0, "confidence": 0}
    
    def get_sentiment_signal(self, symbol):
        """
        Get sentiment signal for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment signal between -1 and 1
        """
        if not self.use_sentiment:
            return 0
        
        # This would normally call the sentiment analysis API
        # For now, return a random value as a placeholder
        # In a real implementation, this would use the NEWS_API_KEY, TWITTER_API_KEY, etc.
        return np.random.uniform(-0.5, 0.5)
    
    def combine_signals(self, technical_signals, ml_signal, sentiment_signal):
        """
        Combine signals from different sources
        
        Args:
            technical_signals: Dictionary with technical signals for each timeframe
            ml_signal: Machine learning signal (can be a dictionary or float)
            sentiment_signal: Sentiment signal
            
        Returns:
            Combined signal between -1 and 1
        """
        # Get weights from config
        weights = self.config.get("signal_weights", {})
        technical_weight = weights.get("technical", 0.6)
        ml_weight = weights.get("ml", 0.4)
        sentiment_weight = weights.get("sentiment", 0.0)
        
        # Normalize weights
        total_weight = technical_weight + ml_weight + sentiment_weight
        if total_weight > 0:
            technical_weight /= total_weight
            ml_weight /= total_weight
            sentiment_weight /= total_weight
        else:
            # Default to equal weights if total is zero
            technical_weight = ml_weight = sentiment_weight = 1/3
        
        # Calculate technical signal across timeframes
        if not technical_signals:
            technical_signal = 0
        else:
            # Get timeframe weights
            timeframe_weights = self.config.get("timeframe_weights", {
                "daily": 0.5,
                "hourly": 0.3,
                "minutes": 0.2
            })
            
            # Calculate weighted average of technical signals
            weighted_sum = 0
            weight_sum = 0
            
            for timeframe, data in technical_signals.items():
                if timeframe in timeframe_weights:
                    weight = timeframe_weights[timeframe]
                    signal = data.get("signal", 0)
                    weighted_sum += signal * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                technical_signal = weighted_sum / weight_sum
            else:
                technical_signal = 0
        
        # Extract ML signal value from dictionary if needed
        ml_signal_value = 0
        if isinstance(ml_signal, dict):
            ml_signal_value = ml_signal.get("signal", 0)
        elif isinstance(ml_signal, (int, float)):
            ml_signal_value = ml_signal
        
        # Combine signals
        combined_signal = (
            technical_signal * technical_weight +
            ml_signal_value * ml_weight +
            sentiment_signal * sentiment_weight
        )
        
        # Ensure signal is between -1 and 1
        combined_signal = max(min(combined_signal, 1.0), -1.0)
        
        return combined_signal
    
    def calculate_position_size(self, symbol, signal, current_price, available_cash):
        """
        Calculate position size based on signal strength, volatility, and available cash
        
        Args:
            symbol: Stock symbol
            signal: Signal strength (-1 to 1)
            current_price: Current price of the stock
            available_cash: Available cash for trading
            
        Returns:
            Target position size in dollars
        """
        # Skip if signal is too weak
        min_signal_threshold = self.config.get("min_signal_threshold", 0.1)
        if abs(signal) < min_signal_threshold:
            return 0
        
        # Get position sizing settings from config
        position_sizing = self.config.get("position_sizing", {})
        max_position_pct = position_sizing.get("max_position_pct", 0.2)
        base_position_pct = position_sizing.get("base_position_pct", 0.05)
        
        # Calculate base position size
        max_position_dollars = self.cash_limit * max_position_pct
        base_position_dollars = self.cash_limit * base_position_pct
        
        # Scale position size by signal strength
        signal_scale = abs(signal) ** 2  # Square to emphasize stronger signals
        position_dollars = base_position_dollars + (max_position_dollars - base_position_dollars) * signal_scale
        
        # Adjust for volatility
        volatility_factor = self.calculate_volatility_factor(symbol)
        position_dollars = position_dollars * (1.0 / volatility_factor)
        
        # Adjust for correlation with existing positions
        correlation_factor = self.calculate_correlation_factor(symbol)
        position_dollars = position_dollars * correlation_factor
        
        # Adjust for market regime
        regime = self.market_regimes.get(symbol, MarketRegime.NEUTRAL)
        regime_factor = 1.0
        
        if regime == MarketRegime.BULLISH_TREND and signal > 0:
            # Increase position size in bullish trend with positive signal
            regime_factor = 1.2
        elif regime == MarketRegime.BEARISH_TREND and signal < 0:
            # Increase position size in bearish trend with negative signal
            regime_factor = 1.2
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Reduce position size in high volatility
            regime_factor = 0.7
        elif regime == MarketRegime.LOW_VOLATILITY:
            # Increase position size in low volatility
            regime_factor = 1.1
            
        position_dollars = position_dollars * regime_factor
        
        # Cap at maximum position size
        position_dollars = min(position_dollars, max_position_dollars)
        
        # Ensure we have enough cash
        position_dollars = min(position_dollars, available_cash)
        
        # Set minimum position size to avoid tiny trades
        min_position_dollars = position_sizing.get("min_position_dollars", 1000)
        if 0 < position_dollars < min_position_dollars:
            logger.info(f"Skipping small position adjustment for {symbol}: ${position_dollars:.2f}")
            position_dollars = 0
        
        # Set position direction based on signal
        if signal < 0:
            position_dollars = -position_dollars
        
        return position_dollars
    
    def calculate_volatility_factor(self, symbol):
        """
        Calculate a scaling factor based on recent volatility
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Scaling factor between 0.5 and 1.5
        """
        try:
            # Get recent data
            data = self.get_historical_data(symbol, days=20)
            
            if data.empty:
                return 1.0
            
            # Calculate daily returns
            data['return'] = data['close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            volatility = data['return'].dropna().std()
            
            # Compare to average market volatility (can be refined with actual market data)
            avg_market_volatility = 0.01  # 1% daily volatility as baseline
            relative_volatility = volatility / avg_market_volatility
            
            # Range: 0.5 (high vol) to 1.5 (low vol)
            factor = 1.0 / (relative_volatility + 0.5)
            factor = max(min(factor, 1.5), 0.5)
            
            return factor
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor for {symbol}: {e}")
            return 1.0
    
    def calculate_correlation_factor(self, symbol):
        """
        Calculate a correlation factor to adjust position size based on
        correlation with existing positions
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Correlation factor between 0.5 and 1.0
        """
        try:
            # Get current positions
            positions = self.get_current_positions()
            
            # If no positions or only this symbol, return 1.0 (no adjustment)
            if not positions or (len(positions) == 1 and symbol in positions):
                return 1.0
                
            # Get historical data for this symbol
            symbol_data = self.get_historical_data(symbol, days=60)
            if symbol_data.empty:
                return 1.0
                
            symbol_returns = symbol_data['close'].pct_change().dropna()
            
            # Calculate correlation with each existing position
            max_correlation = 0
            
            for pos_symbol, position in positions.items():
                # Skip if it's the same symbol or not enough data
                if pos_symbol == symbol:
                    continue
                    
                # Get historical data for position symbol
                pos_data = self.get_historical_data(pos_symbol, days=60)
                if pos_data.empty:
                    continue
                    
                pos_returns = pos_data['close'].pct_change().dropna()
                
                # Align the return series to the same dates
                common_index = symbol_returns.index.intersection(pos_returns.index)
                if len(common_index) < 10:  # Need at least 10 data points
                    continue
                    
                aligned_symbol_returns = symbol_returns.loc[common_index]
                aligned_pos_returns = pos_returns.loc[common_index]
                
                # Calculate correlation
                correlation = abs(aligned_symbol_returns.corr(aligned_pos_returns))
                
                # Track maximum correlation
                max_correlation = max(max_correlation, correlation)
            
            # Calculate factor: high correlation = lower factor
            # Use a threshold from risk manager if available
            max_correlation_threshold = getattr(self.risk_manager, 'max_correlation', 0.7)
            
            if max_correlation > max_correlation_threshold:
                # Reduce position size for highly correlated assets
                factor = 1.0 - ((max_correlation - max_correlation_threshold) / (1.0 - max_correlation_threshold)) * 0.5
                factor = max(factor, 0.5)  # Don't go below 0.5
            else:
                factor = 1.0
                
            return factor
            
        except Exception as e:
            logger.error(f"Error calculating correlation factor for {symbol}: {e}")
            return 1.0
    
    def execute_trade(self, symbol, target_position_dollars, current_position_dollars):
        """
        Execute a trade to reach the target position
        
        Args:
            symbol: Stock symbol
            target_position_dollars: Target position in dollars
            current_position_dollars: Current position in dollars
            
        Returns:
            Dictionary with trade result
        """
        try:
            # Calculate position adjustment
            adjustment_dollars = target_position_dollars - current_position_dollars
            
            # Skip small adjustments
            if abs(adjustment_dollars) < 100:
                logger.info(f"Skipping small position adjustment for {symbol}: ${adjustment_dollars:.2f}")
                return {"trade_executed": False}
            
            # Get current price
            latest_quote = self.api.get_latest_quote(symbol)
            price = latest_quote.ap  # Ask price
            
            # Calculate quantity
            quantity = int(abs(adjustment_dollars) / price)
            
            # Skip if quantity is zero
            if quantity == 0:
                return {"trade_executed": False}
            
            # Determine side
            side = "buy" if adjustment_dollars > 0 else "sell"
            
            # Execute trade
            if ALPACA_PAPER_TRADING:
                # Execute trade on paper trading account
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type="market",
                    time_in_force="day"
                )
                
                logger.info(f"Executed {side} order for {quantity} shares of {symbol} at ${price:.2f}")
                
                return {
                    "trade_executed": True,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_id": order.id
                }
            else:
                # Simulate trade for testing
                logger.info(f"Simulated {side} order for {quantity} shares of {symbol} at ${price:.2f}")
                
                return {
                    "trade_executed": True,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_id": "simulated"
                }
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {"trade_executed": False, "error": str(e)}
    
    def get_current_positions(self):
        """
        Get current positions from Alpaca
        
        Returns:
            Dictionary with positions by symbol
        """
        try:
            positions = {}
            
            # Get positions from Alpaca
            alpaca_positions = self.api.list_positions()
            
            for position in alpaca_positions:
                symbol = position.symbol
                market_value = float(position.market_value)
                positions[symbol] = market_value
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}
    
    def get_account_info(self):
        """
        Get account information from Alpaca
        
        Returns:
            Dictionary with account information
        """
        try:
            account = self.api.get_account()
            
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power)
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"cash": self.cash_limit, "portfolio_value": self.cash_limit}
    
    def run(self):
        """
        Run the strategy once
        
        Returns:
            Dictionary with results
        """
        try:
            logger.info("Running enhanced AI strategy...")
            
            # Get account info
            try:
                account = self.get_account_info()
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                account = {
                    "equity": 100000,
                    "cash": 100000,
                    "buying_power": 200000,
                    "portfolio_value": 100000
                }
            
            # Get current positions
            try:
                positions = self.get_current_positions()
            except Exception as e:
                logger.error(f"Error getting current positions: {e}")
                positions = {}
            
            # Process each symbol
            signals = {}
            trades = []
            
            for symbol in self.symbols:
                try:
                    # Get current market data
                    if hasattr(self, 'use_mock_data') and self.use_mock_data:
                        snapshot = self.generate_mock_snapshot(symbol)
                    else:
                        snapshot = self.get_current_market_data(symbol)
                    
                    # Get historical data or generate synthetic data
                    try:
                        historical_data = self.get_historical_data(symbol)
                    except Exception as hist_error:
                        logger.warning(f"Error getting historical data for {symbol}: {hist_error}")
                        logger.info(f"Generating synthetic data for {symbol} based on current snapshot")
                        historical_data = self.generate_synthetic_data(symbol, 60, snapshot)
                    
                    # Get multi-timeframe data
                    try:
                        multi_tf_data = self.get_multi_timeframe_data(symbol)
                    except Exception as tf_error:
                        logger.warning(f"Error getting multi-timeframe data for {symbol}: {tf_error}")
                        multi_tf_data = {
                            "daily": historical_data,
                            "hourly": historical_data.copy(),
                            "minutes": historical_data.copy()
                        }
                    
                    # Calculate technical signals
                    technical_signals = {}
                    for timeframe, data in multi_tf_data.items():
                        technical_signals[timeframe] = self.calculate_technical_signals(data, symbol)
                    
                    # Get ML prediction
                    ml_signal = self.get_ml_prediction(symbol, historical_data)
                    
                    # Get sentiment signal
                    sentiment_signal = self.get_sentiment_signal(symbol)
                    
                    # Combine signals
                    combined_signal = self.combine_signals(technical_signals, ml_signal, sentiment_signal)
                    
                    # Determine action based on signal
                    action = "HOLD"
                    if combined_signal > 0.3:
                        action = "BUY"
                    elif combined_signal < -0.3:
                        action = "SELL"
                    
                    # Store signal
                    signals[symbol] = {
                        "technical": technical_signals,
                        "ml": ml_signal,
                        "sentiment": sentiment_signal,
                        "combined_signal": combined_signal,
                        "action": action,
                        "price": snapshot["price"],
                        "timestamp": datetime.now()
                    }
                    
                    # Execute trade if not in simulation mode
                    if not self.config.get("simulation_mode", False):
                        current_position_dollars = 0
                        if symbol in positions:
                            current_position_dollars = positions[symbol]
                        
                        # Calculate target position size
                        available_cash = account["cash"]
                        target_position_dollars = self.calculate_position_size(
                            symbol, combined_signal, snapshot["price"], available_cash
                        )
                        
                        # Execute trade if target position is different from current
                        if abs(target_position_dollars - current_position_dollars) > 100:  # Minimum change threshold
                            trade_result = self.execute_trade(
                                symbol, target_position_dollars, current_position_dollars
                            )
                            trades.append(trade_result)
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Calculate performance metrics
            performance = self.get_performance_metrics()
            
            # Return results
            return {
                "timestamp": datetime.now(),
                "account": account,
                "positions": positions,
                "signals": signals,
                "trades": trades,
                "performance": performance
            }
        
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return None
    
    def run_with_data(self, data_frames):
        """
        Run the strategy with provided data frames
        
        Args:
            data_frames: Dictionary of data frames for each symbol
            
        Returns:
            Dictionary with results
        """
        try:
            logger.info("Running strategy with provided data")
            
            # Initialize results
            results = {
                "portfolio_value": self.cash_limit,
                "cash": self.cash_limit,
                "results": {}
            }
            
            # Process each symbol
            for symbol in self.symbols:
                try:
                    # Skip if we don't have data for this symbol
                    if symbol not in data_frames or data_frames[symbol].empty:
                        logger.warning(f"No data available for {symbol}")
                        results["results"][symbol] = {"error": "No data available"}
                        continue
                    
                    # Get data for this symbol
                    df = data_frames[symbol]
                    
                    # Get current price
                    current_price = df['close'].iloc[-1]
                    
                    # Detect market regime
                    self.market_regimes[symbol] = TechnicalIndicators.detect_market_regime(
                        df['close'], df['high'], df['low'], df['volume']
                    )
                    
                    # Calculate technical signals
                    technical_signals = self.calculate_technical_signals(df, symbol)
                    
                    # Get ML prediction
                    ml_signal = self.get_ml_prediction(df, symbol)
                    
                    # Combine signals
                    combined_signal = self.combine_signals(
                        technical_signals=technical_signals,
                        ml_signal=ml_signal,
                        sentiment_signal=0  # No sentiment analysis yet
                    )
                    
                    # Calculate target position
                    target_position = self.calculate_position_size(
                        symbol, combined_signal, current_price, results["cash"]
                    )
                    
                    # Execute trade
                    trade_result = self.execute_trade_simulation(
                        symbol, target_position, current_price
                    )
                    
                    # Update cash
                    if trade_result.get("trade_executed", False):
                        results["cash"] -= trade_result.get("value", 0)
                    
                    # Add to results
                    results["results"][symbol] = {
                        "technical_signal": technical_signals.get("combined", 0) if technical_signals else 0,
                        "ml_signal": ml_signal,
                        "sentiment_signal": 0,  # No sentiment analysis yet
                        "combined_signal": combined_signal,
                        "target_position": target_position,
                        "current_price": current_price,
                        "market_regime": str(self.market_regimes[symbol]),
                        "trade_result": trade_result
                    }
                    
                    logger.info(f"Processed {symbol}: Signal={combined_signal:.2f}, Target=${target_position:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results["results"][symbol] = {"error": str(e)}
            
            # Calculate portfolio value
            portfolio_value = results["cash"]
            for symbol in self.symbols:
                if symbol in results["results"] and "trade_result" in results["results"][symbol]:
                    trade_result = results["results"][symbol]["trade_result"]
                    if trade_result.get("trade_executed", False):
                        portfolio_value += trade_result.get("quantity", 0) * results["results"][symbol]["current_price"]
            
            results["portfolio_value"] = portfolio_value
            
            return results
            
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return {"error": str(e)}
    
    def execute_trade_simulation(self, symbol, target_position, current_price):
        """
        Simulate trade execution for backtesting
        
        Args:
            symbol: Stock symbol
            target_position: Target position in dollars (positive for long, negative for short)
            current_price: Current price
            
        Returns:
            Dictionary with trade result
        """
        try:
            # Get current position
            current_position = 0  # Assume no position initially
            
            # Calculate target quantity
            target_quantity = int(target_position / current_price) if current_price > 0 else 0
            
            # Calculate quantity to trade
            quantity_to_trade = target_quantity - current_position
            
            # Skip if no trade needed
            if quantity_to_trade == 0:
                return {
                    "trade_executed": False,
                    "reason": "No trade needed"
                }
            
            # Determine trade side
            side = "buy" if quantity_to_trade > 0 else "sell"
            
            # Calculate trade value
            trade_value = abs(quantity_to_trade) * current_price
            
            # Return trade result
            return {
                "trade_executed": True,
                "side": side,
                "quantity": abs(quantity_to_trade),
                "price": current_price,
                "value": trade_value
            }
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {
                "trade_executed": False,
                "error": str(e)
            }

    def generate_mock_data(self, symbol, timeframe, days=60):
        """
        Generate mock price data for testing.
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Timeframe for the data
            days: Number of days of data to generate
            
        Returns:
            DataFrame with mock OHLCV data
        """
        logger.info(f"Using mock data for {symbol}")
        
        # Ensure we have at least 60 days of data for ML prediction (30 required + buffer)
        days = max(days, 60)  # Ensure we have at least 60 days of data
        
        # Calculate date range
        end_date_obj = datetime.now()
        start_date_obj = end_date_obj - timedelta(days=days)
        
        # Generate synthetic data
        return self.generate_synthetic_data(symbol, timeframe, start_date_obj, end_date_obj)

    def generate_synthetic_data(self, symbol, timeframe, start_date, end_date):
        """
        Generate synthetic historical data based on current snapshot
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Timeframe for the data
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with synthetic historical price data
        """
        # Get current price or use a default
        try:
            snapshot = self.get_current_market_data(symbol)
            current_price = snapshot.get("price", 100.0)
        except:
            current_price = 100.0
            
        # Generate dates
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        
        # Generate price data with realistic volatility and trend
        prices = []
        volatility = 0.02  # 2% daily volatility
        trend = 0.0001  # Slight upward trend
        
        # Start with current price and work backwards
        price = current_price
        for i in range(len(dates)):
            # Add random noise
            daily_return = np.random.normal(trend, volatility)
            price = price * (1 + daily_return)  # Work forward
            prices.append(price)
            
        # Generate volume data
        volumes = [int(np.random.normal(10000000, 5000000)) for _ in range(len(dates))]
        
        # Create DataFrame
        df = pd.DataFrame({
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            "close": prices,
            "volume": volumes
        }, index=dates)
        
        return df

    def get_performance_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Need at least 2 data points
            if len(self.performance_history) < 2:
                return {
                    "total_return": 0,
                    "annualized_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "profit_factor": 0
                }
            
            # Extract portfolio values
            timestamps = [pd.to_datetime(entry["timestamp"]) for entry in self.performance_history]
            portfolio_values = [entry["portfolio_value"] for entry in self.performance_history]
            
            # Calculate returns
            portfolio_series = pd.Series(portfolio_values, index=timestamps)
            returns = portfolio_series.pct_change().dropna()
            
            # Skip if no returns
            if len(returns) == 0:
                return {
                    "total_return": 0,
                    "annualized_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "profit_factor": 0
                }
            
            # Calculate metrics
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            
            # Total return
            total_return = ((final_value / initial_value) - 1) * 100
            
            # Annualized return
            days = (timestamps[-1] - timestamps[0]).days
            if days > 0:
                annualized_return = ((final_value / initial_value) ** (365 / days) - 1) * 100
            else:
                annualized_return = 0
            
            # Sharpe ratio
            risk_free_rate = self.config.get("risk_free_rate", 0.02) / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = 0
            if len(excess_returns) > 0 and excess_returns.std() > 0:
                sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1) * 100
            max_drawdown = abs(drawdown.min())
            
            # Win rate and profit factor
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            
            profit_factor = 0
            if len(returns[returns < 0]) > 0 and abs(returns[returns < 0].sum()) > 0:
                profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
            
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate * 100,
                "profit_factor": profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "error": str(e),
                "total_return": 0,
                "annualized_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0
            }

class MockAlpacaAPI:
    def get_account(self):
        return MockAccount()

    def list_positions(self):
        return []

    def get_latest_quote(self, symbol):
        return MockQuote()

    def submit_order(self, symbol, qty, side, type, time_in_force):
        return MockOrder()

    def get_bars(self, symbol, timeframe, start, end):
        return MockBars()

class MockAccount:
    def __init__(self):
        self.id = "MOCK_ACCOUNT"
        self.status = "ACTIVE"
        self.cash = 100000.0
        self.portfolio_value = 100000.0
        self.equity = 100000.0
        self.buying_power = 200000.0

class MockQuote:
    def __init__(self):
        self.ap = 100.0  # Ask price

class MockOrder:
    def __init__(self):
        self.id = "MOCK_ORDER"

class MockBars:
    def __init__(self):
        self.df = pd.DataFrame()
