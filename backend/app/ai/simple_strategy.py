"""
Simple AI Trading Strategy

This module implements a basic moving average crossover strategy using Alpaca's paper
trading API. It uses real market data to make trading decisions.
"""

import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKFK3EAPXA9D9CXZ33JD")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET","nqGS1mnSKFYfkE6pLvZhYNZizurzAMCGcWEDnLR3")
ALPACA_API_BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"

# Load strategy configuration
def load_strategy_config():
    """Load strategy configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                              "strategy_config.json")
    
    default_config = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "cash_limit": 10000,
        "short_window": 10,
        "long_window": 30,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
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


class MovingAverageCrossover:
    """Moving Average Crossover Strategy for single symbol real-time trading"""
    
    def __init__(self, symbol, short_window=10, long_window=30, alpaca_api=None):
        """
        Initialize the strategy for a single symbol
        
        Args:
            symbol: Stock symbol to trade
            short_window: Short moving average window length
            long_window: Long moving average window length
            alpaca_api: Alpaca API instance (optional)
        """
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        
        # Initialize Alpaca API if not provided
        if alpaca_api is None:
            self.api = tradeapi.REST(
                ALPACA_API_KEY,
                ALPACA_API_SECRET,
                ALPACA_API_BASE_URL,
                api_version="v2"
            )
        else:
            self.api = alpaca_api
            
        # Track last signal to avoid duplicate trades
        self.last_signal = None
        self.last_position = 0
        
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} with parameters: "
                   f"Short MA={short_window}, Long MA={long_window}")
    
    def get_historical_data(self, days=40):
        """
        Get historical data for the symbol
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Getting historical data for {self.symbol} from {start_date} to {end_date}")
            
            # Get historical data from Alpaca
            bars = self.api.get_bars(
                self.symbol,
                TimeFrame.Day,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            ).df
            
            # Reset index to make timestamp a column
            bars = bars.reset_index()
            
            logger.info(f"Got {len(bars)} bars for {self.symbol}")
            
            return bars
        except Exception as e:
            logger.error(f"Error getting historical data for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_signals(self, df):
        """
        Calculate trading signals based on moving average crossover
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            DataFrame with signals
        """
        try:
            # Create a copy of the dataframe
            df = df.copy()
            
            # Calculate moving averages
            df["short_ma"] = df["close"].rolling(window=self.short_window, min_periods=1).mean()
            df["long_ma"] = df["close"].rolling(window=self.long_window, min_periods=1).mean()
            
            # Calculate signals
            df["signal"] = 0.0
            signal_mask = df.index >= df.index[min(self.short_window, len(df)-1)]
            df.loc[signal_mask, "signal"] = np.where(
                df.loc[signal_mask, "short_ma"] > df.loc[signal_mask, "long_ma"],
                1.0,
                0.0
            )
            
            # Calculate position changes
            df["position"] = df["signal"].diff()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating signals for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self):
        """Get the current price of the symbol"""
        try:
            trade = self.api.get_latest_trade(self.symbol)
            return float(trade.price)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {e}")
            return None
    
    def check_signals(self):
        """
        Check for trading signals based on current market data
        
        Returns:
            Signal string: 'buy', 'sell', or None
        """
        try:
            # Get historical data
            data = self.get_historical_data(days=max(40, self.long_window * 2))
            
            if data.empty:
                logger.error(f"No historical data available for {self.symbol}")
                return None
            
            # Calculate signals
            signals = self.calculate_signals(data)
            
            if signals.empty:
                logger.error(f"Error calculating signals for {self.symbol}")
                return None
            
            # Get the latest signal
            latest_signal = signals.iloc[-1]
            position_change = latest_signal["position"]
            
            # Check for position change
            if position_change > 0:
                # Buy signal
                if self.last_position == 1:
                    # Already have a buy signal, no need to buy again
                    logger.info(f"Already have buy signal for {self.symbol}")
                    return None
                
                logger.info(f"Buy signal for {self.symbol} at {latest_signal['close']}")
                self.last_position = 1
                self.last_signal = "buy"
                return "buy"
                
            elif position_change < 0:
                # Sell signal
                if self.last_position == 0:
                    # Already have a sell signal, no need to sell again
                    logger.info(f"Already have sell signal for {self.symbol}")
                    return None
                
                logger.info(f"Sell signal for {self.symbol} at {latest_signal['close']}")
                self.last_position = 0
                self.last_signal = "sell"
                return "sell"
            
            # No position change
            return None
        
        except Exception as e:
            logger.error(f"Error checking signals for {self.symbol}: {e}")
            return None
    
    def get_performance_metrics(self, days=90):
        """
        Calculate performance metrics for this strategy
        
        Args:
            days: Number of days to lookback
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Get historical data
            data = self.get_historical_data(days=days)
            
            if data.empty:
                logger.error(f"No historical data available for {self.symbol}")
                return {}
            
            # Calculate signals
            signals = self.calculate_signals(data)
            
            if signals.empty:
                logger.error(f"Error calculating signals for {self.symbol}")
                return {}
            
            # Calculate returns
            signals["returns"] = signals["close"].pct_change()
            
            # Calculate strategy returns
            signals["strategy_returns"] = signals["signal"].shift(1) * signals["returns"]
            
            # Calculate cumulative returns
            signals["cumulative_returns"] = (1 + signals["returns"]).cumprod() - 1
            signals["cumulative_strategy_returns"] = (1 + signals["strategy_returns"]).cumprod() - 1
            
            # Calculate metrics
            total_return = signals["cumulative_strategy_returns"].iloc[-1]
            buy_hold_return = signals["cumulative_returns"].iloc[-1]
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = signals["strategy_returns"].mean() / signals["strategy_returns"].std() * np.sqrt(252)
            
            # Calculate maximum drawdown
            cumulative = (1 + signals["strategy_returns"]).cumprod()
            max_value = cumulative.cummax()
            drawdown = (cumulative / max_value) - 1
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            winning_days = signals[signals["strategy_returns"] > 0]
            win_rate = len(winning_days) / len(signals[signals["strategy_returns"] != 0]) if len(signals[signals["strategy_returns"] != 0]) > 0 else 0
            
            return {
                "symbol": self.symbol,
                "total_return": total_return * 100,
                "buy_hold_return": buy_hold_return * 100,
                "outperformance": (total_return - buy_hold_return) * 100,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown * 100,
                "win_rate": win_rate * 100,
                "trades": len(signals[signals["position"] != 0])
            }
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {self.symbol}: {e}")
            return {}


class SimpleMAStrategy:
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, config=None):
        """
        Initialize the strategy
        
        Args:
            config: Strategy configuration (default: load from file)
        """
        # Load configuration
        self.config = config or load_strategy_config()
        
        # Extract configuration values
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        self.cash_limit = self.config.get("cash_limit", 10000)
        self.short_window = self.config.get("short_window", 10)
        self.long_window = self.config.get("long_window", 30)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 2.0)
        self.take_profit_pct = self.config.get("take_profit_pct", 5.0)
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            ALPACA_API_BASE_URL,
            api_version="v2"
        )
        
        logger.info(f"Initialized strategy with symbols: {self.symbols}")
        logger.info(f"Cash limit: ${self.cash_limit}")
        logger.info(f"Paper trading: {ALPACA_PAPER_TRADING}")
        logger.info(f"Strategy parameters: Short MA={self.short_window}, Long MA={self.long_window}")
    
    def get_historical_data(self, symbol, days=40):
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
        
        # Get historical data from Alpaca
        bars = self.api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        ).df
        
        # Reset index to make timestamp a column
        bars = bars.reset_index()
        
        logger.info(f"Got {len(bars)} bars for {symbol}")
        
        return bars
    
    def calculate_signals(self, df):
        """
        Calculate trading signals based on moving average crossover
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            DataFrame with signals
        """
        # Create a copy of the dataframe
        df = df.copy()
        
        # Calculate moving averages
        df["short_ma"] = df["close"].rolling(window=self.short_window, min_periods=1).mean()
        df["long_ma"] = df["close"].rolling(window=self.long_window, min_periods=1).mean()
        
        # Calculate signals
        df["signal"] = 0.0
        df["signal"][self.short_window:] = np.where(
            df["short_ma"][self.short_window:] > df["long_ma"][self.short_window:],
            1.0,
            0.0
        )
        
        # Calculate position changes
        df["position"] = df["signal"].diff()
        
        return df
    
    def execute_trades(self, symbol, position_change):
        """
        Execute trades based on position changes
        
        Args:
            symbol: Stock symbol
            position_change: Position change signal (1 for buy, -1 for sell, 0 for hold)
            
        Returns:
            Order information if trade executed, None otherwise
        """
        # Get account information
        account = self.api.get_account()
        buying_power = float(account.buying_power)
        current_positions = {p.symbol: p for p in self.api.list_positions()}
        
        # Check if we have enough buying power
        if buying_power < 100:
            logger.warning(f"Insufficient buying power: ${buying_power}")
            return None
        
        # Calculate position size (1/3 of available cash or max based on config)
        position_size = min(buying_power / 3, self.cash_limit)
        
        # Get latest price
        latest_price = float(self.api.get_latest_trade(symbol).price)
        
        # Calculate number of shares
        qty = int(position_size / latest_price)
        
        # Execute trade based on position change
        if position_change > 0:  # Buy signal
            if symbol in current_positions:
                logger.info(f"Already have position in {symbol}, not buying more")
                return None
            
            if qty <= 0:
                logger.warning(f"Quantity {qty} too small to buy {symbol}")
                return None
            
            # Place buy order
            logger.info(f"Buying {qty} shares of {symbol} at ${latest_price}")
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"Buy order placed: {order.id}")
                
                # Set stop loss and take profit if enabled
                if self.stop_loss_pct > 0:
                    stop_price = round(latest_price * (1 - self.stop_loss_pct / 100), 2)
                    logger.info(f"Setting stop loss for {symbol} at ${stop_price}")
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="stop",
                        time_in_force="gtc",
                        stop_price=stop_price
                    )
                
                if self.take_profit_pct > 0:
                    limit_price = round(latest_price * (1 + self.take_profit_pct / 100), 2)
                    logger.info(f"Setting take profit for {symbol} at ${limit_price}")
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="limit",
                        time_in_force="gtc",
                        limit_price=limit_price
                    )
                
                return order
            except Exception as e:
                logger.error(f"Error placing buy order: {e}")
                return None
        
        elif position_change < 0:  # Sell signal
            if symbol not in current_positions:
                logger.info(f"No position in {symbol} to sell")
                return None
            
            # Get current position
            position = current_positions[symbol]
            qty = abs(int(float(position.qty)))
            
            if qty <= 0:
                logger.warning(f"Quantity {qty} too small to sell {symbol}")
                return None
            
            # Place sell order
            logger.info(f"Selling {qty} shares of {symbol} at ${latest_price}")
            try:
                # Cancel any existing orders for this symbol
                orders = self.api.list_orders(symbol=symbol, status="open")
                for order in orders:
                    logger.info(f"Canceling existing order {order.id} for {symbol}")
                    self.api.cancel_order(order.id)
                
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"Sell order placed: {order.id}")
                return order
            except Exception as e:
                logger.error(f"Error placing sell order: {e}")
                return None
        
        return None
    
    def run(self):
        """Run the strategy once"""
        logger.info("Running strategy...")
        
        # Check if market is open
        clock = self.api.get_clock()
        if not clock.is_open:
            next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Market is closed. Next open: {next_open}")
            return
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                # Get historical data
                data = self.get_historical_data(symbol)
                
                # Calculate signals
                data = self.calculate_signals(data)
                
                # Check if we have a new signal
                if len(data) > 0:
                    latest = data.iloc[-1]
                    position_change = latest["position"]
                    
                    if position_change != 0:
                        # Execute trade
                        logger.info(f"Signal for {symbol}: {position_change}")
                        self.execute_trades(symbol, position_change)
                    else:
                        logger.info(f"No new signal for {symbol}")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info("Strategy run completed")
    
    def get_portfolio_value(self):
        """Get current portfolio value"""
        account = self.api.get_account()
        return float(account.portfolio_value)
    
    def get_positions(self):
        """Get current positions"""
        return self.api.list_positions()
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        initial_value = float(account.cash)  # Assuming cash is the initial value
        
        # Calculate profit/loss
        pnl = portfolio_value - initial_value
        pnl_pct = (pnl / initial_value) * 100 if initial_value > 0 else 0
        
        # Get positions
        positions = self.api.list_positions()
        num_positions = len(positions)
        
        # Calculate position metrics
        position_metrics = []
        for position in positions:
            symbol = position.symbol
            qty = int(float(position.qty))
            avg_entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price)
            market_value = float(position.market_value)
            cost_basis = float(position.cost_basis)
            
            # Calculate position P&L
            position_pnl = market_value - cost_basis
            position_pnl_pct = (position_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            position_metrics.append({
                "symbol": symbol,
                "qty": qty,
                "avg_entry_price": avg_entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "pnl": position_pnl,
                "pnl_pct": position_pnl_pct
            })
        
        # Return performance metrics
        return {
            "portfolio_value": portfolio_value,
            "initial_value": initial_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "num_positions": num_positions,
            "positions": position_metrics
        }


def main():
    """Main function to run the strategy"""
    # Create strategy instance
    strategy = SimpleMAStrategy()
    
    # Get initial portfolio value
    initial_value = strategy.get_portfolio_value()
    logger.info(f"Initial portfolio value: ${initial_value}")
    
    # Run the strategy once
    strategy.run()
    
    # Get final portfolio value
    final_value = strategy.get_portfolio_value()
    logger.info(f"Final portfolio value: ${final_value}")
    
    # Calculate performance
    performance = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
    logger.info(f"Performance: {performance:.2f}%")
    
    # Get current positions
    positions = strategy.get_positions()
    logger.info(f"Current positions: {len(positions)}")
    for position in positions:
        logger.info(f"  {position.symbol}: {position.qty} shares, Market value: ${position.market_value}")
    
    # Get detailed performance metrics
    metrics = strategy.get_performance_metrics()
    logger.info("Performance metrics:")
    logger.info(f"  Portfolio value: ${metrics['portfolio_value']:.2f}")
    logger.info(f"  P&L: ${metrics['pnl']:.2f} ({metrics['pnl_pct']:.2f}%)")
    logger.info(f"  Number of positions: {metrics['num_positions']}")


if __name__ == "__main__":
    main()
