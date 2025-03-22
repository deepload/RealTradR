"""
Backtesting Module for AI Trading Strategies

This module allows testing trading strategies against historical data
to evaluate performance before deploying with real money.
"""

import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from dotenv import load_dotenv

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


class BacktestEngine:
    """Engine for backtesting trading strategies on historical data"""
    
    def __init__(self, config=None):
        """
        Initialize the backtesting engine
        
        Args:
            config: Configuration dictionary or path to config file
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Extract configuration values
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        self.cash = self.config.get("initial_cash", 100000)
        self.commission = self.config.get("commission", 0.0)
        self.slippage = self.config.get("slippage", 0.0)
        
        # Strategy parameters
        self.short_window = self.config.get("short_window", 10)
        self.long_window = self.config.get("long_window", 30)
        
        # Initialize Alpaca API for historical data
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            ALPACA_API_BASE_URL,
            api_version="v2"
        )
        
        # Initialize portfolio and positions
        self.portfolio = {"cash": self.cash, "equity": self.cash, "total": self.cash}
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        
        logger.info(f"Initialized backtesting engine with symbols: {self.symbols}")
        logger.info(f"Initial cash: ${self.cash}")
        logger.info(f"Strategy parameters: Short MA={self.short_window}, Long MA={self.long_window}")
    
    def _load_config(self, config):
        """Load configuration from file or dictionary"""
        if config is None:
            # Try to load from default location
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                     "strategy_config.json")
            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        logger.info(f"Loaded config from {config_path}")
                        return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        elif isinstance(config, str):
            # Load from specified path
            try:
                with open(config, "r") as f:
                    config = json.load(f)
                    logger.info(f"Loaded config from {config}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config from {config}: {e}")
        
        # Return config if it's a dictionary, otherwise return default config
        if isinstance(config, dict):
            return config
        
        # Default configuration
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "initial_cash": 100000,
            "commission": 0.0,
            "slippage": 0.0,
            "short_window": 10,
            "long_window": 30,
        }
    
    def get_historical_data(self, symbol, start_date, end_date=None):
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with historical data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
        
        try:
            # Get historical data from Alpaca
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment='all'
            ).df
            
            # Reset index to make timestamp a column
            bars = bars.reset_index()
            bars["symbol"] = symbol
            
            logger.info(f"Got {len(bars)} bars for {symbol}")
            return bars
        
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
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
        signal_mask = df.index >= df.index[min(self.short_window, len(df)-1)]
        df.loc[signal_mask, "signal"] = np.where(
            df.loc[signal_mask, "short_ma"] > df.loc[signal_mask, "long_ma"],
            1.0,
            0.0
        )
        
        # Calculate position changes
        df["position"] = df["signal"].diff()
        
        return df
    
    def execute_trade(self, row, symbol):
        """
        Execute a trade in the backtest
        
        Args:
            row: DataFrame row with trade information
            symbol: Symbol to trade
            
        Returns:
            Trade details or None if no trade executed
        """
        timestamp = row["timestamp"]
        price = row["close"]
        position_change = row["position"]
        
        if position_change == 0:
            return None
        
        # Get current position for this symbol
        current_position = self.positions.get(symbol, {"qty": 0, "cost_basis": 0})
        
        # Calculate cash available
        cash_available = self.portfolio["cash"]
        
        if position_change > 0:  # Buy signal
            # Calculate position size (1/5 of available cash per position)
            position_size = min(cash_available * 0.2, cash_available)
            
            if position_size < 100:  # Minimum trade size
                logger.warning(f"{timestamp}: Insufficient cash (${cash_available}) for buy order")
                return None
            
            # Calculate number of shares
            qty = int(position_size / price)
            
            if qty <= 0:
                return None
            
            # Calculate commission and slippage
            commission_cost = qty * price * self.commission
            slippage_cost = qty * price * self.slippage
            total_cost = (qty * price) + commission_cost + slippage_cost
            
            # Update portfolio and position
            self.portfolio["cash"] -= total_cost
            
            # Update position
            if symbol in self.positions:
                # Average down
                total_shares = current_position["qty"] + qty
                total_cost = current_position["cost_basis"] * current_position["qty"] + total_cost
                avg_cost_basis = total_cost / total_shares
                
                self.positions[symbol] = {
                    "qty": total_shares,
                    "cost_basis": avg_cost_basis,
                    "market_value": total_shares * price,
                    "last_price": price
                }
            else:
                # New position
                self.positions[symbol] = {
                    "qty": qty,
                    "cost_basis": total_cost / qty,
                    "market_value": qty * price,
                    "last_price": price
                }
            
            # Record trade
            trade = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "buy",
                "qty": qty,
                "price": price,
                "commission": commission_cost,
                "slippage": slippage_cost,
                "total_cost": total_cost
            }
            
            self.trades.append(trade)
            logger.info(f"{timestamp}: BUY {qty} shares of {symbol} at ${price}")
            
            return trade
        
        elif position_change < 0:  # Sell signal
            if symbol not in self.positions or current_position["qty"] <= 0:
                return None
            
            # Sell all shares
            qty = current_position["qty"]
            
            # Calculate proceeds, commission, and slippage
            gross_proceeds = qty * price
            commission_cost = qty * price * self.commission
            slippage_cost = qty * price * self.slippage
            net_proceeds = gross_proceeds - commission_cost - slippage_cost
            
            # Calculate P&L
            cost_basis = current_position["cost_basis"] * qty
            pnl = net_proceeds - cost_basis
            
            # Update portfolio and position
            self.portfolio["cash"] += net_proceeds
            self.positions.pop(symbol)
            
            # Record trade
            trade = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "sell",
                "qty": qty,
                "price": price,
                "commission": commission_cost,
                "slippage": slippage_cost,
                "net_proceeds": net_proceeds,
                "pnl": pnl,
                "pnl_pct": (pnl / cost_basis) * 100 if cost_basis > 0 else 0
            }
            
            self.trades.append(trade)
            logger.info(f"{timestamp}: SELL {qty} shares of {symbol} at ${price}, P&L: ${pnl:.2f}")
            
            return trade
        
        return None
    
    def update_portfolio_value(self, date, prices):
        """
        Update portfolio value based on current market prices
        
        Args:
            date: Current date
            prices: Dictionary of current prices by symbol
        """
        # Calculate equity value (sum of all positions)
        equity = 0
        for symbol, position in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                position["market_value"] = position["qty"] * price
                position["last_price"] = price
                equity += position["market_value"]
        
        # Update portfolio values
        prev_total = self.portfolio["total"]
        self.portfolio["equity"] = equity
        self.portfolio["total"] = self.portfolio["cash"] + equity
        
        # Calculate daily return
        daily_return = (self.portfolio["total"] / prev_total) - 1 if prev_total > 0 else 0
        
        # Record daily portfolio value
        daily_value = {
            "date": date,
            "cash": self.portfolio["cash"],
            "equity": equity,
            "total": self.portfolio["total"],
            "daily_return": daily_return
        }
        
        self.daily_returns.append(daily_value)
    
    def run_backtest(self, start_date, end_date=None):
        """
        Run backtest for all symbols from start_date to end_date
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD), defaults to today
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date or 'today'}")
        
        # Initialize backtest
        self.portfolio = {"cash": self.cash, "equity": 0, "total": self.cash}
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        
        # Get historical data for all symbols
        all_data = []
        for symbol in self.symbols:
            data = self.get_historical_data(symbol, start_date, end_date)
            if not data.empty:
                all_data.append(data)
        
        if not all_data:
            logger.error("No historical data available for backtest")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data)
        
        # Sort by date
        combined_data = combined_data.sort_values(by="timestamp")
        
        # Group by date to get prices for all symbols on each date
        dates = combined_data["timestamp"].dt.date.unique()
        
        # Process each date
        for date in dates:
            # Get data for this date
            date_data = combined_data[combined_data["timestamp"].dt.date == date]
            
            # Create price dictionary for this date
            prices = {row["symbol"]: row["close"] for _, row in date_data.iterrows()}
            
            # Process each symbol
            for symbol in self.symbols:
                symbol_data = date_data[date_data["symbol"] == symbol]
                
                if not symbol_data.empty:
                    # Calculate signals for this symbol
                    historical_data = combined_data[
                        (combined_data["symbol"] == symbol) & 
                        (combined_data["timestamp"].dt.date <= date)
                    ].copy()
                    
                    if len(historical_data) >= self.long_window:
                        signals = self.calculate_signals(historical_data)
                        
                        # Get the last row (current date)
                        current_row = signals.iloc[-1]
                        
                        # Execute trade based on signal
                        self.execute_trade(current_row, symbol)
            
            # Update portfolio value
            self.update_portfolio_value(date, prices)
        
        # Calculate backtest metrics
        metrics = self.calculate_metrics()
        
        logger.info(f"Backtest completed with final portfolio value: ${self.portfolio['total']:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Annualized return: {metrics['annualized_return']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        return {
            "portfolio": self.portfolio,
            "trades": self.trades,
            "daily_returns": self.daily_returns,
            "metrics": metrics
        }
    
    def calculate_metrics(self):
        """
        Calculate performance metrics from backtest results
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.daily_returns:
            return {}
        
        # Create DataFrame from daily returns
        df = pd.DataFrame(self.daily_returns)
        
        # Calculate metrics
        initial_value = self.cash
        final_value = self.portfolio["total"]
        
        # Total return
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Number of days in backtest
        days = len(df)
        years = days / 252  # Trading days in a year
        
        # Annualized return
        annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Daily returns
        daily_returns = df["daily_return"].values
        
        # Volatility (annualized)
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate of 0%)
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Maximum drawdown
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod()
        df["cumulative_max"] = df["cumulative_return"].cummax()
        df["drawdown"] = (df["cumulative_return"] / df["cumulative_max"]) - 1
        max_drawdown = df["drawdown"].min() * 100
        
        # Winning trades
        winning_trades = [t for t in self.trades if t.get("side") == "sell" and t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("side") == "sell" and t.get("pnl", 0) <= 0]
        
        win_rate = len(winning_trades) / len([t for t in self.trades if t.get("side") == "sell"]) * 100 if len([t for t in self.trades if t.get("side") == "sell"]) > 0 else 0
        
        # Average winning and losing trade
        avg_win = np.mean([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum([t.get("pnl", 0) for t in winning_trades])
        gross_loss = abs(sum([t.get("pnl", 0) for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "num_trades": len([t for t in self.trades if t.get("side") == "sell"]),
            "trading_days": days
        }
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results
        
        Args:
            save_path: Path to save the plot, if None, plot will be displayed
            
        Returns:
            None
        """
        if not self.daily_returns:
            logger.error("No backtest results to plot")
            return
        
        # Create DataFrame from daily returns
        df = pd.DataFrame(self.daily_returns)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot portfolio value
        df[["cash", "equity", "total"]].plot(ax=axes[0], title="Portfolio Value")
        axes[0].set_ylabel("Value ($)")
        axes[0].legend(["Cash", "Equity", "Total"])
        axes[0].grid(True)
        
        # Plot cumulative returns
        cum_returns = (1 + df["daily_return"]).cumprod() - 1
        cum_returns.plot(ax=axes[1], title="Cumulative Returns", color="green")
        axes[1].set_ylabel("Return (%)")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        axes[1].grid(True)
        
        # Plot drawdowns
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod()
        df["cumulative_max"] = df["cumulative_return"].cummax()
        df["drawdown"] = (df["cumulative_return"] / df["cumulative_max"]) - 1
        df["drawdown"].plot(ax=axes[2], title="Drawdowns", color="red")
        axes[2].set_ylabel("Drawdown (%)")
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        axes[2].grid(True)
        
        # Add metrics as text
        metrics = self.calculate_metrics()
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2f}%\n"
            f"Annualized Return: {metrics['annualized_return']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']:.2f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Number of Trades: {metrics['num_trades']}"
        )
        fig.text(0.01, 0.01, metrics_text, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to run the backtest"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)", default="2024-01-01")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--plot", help="Path to save plot", default="backtest_results.png")
    
    args = parser.parse_args()
    
    # Create backtest engine
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    
    engine = BacktestEngine(config)
    
    # Run backtest
    results = engine.run_backtest(args.start, args.end)
    
    if results:
        # Plot results
        engine.plot_results(args.plot)


if __name__ == "__main__":
    main()
