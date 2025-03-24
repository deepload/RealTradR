"""
Backtesting Script for Advanced AI Trading Strategy

This script runs a comprehensive backtest of the advanced AI trading strategy
against historical data and generates performance reports.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.ai.advanced_strategy import AdvancedAIStrategy, load_strategy_config
from backend.app.ai.risk_management import RiskManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Engine for backtesting the advanced AI trading strategy"""
    
    def __init__(self, config=None, start_date=None, end_date=None, initial_capital=100000.0):
        """
        Initialize the backtest engine
        
        Args:
            config: Strategy configuration (default: load from file)
            start_date: Start date for backtest (default: 1 year ago)
            end_date: End date for backtest (default: today)
            initial_capital: Initial capital for backtest
        """
        # Load configuration
        self.config = config or load_strategy_config()
        
        # Set backtest parameters
        self.backtest_settings = self.config.get("backtest_settings", {})
        
        # Set dates
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            start_date_str = self.backtest_settings.get("start_date")
            if start_date_str:
                self.start_date = pd.to_datetime(start_date_str)
            else:
                self.start_date = datetime.now() - timedelta(days=365)
        
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            end_date_str = self.backtest_settings.get("end_date")
            if end_date_str:
                self.end_date = pd.to_datetime(end_date_str)
            else:
                self.end_date = datetime.now()
        
        # Set initial capital
        self.initial_capital = self.backtest_settings.get("initial_capital", initial_capital)
        
        # Set transaction costs
        self.commission = self.backtest_settings.get("commission", 0.001)  # 0.1% commission
        self.slippage = self.backtest_settings.get("slippage", 0.001)      # 0.1% slippage
        
        # Initialize strategy
        self.strategy = AdvancedAIStrategy(config=self.config)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            stop_loss_pct=self.config.get("stop_loss_pct", 2.0),
            take_profit_pct=self.config.get("take_profit_pct", 5.0),
            max_drawdown_pct=self.config.get("max_drawdown_pct", 25.0),
            risk_free_rate=self.config.get("risk_free_rate", 0.02),
            max_correlation=self.config.get("max_correlation", 0.7)
        )
        
        # Initialize portfolio and tracking variables
        self.reset()
        
        logger.info(f"Initialized backtest from {self.start_date.strftime('%Y-%m-%d')} "
                    f"to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")
        logger.info(f"Transaction costs: Commission={self.commission*100:.2f}%, "
                    f"Slippage={self.slippage*100:.2f}%")
    
    def reset(self):
        """Reset the backtest state"""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: {'shares': shares, 'cost_basis': cost_basis}}
        self.trades = []
        self.portfolio_history = []
        self.equity_curve = []
        self.daily_returns = []
        self.peak_equity = self.initial_capital
    
    def load_historical_data(self, symbols=None, timeframe="1D"):
        """
        Load historical data for backtesting
        
        Args:
            symbols: List of symbols to load data for (default: from config)
            timeframe: Time frame for data (default: "1D")
            
        Returns:
            Dictionary of DataFrames with historical data for each symbol
        """
        symbols = symbols or self.config.get("symbols", ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
        
        logger.info(f"Loading historical data for {len(symbols)} symbols: {symbols}")
        
        historical_data = {}
        
        for symbol in symbols:
            try:
                # Use the strategy's get_historical_data method
                # In a real implementation, this would fetch from a database or API
                # For backtesting, we'll use a mock implementation
                df = self._mock_historical_data(symbol, timeframe)
                
                if not df.empty:
                    historical_data[symbol] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return historical_data
    
    def _mock_historical_data(self, symbol, timeframe):
        """
        Mock implementation to generate historical data for backtesting
        
        In a real implementation, this would fetch from a database or API
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Time frame for data
            
        Returns:
            DataFrame with historical data
        """
        # Generate date range
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='B'  # Business days
        )
        
        # Base price depends on symbol
        if symbol == "AAPL":
            base_price = 150.0
            volatility = 2.0
        elif symbol == "MSFT":
            base_price = 300.0
            volatility = 3.0
        elif symbol == "GOOGL":
            base_price = 2000.0
            volatility = 20.0
        elif symbol == "AMZN":
            base_price = 100.0
            volatility = 2.0
        elif symbol == "META":
            base_price = 250.0
            volatility = 5.0
        else:
            base_price = 100.0
            volatility = 2.0
        
        # Generate random walk prices
        n = len(date_range)
        
        # Start with base price
        closes = [base_price]
        
        # Add random daily changes
        for i in range(1, n):
            # Random daily return with slight upward bias
            daily_return = np.random.normal(0.0005, 0.015)  # Mean: 0.05% daily, Std: 1.5%
            new_price = closes[-1] * (1 + daily_return)
            closes.append(new_price)
        
        closes = np.array(closes)
        
        # Generate OHLC data
        highs = closes * (1 + np.random.uniform(0, 0.02, n))
        lows = closes * (1 - np.random.uniform(0, 0.02, n))
        opens = lows + np.random.uniform(0, 1, n) * (highs - lows)
        
        # Generate volume
        volume = np.random.randint(1000000, 10000000, n)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume
        })
        
        return df
    
    def execute_trade(self, symbol, price, shares, side, timestamp):
        """
        Execute a trade in the backtest
        
        Args:
            symbol: Symbol to trade
            price: Price to trade at
            shares: Number of shares to trade
            side: 'buy' or 'sell'
            timestamp: Timestamp of the trade
            
        Returns:
            Dictionary with trade information
        """
        # Apply slippage
        if side == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate trade value
        trade_value = execution_price * shares
        
        # Apply commission
        commission_cost = trade_value * self.commission
        
        # Calculate total cost
        total_cost = trade_value + commission_cost if side == 'buy' else trade_value - commission_cost
        
        # Update cash
        if side == 'buy':
            self.cash -= total_cost
        else:
            self.cash += total_cost
        
        # Update positions
        if side == 'buy':
            if symbol in self.positions:
                # Update existing position
                current_shares = self.positions[symbol]['shares']
                current_cost = self.positions[symbol]['cost_basis'] * current_shares
                new_shares = current_shares + shares
                new_cost = current_cost + trade_value
                self.positions[symbol] = {
                    'shares': new_shares,
                    'cost_basis': new_cost / new_shares
                }
            else:
                # Create new position
                self.positions[symbol] = {
                    'shares': shares,
                    'cost_basis': execution_price
                }
        else:
            if symbol in self.positions:
                # Update existing position
                current_shares = self.positions[symbol]['shares']
                new_shares = current_shares - shares
                
                if new_shares <= 0:
                    # Position closed
                    del self.positions[symbol]
                else:
                    # Position reduced
                    self.positions[symbol]['shares'] = new_shares
            else:
                # Short position (not implemented in this simple backtest)
                logger.warning(f"Attempted to sell {symbol} without a position")
        
        # Record the trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'price': execution_price,
            'value': trade_value,
            'commission': commission_cost,
            'total_cost': total_cost
        }
        
        self.trades.append(trade)
        
        return trade
    
    def update_portfolio_value(self, prices, timestamp):
        """
        Update portfolio value based on current prices
        
        Args:
            prices: Dictionary of current prices {symbol: price}
            timestamp: Current timestamp
            
        Returns:
            Current portfolio value
        """
        # Calculate value of positions
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                positions_value += position['shares'] * prices[symbol]
        
        # Calculate total portfolio value
        self.portfolio_value = self.cash + positions_value
        
        # Update peak equity
        if self.portfolio_value > self.peak_equity:
            self.peak_equity = self.portfolio_value
        
        # Calculate daily return
        if self.equity_curve:
            daily_return = (self.portfolio_value / self.equity_curve[-1]['value']) - 1
        else:
            daily_return = 0
        
        # Record portfolio history
        portfolio_snapshot = {
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': positions_value,
            'value': self.portfolio_value,
            'daily_return': daily_return
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        self.equity_curve.append(portfolio_snapshot)
        self.daily_returns.append(daily_return)
        
        return self.portfolio_value
    
    def run_backtest(self):
        """
        Run the backtest
        
        Returns:
            Dictionary with backtest results
        """
        # Reset the backtest state
        self.reset()
        
        # Load historical data
        historical_data = self.load_historical_data()
        
        if not historical_data:
            logger.error("No historical data available for backtesting")
            return {"error": "No historical data available"}
        
        # Get common date range across all symbols
        common_dates = None
        
        for symbol, df in historical_data.items():
            dates = pd.to_datetime(df['timestamp']).dt.date
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates = common_dates.intersection(set(dates))
        
        common_dates = sorted(list(common_dates))
        
        logger.info(f"Running backtest over {len(common_dates)} trading days")
        
        # Run the backtest day by day
        for date in common_dates:
            # Get prices for this date
            prices = {}
            for symbol, df in historical_data.items():
                day_data = df[pd.to_datetime(df['timestamp']).dt.date == date]
                if not day_data.empty:
                    prices[symbol] = day_data['close'].iloc[-1]
            
            # Update portfolio value
            self.update_portfolio_value(prices, pd.Timestamp(date))
            
            # Check if we need to reduce position sizes due to drawdown
            if self.risk_manager.check_drawdown_limit(self.portfolio_value, self.peak_equity):
                logger.warning(f"Drawdown limit exceeded on {date}: "
                              f"${self.portfolio_value:.2f} vs peak ${self.peak_equity:.2f}")
                
                # Reduce all positions by 50%
                for symbol, position in list(self.positions.items()):
                    if symbol in prices:
                        shares_to_sell = position['shares'] // 2
                        if shares_to_sell > 0:
                            self.execute_trade(
                                symbol, prices[symbol], shares_to_sell, 'sell', pd.Timestamp(date)
                            )
                            logger.info(f"Reduced position in {symbol} by 50% due to drawdown")
            
            # Generate signals and execute trades
            for symbol, price in prices.items():
                # Get historical data for this symbol up to this date
                symbol_data = historical_data[symbol].copy()
                symbol_data = symbol_data[pd.to_datetime(symbol_data['timestamp']).dt.date <= date]
                
                if len(symbol_data) < 30:
                    # Not enough data for analysis
                    continue
                
                # Calculate technical signals
                technical_signal = self.strategy.get_technical_signals(symbol_data)
                
                # For backtesting, we'll use a simplified sentiment signal
                # In a real implementation, this would use historical sentiment data
                sentiment_signal = {'sentiment_signal': 0}
                
                # For backtesting, we'll use a simplified ML signal
                # In a real implementation, this would use historical ML predictions
                ml_signal = {'ml_signal': 0}
                
                # Combine signals
                combined_signal = self.strategy.combine_signals(
                    technical_signal, sentiment_signal, ml_signal
                )
                
                # Calculate target position size
                target_position_size = self.strategy.calculate_position_size(
                    symbol, combined_signal, self.portfolio_value
                )
                
                # Calculate current position size
                current_position_size = 0
                if symbol in self.positions:
                    current_position_size = self.positions[symbol]['shares'] * price
                
                # Calculate difference
                diff = target_position_size - current_position_size
                
                # Skip small adjustments
                if abs(diff) < 1000:
                    continue
                
                # Execute trade
                if diff > 0:
                    # Buy
                    shares_to_buy = int(diff / price)
                    if shares_to_buy > 0 and self.cash >= shares_to_buy * price * (1 + self.commission):
                        self.execute_trade(symbol, price, shares_to_buy, 'buy', pd.Timestamp(date))
                        logger.info(f"Bought {shares_to_buy} shares of {symbol} at ${price:.2f}")
                else:
                    # Sell
                    shares_to_sell = int(abs(diff) / price)
                    if shares_to_sell > 0 and symbol in self.positions and self.positions[symbol]['shares'] >= shares_to_sell:
                        self.execute_trade(symbol, price, shares_to_sell, 'sell', pd.Timestamp(date))
                        logger.info(f"Sold {shares_to_sell} shares of {symbol} at ${price:.2f}")
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        logger.info(f"Backtest completed: Final portfolio value: ${self.portfolio_value:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        return {
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio_value,
            "trades": self.trades,
            "portfolio_history": self.portfolio_history,
            "metrics": metrics
        }
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the backtest
        
        Returns:
            Dictionary with performance metrics
        """
        # Extract equity values
        equity_values = [snapshot['value'] for snapshot in self.equity_curve]
        
        # Calculate total return
        total_return = ((self.portfolio_value / self.initial_capital) - 1) * 100
        
        # Calculate annualized return
        trading_days = len(self.equity_curve)
        if trading_days > 1:
            annualized_return = ((1 + total_return/100) ** (252/trading_days) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio
        risk_free_rate = self.config.get("risk_free_rate", 0.02) / 252  # Daily risk-free rate
        excess_returns = [r - risk_free_rate for r in self.daily_returns]
        sharpe_ratio = 0
        
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = [r for r in excess_returns if r < 0]
        sortino_ratio = 0
        
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        
        # Calculate maximum drawdown
        max_drawdown = self.risk_manager.calculate_max_drawdown(equity_values)
        
        # Calculate win rate and profit factor
        win_count = 0
        loss_count = 0
        win_amount = 0
        loss_amount = 0
        
        for trade in self.trades:
            if trade['side'] == 'buy':
                cost = trade['total_cost']
            else:
                # For sells, we need to calculate profit/loss
                # This is simplified and doesn't account for partial position closes
                cost = -trade['total_cost']
                
                # Find matching buy trade
                symbol = trade['symbol']
                shares = trade['shares']
                buy_cost = 0
                
                for buy_trade in [t for t in self.trades if t['side'] == 'buy' and t['symbol'] == symbol]:
                    buy_cost += buy_trade['price'] * shares
                
                # Calculate profit/loss
                profit = cost - buy_cost
                
                if profit > 0:
                    win_count += 1
                    win_amount += profit
                else:
                    loss_count += 1
                    loss_amount += abs(profit)
        
        # Calculate win rate
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        profit_factor = win_amount / loss_amount if loss_amount > 0 else float('inf')
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate * 100,  # Convert to percentage
            "profit_factor": profit_factor,
            "total_trades": total_trades
        }
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot the equity curve
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            None
        """
        if not self.equity_curve:
            logger.error("No equity curve data available")
            return
        
        # Extract data
        dates = [snapshot['timestamp'] for snapshot in self.equity_curve]
        equity = [snapshot['value'] for snapshot in self.equity_curve]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Backtest Equity Curve')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_dir=None):
        """
        Generate a backtest report
        
        Args:
            output_dir: Directory to save the report (optional)
            
        Returns:
            None
        """
        if not self.equity_curve:
            logger.error("No equity curve data available")
            return
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        # Generate report
        report = {
            "backtest_summary": {
                "start_date": self.start_date.strftime('%Y-%m-%d'),
                "end_date": self.end_date.strftime('%Y-%m-%d'),
                "initial_capital": self.initial_capital,
                "final_value": self.portfolio_value,
                "total_return": metrics['total_return'],
                "annualized_return": metrics['annualized_return'],
                "sharpe_ratio": metrics['sharpe_ratio'],
                "sortino_ratio": metrics['sortino_ratio'],
                "max_drawdown": metrics['max_drawdown'],
                "win_rate": metrics['win_rate'],
                "profit_factor": metrics['profit_factor'],
                "total_trades": metrics['total_trades']
            },
            "strategy_config": self.config,
            "trades": self.trades[:10]  # Include only first 10 trades for brevity
        }
        
        # Save report
        if output_dir:
            report_path = os.path.join(output_dir, 'backtest_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Backtest report saved to {report_path}")
            
            # Generate equity curve plot
            plot_path = os.path.join(output_dir, 'equity_curve.png')
            self.plot_equity_curve(save_path=plot_path)
        
        # Print summary
        print("\n=== Backtest Summary ===")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Value: ${self.portfolio_value:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print("========================\n")


def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='Run a backtest of the advanced AI trading strategy')
    
    parser.add_argument('--config', type=str, help='Path to strategy configuration file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--output', type=str, help='Output directory for report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return
    
    # Create backtest engine
    engine = BacktestEngine(
        config=config,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )
    
    # Run backtest
    engine.run_backtest()
    
    # Generate report
    engine.generate_report(output_dir=args.output)
    
    # Plot equity curve
    if not args.output:
        engine.plot_equity_curve()


if __name__ == '__main__':
    main()
