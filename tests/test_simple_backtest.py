"""
Simple Backtest Script for RealTradR

This script runs a simplified backtest that doesn't rely on ML models
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.ai.risk_management import RiskManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleBacktest:
    """Simple backtest engine that focuses on risk management"""
    
    def __init__(self, start_date=None, end_date=None, initial_capital=100000.0):
        """
        Initialize the backtest engine
        
        Args:
            start_date: Start date for backtest (default: 1 year ago)
            end_date: End date for backtest (default: today)
            initial_capital: Initial capital for backtest
        """
        # Set dates
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = datetime.now() - timedelta(days=365)
        
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = datetime.now()
        
        # Set initial capital
        self.initial_capital = initial_capital
        
        # Set transaction costs
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.001    # 0.1% slippage
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            max_drawdown_pct=25.0,
            risk_free_rate=0.02,
            max_correlation=0.7
        )
        
        # Initialize portfolio and tracking variables
        self.reset()
        
        logger.info(f"Initialized backtest from {self.start_date.strftime('%Y-%m-%d')} "
                    f"to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")
    
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
    
    def generate_mock_data(self, symbols=None):
        """
        Generate mock price data for backtesting
        
        Args:
            symbols: List of symbols to generate data for
            
        Returns:
            Dictionary of DataFrames with mock data for each symbol
        """
        symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        logger.info(f"Generating mock data for {len(symbols)} symbols: {symbols}")
        
        historical_data = {}
        
        for symbol in symbols:
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
            
            historical_data[symbol] = df
            logger.info(f"Generated {len(df)} bars for {symbol}")
        
        return historical_data
    
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
    
    def run_simple_strategy(self, symbols=None):
        """
        Run a simple moving average crossover strategy
        
        Args:
            symbols: List of symbols to trade
            
        Returns:
            Dictionary with backtest results
        """
        # Reset the backtest state
        self.reset()
        
        # Generate mock data
        historical_data = self.generate_mock_data(symbols)
        
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
        
        # Calculate moving averages for each symbol
        for symbol, df in historical_data.items():
            # Calculate 20-day and 50-day moving averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate volatility (20-day standard deviation)
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Calculate market regime (simple version)
            df['market_regime'] = 0  # Unknown
            df.loc[df['ma_20'] > df['ma_50'], 'market_regime'] = 1  # Bullish
            df.loc[df['ma_20'] < df['ma_50'], 'market_regime'] = 2  # Bearish
        
        # Run the backtest day by day
        for date in common_dates:
            # Skip the first 50 days to allow for moving averages to be calculated
            if date < pd.to_datetime(common_dates[50]).date():
                continue
            
            # Get prices and signals for this date
            prices = {}
            signals = {}
            volatilities = {}
            market_regimes = {}
            
            for symbol, df in historical_data.items():
                day_data = df[pd.to_datetime(df['timestamp']).dt.date == date]
                if not day_data.empty:
                    prices[symbol] = day_data['close'].iloc[-1]
                    volatilities[symbol] = day_data['volatility'].iloc[-1]
                    market_regimes[symbol] = day_data['market_regime'].iloc[-1]
                    
                    # Generate signal based on moving average crossover
                    if day_data['ma_20'].iloc[-1] > day_data['ma_50'].iloc[-1]:
                        signals[symbol] = 1  # Buy signal
                    else:
                        signals[symbol] = -1  # Sell signal
            
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
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                if symbol not in prices:
                    continue
                
                price = prices[symbol]
                volatility = volatilities[symbol]
                market_regime = market_regimes[symbol]
                
                # Calculate position size based on volatility
                base_position_pct = 10.0  # 10% of portfolio per position
                position_pct = self.risk_manager.calculate_volatility_adjusted_position_size(
                    volatility, 2.0, base_position_pct
                )
                
                # Adjust position size based on signal strength
                position_pct *= abs(signal)
                
                # Calculate target position size
                target_position_size = self.portfolio_value * (position_pct / 100) * np.sign(signal)
                
                # Calculate current position size
                current_position_size = 0
                if symbol in self.positions:
                    current_position_size = self.positions[symbol]['shares'] * price
                
                # Calculate difference
                diff = target_position_size - current_position_size
                
                # Skip small adjustments
                if abs(diff) < 1000:
                    continue
                
                # Calculate stop loss and take profit levels
                if signal > 0:
                    # Long position
                    stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                        price, volatility, market_regime
                    )
                    take_profit = self.risk_manager.calculate_dynamic_take_profit(
                        price, volatility, market_regime
                    )
                    
                    # Check risk/reward ratio
                    rr_ratio = self.risk_manager.calculate_risk_reward_ratio(
                        price, stop_loss, take_profit
                    )
                    
                    if rr_ratio < 1.5:
                        logger.info(f"Skipping {symbol} due to poor risk/reward ratio: {rr_ratio:.2f}")
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
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
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
            "trades": [
                {
                    "timestamp": trade["timestamp"].strftime('%Y-%m-%d'),
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "shares": trade["shares"],
                    "price": trade["price"],
                    "value": trade["value"],
                    "commission": trade["commission"],
                    "total_cost": trade["total_cost"]
                } for trade in self.trades[:10]  # Include only first 10 trades for brevity
            ]
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a simple backtest')
    
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--output', type=str, help='Output directory for report')
    
    args = parser.parse_args()
    
    # Create backtest engine
    engine = SimpleBacktest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )
    
    # Run backtest
    engine.run_simple_strategy()
    
    # Generate report
    engine.generate_report(output_dir=args.output)
    
    # Plot equity curve
    if not args.output:
        engine.plot_equity_curve()


if __name__ == '__main__':
    main()
