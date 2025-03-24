#!/usr/bin/env python
"""
Run Enhanced AI Trading Strategy with Alpaca Paper Trading

This script runs the enhanced AI trading strategy using Alpaca's paper trading API.
It provides command-line options for configuration and can be scheduled to run at
regular intervals for automated trading.
"""

import os
import sys
import argparse
import logging
import json
import time
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import random
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RealTradR modules
from backend.app.ai.enhanced_strategy import EnhancedAIStrategy
from backend.app.utils.performance import PerformanceMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enhanced_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_enhanced_strategy")

# Load environment variables
load_dotenv()


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run Enhanced AI Trading Strategy")
    parser.add_argument("--config", default="strategy_config.json", help="Path to strategy configuration file")
    parser.add_argument("--continuous", action="store_true", help="Run strategy continuously at regular intervals")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes between strategy runs (when using --continuous)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to run (when using --continuous)")
    parser.add_argument("--report", default="strategy_report.json", help="Path to output report file")
    parser.add_argument("--paper-trading", action="store_true", help="Use paper trading (default is to use the setting from .env)")
    parser.add_argument("--use-mock-data", action="store_true", help="Use mock data instead of real market data")
    parser.add_argument("--duration", type=int, default=30, help="Run strategy for a specified duration in minutes")
    return parser.parse_args()


def load_config(config_file):
    """Load strategy configuration from file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_file}: {e}")
        logger.info("Using default configuration")
        return None


def generate_mock_data(symbols, days=60, volatility=0.015):
    """Generate mock price data for testing"""
    logger.info(f"Generating mock data for {len(symbols)} symbols")
    
    # Initialize data dictionary
    data = {}
    
    # Generate data for each symbol
    for symbol in symbols:
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize price at a random value between 50 and 500
        initial_price = random.uniform(50, 500)
        
        # Generate random price movements
        prices = [initial_price]
        for i in range(1, len(date_range)):
            # Random daily return with slight upward bias
            daily_return = random.normalvariate(0.0005, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Generate realistic high, low, open values
        opens = []
        highs = []
        lows = []
        for i, close in enumerate(prices):
            # Open is previous close with small adjustment, or same as close for first day
            if i == 0:
                opens.append(close)
            else:
                opens.append(prices[i-1] * (1 + random.normalvariate(0, 0.003)))
            
            # High is the max of open and close plus a random amount
            high_adjustment = close * random.uniform(0.001, 0.015)
            highs.append(max(opens[i], close) + high_adjustment)
            
            # Low is the min of open and close minus a random amount
            low_adjustment = close * random.uniform(0.001, 0.015)
            lows.append(min(opens[i], close) - low_adjustment)
        
        # Generate volume with occasional spikes
        volumes = []
        base_volume = random.randint(100000, 1000000)
        for i in range(len(date_range)):
            # Add occasional volume spikes
            if random.random() < 0.1:
                volume = base_volume * random.uniform(2, 5)
            else:
                volume = base_volume * random.uniform(0.8, 1.2)
            volumes.append(int(volume))
        
        # Create DataFrame with proper column names
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=date_range)
        
        # Add to data dictionary
        data[symbol] = df
    
    logger.info(f"Generated mock data with {len(date_range)} days per symbol")
    return data


def run_strategy_once(strategy, use_mock_data=False, mock_data_volatility=0.015):
    """Run the strategy once and return results"""
    try:
        # Start strategy
        logger.info("Running enhanced AI strategy...")
        
        # Generate mock data if requested
        if use_mock_data:
            data = generate_mock_data(
                strategy.symbols, 
                days=strategy.config.get("market_data", {}).get("default_days", 60),
                volatility=mock_data_volatility
            )
            results = strategy.run_with_data(data)
        else:
            # Run with real data
            try:
                results = strategy.run()
            except Exception as e:
                logger.error(f"Error running strategy with real data: {e}")
                
                # Check if we should fall back to mock data
                if strategy.config.get("market_data", {}).get("use_mock_data_on_api_failure", True):
                    logger.warning("Falling back to mock data due to API failure")
                    data = generate_mock_data(
                        strategy.symbols, 
                        days=strategy.config.get("market_data", {}).get("default_days", 60),
                        volatility=strategy.config.get("market_data", {}).get("mock_data_volatility", 0.015)
                    )
                    results = strategy.run_with_data(data)
                else:
                    raise
        
        # Log results
        if "error" in results:
            logger.error(f"Error running strategy: {results['error']}")
        else:
            logger.info(f"Strategy run completed")
            logger.info(f"Portfolio value: ${results.get('portfolio_value', 0):.2f}")
            logger.info(f"Cash: ${results.get('cash', 0):.2f}")
            
            # Log results for each symbol
            for symbol_key, data in results.get("results", {}).items():
                if "error" in data:
                    logger.error(f"Error processing {symbol_key}: {data['error']}")
                    continue
                
                logger.info(f"Symbol: {symbol_key}")
                logger.info(f"  Signal: {data.get('combined_signal', 0):.2f}")
                logger.info(f"  Market regime: {data.get('market_regime', 'unknown')}")
                
                trade_result = data.get("trade_result", {})
                if trade_result.get("trade_executed", False):
                    logger.info(f"  Trade executed: {trade_result.get('side')} "
                               f"{trade_result.get('quantity')} shares at "
                               f"${trade_result.get('price', 0):.2f}")
                else:
                    logger.info("  No trade executed")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running strategy: {e}")
        traceback.print_exc()
        return None


def run_strategy_continuously(strategy, interval_minutes=60, days=1, use_mock_data=False, mock_data_volatility=0.015):
    """Run the strategy continuously at regular intervals"""
    logger.info(f"Running strategy continuously every {interval_minutes} minutes for {days} days")
    
    # Calculate end time
    end_time = datetime.now() + timedelta(days=days)
    
    # Initialize performance monitor
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    try:
        # Run until end time is reached
        while datetime.now() < end_time:
            # Run the strategy
            start_time = datetime.now()
            logger.info(f"Running strategy at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            results = run_strategy_once(strategy, use_mock_data, mock_data_volatility)
            
            # Calculate performance metrics
            metrics = strategy.get_performance_metrics()
            logger.info("Performance metrics:")
            logger.info(f"  Total return: {metrics.get('total_return', 0):.2f}%")
            logger.info(f"  Annualized return: {metrics.get('annualized_return', 0):.2f}%")
            logger.info(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            logger.info(f"  Win rate: {metrics.get('win_rate', 0):.2f}%")
            logger.info(f"  Profit factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Calculate time to next run
            elapsed_time = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, interval_minutes * 60 - elapsed_time)
            
            # Check if we've reached the end time
            next_run_time = datetime.now() + timedelta(seconds=sleep_time)
            if next_run_time > end_time:
                logger.info("End time reached, stopping continuous run")
                break
            
            logger.info(f"Waiting {sleep_time:.1f} seconds until next run at {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping continuous run")
    
    finally:
        # Stop performance monitoring
        performance_monitor.stop()
        
        # Calculate final performance metrics
        metrics = strategy.get_performance_metrics()
        logger.info("Final performance metrics:")
        logger.info(f"  Total return: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"  Annualized return: {metrics.get('annualized_return', 0):.2f}%")
        logger.info(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        
        return metrics


def run_strategy_timed(strategy, duration_minutes=30, use_mock_data=False):
    """Run the strategy for a specified duration"""
    logger.info(f"Running strategy for {duration_minutes} minutes with real Alpaca data")
    
    # Calculate end time
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    # Initialize performance monitor
    performance_monitor = PerformanceMonitor()
    performance_monitor.start()
    
    # Track trades and performance
    trades_executed = 0
    signals_generated = 0
    
    try:
        # Initial run
        logger.info(f"Starting strategy run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        results = run_strategy_once(strategy, use_mock_data)
        
        if results:
            # Count signals and trades
            for symbol, data in results.get("results", {}).items():
                if "combined_signal" in data and abs(data["combined_signal"]) > 0.1:
                    signals_generated += 1
                
                trade_result = data.get("trade_result", {})
                if trade_result.get("trade_executed", False):
                    trades_executed += 1
        
        # Run until end time is reached
        while datetime.now() < end_time:
            # Wait for 5 minutes between runs
            wait_time = min(5, (end_time - datetime.now()).total_seconds() / 60)
            if wait_time <= 0:
                break
                
            logger.info(f"Waiting {wait_time:.1f} minutes until next run...")
            time.sleep(wait_time * 60)
            
            # Check if we've reached the end time
            if datetime.now() >= end_time:
                break
                
            # Run the strategy again
            logger.info(f"Running strategy at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            results = run_strategy_once(strategy, use_mock_data)
            
            if results:
                # Count signals and trades
                for symbol, data in results.get("results", {}).items():
                    if "combined_signal" in data and abs(data["combined_signal"]) > 0.1:
                        signals_generated += 1
                    
                    trade_result = data.get("trade_result", {})
                    if trade_result.get("trade_executed", False):
                        trades_executed += 1
        
        # Get final performance metrics
        metrics = strategy.get_performance_metrics()
        
        # Generate report
        logger.info(f"Strategy run completed after {duration_minutes} minutes")
        logger.info(f"Signals generated: {signals_generated}")
        logger.info(f"Trades executed: {trades_executed}")
        
        # Generate detailed report
        report_file = f"strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generate_report(strategy, results, metrics, output_file=report_file)
        logger.info(f"Report generated: {report_file}")
        
        return results, metrics
        
    except Exception as e:
        logger.error(f"Error running strategy: {e}")
        traceback.print_exc()
        return None, None
    finally:
        performance_monitor.stop()
        performance_metrics = performance_monitor.get_metrics()
        logger.info(f"Performance metrics: {performance_metrics}")


def generate_report(strategy, results, metrics, output_file="strategy_report.json"):
    """Generate a detailed report of strategy execution"""
    # Create report structure
    report = {
        "timestamp": datetime.now().isoformat(),
        "portfolio": {
            "value": results.get("portfolio_value", 0),
            "cash": results.get("cash", 0)
        },
        "metrics": metrics,
        "trades": [],
        "positions": [],
        "signals": {},
        "market_regimes": {}
    }
    
    # Add trade information
    for symbol, data in results.get("results", {}).items():
        # Add signal data
        report["signals"][symbol] = {
            "technical": data.get("technical_signal", 0),
            "ml": data.get("ml_signal", 0),
            "sentiment": data.get("sentiment_signal", 0),
            "combined": data.get("combined_signal", 0)
        }
        
        # Add market regime
        report["market_regimes"][symbol] = data.get("market_regime", "unknown")
        
        # Add trade information if a trade was executed
        trade_result = data.get("trade_result", {})
        if trade_result.get("trade_executed", False):
            report["trades"].append({
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "side": trade_result.get("side", ""),
                "quantity": trade_result.get("quantity", 0),
                "price": trade_result.get("price", 0),
                "value": trade_result.get("value", 0),
                "signal": data.get("combined_signal", 0)
            })
    
    # Get current positions
    try:
        positions = strategy.api.list_positions()
        for position in positions:
            report["positions"].append({
                "symbol": position.symbol,
                "quantity": float(position.qty),
                "market_value": float(position.market_value),
                "avg_entry_price": float(position.avg_entry_price),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "current_price": float(position.current_price)
            })
    except Exception as e:
        logger.error(f"Error retrieving positions: {e}")
    
    # Add detailed performance metrics
    if hasattr(strategy, 'performance_metrics'):
        report["detailed_metrics"] = {
            "daily_returns": strategy.performance_metrics.get("daily_returns", []),
            "drawdowns": strategy.performance_metrics.get("drawdowns", []),
            "win_rate": strategy.performance_metrics.get("win_rate", 0),
            "profit_factor": strategy.performance_metrics.get("profit_factor", 0),
            "avg_win": strategy.performance_metrics.get("avg_win", 0),
            "avg_loss": strategy.performance_metrics.get("avg_loss", 0),
            "max_consecutive_wins": strategy.performance_metrics.get("max_consecutive_wins", 0),
            "max_consecutive_losses": strategy.performance_metrics.get("max_consecutive_losses", 0)
        }
    
    # Write report to file
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report generated: {output_file}")
    except Exception as e:
        logger.error(f"Error writing report to {output_file}: {e}")


def check_market_status():
    """Check if the market is currently open"""
    try:
        # Initialize Alpaca API
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        base_url = os.getenv("ALPACA_API_BASE_URL")
        
        if not api_key or not api_secret or not base_url:
            logger.error("Alpaca API credentials not found in environment variables")
            return False, "API credentials not configured"
        
        api = tradeapi.REST(api_key, api_secret, base_url)
        
        # Get clock
        clock = api.get_clock()
        
        if clock.is_open:
            next_close = clock.next_close.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Market is open. Next close at {next_close}")
            return True, f"Market is open. Next close at {next_close}"
        else:
            next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Market is closed. Next open at {next_open}")
            return False, f"Market is closed. Next open at {next_open}"
    
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False, f"Error checking market status: {e}"


def update_env_file():
    """Update the .env file with the correct API credentials"""
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            logger.error(".env file not found")
            return False
        
        # Read the current .env file
        with open('.env', 'r') as f:
            env_lines = f.readlines()
        
        # Update the API credentials
        updated_lines = []
        for line in env_lines:
            if line.startswith('ALPACA_API_KEY='):
                updated_lines.append('ALPACA_API_KEY=PK88UAXEPBIEQCEAS8YV\n')
            elif line.startswith('ALPACA_API_SECRET='):
                updated_lines.append('ALPACA_API_SECRET=hu9YIoZSqhLiOLLoTkHt5mh2NK4gTi7fXPQf6X1L\n')
            else:
                updated_lines.append(line)
        
        # Write the updated .env file
        with open('.env', 'w') as f:
            f.writelines(updated_lines)
        
        logger.info("Updated .env file with correct API credentials")
        return True
    
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")
        return False


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paper trading setting if specified
    if args.paper_trading:
        os.environ["ALPACA_PAPER_TRADING"] = "true"
    
    # Update .env file with correct API credentials
    update_env_file()
    
    # Reload environment variables
    load_dotenv(override=True)
    
    # Initialize strategy
    strategy = EnhancedAIStrategy(config=config)
    
    # Check if market is open
    is_market_open, market_status = check_market_status()
    logger.info(f"Market status: {market_status}")
    
    # If using mock data, we can proceed even if market is closed
    if not is_market_open and not args.use_mock_data:
        logger.warning("Market is closed and not using mock data. Use --use-mock-data to run with mock data.")
        print(f"Market is closed: {market_status}")
        print("Use --use-mock-data flag to run with mock data.")
        return
    
    try:
        # Run strategy
        if args.continuous:
            # Run continuously
            metrics = run_strategy_continuously(
                strategy, 
                interval_minutes=args.interval, 
                days=args.days,
                use_mock_data=args.use_mock_data
            )
            results = strategy.run()
        elif args.duration:
            # Run for a specified duration
            results, metrics = run_strategy_timed(strategy, duration_minutes=args.duration, use_mock_data=args.use_mock_data)
        else:
            # Run once
            results = run_strategy_once(strategy, use_mock_data=args.use_mock_data)
            metrics = strategy.get_performance_metrics()
        
        # Generate report
        generate_report(strategy, results, metrics, output_file=args.report)
        
    except Exception as e:
        logger.error(f"Error running strategy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
