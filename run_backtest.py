#!/usr/bin/env python
"""
Run Backtest Script

This script runs a backtest of the trading strategy against historical data
to evaluate performance before deploying with real money.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the backtest engine
from backend.app.ai.backtest import BacktestEngine

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run backtest on trading strategy")
    
    # Basic arguments
    parser.add_argument("--config", type=str, default="strategy_config.json",
                      help="Path to strategy configuration file")
    parser.add_argument("--start", type=str, default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                      help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                      help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str,
                      help="Comma-separated list of symbols to test")
    parser.add_argument("--output", type=str, default="backtest_results",
                      help="Output directory for results")
    
    # Strategy parameters
    parser.add_argument("--short-window", type=int,
                      help="Short moving average window")
    parser.add_argument("--long-window", type=int,
                      help="Long moving average window")
    parser.add_argument("--cash", type=float,
                      help="Initial cash amount")
    parser.add_argument("--commission", type=float, default=0.0,
                      help="Commission rate (e.g., 0.001 for 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0,
                      help="Slippage rate (e.g., 0.001 for 0.1%)")
    parser.add_argument("--force", action="store_true",
                      help="Force run even if markets are closed")
    
    return parser.parse_args()


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle pandas Timestamp objects"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'date') and callable(getattr(obj, 'date')):
            return obj.date().isoformat()
        if hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
            return obj.isoformat()
        return super().default(obj)


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = None
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Configuration file {args.config} not found, using defaults")
        config = {}
    
    # Override configuration with command line arguments
    if args.symbols:
        config["symbols"] = [s.strip().upper() for s in args.symbols.split(",")]
    
    if args.short_window:
        config["short_window"] = args.short_window
    
    if args.long_window:
        config["long_window"] = args.long_window
    
    if args.cash:
        config["initial_cash"] = args.cash
    
    if args.commission:
        config["commission"] = args.commission
        
    if args.slippage:
        config["slippage"] = args.slippage
    
    # Display configuration
    print("\n=== Backtest Configuration ===")
    print(f"Start date: {args.start}")
    print(f"End date: {args.end}")
    print(f"Symbols: {config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])}")
    print(f"Short window: {config.get('short_window', 10)}")
    print(f"Long window: {config.get('long_window', 30)}")
    print(f"Initial cash: ${config.get('initial_cash', 100000)}")
    print(f"Commission: {config.get('commission', 0.0) * 100}%")
    print(f"Slippage: {config.get('slippage', 0.0) * 100}%")
    print("===============================\n")
    
    # Create backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest
    print(f"Running backtest from {args.start} to {args.end}...")
    results = engine.run_backtest(args.start, args.end)
    
    if not results:
        print("Backtest failed")
        return
    
    # Save results
    results_file = output_dir / "results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, cls=JSONEncoder)
    
    print(f"Results saved to {results_file}")
    
    # Save trades to CSV
    trades_file = output_dir / "trades.csv"
    trades_df = pd.DataFrame(results["trades"])
    if not trades_df.empty:
        trades_df.to_csv(trades_file, index=False)
        print(f"Trades saved to {trades_file}")
    
    # Save daily returns to CSV
    returns_file = output_dir / "daily_returns.csv"
    returns_df = pd.DataFrame(results["daily_returns"])
    if not returns_df.empty:
        returns_df.to_csv(returns_file, index=False)
        print(f"Daily returns saved to {returns_file}")
    
    # Plot results
    plot_file = output_dir / "backtest_plot.png"
    engine.plot_results(str(plot_file))
    
    # Print summary
    metrics = results["metrics"]
    print("\n=== Backtest Results ===")
    print(f"Initial portfolio value: ${config.get('initial_cash', 100000):.2f}")
    print(f"Final portfolio value: ${results['portfolio']['total']:.2f}")
    print(f"Total return: {metrics['total_return']:.2f}%")
    print(f"Annualized return: {metrics['annualized_return']:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win rate: {metrics['win_rate']:.2f}%")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Number of trades: {metrics['num_trades']}")
    print("=========================\n")
    
    # Display top performing symbols
    if "positions" in results:
        positions = sorted(results["positions"].items(), 
                          key=lambda x: x[1].get("pnl_pct", 0), 
                          reverse=True)
        
        if positions:
            print("\n=== Top Performing Symbols ===")
            for symbol, data in positions[:5]:
                pnl = data.get("pnl", 0)
                pnl_pct = data.get("pnl_pct", 0)
                print(f"{symbol}: ${pnl:.2f} ({pnl_pct:.2f}%)")
            print("=============================\n")
    
    # Provide trading recommendation based on backtest
    if metrics['sharpe_ratio'] >= 1.0 and metrics['total_return'] > 0:
        print("\n RECOMMENDATION: This strategy shows promise and may be suitable for paper trading.")
        if metrics['sharpe_ratio'] >= 1.5 and metrics['win_rate'] > 50:
            print("   Consider using this strategy with a SMALL allocation of real capital after paper trading.")
    elif metrics['total_return'] > 0 and metrics['sharpe_ratio'] < 1.0:
        print("\n RECOMMENDATION: This strategy is profitable but has too much risk relative to return.")
        print("   Continue optimizing parameters before paper trading.")
    else:
        print("\n RECOMMENDATION: This strategy does not perform well enough for real trading.")
        print("   Continue refining the strategy before proceeding.")
    
    print(f"\nBacktest completed. View the results at {plot_file}")


if __name__ == "__main__":
    main()
