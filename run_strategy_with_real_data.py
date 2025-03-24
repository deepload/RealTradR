#!/usr/bin/env python
"""
Run the enhanced strategy with real-time data from Alpaca
"""
import os
import sys
import logging
import json
import argparse
from datetime import datetime, timedelta
import time
import pandas as pd

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import strategy
from app.ai.enhanced_strategy import EnhancedAIStrategy
from app.utils.performance import PerformanceTracker
from app.utils.trade_logger import TradeLogger
from app.utils.alert_manager import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy_log.txt')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the enhanced trading strategy with real-time data')
    parser.add_argument('--config', type=str, default='strategy_config.json', help='Path to strategy configuration file')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to trade')
    parser.add_argument('--paper', action='store_true', help='Use paper trading')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=60, help='Interval between runs in seconds')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def run_strategy(args):
    """Run the enhanced strategy"""
    # Load configuration
    config = load_config(args.config)
    
    # Override symbols if provided
    if args.symbols:
        config['symbols'] = args.symbols.split(',')
    
    # Set paper trading flag
    if args.paper:
        config['paper_trading'] = True
    
    # Set mock data flag
    if args.mock:
        config['use_mock_data'] = True
    
    # Initialize strategy
    strategy = EnhancedAIStrategy(config)
    
    # Initialize performance tracker
    performance_tracker = PerformanceTracker()
    
    # Initialize trade logger
    trade_logger = TradeLogger('trades.csv')
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Run strategy
    try:
        if args.continuous:
            logger.info(f"Running strategy continuously with interval {args.interval} seconds")
            run_count = 0
            while True:
                run_count += 1
                logger.info(f"Run #{run_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run strategy once
                result = strategy.run()
                
                # Log results
                if result:
                    # Update performance metrics
                    performance_tracker.update(result)
                    
                    # Log trades
                    if 'trades' in result:
                        for trade in result['trades']:
                            trade_logger.log_trade(trade)
                    
                    # Send alerts if necessary
                    if 'alerts' in result:
                        for alert in result['alerts']:
                            alert_manager.send_alert(alert)
                    
                    # Print summary
                    print_summary(result)
                
                # Sleep until next run
                logger.info(f"Sleeping for {args.interval} seconds")
                time.sleep(args.interval)
        else:
            logger.info("Running strategy once")
            result = strategy.run()
            
            # Log results
            if result:
                # Update performance metrics
                performance_tracker.update(result)
                
                # Log trades
                if 'trades' in result:
                    for trade in result['trades']:
                        trade_logger.log_trade(trade)
                
                # Print summary
                print_summary(result)
                
                # Generate report if requested
                if args.report:
                    generate_report(result, performance_tracker)
    
    except KeyboardInterrupt:
        logger.info("Strategy execution interrupted by user")
    except Exception as e:
        logger.error(f"Error running strategy: {e}")
    finally:
        logger.info("Strategy execution completed")

def print_summary(result):
    """Print a summary of the strategy results"""
    logger.info("=" * 50)
    logger.info("STRATEGY EXECUTION SUMMARY")
    logger.info("=" * 50)
    
    # Print account info
    if 'account' in result:
        account = result['account']
        logger.info(f"Account Value: ${account.get('equity', 0):.2f}")
        logger.info(f"Cash: ${account.get('cash', 0):.2f}")
        logger.info(f"Buying Power: ${account.get('buying_power', 0):.2f}")
    
    # Print positions
    if 'positions' in result:
        positions = result['positions']
        logger.info(f"Current Positions: {len(positions)}")
        for symbol, position in positions.items():
            logger.info(f"  {symbol}: {position.get('qty', 0)} shares, Market Value: ${position.get('market_value', 0):.2f}")
    
    # Print signals
    if 'signals' in result:
        signals = result['signals']
        logger.info(f"Signals Generated: {len(signals)}")
        for symbol, signal in signals.items():
            logger.info(f"  {symbol}: Signal Strength: {signal.get('combined_signal', 0):.4f}, Action: {signal.get('action', 'HOLD')}")
    
    # Print trades
    if 'trades' in result:
        trades = result['trades']
        logger.info(f"Trades Executed: {len(trades)}")
        for trade in trades:
            logger.info(f"  {trade.get('symbol')}: {trade.get('side')} {trade.get('qty')} shares at ${trade.get('price', 0):.2f}")
    
    # Print performance
    if 'performance' in result:
        performance = result['performance']
        logger.info(f"Performance Metrics:")
        logger.info(f"  Daily Return: {performance.get('daily_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        logger.info(f"  Win Rate: {performance.get('win_rate', 0):.2%}")
    
    logger.info("=" * 50)

def generate_report(result, performance_tracker):
    """Generate a performance report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'account': result.get('account', {}),
        'positions': result.get('positions', {}),
        'performance': performance_tracker.get_metrics(),
        'trades': performance_tracker.get_trade_history()
    }
    
    # Save report to file
    report_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Performance report saved to {report_file}")

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ENHANCED TRADING STRATEGY WITH REAL-TIME DATA")
    logger.info("=" * 80)
    
    # Parse command line arguments
    args = parse_args()
    
    # Run strategy
    run_strategy(args)
    
    logger.info("=" * 80)
