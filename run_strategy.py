#!/usr/bin/env python
"""
RealTradR Strategy Runner

This script runs the enhanced AI trading strategy with options to use either
paper trading or live trading.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import time
from datetime import timedelta

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the strategy
from backend.app.ai.enhanced_strategy import EnhancedAIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"strategy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("strategy_runner")

def load_config(config_file="strategy_config.json"):
    """Load the strategy configuration"""
    with open(config_file, 'r') as f:
        return json.load(f)

def save_config(config, config_file="strategy_config.json"):
    """Save the strategy configuration"""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def run_strategy(use_live_trading=False, use_mock_data=False, config_file="strategy_config.json", duration_minutes=None):
    """Run the trading strategy"""
    # Load the configuration
    config = load_config(config_file)
    
    # Update configuration based on arguments
    config["api_credentials"]["paper_trading"] = not use_live_trading
    config["market_data"]["use_mock_data"] = use_mock_data
    
    # If using live trading, prompt for confirmation
    if use_live_trading:
        logger.warning("*** LIVE TRADING MODE ENABLED ***")
        logger.warning("This will use real money from your Alpaca account!")
        confirm = input("Are you sure you want to proceed with LIVE trading? (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("Live trading cancelled. Exiting.")
            return
        
        # Check if we need to update API keys for live trading
        if config["api_credentials"]["use_test_credentials"]:
            logger.warning("You are currently using test credentials in your config.")
            update_keys = input("Do you want to enter your live API keys? (yes/no): ")
            if update_keys.lower() == "yes":
                api_key = input("Enter your live Alpaca API key: ")
                api_secret = input("Enter your live Alpaca API secret: ")
                
                # Update the configuration
                config["api_credentials"]["use_test_credentials"] = False
                config["api_credentials"]["api_key"] = api_key
                config["api_credentials"]["api_secret"] = api_secret
                
                # Save the updated configuration
                save_config(config, config_file)
    
    # Display the trading mode
    if use_live_trading:
        logger.info("Running strategy in LIVE trading mode")
    else:
        logger.info("Running strategy in PAPER trading mode")
    
    if use_mock_data:
        logger.info("Using mock market data for testing")
    
    # Initialize the strategy
    strategy = EnhancedAIStrategy(config=config)
    
    if duration_minutes is None:
        # Run strategy once
        logger.info("Starting strategy execution...")
        result = strategy.run()
        
        # Log the results
        if result:
            logger.info(f"Strategy execution completed at {result['timestamp']}")
            logger.info(f"Account cash: ${result['account']['cash']:.2f}")
            logger.info(f"Portfolio value: ${result['account']['portfolio_value']:.2f}")
            
            # Log positions
            logger.info("Current positions:")
            for symbol, value in result['positions'].items():
                logger.info(f"  {symbol}: ${value:.2f}")
            
            # Log trades
            if result['trades']:
                logger.info("Trades executed:")
                for trade in result['trades']:
                    logger.info(f"  {trade['symbol']}: {trade['action']} {trade['quantity']} shares at ${trade['price']:.2f}")
            else:
                logger.info("No trades executed")
            
            # Log performance
            logger.info("Performance metrics:")
            for metric, value in result['performance'].items():
                logger.info(f"  {metric}: {value}")
        else:
            logger.error("Strategy execution failed")
    else:
        # Run strategy for specified duration
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        logger.info(f"Running strategy until {end_time.strftime('%Y-%m-%d %H:%M:%S')} (duration: {duration_minutes} minutes)")
        
        run_count = 0
        while datetime.now() < end_time:
            run_count += 1
            logger.info(f"Strategy run #{run_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run strategy
            result = strategy.run()
            
            # Log current status
            if result:
                logger.info(f"Account cash: ${result['account']['cash']:.2f}")
                logger.info(f"Portfolio value: ${result['account']['portfolio_value']:.2f}")
                
                # Log positions
                logger.info("Current positions:")
                for symbol, value in result['positions'].items():
                    logger.info(f"  {symbol}: ${value:.2f}")
            
            # Wait for 1 minute before next run
            if datetime.now() < end_time:
                logger.info(f"Waiting for next run... (time remaining: {(end_time - datetime.now()).seconds // 60} minutes)")
                time.sleep(60)  # Wait for 1 minute
        
        logger.info(f"Strategy execution completed after {duration_minutes} minutes")
        
        # Log final results
        result = strategy.run()
        if result:
            logger.info(f"Final account cash: ${result['account']['cash']:.2f}")
            logger.info(f"Final portfolio value: ${result['account']['portfolio_value']:.2f}")
            logger.info(f"Final positions:")
            
            # Log positions
            logger.info("Current positions:")
            for symbol, value in result['positions'].items():
                logger.info(f"  {symbol}: ${value:.2f}")
            
            # Log trades
            if result['trades']:
                logger.info("Trades executed:")
                for trade in result['trades']:
                    logger.info(f"  {trade['symbol']}: {trade['action']} {trade['quantity']} shares at ${trade['price']:.2f}")
            else:
                logger.info("No trades executed")
            
            # Log performance
            logger.info("Performance metrics:")
            for metric, value in result['performance'].items():
                logger.info(f"  {metric}: {value}")
        else:
            logger.error("Strategy execution failed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run the RealTradR trading strategy")
    parser.add_argument("--live", action="store_true", help="Use live trading instead of paper trading")
    parser.add_argument("--mock", action="store_true", help="Use mock market data for testing")
    parser.add_argument("--config", default="strategy_config.json", help="Path to configuration file")
    parser.add_argument("--duration", type=int, help="Duration to run the strategy in minutes")
    
    args = parser.parse_args()
    
    run_strategy(
        use_live_trading=args.live,
        use_mock_data=args.mock,
        config_file=args.config,
        duration_minutes=args.duration
    )

if __name__ == "__main__":
    main()
