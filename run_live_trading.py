#!/usr/bin/env python
"""
Live Trading Script for RealTradR

This script runs the trading strategy in real-time using Alpaca's API.
It supports both paper trading and live trading modes.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the trading strategy
from backend.app.ai.simple_strategy import MovingAverageCrossover
from backend.app.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_trading")

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run live trading with Alpaca API")
    
    # Basic arguments
    parser.add_argument("--config", type=str, default="strategy_config.json",
                      help="Path to strategy configuration file")
    parser.add_argument("--live", action="store_true",
                      help="Use live trading instead of paper trading")
    parser.add_argument("--symbols", type=str,
                      help="Comma-separated list of symbols to trade")
    parser.add_argument("--check-interval", type=int, default=60,
                      help="Interval in seconds between strategy checks")
    
    # Strategy parameters
    parser.add_argument("--short-window", type=int,
                      help="Short moving average window")
    parser.add_argument("--long-window", type=int,
                      help="Long moving average window")
    
    return parser.parse_args()


class LiveTrader:
    """Live trading implementation of the strategy"""
    
    def __init__(self, live_trading=False, config=None):
        """
        Initialize the live trader
        
        Args:
            live_trading: If True, use live trading API
            config: Configuration dictionary or path to config file
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Extract configuration values
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        
        # Strategy parameters
        self.short_window = self.config.get("short_window", 10)
        self.long_window = self.config.get("long_window", 30)
        
        # Trading parameters
        self.check_interval = self.config.get("check_interval", 60)  # seconds
        self.max_positions = self.config.get("max_positions", 5)
        self.position_size = self.config.get("position_size", 0.2)  # 20% of available cash per position
        
        # Set up API
        self.live_trading = live_trading
        if live_trading:
            logger.warning("USING LIVE TRADING - REAL MONEY WILL BE USED")
            self.api = tradeapi.REST(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                settings.alpaca_base_url,
                api_version="v2"
            )
        else:
            logger.info("Using paper trading mode (no real money at risk)")
            self.api = tradeapi.REST(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                "https://paper-api.alpaca.markets",
                api_version="v2"
            )
        
        # Create strategy instances for each symbol
        self.strategies = {}
        for symbol in self.symbols:
            self.strategies[symbol] = MovingAverageCrossover(
                symbol=symbol,
                short_window=self.short_window,
                long_window=self.long_window,
                alpaca_api=self.api
            )
        
        # Track positions and orders
        self.positions = {}
        self.pending_orders = {}
        
        logger.info(f"Initialized live trader with symbols: {self.symbols}")
        logger.info(f"Strategy parameters: Short MA={self.short_window}, Long MA={self.long_window}")
        logger.info(f"Trading mode: {'LIVE' if live_trading else 'PAPER'}")
    
    def _load_config(self, config):
        """Load configuration from file or dictionary"""
        if config is None:
            # Try to load from default location
            config_path = "strategy_config.json"
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
            "short_window": 10,
            "long_window": 30,
            "check_interval": 60,
            "max_positions": 5,
            "position_size": 0.2,
        }
    
    def get_account(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            logger.info(f"Account cash: ${float(account.cash)}")
            logger.info(f"Account equity: ${float(account.equity)}")
            logger.info(f"Account buying power: ${float(account.buying_power)}")
            return account
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            positions = {p.symbol: p for p in self.api.list_positions()}
            
            logger.info(f"Current positions: {list(positions.keys())}")
            for symbol, position in positions.items():
                market_value = float(position.market_value)
                cost_basis = float(position.cost_basis)
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc) * 100
                
                logger.info(f"{symbol}: {position.qty} shares, Market value: ${market_value:.2f}, "
                           f"P&L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)")
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def update_positions(self):
        """Update position information"""
        try:
            self.positions = {p.symbol: p for p in self.api.list_positions()}
            return self.positions
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return {}
    
    def check_market_status(self):
        """Check if market is open"""
        try:
            clock = self.api.get_clock()
            if clock.is_open:
                next_close = clock.next_close.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Market is OPEN. Will close at {next_close}")
                return True
            else:
                next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Market is CLOSED. Will open at {next_open}")
                return False
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_cash_available(self):
        """Get available cash for trading"""
        try:
            account = self.api.get_account()
            return float(account.cash)
        except Exception as e:
            logger.error(f"Error getting cash available: {e}")
            return 0
    
    def calculate_position_size(self, symbol, price):
        """Calculate number of shares to buy based on position sizing"""
        try:
            # Get account cash
            cash = self.get_cash_available()
            
            # Calculate how much to allocate to this position
            cash_to_use = cash * self.position_size
            
            # Calculate number of shares
            num_shares = int(cash_to_use / price)
            
            # Ensure minimum position size
            if num_shares * price < 100:  # Minimum $100 position
                return 0
            
            logger.info(f"Calculated position size for {symbol}: {num_shares} shares (${num_shares * price:.2f})")
            return num_shares
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def place_order(self, symbol, qty, side):
        """Place an order"""
        try:
            if qty <= 0:
                logger.warning(f"Invalid quantity ({qty}) for {symbol}, skipping order")
                return None
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
            
            logger.info(f"Placed {side} order for {qty} shares of {symbol}, order ID: {order.id}")
            
            # Track pending order
            self.pending_orders[order.id] = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "status": order.status,
                "created_at": order.created_at
            }
            
            return order
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def check_orders(self):
        """Check status of pending orders"""
        if not self.pending_orders:
            return
        
        try:
            # Get all orders
            orders = self.api.list_orders(status="all", limit=100)
            orders_dict = {o.id: o for o in orders}
            
            # Update status of pending orders
            for order_id in list(self.pending_orders.keys()):
                if order_id in orders_dict:
                    order = orders_dict[order_id]
                    prev_status = self.pending_orders[order_id]["status"]
                    curr_status = order.status
                    
                    if curr_status != prev_status:
                        logger.info(f"Order {order_id} for {order.symbol} status changed: {prev_status} -> {curr_status}")
                        self.pending_orders[order_id]["status"] = curr_status
                    
                    # Remove filled or cancelled orders from pending
                    if curr_status in ["filled", "cancelled", "expired", "rejected"]:
                        logger.info(f"Order {order_id} for {order.symbol} {curr_status}")
                        if curr_status == "filled":
                            logger.info(f"Filled {order.side} order for {order.filled_qty} shares of {order.symbol} at avg price ${float(order.filled_avg_price)}")
                        del self.pending_orders[order_id]
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
    
    def run_strategy_check(self, symbol):
        """Run a strategy check for a symbol"""
        try:
            # Get strategy for symbol
            strategy = self.strategies[symbol]
            
            # Run strategy check
            signal = strategy.check_signals()
            
            if signal is not None:
                current_price = strategy.get_current_price()
                if current_price is None:
                    logger.error(f"Could not get current price for {symbol}")
                    return
                
                if symbol in self.positions:
                    # We already have a position
                    if signal == "sell":
                        # Sell the position
                        qty = int(float(self.positions[symbol].qty))
                        self.place_order(symbol, qty, "sell")
                    else:
                        logger.info(f"Holding position in {symbol}")
                else:
                    # We don't have a position
                    if signal == "buy":
                        # Calculate position size
                        qty = self.calculate_position_size(symbol, current_price)
                        if qty > 0:
                            # Place buy order
                            self.place_order(symbol, qty, "buy")
                        else:
                            logger.info(f"Not enough cash for {symbol} position")
                    else:
                        logger.info(f"No signal for {symbol}")
            else:
                logger.info(f"No signal for {symbol}")
        except Exception as e:
            logger.error(f"Error running strategy check for {symbol}: {e}")
    
    def run(self):
        """Run the live trading loop"""
        try:
            logger.info("Starting live trading...")
            
            # Get account info
            self.get_account()
            
            # Main trading loop
            while True:
                try:
                    # Check if market is open
                    market_open = self.check_market_status()
                    if not market_open:
                        logger.info(f"Market is closed. Waiting for {self.check_interval} seconds...")
                        time.sleep(self.check_interval)
                        continue
                    
                    # Check existing orders
                    self.check_orders()
                    
                    # Update positions
                    self.update_positions()
                    
                    # Run strategy check for each symbol
                    for symbol in self.symbols:
                        self.run_strategy_check(symbol)
                    
                    # Sleep until next check
                    logger.info(f"Waiting for {self.check_interval} seconds until next check...")
                    time.sleep(self.check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt detected, exiting...")
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(self.check_interval)
        except Exception as e:
            logger.error(f"Error running live trading: {e}")
    
    def stop(self):
        """Stop the live trading loop and clean up"""
        logger.info("Stopping live trading...")
        
        # Get final positions and account info
        self.update_positions()
        self.get_positions()
        self.get_account()
        
        logger.info("Live trading stopped")


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
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
    
    if args.check_interval:
        config["check_interval"] = args.check_interval
    
    # Display configuration
    print("\n=== Live Trading Configuration ===")
    print(f"Trading mode: {'LIVE' if args.live else 'PAPER (no real money)'}")
    print(f"Symbols: {config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])}")
    print(f"Short window: {config.get('short_window', 10)}")
    print(f"Long window: {config.get('long_window', 30)}")
    print(f"Check interval: {config.get('check_interval', 60)} seconds")
    print("====================================\n")
    
    # Confirm if using live trading
    if args.live:
        confirmation = input("⚠️ WARNING: You are about to trade with REAL MONEY. Type 'YES' to confirm: ")
        if confirmation.upper() != "YES":
            print("Live trading not confirmed. Exiting.")
            return
    
    # Create live trader
    trader = LiveTrader(live_trading=args.live, config=config)
    
    try:
        # Run the trader
        trader.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping live trader...")
    finally:
        # Stop the trader
        trader.stop()


if __name__ == "__main__":
    main()
