#!/usr/bin/env python
"""
Test script for verifying Alpaca connection

This script tests the connection to Alpaca Markets and performs
basic operations to verify functionality.
"""

import sys
import os
import logging
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Add the parent directory to the Python path so we can import from app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services.alpaca_broker import AlpacaBroker
from app.core.config import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_connection(api_key=None, api_secret=None, paper_trading=True):
    """Test connection to Alpaca Markets"""
    settings = get_settings()
    
    # Get configuration from settings or provided parameters
    api_key = api_key or os.getenv("ALPACA_API_KEY", settings.ALPACA_API_KEY)
    api_secret = api_secret or os.getenv("ALPACA_API_SECRET", settings.ALPACA_API_SECRET)
    paper_trading = paper_trading if paper_trading is not None else \
                    (os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true")
    
    logger.info(f"Connecting to Alpaca Markets with API key: {api_key[:4]}...{api_key[-4:]}")
    logger.info(f"Paper trading: {paper_trading}")
    
    try:
        # Create Alpaca broker instance
        broker = AlpacaBroker(
            api_key=api_key,
            api_secret=api_secret,
            paper_trading=paper_trading
        )
        
        # Try to connect and get account
        logger.info("Retrieving account information...")
        account = broker.get_account()
        logger.info(f"Account ID: {account['account_id']}")
        logger.info(f"Status: {account['status']}")
        logger.info(f"Cash: {account['cash']}")
        logger.info(f"Portfolio Value: {account['portfolio_value']}")
        logger.info(f"Buying Power: {account['buying_power']}")
        
        # Check if we can get market data
        logger.info("Retrieving market data for AAPL...")
        quote = broker.get_quote(symbol="AAPL")
        logger.info(f"AAPL Last Price: {quote['last_price']}")
        
        # Get positions
        logger.info("Retrieving current positions...")
        positions = broker.get_positions()
        logger.info(f"Number of positions: {len(positions)}")
        for position in positions:
            logger.info(f"Position: {position['symbol']}, Qty: {position['quantity']}, Value: {position['market_value']}")
        
        # Get orders
        logger.info("Retrieving recent orders...")
        orders = broker.get_orders(status="all", limit=5)
        logger.info(f"Number of recent orders: {len(orders)}")
        for order in orders:
            logger.info(f"Order ID: {order['id']}, Symbol: {order['symbol']}, Status: {order['status']}")
        
        # Get historical data
        logger.info("Retrieving historical data for AAPL...")
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        bars = broker.get_historical_data(
            symbol="AAPL",
            timeframe="1d",
            start=week_ago,
            end=now
        )
        logger.info(f"Number of bars retrieved: {len(bars)}")
        for bar in bars[:3]:  # Show first 3 bars
            logger.info(f"Date: {bar['timestamp']}, Close: {bar['close']}")
        
        logger.info("Alpaca connection and basic functions test successful!")
        return True
    
    except Exception as e:
        logger.error(f"Error connecting to Alpaca: {e}")
        return False


def test_order_flow(api_key=None, api_secret=None, paper_trading=True, execute_orders=False):
    """Test the full order flow with Alpaca Markets"""
    settings = get_settings()
    
    # Get configuration from settings or provided parameters
    api_key = api_key or os.getenv("ALPACA_API_KEY", settings.ALPACA_API_KEY)
    api_secret = api_secret or os.getenv("ALPACA_API_SECRET", settings.ALPACA_API_SECRET)
    paper_trading = paper_trading if paper_trading is not None else \
                    (os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true")
    
    # Safety check
    if not paper_trading and not execute_orders:
        logger.warning("Live account detected but execute_orders is False. No orders will be placed.")
    
    try:
        # Create Alpaca broker instance
        broker = AlpacaBroker(
            api_key=api_key,
            api_secret=api_secret,
            paper_trading=paper_trading
        )
        
        # Get account info first
        account = broker.get_account()
        logger.info(f"Account ID: {account['account_id']}")
        logger.info(f"Cash: {account['cash']}")
        
        # Test with a cheap, liquid stock
        symbol = "AAPL"
        
        # Get current price
        quote = broker.get_quote(symbol=symbol)
        current_price = quote['last_price']
        logger.info(f"{symbol} Current Price: ${current_price}")
        
        # Calculate small position size (just 1 share for testing)
        qty = 1
        
        # Only execute orders if explicitly requested
        if execute_orders:
            # Place a market order
            logger.info(f"Placing market order: BUY {qty} {symbol}")
            market_order = broker.place_market_order(
                symbol=symbol,
                qty=qty,
                side="buy"
            )
            logger.info(f"Market order placed: ID {market_order['id']}, Status: {market_order['status']}")
            
            # Wait for order to fill if needed
            if market_order['status'] != 'filled':
                logger.info("Waiting for market order to fill...")
                # In a real implementation, you would poll the order status here
            
            # Get updated positions
            logger.info("Retrieving positions after market order...")
            positions = broker.get_positions()
            for position in positions:
                if position['symbol'] == symbol:
                    logger.info(f"Position: {position['symbol']}, Qty: {position['quantity']}")
            
            # Place a limit order to sell at higher price
            limit_price = round(current_price * 1.01, 2)  # 1% above current price
            logger.info(f"Placing limit order: SELL {qty} {symbol} @ ${limit_price}")
            limit_order = broker.place_limit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                limit_price=limit_price
            )
            logger.info(f"Limit order placed: ID {limit_order['id']}, Status: {limit_order['status']}")
            
            # Cancel the limit order
            logger.info(f"Cancelling limit order: {limit_order['id']}")
            cancel_result = broker.cancel_order(order_id=limit_order['id'])
            logger.info(f"Cancellation result: {cancel_result}")
            
            # Place a market order to sell and close position
            logger.info(f"Placing market order to close position: SELL {qty} {symbol}")
            close_order = broker.place_market_order(
                symbol=symbol,
                qty=qty,
                side="sell"
            )
            logger.info(f"Close order placed: ID {close_order['id']}, Status: {close_order['status']}")
        else:
            logger.info("Order execution skipped. Run with --execute-orders to place real orders.")
            
            # Just simulate what would happen
            logger.info(f"Would place market BUY order for {qty} {symbol} @ ~${current_price}")
            logger.info(f"Would place limit SELL order for {qty} {symbol} @ ${round(current_price * 1.01, 2)}")
            logger.info(f"Would cancel limit order")
            logger.info(f"Would place market SELL order for {qty} {symbol} to close position")
        
        logger.info("Order flow test completed!")
        return True
    
    except Exception as e:
        logger.error(f"Error in order flow test: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Alpaca Markets integration')
    parser.add_argument('--api-key', help='Alpaca API Key')
    parser.add_argument('--api-secret', help='Alpaca API Secret')
    parser.add_argument('--live', action='store_true', help='Use live trading instead of paper')
    parser.add_argument('--execute-orders', action='store_true', help='Actually execute orders (USE WITH CAUTION)')
    parser.add_argument('--test-orders', action='store_true', help='Test order flow')
    
    args = parser.parse_args()
    
    logger.info("Starting Alpaca Markets integration test")
    paper_trading = not args.live
    
    # Safety warning for live trading
    if not paper_trading:
        logger.warning("⚠️  LIVE TRADING MODE ENABLED ⚠️")
        if args.execute_orders:
            logger.warning("⚠️  REAL ORDERS WILL BE PLACED! ⚠️")
            confirmation = input("Are you sure you want to continue with LIVE trading? (yes/no): ")
            if confirmation.lower() != 'yes':
                logger.info("Test aborted by user.")
                sys.exit(0)
    
    # Test basic connection
    success = test_connection(
        api_key=args.api_key,
        api_secret=args.api_secret,
        paper_trading=paper_trading
    )
    
    # Test order flow if requested
    if args.test_orders:
        order_success = test_order_flow(
            api_key=args.api_key,
            api_secret=args.api_secret,
            paper_trading=paper_trading,
            execute_orders=args.execute_orders
        )
        success = success and order_success
    
    if success:
        logger.info("Alpaca Markets integration test successful!")
        sys.exit(0)
    else:
        logger.error("Alpaca Markets integration test failed!")
        sys.exit(1)
