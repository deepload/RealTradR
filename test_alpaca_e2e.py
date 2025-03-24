#!/usr/bin/env python
"""
End-to-end test for Alpaca API integration
Tests both Trading API and Market Data API separately
"""
import os
import sys
import logging
import json
from datetime import datetime, timedelta
import time
import requests
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trading_api_direct():
    """Test Alpaca Trading API directly using requests"""
    # API credentials
    api_key = "PK88UAXEPBIEQCEAS8YV"
    api_secret = "hu9YIoZSqhLiOLLoTkHt5mh2NK4gTi7fXPQf6X1L"
    base_url = "https://paper-api.alpaca.markets"
    
    # API endpoints to test
    endpoints = [
        "/v2/account",
        "/v2/positions",
        "/v2/orders",
        "/v2/clock",
        "/v2/calendar"
    ]
    
    logger.info("Testing Alpaca Trading API directly...")
    logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
    logger.info(f"Base URL: {base_url}")
    
    # Test each endpoint
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        
        try:
            logger.info(f"Testing endpoint: {endpoint}")
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"✅ Success! Status code: {response.status_code}")
                # Print a sample of the response
                response_data = response.json()
                if isinstance(response_data, dict):
                    sample = {k: response_data[k] for k in list(response_data.keys())[:3]} if response_data else {}
                    logger.info(f"Sample response: {json.dumps(sample, indent=2)}")
                elif isinstance(response_data, list):
                    sample = response_data[:2] if response_data else []
                    logger.info(f"Sample response: {json.dumps(sample, indent=2)}")
            else:
                logger.error(f"❌ Failed! Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
        
        except Exception as e:
            logger.error(f"❌ Error testing endpoint {endpoint}: {e}")
    
    return True

def test_market_data_api_direct():
    """Test Alpaca Market Data API directly using requests"""
    # API credentials
    api_key = "PK88UAXEPBIEQCEAS8YV"
    api_secret = "hu9YIoZSqhLiOLLoTkHt5mh2NK4gTi7fXPQf6X1L"
    base_url = "https://data.alpaca.markets"
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    # Format dates
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # API endpoints to test
    endpoints = [
        f"/v2/stocks/AAPL/bars?timeframe=1Day&start={start_str}&end={end_str}",
        f"/v2/stocks/AAPL/trades?start={start_str}&end={end_str}&limit=10",
        f"/v2/stocks/AAPL/quotes?start={start_str}&end={end_str}&limit=10",
        "/v2/stocks/snapshots?symbols=AAPL",
        "/v2/stocks/AAPL/snapshot"
    ]
    
    logger.info("\nTesting Alpaca Market Data API directly...")
    logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Date range: {start_str} to {end_str}")
    
    # Test each endpoint
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        
        try:
            logger.info(f"Testing endpoint: {endpoint}")
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"✅ Success! Status code: {response.status_code}")
                # Print a sample of the response
                response_data = response.json()
                logger.info(f"Sample response: {json.dumps(response_data, indent=2)[:500]}...")
            else:
                logger.error(f"❌ Failed! Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
        
        except Exception as e:
            logger.error(f"❌ Error testing endpoint {endpoint}: {e}")
    
    return True

def test_alpaca_api_sdk():
    """Test Alpaca API using the official SDK"""
    try:
        # Import the SDK
        import alpaca_trade_api as tradeapi
        
        # API credentials
        api_key = "PK88UAXEPBIEQCEAS8YV"
        api_secret = "hu9YIoZSqhLiOLLoTkHt5mh2NK4gTi7fXPQf6X1L"
        trading_url = "https://paper-api.alpaca.markets"
        data_url = "https://data.alpaca.markets"
        
        logger.info("\nTesting Alpaca API using the official SDK...")
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        
        # Initialize Trading API
        trading_api = tradeapi.REST(api_key, api_secret, trading_url, api_version='v2')
        
        # Test Trading API
        logger.info("Testing Trading API...")
        try:
            account = trading_api.get_account()
            logger.info(f"✅ Account API successful!")
            logger.info(f"Account ID: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Account cash: ${float(account.cash):.2f}")
        except Exception as e:
            logger.error(f"❌ Error getting account: {e}")
        
        # Test positions
        try:
            positions = trading_api.list_positions()
            logger.info(f"✅ Positions API successful!")
            logger.info(f"Number of positions: {len(positions)}")
            if positions:
                logger.info(f"Sample position: {positions[0].__dict__}")
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
        
        # Test orders
        try:
            orders = trading_api.list_orders()
            logger.info(f"✅ Orders API successful!")
            logger.info(f"Number of orders: {len(orders)}")
            if orders:
                logger.info(f"Sample order: {orders[0].__dict__}")
        except Exception as e:
            logger.error(f"❌ Error getting orders: {e}")
        
        # Test clock
        try:
            clock = trading_api.get_clock()
            logger.info(f"✅ Clock API successful!")
            logger.info(f"Market is {'open' if clock.is_open else 'closed'}")
            logger.info(f"Next open: {clock.next_open}")
            logger.info(f"Next close: {clock.next_close}")
        except Exception as e:
            logger.error(f"❌ Error getting clock: {e}")
        
        # Initialize Market Data API
        data_api = tradeapi.REST(api_key, api_secret, data_url, api_version='v2')
        
        # Test Market Data API
        logger.info("\nTesting Market Data API...")
        
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Test bars
        try:
            logger.info(f"Getting bars for AAPL from {start_str} to {end_str}...")
            bars = data_api.get_bars(
                "AAPL", 
                tradeapi.TimeFrame.Day, 
                start=start_str,
                end=end_str,
                adjustment='raw'
            ).df
            
            logger.info(f"✅ Bars API successful!")
            logger.info(f"Number of bars: {len(bars)}")
            if not bars.empty:
                logger.info(f"Sample bars:\n{bars.head(2)}")
        except Exception as e:
            logger.error(f"❌ Error getting bars: {e}")
        
        # Test quotes
        try:
            logger.info(f"Getting quotes for AAPL from {start_str} to {end_str}...")
            quotes = data_api.get_quotes(
                "AAPL",
                start=start_str,
                end=end_str,
                limit=10
            ).df
            
            logger.info(f"✅ Quotes API successful!")
            logger.info(f"Number of quotes: {len(quotes)}")
            if not quotes.empty:
                logger.info(f"Sample quotes:\n{quotes.head(2)}")
        except Exception as e:
            logger.error(f"❌ Error getting quotes: {e}")
        
        # Test trades
        try:
            logger.info(f"Getting trades for AAPL from {start_str} to {end_str}...")
            trades = data_api.get_trades(
                "AAPL",
                start=start_str,
                end=end_str,
                limit=10
            ).df
            
            logger.info(f"✅ Trades API successful!")
            logger.info(f"Number of trades: {len(trades)}")
            if not trades.empty:
                logger.info(f"Sample trades:\n{trades.head(2)}")
        except Exception as e:
            logger.error(f"❌ Error getting trades: {e}")
        
        # Test snapshot
        try:
            logger.info(f"Getting snapshot for AAPL...")
            snapshot = data_api.get_snapshot("AAPL")
            
            logger.info(f"✅ Snapshot API successful!")
            logger.info(f"Snapshot: {snapshot.__dict__}")
        except Exception as e:
            logger.error(f"❌ Error getting snapshot: {e}")
        
        return True
    
    except ImportError as e:
        logger.error(f"❌ Error importing alpaca_trade_api: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ALPACA API END-TO-END TEST")
    logger.info("=" * 80)
    
    # Test Trading API directly
    trading_api_result = test_trading_api_direct()
    
    # Test Market Data API directly
    market_data_api_result = test_market_data_api_direct()
    
    # Test Alpaca API SDK
    sdk_result = test_alpaca_api_sdk()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY:")
    logger.info(f"Trading API Direct Test: {'SUCCESS' if trading_api_result else 'FAILED'}")
    logger.info(f"Market Data API Direct Test: {'SUCCESS' if market_data_api_result else 'FAILED'}")
    logger.info(f"Alpaca SDK Test: {'SUCCESS' if sdk_result else 'FAILED'}")
    logger.info("=" * 80)
    
    # Recommendations
    logger.info("\nRECOMMENDATIONS:")
    if not trading_api_result:
        logger.info("- Check Trading API credentials and permissions")
    if not market_data_api_result:
        logger.info("- Check Market Data API credentials and permissions")
        logger.info("- Verify that your Alpaca account has access to market data")
        logger.info("- Consider upgrading to a paid plan if you need more market data access")
    if not sdk_result:
        logger.info("- Check SDK installation and dependencies")
    
    logger.info("=" * 80)
