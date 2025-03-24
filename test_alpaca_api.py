#!/usr/bin/env python
"""
Test script for Alpaca API connectivity
"""
import os
import sys
import logging
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trading_api():
    """Test connection to Alpaca Trading API"""
    try:
        logger.info("Testing Alpaca Trading API...")
        
        # Get API credentials
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        base_url = os.getenv("ALPACA_API_BASE_URL")
        
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        logger.info(f"Using Base URL: {base_url}")
        
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        # Test account info
        account = api.get_account()
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Account Cash: ${float(account.cash):.2f}")
        logger.info(f"Account Equity: ${float(account.equity):.2f}")
        
        # Test market clock
        clock = api.get_clock()
        logger.info(f"Market is {'open' if clock.is_open else 'closed'}")
        logger.info(f"Next market open: {clock.next_open}")
        logger.info(f"Next market close: {clock.next_close}")
        
        # Test positions
        positions = api.list_positions()
        logger.info(f"Current positions: {len(positions)}")
        for position in positions:
            logger.info(f"  {position.symbol}: {position.qty} shares at ${float(position.avg_entry_price):.2f}")
        
        logger.info("Trading API test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Trading API: {e}")
        return False

def test_market_data_api():
    """Test connection to Alpaca Market Data API"""
    try:
        logger.info("Testing Alpaca Market Data API...")
        
        # Get API credentials
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        
        # For market data, we need to use the data API URL
        data_url = "https://data.alpaca.markets"
        
        logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
        logger.info(f"Using Data URL: {data_url}")
        
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, data_url, api_version='v2')
        
        # Test getting bars
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        logger.info(f"Getting bars for {symbols} from {start_date.date()} to {end_date.date()}")
        
        # Get daily bars
        bars = api.get_bars(
            symbols,
            tradeapi.TimeFrame.Day,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            adjustment='raw'
        ).df
        
        if len(bars) > 0:
            logger.info(f"Received {len(bars)} bars")
            logger.info(f"Sample data:\n{bars.head()}")
            logger.info("Market Data API test successful!")
            return True
        else:
            logger.warning("No bars received from Market Data API")
            return False
        
    except Exception as e:
        logger.error(f"Error testing Market Data API: {e}")
        return False

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    # Print the first and last 4 characters of the credentials for verification
    logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
    logger.info(f"Using API Secret: {api_secret[:4]}...{api_secret[-4:]}")
    
    # Test Trading API
    trading_success = test_trading_api()
    
    # Add a separator
    logger.info("\n" + "="*80 + "\n")
    
    # Test Market Data API
    data_success = test_market_data_api()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("API Test Summary:")
    logger.info(f"Trading API: {'SUCCESS' if trading_success else 'FAILED'}")
    logger.info(f"Market Data API: {'SUCCESS' if data_success else 'FAILED'}")
    logger.info("="*80)
    
    if not trading_success or not data_success:
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Check that your API key and secret are correct")
        logger.info("2. For paper trading, use https://paper-api.alpaca.markets")
        logger.info("3. For live trading, use https://api.alpaca.markets")
        logger.info("4. For market data, use https://data.alpaca.markets")
        logger.info("5. Make sure your API key has the necessary permissions")
        logger.info("6. Check Alpaca's status page: https://status.alpaca.markets/")

if __name__ == "__main__":
    main()
