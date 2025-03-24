#!/usr/bin/env python
"""
Test script for Alpaca API connectivity with direct credential input
"""
import sys
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_alpaca_connection(api_key, api_secret):
    """Test connection to Alpaca APIs with provided credentials"""
    # Test Trading API
    trading_success = test_trading_api(api_key, api_secret)
    
    # Add a separator
    logger.info("\n" + "="*80 + "\n")
    
    # Test Market Data API
    data_success = test_market_data_api(api_key, api_secret)
    
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

def test_trading_api(api_key, api_secret):
    """Test connection to Alpaca Trading API"""
    try:
        logger.info("Testing Alpaca Trading API...")
        
        # Paper trading URL
        base_url = "https://paper-api.alpaca.markets"
        
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

def test_market_data_api(api_key, api_secret):
    """Test connection to Alpaca Market Data API"""
    try:
        logger.info("Testing Alpaca Market Data API...")
        
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
    # Check if API key and secret are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python test_alpaca_direct.py <API_KEY> <API_SECRET>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    api_secret = sys.argv[2]
    
    # Test Alpaca connection
    test_alpaca_connection(api_key, api_secret)

if __name__ == "__main__":
    main()
