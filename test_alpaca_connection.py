import os
import json
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from JSON file"""
    try:
        with open("strategy_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    config = load_config()
    
    # Get API credentials from config
    api_credentials = config.get("api_credentials", {})
    api_key = api_credentials.get("api_key")
    api_secret = api_credentials.get("api_secret")
    paper_trading = api_credentials.get("paper_trading", True)
    
    # Set base URL based on paper trading flag
    base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
    
    logger.info(f"Testing connection to Alpaca API at {base_url}")
    logger.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        # Test account access
        logger.info("Testing account access...")
        account = api.get_account()
        logger.info(f"✅ Account access successful")
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Account cash: ${float(account.cash):.2f}")
        
        # Test positions access
        logger.info("Testing positions access...")
        positions = api.list_positions()
        logger.info(f"✅ Positions access successful")
        logger.info(f"Number of positions: {len(positions)}")
        
        # Test data API access
        logger.info("Testing market data API access...")
        data_url = "https://data.alpaca.markets"
        data_api = tradeapi.REST(api_key, api_secret, data_url, api_version='v2')
        
        # Test getting a snapshot
        symbol = "AAPL"
        logger.info(f"Testing snapshot data for {symbol}...")
        try:
            snapshot = data_api.get_snapshot(symbol)
            logger.info(f"✅ Snapshot data access successful")
            logger.info(f"Latest price for {symbol}: ${snapshot.latest_trade.price:.2f}")
        except Exception as e:
            logger.error(f"❌ Error getting snapshot: {e}")
        
        # Test getting historical data
        logger.info(f"Testing historical data for {symbol}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        try:
            bars = data_api.get_bars(symbol, "1D", start=start_str, end=end_str).df
            logger.info(f"✅ Historical data access successful")
            logger.info(f"Number of bars: {len(bars)}")
            if not bars.empty:
                logger.info(f"Latest close price: ${bars['close'].iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
        
        logger.info("All tests completed")
        
    except Exception as e:
        logger.error(f"❌ Error connecting to Alpaca API: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ALPACA API CONNECTION TEST")
    logger.info("=" * 80)
    
    success = test_alpaca_connection()
    
    if success:
        logger.info("✅ All tests passed successfully!")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    logger.info("=" * 80)
