#!/usr/bin/env python
"""
Test script for running the enhanced strategy with real Alpaca data
"""
import os
import sys
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import alpaca_trade_api as tradeapi

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RealTradR modules
from backend.app.ai.enhanced_strategy import EnhancedAIStrategy
from backend.app.ai.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file="strategy_config.json"):
    """Load strategy configuration from file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def test_real_data():
    """Test getting real data from Alpaca"""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    api_key = "PK88UAXEPBIEQCEAS8YV"
    api_secret = "hu9YIoZSqhLiOLLoTkHt5mh2NK4gTi7fXPQf6X1L"
    
    # Check if credentials are available
    if not api_key or not api_secret:
        logger.error("API credentials not found in environment variables")
        return False
    
    logger.info(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")
    
    # Initialize API for market data
    data_url = "https://data.alpaca.markets"
    api = tradeapi.REST(api_key, api_secret, data_url, api_version='v2')
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Format dates
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Getting data for {symbols} from {start_str} to {end_str}")
    
    try:
        # Get daily bars
        bars = api.get_bars(
            symbols,
            tradeapi.TimeFrame.Day,
            start=start_str,
            end=end_str,
            adjustment='raw'
        ).df
        
        if len(bars) > 0:
            logger.info(f"Successfully retrieved {len(bars)} bars")
            logger.info(f"Sample data:\n{bars.head()}")
            
            # Save to CSV for inspection
            csv_file = "alpaca_real_data.csv"
            bars.to_csv(csv_file)
            logger.info(f"Saved data to {csv_file}")
            
            # Process data for each symbol
            for symbol in symbols:
                if symbol in bars.index.get_level_values('symbol'):
                    symbol_data = bars.loc[symbol]
                    logger.info(f"\nData for {symbol}:")
                    logger.info(f"Shape: {symbol_data.shape}")
                    logger.info(f"Columns: {symbol_data.columns.tolist()}")
                    logger.info(f"First 3 rows:\n{symbol_data.head(3)}")
                    
                    # Calculate technical indicators
                    logger.info(f"\nCalculating technical indicators for {symbol}")
                    try:
                        indicators = TechnicalIndicators.add_all_indicators(symbol_data.copy())
                        logger.info(f"Technical indicators calculated successfully")
                        logger.info(f"Indicator columns: {[col for col in indicators.columns if col not in symbol_data.columns]}")
                    except Exception as e:
                        logger.error(f"Error calculating indicators: {e}")
                else:
                    logger.warning(f"No data found for {symbol}")
            
            return True
        else:
            logger.warning("No data received from Alpaca")
            return False
    
    except Exception as e:
        logger.error(f"Error getting real data: {e}")
        return False

def test_strategy_with_real_data():
    """Test running the enhanced strategy with real data"""
    # Load configuration
    config = load_config()
    
    # Initialize strategy
    strategy = EnhancedAIStrategy(config=config)
    
    # Run strategy
    logger.info("Running strategy with real data...")
    results = strategy.run()
    
    if results:
        logger.info("Strategy executed successfully")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        return True
    else:
        logger.error("Strategy execution failed")
        return False

if __name__ == "__main__":
    # Test getting real data
    data_success = test_real_data()
    
    # Add separator
    logger.info("\n" + "="*80 + "\n")
    
    # Test running strategy with real data
    if data_success:
        strategy_success = test_strategy_with_real_data()
    else:
        logger.warning("Skipping strategy test due to data retrieval failure")
        strategy_success = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Test Summary:")
    logger.info(f"Real Data Retrieval: {'SUCCESS' if data_success else 'FAILED'}")
    logger.info(f"Strategy Execution: {'SUCCESS' if strategy_success else 'FAILED'}")
    logger.info("="*80)
