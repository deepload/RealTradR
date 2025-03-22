#!/usr/bin/env python
"""
Test script for verifying IBKR connection

This script tests the connection to Interactive Brokers using the IBKR Broker
service. It's a simple utility to verify that the connection can be established.
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the Python path so we can import from app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ibkr_broker import IBKRBroker
from app.core.config import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_connection():
    """Test connection to IBKR"""
    settings = get_settings()
    
    # Get configuration from settings
    host = os.getenv("IBKR_HOST", settings.IBKR_HOST)
    port = int(os.getenv("IBKR_PORT", settings.IBKR_PORT))
    client_id = int(os.getenv("IBKR_CLIENT_ID", settings.IBKR_CLIENT_ID))
    paper_trading = os.getenv("IBKR_PAPER_TRADING", "true").lower() == "true"
    
    logger.info(f"Connecting to IBKR at {host}:{port} with client ID {client_id}")
    logger.info(f"Paper trading: {paper_trading}")
    
    try:
        # Create IBKR broker instance
        broker = IBKRBroker(
            host=host,
            port=port,
            client_id=client_id,
            paper_trading=paper_trading
        )
        
        # Try to connect
        logger.info("Attempting to connect to IBKR...")
        success = broker.connect()
        
        if success:
            logger.info("Successfully connected to IBKR!")
            
            # Check if we can get account information
            logger.info("Retrieving account information...")
            account = broker.get_account()
            logger.info(f"Account ID: {account['account_id']}")
            logger.info(f"Cash: {account['cash']}")
            logger.info(f"Portfolio Value: {account['portfolio_value']}")
            
            # Check if we can get market data
            logger.info("Retrieving market data for AAPL...")
            quote = broker.get_quote(symbol="AAPL")
            logger.info(f"AAPL Last Price: {quote['last_price']}")
            
            # Disconnect
            logger.info("Disconnecting from IBKR...")
            broker.disconnect()
            logger.info("Disconnected from IBKR.")
            
            return True
        else:
            logger.error("Failed to connect to IBKR")
            return False
    
    except Exception as e:
        logger.error(f"Error connecting to IBKR: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting IBKR connection test")
    
    if test_connection():
        logger.info("IBKR connection test successful!")
        sys.exit(0)
    else:
        logger.error("IBKR connection test failed!")
        sys.exit(1)
