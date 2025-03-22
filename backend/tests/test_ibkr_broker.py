"""
Unit tests for IBKR broker service

This module tests the IBKR broker service using a mock IBKR client.
"""

import unittest
from unittest.mock import patch, MagicMock
import datetime
from decimal import Decimal

from app.services.ibkr_broker import IBKRBroker
from tests.utils.mock_ibkr import MockIBKRClient


class TestIBKRBroker(unittest.TestCase):
    """Test case for IBKR broker service"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create a patcher for the ibapi.client.EClient
        self.ib_client_patcher = patch('app.services.ibkr_broker.EClient')
        # Create a patcher for the ibapi.wrapper.EWrapper
        self.ib_wrapper_patcher = patch('app.services.ibkr_broker.EWrapper')
        # Create a patcher for the threading.Thread
        self.thread_patcher = patch('app.services.ibkr_broker.Thread')
        
        # Start the patchers
        self.mock_client = self.ib_client_patcher.start()
        self.mock_wrapper = self.ib_wrapper_patcher.start()
        self.mock_thread = self.thread_patcher.start()
        
        # Create a mock IBKR client
        self.mock_ibkr = MockIBKRClient()
        
        # Create an IBKRBroker instance with default settings
        self.broker = IBKRBroker(
            host="127.0.0.1",
            port=7497,
            client_id=1,
            paper_trading=True
        )
        
        # Replace the real IB client with our mock
        self.broker.client = self.mock_ibkr
    
    def tearDown(self):
        """Clean up after each test"""
        # Stop the patchers
        self.ib_client_patcher.stop()
        self.ib_wrapper_patcher.stop()
        self.thread_patcher.stop()
    
    def test_connect_and_disconnect(self):
        """Test connection and disconnection"""
        # Connect to IBKR
        self.broker.connect()
        self.assertTrue(self.mock_ibkr.is_connected())
        
        # Disconnect from IBKR
        self.broker.disconnect()
        self.assertFalse(self.mock_ibkr.is_connected())
    
    def test_is_connected(self):
        """Test is_connected method"""
        # Should be False initially
        self.assertFalse(self.broker.is_connected())
        
        # Connect and check again
        self.broker.connect()
        self.assertTrue(self.broker.is_connected())
        
        # Disconnect and check again
        self.broker.disconnect()
        self.assertFalse(self.broker.is_connected())
    
    def test_place_market_order(self):
        """Test placing market orders"""
        # Connect first
        self.broker.connect()
        
        # Place a buy market order
        buy_order = self.broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side="buy"
        )
        
        # Verify the order
        self.assertIsNotNone(buy_order)
        self.assertEqual(buy_order["symbol"], "AAPL")
        self.assertEqual(buy_order["quantity"], 10)
        self.assertEqual(buy_order["side"], "buy")
        self.assertEqual(buy_order["order_type"], "market")
        
        # Place a sell market order
        sell_order = self.broker.place_market_order(
            symbol="MSFT",
            qty=5,
            side="sell"
        )
        
        # Verify the order
        self.assertIsNotNone(sell_order)
        self.assertEqual(sell_order["symbol"], "MSFT")
        self.assertEqual(sell_order["quantity"], 5)
        self.assertEqual(sell_order["side"], "sell")
        self.assertEqual(sell_order["order_type"], "market")
    
    def test_place_limit_order(self):
        """Test placing limit orders"""
        # Connect first
        self.broker.connect()
        
        # Place a buy limit order
        buy_order = self.broker.place_limit_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            limit_price=150.0
        )
        
        # Verify the order
        self.assertIsNotNone(buy_order)
        self.assertEqual(buy_order["symbol"], "AAPL")
        self.assertEqual(buy_order["quantity"], 10)
        self.assertEqual(buy_order["side"], "buy")
        self.assertEqual(buy_order["order_type"], "limit")
        self.assertEqual(buy_order["limit_price"], 150.0)
        
        # Place a sell limit order
        sell_order = self.broker.place_limit_order(
            symbol="MSFT",
            qty=5,
            side="sell",
            limit_price=300.0
        )
        
        # Verify the order
        self.assertIsNotNone(sell_order)
        self.assertEqual(sell_order["symbol"], "MSFT")
        self.assertEqual(sell_order["quantity"], 5)
        self.assertEqual(sell_order["side"], "sell")
        self.assertEqual(sell_order["order_type"], "limit")
        self.assertEqual(sell_order["limit_price"], 300.0)
    
    def test_place_stop_order(self):
        """Test placing stop orders"""
        # Connect first
        self.broker.connect()
        
        # Place a buy stop order
        buy_order = self.broker.place_stop_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            stop_price=160.0
        )
        
        # Verify the order
        self.assertIsNotNone(buy_order)
        self.assertEqual(buy_order["symbol"], "AAPL")
        self.assertEqual(buy_order["quantity"], 10)
        self.assertEqual(buy_order["side"], "buy")
        self.assertEqual(buy_order["order_type"], "stop")
        self.assertEqual(buy_order["stop_price"], 160.0)
        
        # Place a sell stop order
        sell_order = self.broker.place_stop_order(
            symbol="MSFT",
            qty=5,
            side="sell",
            stop_price=280.0
        )
        
        # Verify the order
        self.assertIsNotNone(sell_order)
        self.assertEqual(sell_order["symbol"], "MSFT")
        self.assertEqual(sell_order["quantity"], 5)
        self.assertEqual(sell_order["side"], "sell")
        self.assertEqual(sell_order["order_type"], "stop")
        self.assertEqual(sell_order["stop_price"], 280.0)
    
    def test_place_stop_limit_order(self):
        """Test placing stop-limit orders"""
        # Connect first
        self.broker.connect()
        
        # Place a buy stop-limit order
        buy_order = self.broker.place_stop_limit_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            limit_price=155.0,
            stop_price=160.0
        )
        
        # Verify the order
        self.assertIsNotNone(buy_order)
        self.assertEqual(buy_order["symbol"], "AAPL")
        self.assertEqual(buy_order["quantity"], 10)
        self.assertEqual(buy_order["side"], "buy")
        self.assertEqual(buy_order["order_type"], "stop_limit")
        self.assertEqual(buy_order["limit_price"], 155.0)
        self.assertEqual(buy_order["stop_price"], 160.0)
        
        # Place a sell stop-limit order
        sell_order = self.broker.place_stop_limit_order(
            symbol="MSFT",
            qty=5,
            side="sell",
            limit_price=275.0,
            stop_price=280.0
        )
        
        # Verify the order
        self.assertIsNotNone(sell_order)
        self.assertEqual(sell_order["symbol"], "MSFT")
        self.assertEqual(sell_order["quantity"], 5)
        self.assertEqual(sell_order["side"], "sell")
        self.assertEqual(sell_order["order_type"], "stop_limit")
        self.assertEqual(sell_order["limit_price"], 275.0)
        self.assertEqual(sell_order["stop_price"], 280.0)
    
    def test_cancel_order(self):
        """Test cancelling orders"""
        # Connect first
        self.broker.connect()
        
        # Place an order
        order = self.broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side="buy"
        )
        
        # Cancel the order
        result = self.broker.cancel_order(order_id=order["id"])
        
        # Verify the result
        self.assertTrue(result)
        
        # Get the order and verify its status
        cancelled_order = self.broker.get_order(order_id=order["id"])
        self.assertEqual(cancelled_order["status"], "Cancelled")
    
    def test_get_order(self):
        """Test getting order details"""
        # Connect first
        self.broker.connect()
        
        # Place an order
        order = self.broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side="buy"
        )
        
        # Get the order
        retrieved_order = self.broker.get_order(order_id=order["id"])
        
        # Verify the order
        self.assertIsNotNone(retrieved_order)
        self.assertEqual(retrieved_order["id"], order["id"])
        self.assertEqual(retrieved_order["symbol"], "AAPL")
        self.assertEqual(retrieved_order["action"], "BUY")
    
    def test_get_positions(self):
        """Test getting positions"""
        # Connect first
        self.broker.connect()
        
        # Get positions
        positions = self.broker.get_positions()
        
        # Verify the positions
        self.assertIsNotNone(positions)
        self.assertIsInstance(positions, list)
        self.assertGreater(len(positions), 0)
        
        # Check structure of a position
        position = positions[0]
        self.assertIn("symbol", position)
        self.assertIn("quantity", position)
        self.assertIn("market_price", position)
        self.assertIn("market_value", position)
        self.assertIn("avg_cost", position)
        self.assertIn("unrealized_pl", position)
    
    def test_get_account(self):
        """Test getting account information"""
        # Connect first
        self.broker.connect()
        
        # Get account information
        account = self.broker.get_account()
        
        # Verify the account information
        self.assertIsNotNone(account)
        self.assertIn("account_id", account)
        self.assertIn("cash", account)
        self.assertIn("portfolio_value", account)
        self.assertIn("buying_power", account)
    
    def test_get_historical_data(self):
        """Test getting historical data"""
        # Connect first
        self.broker.connect()
        
        # Get historical data for AAPL
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
        
        historical_data = self.broker.get_historical_data(
            symbol="AAPL",
            timeframe="1d",
            start=start_date,
            end=end_date
        )
        
        # Verify the historical data
        self.assertIsNotNone(historical_data)
        self.assertIsInstance(historical_data, list)
        self.assertGreater(len(historical_data), 0)
        
        # Check structure of a bar
        bar = historical_data[0]
        self.assertIn("timestamp", bar)
        self.assertIn("open", bar)
        self.assertIn("high", bar)
        self.assertIn("low", bar)
        self.assertIn("close", bar)
        self.assertIn("volume", bar)
    
    def test_get_quote(self):
        """Test getting quotes"""
        # Connect first
        self.broker.connect()
        
        # Get quote for AAPL
        quote = self.broker.get_quote(symbol="AAPL")
        
        # Verify the quote
        self.assertIsNotNone(quote)
        self.assertEqual(quote["symbol"], "AAPL")
        self.assertIn("last_price", quote)
        self.assertIn("bid", quote)
        self.assertIn("ask", quote)
        self.assertIn("bid_size", quote)
        self.assertIn("ask_size", quote)
        self.assertIn("volume", quote)
        self.assertIn("timestamp", quote)


if __name__ == "__main__":
    unittest.main()
