"""
Integration tests for trading API endpoints

This module tests the trading API endpoints with a mock IBKR broker.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from app.main import app
from app.services.broker_factory import BrokerFactory
from tests.utils.mock_ibkr import MockIBKRClient


class TestTradingAPI(unittest.TestCase):
    """Test case for trading API endpoints"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.client = TestClient(app)
        
        # Mock authentication
        self.mock_auth()
        
        # Create a patcher for the BrokerFactory.create_broker method
        self.broker_factory_patcher = patch('app.api.endpoints.trading.BrokerFactory.create_broker')
        self.mock_create_broker = self.broker_factory_patcher.start()
        
        # Create a mock IBKRBroker with our MockIBKRClient
        self.mock_ibkr_broker = MagicMock()
        self.mock_ibkr_broker.client = MockIBKRClient()
        
        # Connect the mock broker and set up its methods to use the MockIBKRClient
        self.mock_ibkr_broker.connect.return_value = True
        self.mock_ibkr_broker.disconnect.return_value = None
        self.mock_ibkr_broker.is_connected.return_value = True
        
        # Set up place order methods
        self.mock_ibkr_broker.place_market_order.side_effect = self._mock_place_market_order
        self.mock_ibkr_broker.place_limit_order.side_effect = self._mock_place_limit_order
        self.mock_ibkr_broker.place_stop_order.side_effect = self._mock_place_stop_order
        self.mock_ibkr_broker.place_stop_limit_order.side_effect = self._mock_place_stop_limit_order
        
        # Set up other methods
        self.mock_ibkr_broker.cancel_order.side_effect = self._mock_cancel_order
        self.mock_ibkr_broker.get_order.side_effect = self._mock_get_order
        self.mock_ibkr_broker.get_positions.side_effect = self._mock_get_positions
        self.mock_ibkr_broker.get_account.side_effect = self._mock_get_account
        self.mock_ibkr_broker.get_historical_data.side_effect = self._mock_get_historical_data
        self.mock_ibkr_broker.get_quote.side_effect = self._mock_get_quote
        
        # Set the broker factory to return our mock broker
        self.mock_create_broker.return_value = self.mock_ibkr_broker
    
    def tearDown(self):
        """Clean up after each test"""
        # Stop the patchers
        self.broker_factory_patcher.stop()
    
    def mock_auth(self):
        """Mock authentication for API requests"""
        # Create a patcher for the get_current_user dependency
        self.auth_patcher = patch('app.api.deps.get_current_user')
        self.mock_get_current_user = self.auth_patcher.start()
        
        # Create a mock user
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_user.username = "testuser"
        mock_user.is_active = True
        mock_user.is_superuser = False
        
        # Return the mock user from the dependency
        self.mock_get_current_user.return_value = mock_user
    
    def _mock_place_market_order(self, symbol, qty, side):
        """Mock for place_market_order"""
        return {
            "id": "123456",
            "symbol": symbol,
            "quantity": qty,
            "side": side,
            "order_type": "market",
            "status": "submitted",
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_place_limit_order(self, symbol, qty, side, limit_price, time_in_force="day"):
        """Mock for place_limit_order"""
        return {
            "id": "123456",
            "symbol": symbol,
            "quantity": qty,
            "side": side,
            "order_type": "limit",
            "limit_price": limit_price,
            "time_in_force": time_in_force,
            "status": "submitted",
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_place_stop_order(self, symbol, qty, side, stop_price, time_in_force="day"):
        """Mock for place_stop_order"""
        return {
            "id": "123456",
            "symbol": symbol,
            "quantity": qty,
            "side": side,
            "order_type": "stop",
            "stop_price": stop_price,
            "time_in_force": time_in_force,
            "status": "submitted",
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_place_stop_limit_order(self, symbol, qty, side, limit_price, stop_price, time_in_force="day"):
        """Mock for place_stop_limit_order"""
        return {
            "id": "123456",
            "symbol": symbol,
            "quantity": qty,
            "side": side,
            "order_type": "stop_limit",
            "limit_price": limit_price,
            "stop_price": stop_price,
            "time_in_force": time_in_force,
            "status": "submitted",
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_cancel_order(self, order_id):
        """Mock for cancel_order"""
        return True
    
    def _mock_get_order(self, order_id):
        """Mock for get_order"""
        return {
            "id": order_id,
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market",
            "status": "filled",
            "filled_qty": 10,
            "filled_avg_price": 150.0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_get_positions(self):
        """Mock for get_positions"""
        return [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "market_price": 150.0,
                "market_value": 15000.0,
                "avg_cost": 140.0,
                "unrealized_pl": 1000.0,
                "unrealized_pl_pct": 7.14
            },
            {
                "symbol": "MSFT",
                "quantity": 50,
                "market_price": 300.0,
                "market_value": 15000.0,
                "avg_cost": 280.0,
                "unrealized_pl": 1000.0,
                "unrealized_pl_pct": 7.14
            }
        ]
    
    def _mock_get_account(self):
        """Mock for get_account"""
        return {
            "account_id": "DU123456",
            "status": "Active",
            "currency": "USD",
            "cash": 100000.0,
            "buying_power": 200000.0,
            "equity": 130000.0,
            "portfolio_value": 130000.0,
            "initial_margin": 0.0,
            "maintenance_margin": 0.0,
            "day_trades_remaining": 3
        }
    
    def _mock_get_historical_data(self, symbol, timeframe, start, end):
        """Mock for get_historical_data"""
        data = []
        current_date = start
        
        while current_date <= end:
            data.append({
                "timestamp": current_date.isoformat(),
                "open": 150.0,
                "high": 155.0,
                "low": 145.0,
                "close": 152.0,
                "volume": 1000000
            })
            
            # Increment date based on timeframe
            if timeframe == "1d":
                current_date += timedelta(days=1)
            elif timeframe == "1h":
                current_date += timedelta(hours=1)
            else:
                current_date += timedelta(days=1)
        
        return data
    
    def _mock_get_quote(self, symbol):
        """Mock for get_quote"""
        return {
            "symbol": symbol,
            "last_price": 152.0,
            "bid": 151.95,
            "ask": 152.05,
            "bid_size": 100,
            "ask_size": 100,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_create_market_order(self):
        """Test creating a market order"""
        # Create a market order
        order_data = {
            "symbol_id": 1,
            "broker": "ibkr",
            "order_type": "market",
            "side": "buy",
            "quantity": 10,
            "time_in_force": "day",
            "is_paper": True
        }
        
        with patch('app.repositories.trading.TradingSymbolRepository.get') as mock_get_symbol:
            # Mock the symbol
            mock_symbol = MagicMock()
            mock_symbol.id = 1
            mock_symbol.symbol = "AAPL"
            mock_get_symbol.return_value = mock_symbol
            
            # Mock creating an order in the database
            with patch('app.repositories.trading.OrderRepository.create_from_dict') as mock_create_order:
                mock_order = MagicMock()
                mock_order.id = 1
                mock_order.broker_order_id = None
                mock_create_order.return_value = mock_order
                
                # Mock updating order status
                with patch('app.repositories.trading.OrderRepository.update_order_status') as mock_update_order:
                    mock_updated_order = MagicMock()
                    mock_updated_order.id = 1
                    mock_updated_order.broker_order_id = "123456"
                    mock_updated_order.status = "submitted"
                    mock_update_order.return_value = mock_updated_order
                    
                    # Send the request
                    response = self.client.post("/api/trading/orders", json=order_data)
                    
                    # Verify the response
                    self.assertEqual(response.status_code, 201)
                    self.assertEqual(response.json()["status"], "submitted")
    
    def test_get_orders(self):
        """Test getting orders"""
        with patch('app.repositories.trading.OrderRepository.get_user_orders') as mock_get_orders:
            # Mock the orders
            mock_orders = [
                MagicMock(id=1, symbol_id=1, broker="ibkr", order_type="market", side="buy", quantity=10),
                MagicMock(id=2, symbol_id=1, broker="ibkr", order_type="limit", side="sell", quantity=5)
            ]
            mock_get_orders.return_value = mock_orders
            
            # Send the request
            response = self.client.get("/api/trading/orders")
            
            # Verify the response
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()), 2)
    
    def test_get_positions(self):
        """Test getting positions"""
        # Send the request
        response = self.client.get("/api/trading/positions?broker_name=ibkr&paper_trading=true")
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertGreater(len(response.json()), 0)
        
        # Check a position
        position = response.json()[0]
        self.assertIn("symbol", position)
        self.assertIn("quantity", position)
        self.assertIn("market_price", position)
        self.assertIn("market_value", position)
        self.assertIn("avg_cost", position)
        self.assertIn("unrealized_pl", position)
    
    def test_get_account_info(self):
        """Test getting account information"""
        # Send the request
        response = self.client.get("/api/trading/account?broker_name=ibkr&paper_trading=true")
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertIn("account_id", response.json())
        self.assertIn("cash", response.json())
        self.assertIn("portfolio_value", response.json())
        self.assertIn("buying_power", response.json())
    
    def test_get_market_data(self):
        """Test getting market data"""
        with patch('app.repositories.trading.TradingSymbolRepository.get_by_symbol') as mock_get_symbol:
            # Mock the symbol
            mock_symbol = MagicMock()
            mock_symbol.id = 1
            mock_symbol.symbol = "AAPL"
            mock_get_symbol.return_value = mock_symbol
            
            # Mock getting market data from the repository
            with patch('app.repositories.trading.MarketDataRepository.get_history') as mock_get_history:
                # Initially return empty list to trigger fetching from broker
                mock_get_history.return_value = []
                
                # Mock getting latest market data
                with patch('app.repositories.trading.MarketDataRepository.get_latest') as mock_get_latest:
                    mock_get_latest.return_value = None
                    
                    # Mock batch inserting market data
                    with patch('app.repositories.trading.MarketDataRepository.batch_insert') as mock_batch_insert:
                        # Update mock_get_history to return data after fetching from broker
                        mock_get_history.side_effect = [[], [
                            MagicMock(
                                symbol_id=1,
                                timestamp=datetime.now() - timedelta(days=1),
                                open=150.0,
                                high=155.0,
                                low=145.0,
                                close=152.0,
                                volume=1000000,
                                timeframe="1d"
                            )
                        ]]
                        
                        # Send the request
                        response = self.client.get("/api/trading/market-data/AAPL?timeframe=1d&limit=10")
                        
                        # Verify the response
                        self.assertEqual(response.status_code, 200)
                        self.assertIsInstance(response.json(), list)
    
    def test_get_market_quote(self):
        """Test getting market quote"""
        with patch('app.repositories.trading.TradingSymbolRepository.get_by_symbol') as mock_get_symbol:
            # Mock the symbol
            mock_symbol = MagicMock()
            mock_symbol.id = 1
            mock_symbol.symbol = "AAPL"
            mock_get_symbol.return_value = mock_symbol
            
            # Send the request
            response = self.client.get("/api/trading/market-data/AAPL/quote?broker_name=ibkr")
            
            # Verify the response
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["symbol"], "AAPL")
            self.assertIn("last_price", response.json())
            self.assertIn("bid", response.json())
            self.assertIn("ask", response.json())


if __name__ == "__main__":
    unittest.main()
