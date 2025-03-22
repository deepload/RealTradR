"""
Mock IBKR client for testing the IBKR broker service.

This module provides a mock implementation of the Interactive Brokers API client
that can be used for testing purposes without connecting to actual IBKR services.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import random


class MockIBKRClient:
    """Mock implementation of the IB API client"""
    
    def __init__(self):
        """Initialize the mock client with default values"""
        self.connected = False
        self.client_id = random.randint(1000, 9999)
        self.req_id = 1
        self.next_order_id = 1
        self.orders = {}
        self.positions = {}
        self.account_info = {
            "accountId": f"DU{random.randint(100000, 999999)}",
            "totalCashValue": 100000.0,
            "buyingPower": 200000.0,
            "equityWithLoanValue": 100000.0,
            "initMarginReq": 0.0,
            "maintMarginReq": 0.0,
            "availableFunds": 100000.0,
            "excessLiquidity": 100000.0,
            "currency": "USD"
        }
        self.market_data = {}
        self.quotes = {}
        
        # Populate some sample data
        self._populate_sample_data()
    
    def _populate_sample_data(self):
        """Populate sample market data and positions"""
        # Sample positions
        self.positions = {
            "AAPL": {
                "symbol": "AAPL",
                "position": 100,
                "marketPrice": 150.0,
                "marketValue": 15000.0,
                "averageCost": 140.0,
                "unrealizedPNL": 1000.0,
                "realizedPNL": 500.0,
                "accountName": self.account_info["accountId"]
            },
            "MSFT": {
                "symbol": "MSFT",
                "position": 50,
                "marketPrice": 300.0,
                "marketValue": 15000.0,
                "averageCost": 280.0,
                "unrealizedPNL": 1000.0,
                "realizedPNL": 200.0,
                "accountName": self.account_info["accountId"]
            }
        }
        
        # Sample market data for common symbols
        for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
            base_price = random.uniform(100.0, 500.0)
            now = datetime.now()
            
            # Generate historical data for the last 30 days
            history = []
            for i in range(30):
                date = now - timedelta(days=i)
                daily_volatility = base_price * 0.02  # 2% daily volatility
                
                open_price = base_price + random.uniform(-daily_volatility, daily_volatility)
                high_price = open_price * (1 + random.uniform(0, 0.02))
                low_price = open_price * (1 - random.uniform(0, 0.02))
                close_price = low_price + random.uniform(0, high_price - low_price)
                volume = random.randint(1000000, 10000000)
                
                bar = {
                    "timestamp": date,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume
                }
                history.append(bar)
                
                # Update base price for next day
                base_price = close_price
            
            self.market_data[symbol] = history
            
            # Set current quote
            last_bar = history[0]
            self.quotes[symbol] = {
                "symbol": symbol,
                "last": last_bar["close"],
                "bid": last_bar["close"] - 0.01,
                "ask": last_bar["close"] + 0.01,
                "bidSize": random.randint(100, 1000),
                "askSize": random.randint(100, 1000),
                "volume": last_bar["volume"],
                "timestamp": datetime.now()
            }
    
    def connect(self) -> bool:
        """Mock connection to IBKR API"""
        self.connected = True
        return True
    
    def disconnect(self) -> None:
        """Mock disconnection from IBKR API"""
        self.connected = False
    
    def is_connected(self) -> bool:
        """Check if mock client is connected"""
        return self.connected
    
    def get_next_order_id(self) -> int:
        """Get next valid order ID"""
        order_id = self.next_order_id
        self.next_order_id += 1
        return order_id
    
    def place_order(self, contract: Dict[str, Any], order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a mock order
        
        Args:
            contract: Contract details
            order: Order details
            
        Returns:
            Dict[str, Any]: Order details
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        order_id = self.get_next_order_id()
        
        # Create an order in our mock system
        mock_order = {
            "id": str(order_id),
            "client_id": self.client_id,
            "symbol": contract.get("symbol"),
            "action": order.get("action"),
            "order_type": order.get("orderType"),
            "total_quantity": order.get("totalQuantity"),
            "limit_price": order.get("lmtPrice"),
            "aux_price": order.get("auxPrice"),
            "tif": order.get("tif"),
            "status": "Submitted",
            "filled": 0,
            "remaining": order.get("totalQuantity"),
            "avg_fill_price": 0.0,
            "perm_id": random.randint(100000, 999999),
            "parent_id": order.get("parentId"),
            "account": self.account_info["accountId"],
            "created_at": datetime.now()
        }
        
        self.orders[order_id] = mock_order
        return mock_order
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a mock order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        if order_id in self.orders:
            order = self.orders[order_id]
            
            # Only allow cancellation if not already filled
            if order["status"] not in ["Filled", "Cancelled"]:
                order["status"] = "Cancelled"
                return True
        
        return False
    
    def get_order(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a mock order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Optional[Dict[str, Any]]: Order or None if not found
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        return self.orders.get(order_id)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get mock positions
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        return list(self.positions.values())
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get mock account summary
        
        Returns:
            Dict[str, Any]: Account summary
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        return self.account_info
    
    def get_historical_data(
        self, 
        contract: Dict[str, Any],
        end_date_time: str,
        duration_str: str,
        bar_size_setting: str,
        what_to_show: str,
        use_rth: int
    ) -> List[Dict[str, Any]]:
        """
        Get mock historical data
        
        Args:
            contract: Contract details
            end_date_time: End date and time
            duration_str: Duration
            bar_size_setting: Bar size
            what_to_show: Data type
            use_rth: Use regular trading hours
            
        Returns:
            List[Dict[str, Any]]: Historical data
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        symbol = contract.get("symbol")
        
        # Get data for symbol or generate random data if not in our sample data
        if symbol in self.market_data:
            return self.market_data[symbol]
        else:
            # Generate random data
            now = datetime.now()
            base_price = random.uniform(100.0, 500.0)
            data = []
            
            for i in range(30):
                date = now - timedelta(days=i)
                daily_volatility = base_price * 0.02
                
                open_price = base_price + random.uniform(-daily_volatility, daily_volatility)
                high_price = open_price * (1 + random.uniform(0, 0.02))
                low_price = open_price * (1 - random.uniform(0, 0.02))
                close_price = low_price + random.uniform(0, high_price - low_price)
                volume = random.randint(1000000, 10000000)
                
                bar = {
                    "timestamp": date,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume
                }
                data.append(bar)
                
                # Update base price for next day
                base_price = close_price
            
            return data
    
    def get_market_data(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get mock market data
        
        Args:
            contract: Contract details
            
        Returns:
            Dict[str, Any]: Market data
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")
        
        symbol = contract.get("symbol")
        
        if symbol in self.quotes:
            return self.quotes[symbol]
        else:
            # Generate random quote
            price = random.uniform(100, 500)
            return {
                "symbol": symbol,
                "last": price,
                "bid": price - 0.01,
                "ask": price + 0.01,
                "bidSize": random.randint(100, 1000),
                "askSize": random.randint(100, 1000),
                "volume": random.randint(100000, 1000000),
                "timestamp": datetime.now()
            }
