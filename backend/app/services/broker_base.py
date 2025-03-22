"""
Broker Base Service for RealTradR

This module defines the base broker service interface that all broker implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd


class BrokerBase(ABC):
    """Abstract base class for broker implementations"""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker API
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dict[str, Any]: Account information including cash, equity, buying power, etc.
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
            List[Dict[str, Any]]: List of positions with symbol, quantity, entry price, current price, etc.
        """
        pass
    
    @abstractmethod
    def place_market_order(
        self, 
        symbol: str, 
        qty: float, 
        side: str, 
        time_in_force: str = 'day'
    ) -> Dict[str, Any]:
        """
        Place a market order
        
        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            time_in_force: Time in force (day, gtc, etc.)
            
        Returns:
            Dict[str, Any]: Order information
        """
        pass
    
    @abstractmethod
    def place_limit_order(
        self, 
        symbol: str, 
        qty: float, 
        side: str, 
        limit_price: float, 
        time_in_force: str = 'day'
    ) -> Dict[str, Any]:
        """
        Place a limit order
        
        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            limit_price: Limit price
            time_in_force: Time in force (day, gtc, etc.)
            
        Returns:
            Dict[str, Any]: Order information
        """
        pass
    
    @abstractmethod
    def place_stop_order(
        self, 
        symbol: str, 
        qty: float, 
        side: str, 
        stop_price: float, 
        time_in_force: str = 'day'
    ) -> Dict[str, Any]:
        """
        Place a stop order
        
        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            stop_price: Stop price
            time_in_force: Time in force (day, gtc, etc.)
            
        Returns:
            Dict[str, Any]: Order information
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order information
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all orders, optionally filtered by status
        
        Args:
            status: Optional filter by status (open, closed, all)
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        pass
    
    @abstractmethod
    def get_bars(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: Optional[datetime] = None, 
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical price bars
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 1d, etc.)
            start: Start time
            end: End time (optional)
            limit: Maximum number of bars (optional)
            
        Returns:
            pd.DataFrame: Dataframe with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the last quote for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Last quote information
        """
        pass
    
    @abstractmethod
    def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Get the last trade for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Last trade information
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        pass
    
    @abstractmethod
    def get_tradable_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of tradable assets
        
        Returns:
            List[Dict[str, Any]]: List of assets with metadata
        """
        pass
