"""
Alpaca Broker Service for RealTradR

This module implements the broker interface for Alpaca, providing access to US stocks.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from app.services.broker_base import BrokerBase
from app.core.config import settings

logger = logging.getLogger(__name__)

class AlpacaBroker(BrokerBase):
    """Alpaca broker implementation"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        data_url: Optional[str] = None,
        paper_trading: Optional[bool] = None
    ):
        """
        Initialize the Alpaca broker
        
        Args:
            api_key: Alpaca API key (optional, defaults to config)
            api_secret: Alpaca API secret (optional, defaults to config)
            base_url: Alpaca base URL (optional, defaults to config)
            data_url: Alpaca data URL (optional, defaults to config)
            paper_trading: Whether to use paper trading (optional, defaults to config)
        """
        self.api_key = api_key or settings.alpaca_api_key
        self.api_secret = api_secret or settings.alpaca_api_secret
        self.base_url = base_url or settings.alpaca_base_url
        self.data_url = data_url or settings.alpaca_data_url
        self.paper_trading = paper_trading if paper_trading is not None else settings.alpaca_paper_trading
        
        self.api = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to the Alpaca API
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.connected = True
            logger.info(f"Connected to Alpaca. Account ID: {account.id}, Status: {account.status}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Alpaca API
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        # Alpaca REST API doesn't require explicit disconnection
        self.api = None
        self.connected = False
        logger.info("Disconnected from Alpaca")
        return True
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dict[str, Any]: Account information including cash, equity, buying power, etc.
        """
        if not self.connected:
            self.connect()
            
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'status': account.status,
                'currency': account.currency,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': account.daytrade_count,
                'last_equity': float(account.last_equity),
                'multiplier': account.multiplier,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'is_paper': self.paper_trading
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            return {'error': str(e)}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
            List[Dict[str, Any]]: List of positions with symbol, quantity, entry price, current price, etc.
        """
        if not self.connected:
            self.connect()
            
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': 'long' if float(position.qty) > 0 else 'short',
                    'exchange': position.exchange
                }
                for position in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []
    
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
        if not self.connected:
            self.connect()
            
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force=time_in_force
            )
            return self._format_order(order)
        except Exception as e:
            logger.error(f"Failed to place market order: {str(e)}")
            return {'error': str(e)}
    
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
        if not self.connected:
            self.connect()
            
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force=time_in_force,
                limit_price=limit_price
            )
            return self._format_order(order)
        except Exception as e:
            logger.error(f"Failed to place limit order: {str(e)}")
            return {'error': str(e)}
    
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
        if not self.connected:
            self.connect()
            
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='stop',
                time_in_force=time_in_force,
                stop_price=stop_price
            )
            return self._format_order(order)
        except Exception as e:
            logger.error(f"Failed to place stop order: {str(e)}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        if not self.connected:
            self.connect()
            
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order information
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        if not self.connected:
            self.connect()
            
        try:
            order = self.api.get_order(order_id)
            return self._format_order(order)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {str(e)}")
            return {'error': str(e)}
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all orders, optionally filtered by status
        
        Args:
            status: Optional filter by status ('open', 'closed', 'all')
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        if not self.connected:
            self.connect()
            
        try:
            if status == 'open':
                orders = self.api.list_orders(status='open')
            elif status == 'closed':
                orders = self.api.list_orders(status='closed')
            else:
                orders = self.api.list_orders()
                
            return [self._format_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Failed to get orders: {str(e)}")
            return []
    
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
        if not self.connected:
            self.connect()
            
        try:
            # Map timeframe to Alpaca format
            timeframe_map = {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '1h': '1Hour',
                '1d': 'Day',
                '1w': 'Week',
            }
            alpaca_timeframe = timeframe_map.get(timeframe.lower(), timeframe)
            
            # Set end time to now if not provided
            end = end or datetime.now()
            
            # Adjust limit if needed
            limit = min(limit or 1000, 10000)  # Alpaca's max is 10000
            
            # Get bars
            bars = self.api.get_bars(
                symbol=symbol,
                timeframe=alpaca_timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=limit
            ).df
            
            if bars.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Format columns
            formatted_bars = pd.DataFrame({
                'timestamp': bars.index,
                'open': bars['open'],
                'high': bars['high'],
                'low': bars['low'],
                'close': bars['close'],
                'volume': bars['volume']
            })
            
            return formatted_bars
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the last quote for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Last quote information
        """
        if not self.connected:
            self.connect()
            
        try:
            quote = self.api.get_last_quote(symbol)
            return {
                'symbol': symbol,
                'bid_price': float(quote.bidprice),
                'bid_size': quote.bidsize,
                'ask_price': float(quote.askprice),
                'ask_size': quote.asksize,
                'timestamp': quote.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to get last quote for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Get the last trade for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Last trade information
        """
        if not self.connected:
            self.connect()
            
        try:
            trade = self.api.get_last_trade(symbol)
            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': trade.size,
                'exchange': trade.exchange,
                'timestamp': trade.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to get last trade for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        if not self.connected:
            self.connect()
            
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check if market is open: {str(e)}")
            return False
    
    def get_tradable_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of tradable assets
        
        Returns:
            List[Dict[str, Any]]: List of assets with metadata
        """
        if not self.connected:
            self.connect()
            
        try:
            assets = self.api.list_assets(status='active')
            return [
                {
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': asset.exchange,
                    'asset_class': asset.class_,
                    'tradable': asset.tradable,
                    'marginable': asset.marginable,
                    'shortable': asset.shortable,
                    'easy_to_borrow': asset.easy_to_borrow
                }
                for asset in assets
            ]
        except Exception as e:
            logger.error(f"Failed to get tradable assets: {str(e)}")
            return []
    
    def _format_order(self, order) -> Dict[str, Any]:
        """
        Format an Alpaca order object to a dictionary
        
        Args:
            order: Alpaca order object
            
        Returns:
            Dict[str, Any]: Formatted order information
        """
        return {
            'id': order.id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'qty': float(order.qty),
            'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
            'side': order.side,
            'type': order.type,
            'time_in_force': order.time_in_force,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None,
            'status': order.status,
            'created_at': order.created_at,
            'updated_at': order.updated_at,
            'submitted_at': order.submitted_at,
            'filled_at': order.filled_at,
            'expired_at': order.expired_at,
            'canceled_at': order.canceled_at,
            'failed_at': order.failed_at,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'is_paper': self.paper_trading
        }
