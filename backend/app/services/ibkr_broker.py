"""
Interactive Brokers Service for RealTradR

This module implements the broker interface for Interactive Brokers.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import pytz
import time

# Import ib_insync library for IBKR connectivity
try:
    from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, StopOrder, util
    from ib_insync.contract import ContractDetails
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False

from app.services.broker_base import BrokerBase
from app.core.config import settings

logger = logging.getLogger(__name__)

class IBKRBroker(BrokerBase):
    """Interactive Brokers implementation"""
    
    def __init__(
        self, 
        host: Optional[str] = None, 
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        paper_trading: Optional[bool] = None
    ):
        """
        Initialize the IBKR broker
        
        Args:
            host: IBKR TWS/Gateway host (optional, defaults to config)
            port: IBKR TWS/Gateway port (optional, defaults to config)
            client_id: Client ID (optional, defaults to config)
            paper_trading: Whether to use paper trading (optional, defaults to config)
        """
        if not HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync not installed. Please install it with pip install ib_insync."
            )
            
        self.host = host or settings.ibkr_host
        self.port = port or settings.ibkr_port
        self.client_id = client_id or settings.ibkr_client_id
        self.paper_trading = paper_trading if paper_trading is not None else settings.ibkr_paper_trading
        
        self.ib = IB()
        self.connected = False
        self.contract_cache = {}  # Cache for contracts to avoid repeated lookups
        
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers API (TWS or IB Gateway)
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.connected and self.ib.isConnected():
                return True
                
            self.ib.connect(
                host=self.host, 
                port=self.port, 
                clientId=self.client_id,
                readonly=False
            )
            
            if self.ib.isConnected():
                self.connected = True
                account_value = self.ib.accountSummary()
                account_id = account_value[0].account if account_value else "Unknown"
                logger.info(f"Connected to Interactive Brokers. Account ID: {account_id}")
                return True
            else:
                logger.error("Failed to connect to Interactive Brokers.")
                self.connected = False
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Interactive Brokers: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Interactive Brokers API
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from Interactive Brokers")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Interactive Brokers: {str(e)}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dict[str, Any]: Account information including cash, equity, buying power, etc.
        """
        if not self.connected:
            self.connect()
            
        try:
            account_values = self.ib.accountSummary()
            if not account_values:
                return {'error': 'No account data available'}
                
            # Create a dictionary from account values
            account_data = {}
            account_id = account_values[0].account
            
            for value in account_values:
                account_data[value.tag] = value.value
            
            # Format the account information
            return {
                'id': account_id,
                'status': 'active',  # IBKR doesn't provide this directly
                'currency': account_data.get('BaseCurrency', 'USD'),
                'cash': float(account_data.get('TotalCashValue', 0)),
                'portfolio_value': float(account_data.get('NetLiquidation', 0)),
                'equity': float(account_data.get('EquityWithLoanValue', 0)),
                'buying_power': float(account_data.get('BuyingPower', 0)),
                'initial_margin': float(account_data.get('InitMarginReq', 0)),
                'maintenance_margin': float(account_data.get('MaintMarginReq', 0)),
                'available_funds': float(account_data.get('AvailableFunds', 0)),
                'excess_liquidity': float(account_data.get('ExcessLiquidity', 0)),
                'day_trades_remaining': float(account_data.get('DayTradesRemaining', 0)) if 'DayTradesRemaining' in account_data else None,
                'leverage': float(account_data.get('GrossLeverage', 0)),
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
            portfolio = self.ib.portfolio()
            return [
                {
                    'symbol': position.contract.symbol,
                    'exchange': position.contract.exchange,
                    'currency': position.contract.currency,
                    'qty': float(position.position),
                    'avg_entry_price': float(position.avgCost / position.position if position.position != 0 else 0),
                    'current_price': float(position.marketPrice),
                    'market_value': float(position.marketValue),
                    'unrealized_pl': float(position.unrealizedPNL),
                    'realized_pl': float(position.realizedPNL),
                    'side': 'long' if float(position.position) > 0 else 'short',
                    'contract_type': position.contract.secType
                }
                for position in portfolio if position.position != 0
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []
    
    def _get_contract(self, symbol: str) -> Contract:
        """
        Get a contract object for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Contract: IBKR contract object
        """
        if symbol in self.contract_cache:
            return self.contract_cache[symbol]
            
        # Create a stock contract
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify the contract
        qualified_contracts = self.ib.qualifyContracts(contract)
        if not qualified_contracts:
            raise ValueError(f"Could not qualify contract for {symbol}")
            
        qualified_contract = qualified_contracts[0]
        self.contract_cache[symbol] = qualified_contract
        return qualified_contract
    
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
            # Get contract
            contract = self._get_contract(symbol)
            
            # Create order
            action = 'BUY' if side.lower() == 'buy' else 'SELL'
            tif = 'DAY' if time_in_force.lower() == 'day' else 'GTC'
            order = MarketOrder(action, abs(float(qty)), tif=tif)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give the order a moment to be processed
            
            # Return order information
            return self._format_order(trade)
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
            # Get contract
            contract = self._get_contract(symbol)
            
            # Create order
            action = 'BUY' if side.lower() == 'buy' else 'SELL'
            tif = 'DAY' if time_in_force.lower() == 'day' else 'GTC'
            order = LimitOrder(action, abs(float(qty)), limit_price, tif=tif)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give the order a moment to be processed
            
            # Return order information
            return self._format_order(trade)
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
            # Get contract
            contract = self._get_contract(symbol)
            
            # Create order
            action = 'BUY' if side.lower() == 'buy' else 'SELL'
            tif = 'DAY' if time_in_force.lower() == 'day' else 'GTC'
            order = StopOrder(action, abs(float(qty)), stop_price, tif=tif)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give the order a moment to be processed
            
            # Return order information
            return self._format_order(trade)
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
            # Find the order
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    self.ib.cancelOrder(trade.order)
                    return True
                    
            logger.warning(f"Order {order_id} not found for cancellation")
            return False
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
            # Find the order
            for trade in self.ib.trades():
                if str(trade.order.orderId) == str(order_id):
                    return self._format_order(trade)
                    
            logger.warning(f"Order {order_id} not found")
            return {'error': 'Order not found'}
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
            all_trades = self.ib.trades()
            
            if status == 'open':
                trades = [t for t in all_trades if t.isActive()]
            elif status == 'closed':
                trades = [t for t in all_trades if not t.isActive()]
            else:
                trades = all_trades
                
            return [self._format_order(trade) for trade in trades]
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
            # Get contract
            contract = self._get_contract(symbol)
            
            # Set end time to now if not provided
            end = end or datetime.now()
            
            # Map timeframe to IB format
            timeframe_map = {
                '1m': '1 min',
                '5m': '5 mins',
                '15m': '15 mins',
                '1h': '1 hour',
                '1d': '1 day',
                '1w': '1 week',
            }
            ib_timeframe = timeframe_map.get(timeframe.lower(), timeframe)
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end,
                durationStr=self._get_duration_string(start, end),
                barSizeSetting=ib_timeframe,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to dataframe
            df = util.df(bars)
            
            # Format columns
            formatted_bars = pd.DataFrame({
                'timestamp': pd.to_datetime(df['date']),
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume']
            })
            
            # Apply limit if needed
            if limit and len(formatted_bars) > limit:
                formatted_bars = formatted_bars.tail(limit)
                
            return formatted_bars
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def _get_duration_string(self, start: datetime, end: datetime) -> str:
        """
        Calculate duration string for historical data request
        
        Args:
            start: Start time
            end: End time
            
        Returns:
            str: IB duration string
        """
        delta = end - start
        days = delta.days
        
        if days <= 1:
            seconds = delta.total_seconds()
            if seconds <= 86400:  # 1 day
                return f"{int(seconds)} S"
        elif days <= 30:
            return f"{days} D"
        elif days <= 365:
            return f"{days//30 + 1} M"
        else:
            return f"{days//365 + 1} Y"
    
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
            # Get contract
            contract = self._get_contract(symbol)
            
            # Request market data
            self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data to arrive
            
            # Get quote data
            ticker = self.ib.ticker(contract)
            
            return {
                'symbol': symbol,
                'bid_price': float(ticker.bid) if ticker.bid else 0,
                'bid_size': ticker.bidSize if ticker.bidSize else 0,
                'ask_price': float(ticker.ask) if ticker.ask else 0,
                'ask_size': ticker.askSize if ticker.askSize else 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Failed to get last quote for {symbol}: {str(e)}")
            return {'error': str(e)}
        finally:
            # Cancel market data
            self.ib.cancelMktData(contract)
    
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
            # Get contract
            contract = self._get_contract(symbol)
            
            # Request market data
            self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data to arrive
            
            # Get last trade data
            ticker = self.ib.ticker(contract)
            
            return {
                'symbol': symbol,
                'price': float(ticker.last) if ticker.last else 0,
                'size': ticker.lastSize if ticker.lastSize else 0,
                'exchange': ticker.lastExchange if ticker.lastExchange else 'Unknown',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Failed to get last trade for {symbol}: {str(e)}")
            return {'error': str(e)}
        finally:
            # Cancel market data
            self.ib.cancelMktData(contract)
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        # IBKR doesn't have a direct API for this, but we can approximate
        if not self.connected:
            self.connect()
            
        try:
            # US market hours are typically 9:30 AM to 4:00 PM Eastern Time on weekdays
            now = datetime.now(pytz.timezone('US/Eastern'))
            is_weekday = now.weekday() < 5  # 0-4 are Monday to Friday
            is_market_hours = 9 * 60 + 30 <= now.hour * 60 + now.minute <= 16 * 60
            
            return is_weekday and is_market_hours
        except Exception as e:
            logger.error(f"Failed to check if market is open: {str(e)}")
            return False
    
    def get_tradable_assets(self) -> List[Dict[str, Any]]:
        """
        Get list of tradable assets
        
        Returns:
            List[Dict[str, Any]]: List of assets with metadata
        """
        # IBKR doesn't have a straightforward API for this
        # We could potentially query for specific symbols or use a predefined list
        return [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'exchange': 'NASDAQ',
                'asset_class': 'STK',
                'tradable': True
            },
            {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'exchange': 'NASDAQ',
                'asset_class': 'STK',
                'tradable': True
            },
            # Add more default assets or implement a more comprehensive solution
        ]
    
    def _format_order(self, trade) -> Dict[str, Any]:
        """
        Format an IBKR trade object to a dictionary
        
        Args:
            trade: IBKR trade object
            
        Returns:
            Dict[str, Any]: Formatted order information
        """
        order = trade.order
        contract = trade.contract
        fill = trade.fills[0] if trade.fills else None
        
        status_map = {
            'ApiPending': 'submitted',
            'PendingSubmit': 'submitted',
            'PendingCancel': 'pending_cancel',
            'Submitted': 'open',
            'ApiCancelled': 'canceled',
            'Cancelled': 'canceled',
            'Filled': 'filled',
            'Inactive': 'expired'
        }
        
        return {
            'id': str(order.orderId),
            'client_order_id': order.clientId,
            'symbol': contract.symbol,
            'qty': float(order.totalQuantity),
            'filled_qty': sum(fill.execution.shares for fill in trade.fills) if trade.fills else 0,
            'side': 'buy' if order.action.lower() == 'buy' else 'sell',
            'type': order.orderType.lower(),
            'time_in_force': order.tif.lower(),
            'limit_price': float(order.lmtPrice) if hasattr(order, 'lmtPrice') and order.lmtPrice else None,
            'stop_price': float(order.auxPrice) if hasattr(order, 'auxPrice') and order.auxPrice else None,
            'status': status_map.get(order.status, order.status.lower()),
            'created_at': trade.log[0].time if trade.log else datetime.now(),
            'updated_at': trade.log[-1].time if trade.log else datetime.now(),
            'filled_avg_price': float(fill.execution.price) if fill else None,
            'is_paper': self.paper_trading
        }
