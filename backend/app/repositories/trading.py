"""
Trading repository for RealTradR

This module provides repositories for trading-related operations.
"""

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from datetime import datetime
from sqlalchemy import desc, func

from app.models.trading_symbol import TradingSymbol
from app.models.order import Order
from app.models.trade import Trade
from app.models.market_data import MarketData, NewsSentiment

from app.schemas.trading import (
    TradingSymbolCreate, TradingSymbolUpdate,
    OrderCreate, OrderUpdate,
    TradeCreate,
    MarketDataCreate
)

from app.repositories.base import BaseRepository


class TradingSymbolRepository(BaseRepository[TradingSymbol, TradingSymbolCreate, TradingSymbolUpdate]):
    """Repository for trading symbol operations"""
    
    def __init__(self, db: Session):
        """
        Initialize the repository
        
        Args:
            db: Database session
        """
        super().__init__(TradingSymbol, db)
    
    def get_by_symbol(self, symbol: str) -> Optional[TradingSymbol]:
        """
        Get a trading symbol by symbol
        
        Args:
            symbol: Symbol string
            
        Returns:
            Optional[TradingSymbol]: Trading symbol or None if not found
        """
        return self.db.query(TradingSymbol).filter(TradingSymbol.symbol == symbol).first()
    
    def get_active_symbols(self) -> List[TradingSymbol]:
        """
        Get all active trading symbols
        
        Returns:
            List[TradingSymbol]: List of active trading symbols
        """
        return self.db.query(TradingSymbol).filter(
            TradingSymbol.enabled_for_trading == True
        ).all()
    
    def search_symbols(self, query: str) -> List[TradingSymbol]:
        """
        Search for trading symbols
        
        Args:
            query: Search query
            
        Returns:
            List[TradingSymbol]: List of matching trading symbols
        """
        return self.db.query(TradingSymbol).filter(
            TradingSymbol.symbol.ilike(f"%{query}%") | 
            TradingSymbol.name.ilike(f"%{query}%")
        ).limit(50).all()


class OrderRepository(BaseRepository[Order, OrderCreate, OrderUpdate]):
    """Repository for order operations"""
    
    def __init__(self, db: Session):
        """
        Initialize the repository
        
        Args:
            db: Database session
        """
        super().__init__(Order, db)
    
    def get_by_broker_order_id(self, broker_order_id: str) -> Optional[Order]:
        """
        Get an order by broker order ID
        
        Args:
            broker_order_id: Broker's order ID
            
        Returns:
            Optional[Order]: Order or None if not found
        """
        return self.db.query(Order).filter(Order.broker_order_id == broker_order_id).first()
    
    def get_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """
        Get an order by client order ID
        
        Args:
            client_order_id: Client's order ID
            
        Returns:
            Optional[Order]: Order or None if not found
        """
        return self.db.query(Order).filter(Order.client_order_id == client_order_id).first()
    
    def get_user_orders(
        self, 
        user_id: int, 
        status: Optional[str] = None,
        symbol_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        Get orders for a user
        
        Args:
            user_id: User ID
            status: Optional order status
            symbol_id: Optional symbol ID
            limit: Maximum number of orders to return
            
        Returns:
            List[Order]: List of orders
        """
        query = self.db.query(Order).filter(Order.user_id == user_id)
        
        if status:
            query = query.filter(Order.status == status)
            
        if symbol_id:
            query = query.filter(Order.symbol_id == symbol_id)
            
        return query.order_by(desc(Order.created_at)).limit(limit).all()
    
    def get_active_orders(self, user_id: int) -> List[Order]:
        """
        Get active orders for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List[Order]: List of active orders
        """
        return self.db.query(Order).filter(
            Order.user_id == user_id,
            Order.status.in_(["pending", "submitted", "accepted", "partial"])
        ).all()
    
    def create_from_dict(self, order_data: Dict[str, Any]) -> Order:
        """
        Create an order from dictionary
        
        Args:
            order_data: Order data
            
        Returns:
            Order: Created order
        """
        db_obj = Order(**order_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def update_order_status(
        self, 
        order_id: int, 
        status: str, 
        filled_quantity: Optional[float] = None,
        filled_price: Optional[float] = None,
        broker_order_id: Optional[str] = None
    ) -> Order:
        """
        Update an order's status
        
        Args:
            order_id: Order ID
            status: New status
            filled_quantity: Optional filled quantity
            filled_price: Optional filled price
            broker_order_id: Optional broker order ID
            
        Returns:
            Order: Updated order
        """
        order = self.get(id=order_id)
        
        if not order:
            return None
            
        # Update status
        order.status = status
        order.updated_at = datetime.utcnow()
        
        # Update status-specific fields
        if status == "submitted":
            order.submitted_at = datetime.utcnow()
            
        if status == "filled":
            order.filled_at = datetime.utcnow()
            
        if status == "cancelled":
            order.cancelled_at = datetime.utcnow()
            
        if status == "expired":
            order.expired_at = datetime.utcnow()
            
        # Update filled information
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
            
        if filled_price is not None:
            order.filled_price = filled_price
            
        # Update broker order ID
        if broker_order_id:
            order.broker_order_id = broker_order_id
            
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        
        return order


class TradeRepository(BaseRepository[Trade, TradeCreate, Dict[str, Any]]):
    """Repository for trade operations"""
    
    def __init__(self, db: Session):
        """
        Initialize the repository
        
        Args:
            db: Database session
        """
        super().__init__(Trade, db)
    
    def get_by_broker_trade_id(self, broker_trade_id: str) -> Optional[Trade]:
        """
        Get a trade by broker trade ID
        
        Args:
            broker_trade_id: Broker's trade ID
            
        Returns:
            Optional[Trade]: Trade or None if not found
        """
        return self.db.query(Trade).filter(Trade.broker_trade_id == broker_trade_id).first()
    
    def get_user_trades(
        self, 
        user_id: int, 
        symbol_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trades for a user
        
        Args:
            user_id: User ID
            symbol_id: Optional symbol ID
            start_date: Optional start date
            end_date: Optional end date
            limit: Maximum number of trades to return
            
        Returns:
            List[Trade]: List of trades
        """
        query = self.db.query(Trade).filter(Trade.user_id == user_id)
        
        if symbol_id:
            query = query.filter(Trade.symbol_id == symbol_id)
            
        if start_date:
            query = query.filter(Trade.executed_at >= start_date)
            
        if end_date:
            query = query.filter(Trade.executed_at <= end_date)
            
        return query.order_by(desc(Trade.executed_at)).limit(limit).all()
    
    def create_from_order(
        self, 
        order: Order, 
        execution_price: float,
        commission: Optional[float] = 0
    ) -> Trade:
        """
        Create a trade from an order
        
        Args:
            order: Order
            execution_price: Execution price
            commission: Optional commission
            
        Returns:
            Trade: Created trade
        """
        profit_loss = None
        
        # If this is a closing trade, calculate profit/loss
        if order.side == "sell":
            # Find matching buys
            matching_buys = self.db.query(Trade).filter(
                Trade.user_id == order.user_id,
                Trade.symbol_id == order.symbol_id,
                Trade.side == "buy"
            ).order_by(Trade.executed_at.asc()).all()
            
            # Calculate P&L if we have matching buys
            if matching_buys:
                total_cost = 0
                for buy in matching_buys:
                    total_cost += buy.execution_price * buy.quantity
                
                avg_cost = total_cost / sum(buy.quantity for buy in matching_buys)
                profit_loss = (execution_price - avg_cost) * order.quantity - commission
        
        # Create the trade
        trade_data = {
            "order_id": order.id,
            "user_id": order.user_id,
            "symbol_id": order.symbol_id,
            "execution_price": execution_price,
            "quantity": order.quantity,
            "side": order.side,
            "commission": commission,
            "executed_at": datetime.utcnow(),
            "profit_loss": profit_loss,
            "is_paper": order.is_paper,
            "broker_trade_id": order.broker_order_id
        }
        
        db_obj = Trade(**trade_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def get_user_performance(self, user_id: int) -> Dict[str, Any]:
        """
        Get performance metrics for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Get total trades
        trade_count = self.db.query(func.count(Trade.id)).filter(
            Trade.user_id == user_id
        ).scalar()
        
        # Get profitable trades
        profitable_trades = self.db.query(func.count(Trade.id)).filter(
            Trade.user_id == user_id,
            Trade.profit_loss > 0
        ).scalar()
        
        # Get total profit/loss
        total_pl = self.db.query(func.sum(Trade.profit_loss)).filter(
            Trade.user_id == user_id
        ).scalar()
        
        # Get best trade
        best_trade = self.db.query(Trade).filter(
            Trade.user_id == user_id
        ).order_by(desc(Trade.profit_loss)).first()
        
        # Get worst trade
        worst_trade = self.db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.profit_loss.isnot(None)
        ).order_by(Trade.profit_loss).first()
        
        return {
            "total_trades": trade_count or 0,
            "profitable_trades": profitable_trades or 0,
            "win_rate": (profitable_trades / trade_count) if trade_count > 0 else 0,
            "total_profit_loss": total_pl or 0,
            "average_profit_loss": (total_pl / trade_count) if trade_count > 0 else 0,
            "best_trade": {
                "id": best_trade.id,
                "symbol_id": best_trade.symbol_id,
                "profit_loss": best_trade.profit_loss,
                "executed_at": best_trade.executed_at
            } if best_trade and best_trade.profit_loss else None,
            "worst_trade": {
                "id": worst_trade.id,
                "symbol_id": worst_trade.symbol_id,
                "profit_loss": worst_trade.profit_loss,
                "executed_at": worst_trade.executed_at
            } if worst_trade else None
        }


class MarketDataRepository(BaseRepository[MarketData, MarketDataCreate, Dict[str, Any]]):
    """Repository for market data operations"""
    
    def __init__(self, db: Session):
        """
        Initialize the repository
        
        Args:
            db: Database session
        """
        super().__init__(MarketData, db)
    
    def get_history(
        self, 
        symbol_id: int,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MarketData]:
        """
        Get market data history
        
        Args:
            symbol_id: Symbol ID
            timeframe: Timeframe
            start_date: Start date
            end_date: Optional end date
            limit: Maximum number of records to return
            
        Returns:
            List[MarketData]: List of market data
        """
        query = self.db.query(MarketData).filter(
            MarketData.symbol_id == symbol_id,
            MarketData.timeframe == timeframe,
            MarketData.timestamp >= start_date
        )
        
        if end_date:
            query = query.filter(MarketData.timestamp <= end_date)
            
        return query.order_by(MarketData.timestamp).limit(limit).all()
    
    def get_latest(self, symbol_id: int, timeframe: str) -> Optional[MarketData]:
        """
        Get latest market data
        
        Args:
            symbol_id: Symbol ID
            timeframe: Timeframe
            
        Returns:
            Optional[MarketData]: Latest market data or None if not found
        """
        return self.db.query(MarketData).filter(
            MarketData.symbol_id == symbol_id,
            MarketData.timeframe == timeframe
        ).order_by(desc(MarketData.timestamp)).first()
    
    def batch_insert(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Insert multiple market data records
        
        Args:
            data_list: List of market data dictionaries
        """
        self.db.bulk_insert_mappings(MarketData, data_list)
        self.db.commit()


class NewsSentimentRepository(BaseRepository[NewsSentiment, Dict[str, Any], Dict[str, Any]]):
    """Repository for news sentiment operations"""
    
    def __init__(self, db: Session):
        """
        Initialize the repository
        
        Args:
            db: Database session
        """
        super().__init__(NewsSentiment, db)
    
    def get_for_symbol(
        self, 
        symbol_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsSentiment]:
        """
        Get news sentiment for a symbol
        
        Args:
            symbol_id: Symbol ID
            start_date: Optional start date
            end_date: Optional end date
            limit: Maximum number of records to return
            
        Returns:
            List[NewsSentiment]: List of news sentiment
        """
        query = self.db.query(NewsSentiment).filter(
            NewsSentiment.symbol_id == symbol_id
        )
        
        if start_date:
            query = query.filter(NewsSentiment.published_at >= start_date)
            
        if end_date:
            query = query.filter(NewsSentiment.published_at <= end_date)
            
        return query.order_by(desc(NewsSentiment.published_at)).limit(limit).all()
    
    def get_sentiment_score(
        self, 
        symbol_id: int,
        days: int = 7
    ) -> float:
        """
        Get average sentiment score for a symbol
        
        Args:
            symbol_id: Symbol ID
            days: Number of days to look back
            
        Returns:
            float: Average sentiment score
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = self.db.query(func.avg(NewsSentiment.sentiment_score)).filter(
            NewsSentiment.symbol_id == symbol_id,
            NewsSentiment.published_at >= cutoff_date
        ).scalar()
        
        return result or 0.0
