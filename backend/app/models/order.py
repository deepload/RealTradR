"""
Order model for RealTradR

This module defines the Order model for tracking trading orders.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class Order(Base):
    """Model for trading orders"""
    __tablename__ = "orders"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    user_id = sa.Column(sa.Integer, sa.ForeignKey("users.id"))
    symbol_id = sa.Column(sa.Integer, sa.ForeignKey("trading_symbols.id"))
    broker = sa.Column(sa.String(20), nullable=False)  # 'alpaca' or 'ibkr'
    order_id = sa.Column(sa.String(100))  # Broker's order ID
    order_type = sa.Column(sa.String(20), nullable=False)  # 'market', 'limit', 'stop', 'stop_limit'
    side = sa.Column(sa.String(10), nullable=False)  # 'buy' or 'sell'
    quantity = sa.Column(sa.Numeric(16, 8), nullable=False)
    price = sa.Column(sa.Numeric(16, 8))  # For limit orders
    stop_price = sa.Column(sa.Numeric(16, 8))  # For stop orders
    status = sa.Column(sa.String(20), nullable=False)  # 'open', 'filled', 'canceled', 'rejected', 'partial'
    strategy_id = sa.Column(sa.Integer, sa.ForeignKey("trading_strategies.id"))
    ai_model_id = sa.Column(sa.Integer, sa.ForeignKey("ai_models.id"))
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_at = sa.Column(sa.DateTime)
    is_paper = sa.Column(sa.Boolean, default=True)  # Paper trading or live trading
    notes = sa.Column(sa.Text)  # Additional information
    
    # Relationships
    user = relationship("User", back_populates="orders")
    symbol = relationship("TradingSymbol", back_populates="orders")
    strategy = relationship("TradingStrategy", back_populates="orders")
    ai_model = relationship("AIModel")
    trades = relationship("Trade", back_populates="order")
    
    def __repr__(self):
        return f"<Order {self.id}: {self.side} {self.quantity} {self.symbol_id} @ {self.price or 'market'} ({self.status})>"
