"""
Trade model for RealTradR

This module defines the Trade model for tracking executed trades.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class Trade(Base):
    """Model for executed trades"""
    __tablename__ = "trades"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    order_id = sa.Column(sa.Integer, sa.ForeignKey("orders.id"))
    user_id = sa.Column(sa.Integer, sa.ForeignKey("users.id"))
    symbol_id = sa.Column(sa.Integer, sa.ForeignKey("trading_symbols.id"))
    execution_price = sa.Column(sa.Numeric(16, 8), nullable=False)
    quantity = sa.Column(sa.Numeric(16, 8), nullable=False)
    side = sa.Column(sa.String(10), nullable=False)  # 'buy' or 'sell'
    commission = sa.Column(sa.Numeric(16, 8))
    executed_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    profit_loss = sa.Column(sa.Numeric(16, 8))  # Realized P&L if closing trade
    is_paper = sa.Column(sa.Boolean, default=True)  # Paper trading or live trading
    broker_trade_id = sa.Column(sa.String(100))  # Broker's trade ID
    
    # Relationships
    user = relationship("User", back_populates="trades")
    symbol = relationship("TradingSymbol", back_populates="trades")
    order = relationship("Order", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade {self.id}: {self.side} {self.quantity} {self.symbol_id} @ {self.execution_price}>"
