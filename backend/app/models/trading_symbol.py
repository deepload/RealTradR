"""
Trading Symbol model for RealTradR

This module defines the TradingSymbol model for storing information about tradable assets.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class TradingSymbol(Base):
    """Model for tradable financial instruments"""
    __tablename__ = "trading_symbols"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    symbol = sa.Column(sa.String(20), unique=True, index=True, nullable=False)
    name = sa.Column(sa.String(100))
    exchange = sa.Column(sa.String(20))
    asset_class = sa.Column(sa.String(20))  # e.g., 'stock', 'crypto', 'forex'
    is_active = sa.Column(sa.Boolean, default=True)
    last_price = sa.Column(sa.Numeric(12, 4))
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="symbol")
    orders = relationship("Order", back_populates="symbol")
    trades = relationship("Trade", back_populates="symbol")
    ai_models = relationship("AIModel", back_populates="symbol")
    news_sentiment = relationship("NewsSentiment", back_populates="symbol")
    
    def __repr__(self):
        return f"<TradingSymbol {self.symbol} ({self.exchange})>"
