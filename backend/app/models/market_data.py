"""
Market Data model for RealTradR

This module defines the MarketData model for storing historical price data.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class MarketData(Base):
    """Model for historical price data"""
    __tablename__ = "market_data"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    symbol_id = sa.Column(sa.Integer, sa.ForeignKey("trading_symbols.id"))
    timestamp = sa.Column(sa.DateTime, nullable=False)
    open = sa.Column(sa.Numeric(16, 8), nullable=False)
    high = sa.Column(sa.Numeric(16, 8), nullable=False)
    low = sa.Column(sa.Numeric(16, 8), nullable=False)
    close = sa.Column(sa.Numeric(16, 8), nullable=False)
    volume = sa.Column(sa.Numeric(20, 2), nullable=False)
    timeframe = sa.Column(sa.String(10), nullable=False)  # '1m', '5m', '15m', '1h', '1d', etc.
    
    # Unique constraint for symbol, timestamp, and timeframe
    __table_args__ = (
        sa.UniqueConstraint('symbol_id', 'timestamp', 'timeframe', name='unique_market_data'),
    )
    
    # Relationships
    symbol = relationship("TradingSymbol", back_populates="market_data")
    
    def __repr__(self):
        return f"<MarketData {self.symbol_id} @ {self.timestamp} ({self.timeframe})>"

class NewsSentiment(Base):
    """Model for news sentiment analysis"""
    __tablename__ = "news_sentiment"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    symbol_id = sa.Column(sa.Integer, sa.ForeignKey("trading_symbols.id"))
    headline = sa.Column(sa.Text, nullable=False)
    source = sa.Column(sa.String(50))
    url = sa.Column(sa.Text)
    published_at = sa.Column(sa.DateTime)
    sentiment_score = sa.Column(sa.Numeric(5, 4))  # -1.0 to 1.0
    processed_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    content_summary = sa.Column(sa.Text)
    
    # Relationships
    symbol = relationship("TradingSymbol", back_populates="news_sentiment")
    
    def __repr__(self):
        return f"<NewsSentiment {self.id}: {self.symbol_id} ({self.sentiment_score})>"
