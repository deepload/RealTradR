"""
Trading Strategy model for RealTradR

This module defines the TradingStrategy model for algorithmic trading strategies.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class TradingStrategy(Base):
    """Model for trading strategies"""
    __tablename__ = "trading_strategies"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    name = sa.Column(sa.String(50), nullable=False)
    description = sa.Column(sa.Text)
    strategy_type = sa.Column(sa.String(50))  # e.g., 'technical', 'ai', 'sentiment', 'combined'
    is_active = sa.Column(sa.Boolean, default=True)
    parameters = sa.Column(sa.JSON)  # Strategy parameters
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    orders = relationship("Order", back_populates="strategy")
    backtests = relationship("Backtest", back_populates="strategy")
    
    def __repr__(self):
        return f"<TradingStrategy {self.name}>"

class Backtest(Base):
    """Model for strategy backtesting results"""
    __tablename__ = "backtests"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    user_id = sa.Column(sa.Integer, sa.ForeignKey("users.id"))
    strategy_id = sa.Column(sa.Integer, sa.ForeignKey("trading_strategies.id"))
    symbol = sa.Column(sa.String(20), sa.ForeignKey("trading_symbols.symbol"))
    start_date = sa.Column(sa.DateTime, nullable=False)
    end_date = sa.Column(sa.DateTime, nullable=False)
    initial_capital = sa.Column(sa.Numeric(16, 2), nullable=False)
    final_capital = sa.Column(sa.Numeric(16, 2), nullable=False)
    total_trades = sa.Column(sa.Integer, nullable=False)
    winning_trades = sa.Column(sa.Integer, nullable=False)
    profit_loss = sa.Column(sa.Numeric(16, 2), nullable=False)
    sharpe_ratio = sa.Column(sa.Numeric(8, 4))
    max_drawdown = sa.Column(sa.Numeric(8, 4))
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    parameters = sa.Column(sa.JSON)  # Strategy parameters used for backtest
    
    # Relationships
    user = relationship("User", back_populates="backtests")
    strategy = relationship("TradingStrategy", back_populates="backtests")
    
    def __repr__(self):
        return f"<Backtest {self.id}: {self.symbol} ({self.start_date} to {self.end_date})>"
