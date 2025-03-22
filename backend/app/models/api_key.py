"""
API Key model for RealTradR

This module defines the API Key model for storing broker API credentials.
"""

from typing import Optional
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class APIKey(Base):
    """Model for storing broker API credentials"""
    __tablename__ = "api_keys"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    user_id = sa.Column(sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    broker = sa.Column(sa.String(20), nullable=False)  # 'alpaca' or 'ibkr'
    api_key = sa.Column(sa.String(100), nullable=False)
    api_secret = sa.Column(sa.String(100), nullable=False)
    is_paper = sa.Column(sa.Boolean, default=True)  # Paper trading or live trading
    is_active = sa.Column(sa.Boolean, default=True)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    last_used = sa.Column(sa.DateTime)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey {self.id}: {self.broker} ({'paper' if self.is_paper else 'live'})>"
