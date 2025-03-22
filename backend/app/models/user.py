"""
User model for RealTradR

This module defines the User model and related database operations.
"""

from typing import Optional, List
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base
from app.core.security import get_password_hash, verify_password

class User(Base):
    """User model for authentication and profile information"""
    __tablename__ = "users"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    username = sa.Column(sa.String(50), unique=True, index=True, nullable=False)
    email = sa.Column(sa.String(100), unique=True, index=True, nullable=False)
    hashed_password = sa.Column(sa.String(100), nullable=False)
    full_name = sa.Column(sa.String(100))
    is_active = sa.Column(sa.Boolean, default=True)
    is_superuser = sa.Column(sa.Boolean, default=False)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    backtests = relationship("Backtest", back_populates="user")
    
    @property
    def password(self):
        """Password getter - raises error as password should not be retrievable"""
        raise AttributeError("Password is not a readable attribute")
    
    @password.setter
    def password(self, password: str):
        """Hash password when setting it"""
        self.hashed_password = get_password_hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash"""
        return verify_password(password, self.hashed_password)
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.is_superuser
    
    def __repr__(self):
        return f"<User {self.username}>"
