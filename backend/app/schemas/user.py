"""
User schema for RealTradR API

This module defines Pydantic models for user-related API interactions.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator, Field

class UserBase(BaseModel):
    """Base schema for user data"""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    """Schema for user creation"""
    password: str
    
    @validator('password')
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserUpdate(BaseModel):
    """Schema for user updates"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None

class UserInDBBase(UserBase):
    """Base schema for user data from database"""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class User(UserInDBBase):
    """Schema for user data returned to API"""
    pass

class UserWithStats(User):
    """Schema for user data with trading statistics"""
    total_trades: int = 0
    successful_trades: int = 0
    win_rate: float = 0.0
    portfolio_value: float = 0.0
    account_balance: float = 0.0
    
    class Config:
        orm_mode = True
