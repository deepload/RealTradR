"""
Token schema for RealTradR API

This module defines Pydantic models for authentication token handling.
"""

from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    """Schema for authentication token"""
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    """Schema for token payload"""
    sub: Optional[int] = None
