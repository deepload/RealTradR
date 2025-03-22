"""
Main API router for RealTradR

This module combines all API endpoint routers.
"""

from fastapi import APIRouter

from app.api.endpoints import auth, trading

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
