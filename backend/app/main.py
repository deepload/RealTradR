"""
RealTradR - AI-Powered Stock Trading Simulation Bot
FastAPI Backend Application
"""

import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import os
from typing import List

from app.api.endpoints import (
    users, auth, trading, market_data, backtest, 
    strategies, ai_models, brokers
)
from app.core.config import settings
from app.db.session import engine
from app.db.base import Base
from app.core.security import get_current_active_user

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.log_file)
    ]
)
logger = logging.getLogger(__name__)

# Create all database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RealTradR API",
    description="AI-Powered Stock Trading Simulation Bot API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(market_data.router, prefix="/api/market-data", tags=["Market Data"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtesting"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["Trading Strategies"])
app.include_router(ai_models.router, prefix="/api/ai", tags=["AI Models"])
app.include_router(brokers.router, prefix="/api/brokers", tags=["Brokers"])

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/config", tags=["Configuration"])
async def get_configuration(current_user = Depends(get_current_active_user)):
    """Get application configuration (admin only)"""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Safe configuration without sensitive information
    safe_config = {
        "server": settings.dict(exclude={"secret_key"}),
        "trading": {
            "default_symbols": settings.default_symbols,
            "trading_hours": settings.trading_hours,
            "risk_settings": {
                "risk_per_trade": settings.risk_per_trade,
                "max_positions": settings.max_positions
            }
        },
        "ai": {
            "enabled_models": settings.enabled_ai_models,
            "update_frequency": settings.model_update_frequency
        },
        "brokers": {
            "available": settings.available_brokers,
            "default": settings.default_broker
        }
    }
    return safe_config

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
