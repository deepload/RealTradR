"""
AI schemas for RealTradR API

This module defines Pydantic models for AI model and strategy-related API interactions.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ModelTypeEnum(str, Enum):
    """Types of AI models"""
    LSTM = "lstm"
    CNN = "cnn"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"


class ModelStatusEnum(str, Enum):
    """Status of AI models"""
    TRAINING = "training"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


class StrategyTypeEnum(str, Enum):
    """Types of trading strategies"""
    AI_PREDICTION = "ai_prediction"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    HYBRID = "hybrid"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    CUSTOM = "custom"


class StrategyStatusEnum(str, Enum):
    """Status of trading strategies"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"


class AIModelBase(BaseModel):
    """Base schema for AI model"""
    name: str
    model_type: ModelTypeEnum
    description: Optional[str] = None
    symbol_id: Optional[int] = None
    timeframe: str = "1d"
    lookback_periods: int = 30
    prediction_periods: int = 1
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    features: List[str] = Field(default_factory=list)


class AIModelCreate(AIModelBase):
    """Schema for creating an AI model"""
    user_id: Optional[int] = None


class AIModelUpdate(BaseModel):
    """Schema for updating an AI model"""
    name: Optional[str] = None
    description: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    status: Optional[ModelStatusEnum] = None
    is_active: Optional[bool] = None


class AIModel(AIModelBase):
    """Schema for AI model from database"""
    id: int
    user_id: int
    status: ModelStatusEnum
    accuracy: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    last_trained: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    model_path: Optional[str] = None

    class Config:
        orm_mode = True


class ModelPredictionBase(BaseModel):
    """Base schema for model prediction"""
    model_id: int
    symbol_id: int
    timestamp: datetime
    prediction_for: datetime
    predicted_value: float
    prediction_type: str = "price"  # price, direction, sentiment, etc.
    confidence: Optional[float] = None


class ModelPredictionCreate(ModelPredictionBase):
    """Schema for creating a model prediction"""
    pass


class ModelPrediction(ModelPredictionBase):
    """Schema for model prediction from database"""
    id: int
    actual_value: Optional[float] = None
    error: Optional[float] = None
    created_at: datetime

    class Config:
        orm_mode = True


class TradingStrategyBase(BaseModel):
    """Base schema for trading strategy"""
    name: str
    strategy_type: StrategyTypeEnum
    description: Optional[str] = None
    symbols: List[int] = Field(default_factory=list)
    timeframe: str = "1d"
    entry_conditions: Dict[str, Any] = Field(default_factory=dict)
    exit_conditions: Dict[str, Any] = Field(default_factory=dict)
    position_sizing: Dict[str, Any] = Field(default_factory=dict)
    risk_management: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False


class TradingStrategyCreate(TradingStrategyBase):
    """Schema for creating a trading strategy"""
    user_id: Optional[int] = None
    ai_models: Optional[List[int]] = None


class TradingStrategyUpdate(BaseModel):
    """Schema for updating a trading strategy"""
    name: Optional[str] = None
    description: Optional[str] = None
    entry_conditions: Optional[Dict[str, Any]] = None
    exit_conditions: Optional[Dict[str, Any]] = None
    position_sizing: Optional[Dict[str, Any]] = None
    risk_management: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    status: Optional[StrategyStatusEnum] = None
    ai_models: Optional[List[int]] = None


class TradingStrategy(TradingStrategyBase):
    """Schema for trading strategy from database"""
    id: int
    user_id: int
    status: StrategyStatusEnum
    created_at: datetime
    updated_at: datetime
    last_run: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    ai_models: Optional[List[AIModel]] = None

    class Config:
        orm_mode = True


class BacktestResultBase(BaseModel):
    """Base schema for backtest result"""
    strategy_id: int
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    annual_return: float
    params: Dict[str, Any] = Field(default_factory=dict)


class BacktestResultCreate(BacktestResultBase):
    """Schema for creating a backtest result"""
    user_id: Optional[int] = None


class BacktestResult(BacktestResultBase):
    """Schema for backtest result from database"""
    id: int
    user_id: int
    created_at: datetime
    strategy: Optional[TradingStrategy] = None

    class Config:
        orm_mode = True


class NewsSentimentBase(BaseModel):
    """Base schema for news sentiment"""
    symbol_id: int
    headline: str
    source: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    sentiment_score: float
    content_summary: Optional[str] = None


class NewsSentimentCreate(NewsSentimentBase):
    """Schema for creating a news sentiment"""
    pass


class NewsSentiment(NewsSentimentBase):
    """Schema for news sentiment from database"""
    id: int
    processed_at: datetime
    
    class Config:
        orm_mode = True
