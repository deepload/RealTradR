"""
AI Model for RealTradR

This module defines the AIModel for machine learning models used for prediction.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base

class AIModel(Base):
    """Model for AI/ML models used for predictions"""
    __tablename__ = "ai_models"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    name = sa.Column(sa.String(50), nullable=False)
    model_type = sa.Column(sa.String(50), nullable=False)  # 'lstm', 'sentiment', 'reinforcement', etc.
    symbol_id = sa.Column(sa.Integer, sa.ForeignKey("trading_symbols.id"))
    model_path = sa.Column(sa.String(255))  # Path to saved model
    accuracy = sa.Column(sa.Numeric(5, 2))  # Accuracy metric (0-100%)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    last_trained = sa.Column(sa.DateTime)
    is_active = sa.Column(sa.Boolean, default=True)
    parameters = sa.Column(sa.JSON)  # Model parameters
    
    # Relationships
    symbol = relationship("TradingSymbol", back_populates="ai_models")
    predictions = relationship("AIPrediction", back_populates="model")
    
    def __repr__(self):
        return f"<AIModel {self.name} ({self.model_type})>"

class AIPrediction(Base):
    """Model for storing AI model predictions"""
    __tablename__ = "ai_predictions"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    model_id = sa.Column(sa.Integer, sa.ForeignKey("ai_models.id"), nullable=False)
    symbol = sa.Column(sa.String(20), nullable=False)
    prediction_time = sa.Column(sa.DateTime, default=datetime.utcnow)
    target_time = sa.Column(sa.DateTime, nullable=False)  # Time for which prediction is made
    prediction_value = sa.Column(sa.Numeric(16, 8), nullable=False)  # Predicted value
    confidence = sa.Column(sa.Numeric(5, 4))  # Confidence score (0-1)
    actual_value = sa.Column(sa.Numeric(16, 8))  # Actual value when known
    accuracy = sa.Column(sa.Numeric(8, 4))  # Prediction accuracy when known
    
    # Relationships
    model = relationship("AIModel", back_populates="predictions")
    
    def __repr__(self):
        return f"<AIPrediction {self.id}: {self.symbol} @ {self.target_time}>"
