"""
Fallback ML Models for RealTradR

This module provides machine learning models that don't rely on TensorFlow,
using scikit-learn instead. This allows the advanced strategy to function
even when TensorFlow has DLL loading issues on Windows.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FallbackPricePredictor:
    """
    Price prediction model using scikit-learn instead of TensorFlow.
    Uses an ensemble of RandomForest, GradientBoosting, and SVR models.
    """
    
    def __init__(self, model_dir="./models"):
        """
        Initialize the price predictor.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.prediction_horizon = 5  # Default: predict 5 days ahead
        self.lookback_window = 30    # Default: use 30 days of history
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Initialized FallbackPricePredictor with model_dir: {model_dir}")
    
    def _create_features(self, df):
        """
        Create features for the ML model from price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Basic price features
        data['return_1d'] = data['close'].pct_change(1)
        data['return_5d'] = data['close'].pct_change(5)
        data['return_10d'] = data['close'].pct_change(10)
        data['return_20d'] = data['close'].pct_change(20)
        
        # Moving averages
        data['ma_5'] = data['close'].rolling(window=5).mean()
        data['ma_10'] = data['close'].rolling(window=10).mean()
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['ma_50'] = data['close'].rolling(window=50).mean()
        
        # Price relative to moving averages
        data['close_ma_5_ratio'] = data['close'] / data['ma_5']
        data['close_ma_10_ratio'] = data['close'] / data['ma_10']
        data['close_ma_20_ratio'] = data['close'] / data['ma_20']
        data['close_ma_50_ratio'] = data['close'] / data['ma_50']
        
        # Volatility
        data['volatility_5d'] = data['return_1d'].rolling(window=5).std()
        data['volatility_10d'] = data['return_1d'].rolling(window=10).std()
        data['volatility_20d'] = data['return_1d'].rolling(window=20).std()
        
        # Volume features
        data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_5']
        
        # Price range
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Trend features
        data['ma_5_10_ratio'] = data['ma_5'] / data['ma_10']
        data['ma_10_20_ratio'] = data['ma_10'] / data['ma_20']
        data['ma_20_50_ratio'] = data['ma_20'] / data['ma_50']
        
        # Drop NaN values
        data = data.dropna()
        
        # Store feature columns (excluding the target)
        self.feature_columns = [col for col in data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return data
    
    def _prepare_sequences(self, data, target_column='close', sequence_length=None):
        """
        Prepare sequences for training/prediction.
        
        Args:
            data: DataFrame with features
            target_column: Column to predict
            sequence_length: Length of input sequences
            
        Returns:
            X: Features
            y: Target values
            dates: Corresponding dates
        """
        if sequence_length is None:
            sequence_length = self.lookback_window
        
        # Get the target values (shifted by prediction_horizon)
        target_values = data[target_column].shift(-self.prediction_horizon).values
        
        # Create sequences
        X = []
        y = []
        dates = []
        
        for i in range(len(data) - sequence_length - self.prediction_horizon + 1):
            # Get sequence of features
            sequence = data[self.feature_columns].iloc[i:i+sequence_length].values
            
            # Flatten the sequence
            X.append(sequence.flatten())
            
            # Get the target value
            y.append(target_values[i+sequence_length-1])
            
            # Get the date
            dates.append(data.index[i+sequence_length-1])
        
        return np.array(X), np.array(y), dates
    
    def train(self, price_data, symbol, force_retrain=False):
        """
        Train the model for a specific symbol.
        
        Args:
            price_data: DataFrame with OHLCV data
            symbol: Symbol to train for
            force_retrain: Whether to force retraining even if model exists
            
        Returns:
            Dictionary with training metrics
        """
        model_path = os.path.join(self.model_dir, f"{symbol}_ensemble.pkl")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
        
        # Check if model already exists
        if os.path.exists(model_path) and not force_retrain:
            logger.info(f"Model for {symbol} already exists. Skipping training.")
            return {"status": "skipped", "symbol": symbol}
        
        logger.info(f"Training model for {symbol}...")
        
        # Prepare data
        df = price_data.copy()
        df = df.set_index('timestamp')
        
        # Create features
        data = self._create_features(df)
        
        # Prepare sequences
        X, y, dates = self._prepare_sequences(data)
        
        if len(X) == 0:
            logger.warning(f"Not enough data to train model for {symbol}")
            return {"status": "error", "message": "Not enough data", "symbol": symbol}
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        self.scalers[symbol] = scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        }
        
        # Train each model
        for name, model in models.items():
            logger.info(f"Training {name} model for {symbol}...")
            model.fit(X_train_scaled, y_train)
        
        # Save the ensemble
        self.models[symbol] = models
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)
        
        # Evaluate the models
        predictions = {}
        metrics = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        # Calculate ensemble prediction (average)
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        
        # Calculate ensemble metrics
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        metrics['ensemble'] = {
            'mse': ensemble_mse,
            'mae': ensemble_mae,
            'r2': ensemble_r2
        }
        
        logger.info(f"Model training completed for {symbol}")
        logger.info(f"Ensemble MSE: {ensemble_mse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "metrics": metrics,
            "feature_count": len(self.feature_columns)
        }
    
    def predict(self, price_data, symbol):
        """
        Make price predictions for a specific symbol.
        
        Args:
            price_data: DataFrame with OHLCV data
            symbol: Symbol to predict for
            
        Returns:
            Dictionary with predictions and confidence
        """
        model_path = os.path.join(self.model_dir, f"{symbol}_ensemble.pkl")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
        
        # Load model and scaler if not already loaded
        if symbol not in self.models:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.models[symbol] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
            else:
                logger.warning(f"No trained model found for {symbol}. Training now...")
                self.train(price_data, symbol)
        
        if symbol not in self.models:
            logger.error(f"Failed to load or train model for {symbol}")
            return {
                "prediction": 0,
                "confidence": 0,
                "ml_signal": 0,
                "status": "error"
            }
        
        # Prepare data
        df = price_data.copy()
        
        # Check if timestamp is already the index
        if df.index.name == 'timestamp':
            # Already indexed by timestamp
            pass
        elif 'timestamp' in df.columns:
            # Set timestamp as index if it's a column
            df = df.set_index('timestamp')
        else:
            # If no timestamp column, use the existing index
            logger.warning(f"No timestamp column found for {symbol}, using existing index")
            df.index.name = 'timestamp'  # Rename the index for consistency
        
        # Create features
        data = self._create_features(df)
        
        # Get the most recent sequence
        if len(data) < self.lookback_window:
            logger.warning(f"Not enough data for prediction. Need {self.lookback_window} data points.")
            return {
                "prediction": 0,
                "confidence": 0,
                "ml_signal": 0,
                "status": "error"
            }
        
        # Get the most recent sequence
        recent_data = data.iloc[-self.lookback_window:][self.feature_columns].values
        X = recent_data.flatten().reshape(1, -1)
        
        # Scale the features
        X_scaled = self.scalers[symbol].transform(X)
        
        # Make predictions with each model
        predictions = []
        
        for name, model in self.models[symbol].items():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
        
        # Calculate ensemble prediction (average)
        ensemble_pred = np.mean(predictions)
        
        # Calculate confidence (inverse of standard deviation)
        std = np.std(predictions)
        confidence = 1.0 / (1.0 + std) if std > 0 else 0.9
        
        # Calculate current price
        current_price = data['close'].iloc[-1]
        
        # Calculate predicted return
        predicted_return = (ensemble_pred / current_price) - 1
        
        # Convert to signal (-1 to 1)
        signal = np.clip(predicted_return * 10, -1, 1)
        
        logger.info(f"Prediction for {symbol}: {ensemble_pred:.2f} (current: {current_price:.2f})")
        logger.info(f"Predicted return: {predicted_return:.2%}, Signal: {signal:.2f}, Confidence: {confidence:.2f}")
        
        return {
            "prediction": float(ensemble_pred),
            "current_price": float(current_price),
            "predicted_return": float(predicted_return),
            "confidence": float(confidence),
            "ml_signal": float(signal),
            "status": "success"
        }


class ModelManager:
    """
    Manager for ML models, providing a unified interface for training and prediction.
    """
    
    def __init__(self, model_dir="./models"):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.price_predictor = FallbackPricePredictor(model_dir=model_dir)
        
        logger.info(f"Initialized ModelManager with model_dir: {model_dir}")
    
    def has_model(self, symbol):
        """
        Check if a model exists for a specific symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Boolean indicating if model exists
        """
        model_path = os.path.join(self.model_dir, f"{symbol}_ensemble.pkl")
        return os.path.exists(model_path)
    
    def train_models(self, price_data, symbols, force_retrain=False):
        """
        Train models for multiple symbols.
        
        Args:
            price_data: Dictionary of DataFrames with OHLCV data for each symbol
            symbols: List of symbols to train for
            force_retrain: Whether to force retraining even if models exist
            
        Returns:
            Dictionary with training results for each symbol
        """
        results = {}
        
        for symbol in symbols:
            if symbol in price_data:
                result = self.price_predictor.train(price_data[symbol], symbol, force_retrain)
                results[symbol] = result
            else:
                logger.warning(f"No price data available for {symbol}")
                results[symbol] = {"status": "error", "message": "No price data available"}
        
        return results
    
    def predict(self, price_data, symbol):
        """
        Make predictions for a specific symbol.
        
        Args:
            price_data: DataFrame with OHLCV data
            symbol: Symbol to predict for
            
        Returns:
            Dictionary with predictions
        """
        return self.price_predictor.predict(price_data, symbol)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200)
    
    # Create sample price data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0.001, 0.02, 200))
    high = close * (1 + np.random.uniform(0, 0.02, 200))
    low = close * (1 - np.random.uniform(0, 0.02, 200))
    open_price = low + np.random.uniform(0, 1, 200) * (high - low)
    volume = np.random.randint(1000000, 10000000, 200)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Create model manager
    model_manager = ModelManager(model_dir="./test_models")
    
    # Train model
    result = model_manager.train_models({"AAPL": df}, ["AAPL"], force_retrain=True)
    print(result)
    
    # Make prediction
    prediction = model_manager.predict(df, "AAPL")
    print(prediction)
