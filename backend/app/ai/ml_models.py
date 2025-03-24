"""
Machine Learning Models for RealTradR

This module provides advanced machine learning models for price prediction.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure TensorFlow logs are not too verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TimeSeriesDataGenerator:
    """Class for generating time series data for ML models"""
    
    def __init__(self, sequence_length=10):
        """
        Initialize the data generator
        
        Args:
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
    def prepare_data(self, df, target_column='close', feature_columns=None, 
                    scale_features=True, test_size=0.2):
        """
        Prepare data for time series prediction
        
        Args:
            df: DataFrame with time series data
            target_column: Target column to predict
            feature_columns: List of feature columns to use
            scale_features: Whether to scale features
            test_size: Test set size as a fraction
            
        Returns:
            Dictionary with prepared data
        """
        if feature_columns is None:
            # Use all numeric columns except the target
            feature_columns = [col for col in df.columns if col != target_column 
                              and pd.api.types.is_numeric_dtype(df[col])]
        
        # Make sure target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Make sure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found in DataFrame")
        
        # Create a copy of the data
        data = df.copy()
        
        # Drop rows with NaN values
        data = data.dropna(subset=[target_column] + feature_columns)
        
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points after removing NaNs. Need at least {self.sequence_length + 1}, got {len(data)}")
        
        # Extract target and features
        y = data[target_column].values.reshape(-1, 1)
        X = data[feature_columns].values
        
        # Scale target
        y_scaled = self.price_scaler.fit_transform(y)
        
        # Scale features if requested
        if scale_features:
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Split into train and test sets
        split_idx = int(len(X_seq) * (1 - test_size))
        
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Get the corresponding dates
        dates = data.index[self.sequence_length:].values
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'feature_columns': feature_columns,
            'target_column': target_column
        }
    
    def _create_sequences(self, X, y):
        """
        Create sequences for time series prediction
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_seq, y_seq)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def inverse_transform_price(self, scaled_price):
        """
        Inverse transform scaled price
        
        Args:
            scaled_price: Scaled price array
            
        Returns:
            Original price array
        """
        # Reshape if needed
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)
        
        return self.price_scaler.inverse_transform(scaled_price)
    
    def create_prediction_sequence(self, df, feature_columns=None, scale_features=True):
        """
        Create a sequence for prediction from the latest data
        
        Args:
            df: DataFrame with latest data
            feature_columns: List of feature columns to use
            scale_features: Whether to scale features
            
        Returns:
            Sequence for prediction
        """
        if feature_columns is None:
            # Use all numeric columns
            feature_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Make sure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found in DataFrame")
        
        # Create a copy of the data
        data = df.copy()
        
        # Drop rows with NaN values
        data = data.dropna(subset=feature_columns)
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Not enough data points after removing NaNs. Need at least {self.sequence_length}, got {len(data)}")
        
        # Extract features
        X = data[feature_columns].values
        
        # Scale features if requested
        if scale_features:
            X_scaled = self.feature_scaler.transform(X)
        else:
            X_scaled = X
        
        # Create sequence
        X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, len(feature_columns))
        
        return X_seq


class LSTMModel:
    """LSTM model for time series prediction"""
    
    def __init__(self, sequence_length=10, units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize the LSTM model
        
        Args:
            sequence_length: Length of input sequences
            units: Number of LSTM units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.data_generator = TimeSeriesDataGenerator(sequence_length)
        
    def build_model(self, input_dim):
        """
        Build the LSTM model
        
        Args:
            input_dim: Input dimension (number of features)
        """
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=self.units, return_sequences=True, 
                      input_shape=(self.sequence_length, input_dim)))
        model.add(Dropout(self.dropout_rate))
        
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        self.model = model
        logger.info(f"Built LSTM model with {self.units} units and {self.dropout_rate} dropout rate")
        
        return model
    
    def train(self, df, target_column='close', feature_columns=None, epochs=100, 
             batch_size=32, validation_split=0.1, early_stopping_patience=10,
             model_path=None):
        """
        Train the LSTM model
        
        Args:
            df: DataFrame with time series data
            target_column: Target column to predict
            feature_columns: List of feature columns to use
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation set size as a fraction
            early_stopping_patience: Patience for early stopping
            model_path: Path to save the trained model
            
        Returns:
            Dictionary with training history and evaluation metrics
        """
        # Prepare data
        data = self.data_generator.prepare_data(
            df, target_column, feature_columns, scale_features=True
        )
        
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # Set up callbacks
        callbacks = []
        
        # Early stopping
        if early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(model_checkpoint)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        # Inverse transform predictions
        train_predictions = self.data_generator.inverse_transform_price(train_predictions)
        test_predictions = self.data_generator.inverse_transform_price(test_predictions)
        
        # Inverse transform actual values
        train_actual = self.data_generator.inverse_transform_price(y_train)
        test_actual = self.data_generator.inverse_transform_price(y_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
        
        train_mae = mean_absolute_error(train_actual, train_predictions)
        test_mae = mean_absolute_error(test_actual, test_predictions)
        
        train_r2 = r2_score(train_actual, train_predictions)
        test_r2 = r2_score(test_actual, test_predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        train_mape = np.mean(np.abs((train_actual - train_predictions) / train_actual)) * 100
        test_mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape
        }
        
        logger.info(f"LSTM model training completed with test RMSE: {test_rmse:.4f}, test MAPE: {test_mape:.2f}%")
        
        # Save model if path is provided
        if model_path and not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            self.model.save(model_path)
            logger.info(f"Saved LSTM model to {model_path}")
        
        return {
            'history': history.history,
            'metrics': metrics,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_dates': data['train_dates'],
            'test_dates': data['test_dates']
        }
    
    def predict(self, df, feature_columns=None):
        """
        Make predictions with the LSTM model
        
        Args:
            df: DataFrame with latest data
            feature_columns: List of feature columns to use
            
        Returns:
            Predicted value
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create prediction sequence
        X_seq = self.data_generator.create_prediction_sequence(df, feature_columns)
        
        # Make prediction
        prediction_scaled = self.model.predict(X_seq)
        
        # Inverse transform prediction
        prediction = self.data_generator.inverse_transform_price(prediction_scaled)
        
        return prediction[0, 0]
    
    def save(self, model_path, data_path=None):
        """
        Save the model and data generator
        
        Args:
            model_path: Path to save the model
            data_path: Path to save the data generator
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Saved LSTM model to {model_path}")
        
        # Save data generator if path is provided
        if data_path:
            with open(data_path, 'wb') as f:
                pickle.dump(self.data_generator, f)
            logger.info(f"Saved data generator to {data_path}")
    
    @classmethod
    def load(cls, model_path, data_path=None):
        """
        Load a saved model and data generator
        
        Args:
            model_path: Path to the saved model
            data_path: Path to the saved data generator
            
        Returns:
            Loaded LSTMModel instance
        """
        # Create a new instance
        instance = cls()
        
        # Load model
        instance.model = load_model(model_path)
        logger.info(f"Loaded LSTM model from {model_path}")
        
        # Load data generator if path is provided
        if data_path:
            with open(data_path, 'rb') as f:
                instance.data_generator = pickle.load(f)
            logger.info(f"Loaded data generator from {data_path}")
        
        return instance


class EnsembleModel:
    """Ensemble model combining multiple prediction models"""
    
    def __init__(self, models=None, weights=None):
        """
        Initialize the ensemble model
        
        Args:
            models: List of models
            weights: List of weights for each model
        """
        self.models = models or []
        self.weights = weights
        
        if self.weights is not None and len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        if self.weights is None and len(self.models) > 0:
            # Equal weights by default
            self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def add_model(self, model, weight=None):
        """
        Add a model to the ensemble
        
        Args:
            model: Model to add
            weight: Weight for the model
        """
        self.models.append(model)
        
        if weight is not None:
            if self.weights is None:
                self.weights = [weight]
            else:
                self.weights.append(weight)
                
            # Normalize weights
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        else:
            # Equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def predict(self, df, feature_columns=None):
        """
        Make predictions with the ensemble model
        
        Args:
            df: DataFrame with latest data
            feature_columns: List of feature columns to use
            
        Returns:
            Weighted average of predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(df, feature_columns)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # If some models failed, adjust weights
        valid_weights = [w for i, w in enumerate(self.weights) if i < len(predictions)]
        total = sum(valid_weights)
        valid_weights = [w / total for w in valid_weights]
        
        # Calculate weighted average
        weighted_pred = sum(p * w for p, w in zip(predictions, valid_weights))
        
        return weighted_pred
    
    def save(self, directory):
        """
        Save all models in the ensemble
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save weights
        weights_path = os.path.join(directory, 'weights.pkl')
        with open(weights_path, 'wb') as f:
            pickle.dump(self.weights, f)
        
        # Save each model
        for i, model in enumerate(self.models):
            model_dir = os.path.join(directory, f'model_{i}')
            os.makedirs(model_dir, exist_ok=True)
            
            if isinstance(model, LSTMModel):
                model.save(
                    os.path.join(model_dir, 'model.h5'),
                    os.path.join(model_dir, 'data_generator.pkl')
                )
            elif hasattr(model, 'save'):
                model.save(os.path.join(model_dir, 'model.pkl'))
            else:
                with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
        
        logger.info(f"Saved ensemble model with {len(self.models)} models to {directory}")
    
    @classmethod
    def load(cls, directory):
        """
        Load a saved ensemble model
        
        Args:
            directory: Directory with saved models
            
        Returns:
            Loaded EnsembleModel instance
        """
        # Create a new instance
        instance = cls()
        
        # Load weights
        weights_path = os.path.join(directory, 'weights.pkl')
        with open(weights_path, 'rb') as f:
            instance.weights = pickle.load(f)
        
        # Load each model
        i = 0
        while True:
            model_dir = os.path.join(directory, f'model_{i}')
            if not os.path.exists(model_dir):
                break
            
            # Check if it's an LSTM model
            if os.path.exists(os.path.join(model_dir, 'model.h5')):
                model = LSTMModel.load(
                    os.path.join(model_dir, 'model.h5'),
                    os.path.join(model_dir, 'data_generator.pkl')
                )
            else:
                # Load other model types
                with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
                    model = pickle.load(f)
            
            instance.models.append(model)
            i += 1
        
        logger.info(f"Loaded ensemble model with {len(instance.models)} models from {directory}")
        
        return instance


class SentimentAwareModel:
    """Model that combines price prediction with sentiment analysis"""
    
    def __init__(self, price_model=None, sentiment_weight=0.3):
        """
        Initialize the sentiment-aware model
        
        Args:
            price_model: Price prediction model
            sentiment_weight: Weight for sentiment in the final prediction
        """
        self.price_model = price_model
        self.sentiment_weight = sentiment_weight
    
    def predict(self, df, sentiment_score, feature_columns=None):
        """
        Make predictions with sentiment adjustment
        
        Args:
            df: DataFrame with latest data
            sentiment_score: Sentiment score (-1 to 1)
            feature_columns: List of feature columns to use
            
        Returns:
            Sentiment-adjusted prediction
        """
        if self.price_model is None:
            raise ValueError("Price model not set")
        
        # Get base price prediction
        base_prediction = self.price_model.predict(df, feature_columns)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Calculate predicted percent change
        predicted_change = (base_prediction - current_price) / current_price
        
        # Adjust prediction based on sentiment
        # Sentiment score is between -1 and 1
        # Higher sentiment should increase the prediction
        sentiment_adjustment = sentiment_score * self.sentiment_weight
        
        # Apply adjustment to the predicted change
        adjusted_change = predicted_change * (1 + sentiment_adjustment)
        
        # Calculate adjusted prediction
        adjusted_prediction = current_price * (1 + adjusted_change)
        
        return adjusted_prediction
    
    def save(self, directory):
        """
        Save the model
        
        Args:
            directory: Directory to save model
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save sentiment weight
        with open(os.path.join(directory, 'sentiment_weight.pkl'), 'wb') as f:
            pickle.dump(self.sentiment_weight, f)
        
        # Save price model
        if self.price_model is not None:
            price_model_dir = os.path.join(directory, 'price_model')
            os.makedirs(price_model_dir, exist_ok=True)
            
            if isinstance(self.price_model, LSTMModel):
                self.price_model.save(
                    os.path.join(price_model_dir, 'model.h5'),
                    os.path.join(price_model_dir, 'data_generator.pkl')
                )
            elif isinstance(self.price_model, EnsembleModel):
                self.price_model.save(price_model_dir)
            elif hasattr(self.price_model, 'save'):
                self.price_model.save(os.path.join(price_model_dir, 'model.pkl'))
            else:
                with open(os.path.join(price_model_dir, 'model.pkl'), 'wb') as f:
                    pickle.dump(self.price_model, f)
        
        logger.info(f"Saved sentiment-aware model to {directory}")
    
    @classmethod
    def load(cls, directory):
        """
        Load a saved sentiment-aware model
        
        Args:
            directory: Directory with saved model
            
        Returns:
            Loaded SentimentAwareModel instance
        """
        # Create a new instance
        instance = cls()
        
        # Load sentiment weight
        with open(os.path.join(directory, 'sentiment_weight.pkl'), 'rb') as f:
            instance.sentiment_weight = pickle.load(f)
        
        # Load price model
        price_model_dir = os.path.join(directory, 'price_model')
        
        # Check if it's an LSTM model
        if os.path.exists(os.path.join(price_model_dir, 'model.h5')):
            instance.price_model = LSTMModel.load(
                os.path.join(price_model_dir, 'model.h5'),
                os.path.join(price_model_dir, 'data_generator.pkl')
            )
        # Check if it's an ensemble model
        elif os.path.exists(os.path.join(price_model_dir, 'weights.pkl')):
            instance.price_model = EnsembleModel.load(price_model_dir)
        else:
            # Load other model types
            with open(os.path.join(price_model_dir, 'model.pkl'), 'rb') as f:
                instance.price_model = pickle.load(f)
        
        logger.info(f"Loaded sentiment-aware model from {directory}")
        
        return instance
