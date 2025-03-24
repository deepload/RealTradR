"""
Tests for the Advanced AI Trading Strategy
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.ai.advanced_strategy import AdvancedAIStrategy, load_strategy_config
from backend.app.ai.technical_indicators import MarketRegime


class TestAdvancedStrategy(unittest.TestCase):
    """Test cases for the Advanced AI Trading Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a test configuration
        self.test_config = {
            "symbols": ["AAPL", "MSFT"],
            "cash_limit": 10000.0,
            "short_window": 10,
            "long_window": 30,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
            "sentiment_weight": 0.3,
            "technical_weight": 0.4,
            "ml_weight": 0.3,
            "position_sizing": "equal",
            "max_position_pct": 20.0,
            "use_market_regime": True,
            "use_sentiment": True,
            "use_ml_models": False,  # Disable ML models for testing
            "risk_free_rate": 0.02
        }
        
        # Create a sample historical data DataFrame
        dates = pd.date_range(start='2023-01-01', periods=60)
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 5, 60),
            'high': np.random.normal(105, 5, 60),
            'low': np.random.normal(95, 5, 60),
            'close': np.random.normal(100, 5, 60),
            'volume': np.random.randint(1000, 10000, 60)
        })
        
        # Mock Alpaca API
        self.mock_api_patcher = patch('alpaca_trade_api.REST')
        self.mock_api = self.mock_api_patcher.start()
        
        # Mock API instance
        self.mock_api_instance = MagicMock()
        self.mock_api.return_value = self.mock_api_instance
        
        # Mock account info
        mock_account = MagicMock()
        mock_account.portfolio_value = "110000.0"
        mock_account.cash = "50000.0"
        mock_account.equity = "110000.0"
        mock_account.last_equity = "100000.0"
        mock_account.buying_power = "100000.0"
        self.mock_api_instance.get_account.return_value = mock_account
        
        # Mock position
        mock_position = MagicMock()
        mock_position.market_value = "5000.0"
        self.mock_api_instance.get_position.return_value = mock_position
        
        # Mock latest trade
        mock_trade = MagicMock()
        mock_trade.price = "150.0"
        self.mock_api_instance.get_latest_trade.return_value = mock_trade
        
        # Mock portfolio history
        mock_history = MagicMock()
        mock_history.equity = [100000.0, 101000.0, 102000.0, 103000.0, 102000.0, 104000.0, 110000.0]
        self.mock_api_instance.get_portfolio_history.return_value = mock_history
        
        # Mock order submission
        mock_order = MagicMock()
        mock_order.id = "test-order-id"
        self.mock_api_instance.submit_order.return_value = mock_order
        
        # Initialize the strategy with mocked components
        self.strategy = AdvancedAIStrategy(config=self.test_config)
        
        # Mock the get_historical_data method
        self.strategy.get_historical_data = MagicMock(return_value=self.sample_data)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.mock_api_patcher.stop()
    
    def test_load_strategy_config(self):
        """Test loading strategy configuration"""
        # Test with default config
        config = load_strategy_config()
        self.assertIsInstance(config, dict)
        self.assertIn("symbols", config)
        self.assertIn("cash_limit", config)
    
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.symbols, ["AAPL", "MSFT"])
        self.assertEqual(self.strategy.cash_limit, 10000.0)
        self.assertEqual(self.strategy.sentiment_weight, 0.3)
        self.assertEqual(self.strategy.technical_weight, 0.4)
        self.assertEqual(self.strategy.ml_weight, 0.3)
        self.assertEqual(self.strategy.position_sizing, "equal")
        self.assertEqual(self.strategy.max_position_pct, 20.0)
        self.assertTrue(self.strategy.use_market_regime)
        self.assertTrue(self.strategy.use_sentiment)
        self.assertFalse(self.strategy.use_ml_models)
    
    def test_get_technical_signals(self):
        """Test getting technical signals from price data"""
        # Call the method
        signals = self.strategy.get_technical_signals(self.sample_data)
        
        # Check the structure of the returned signals
        self.assertIn('individual_signals', signals)
        self.assertIn('combined_signal', signals)
        self.assertIn('market_regime', signals)
        self.assertIn('indicators', signals)
        
        # Check individual signals
        self.assertIn('ma_signal', signals['individual_signals'])
        self.assertIn('macd_signal', signals['individual_signals'])
        self.assertIn('rsi_signal', signals['individual_signals'])
        self.assertIn('bb_signal', signals['individual_signals'])
        self.assertIn('adx_signal', signals['individual_signals'])
        
        # Check that combined signal is a float
        self.assertIsInstance(signals['combined_signal'], float)
        
        # Check that market regime is a valid enum value
        self.assertIsInstance(signals['market_regime'], MarketRegime)
    
    def test_get_sentiment_signal(self):
        """Test getting sentiment signal for a symbol"""
        # Mock the get_symbol_sentiment function
        with patch('backend.app.ai.advanced_strategy.get_symbol_sentiment') as mock_sentiment:
            # Set up the mock to return a sentiment score
            mock_sentiment.return_value = {
                'compound': 0.5,
                'positive': 0.7,
                'negative': 0.1,
                'neutral': 0.2
            }
            
            # Call the method
            sentiment = self.strategy.get_sentiment_signal("AAPL")
            
            # Check the structure of the returned sentiment
            self.assertIn('sentiment_score', sentiment)
            self.assertIn('sentiment_signal', sentiment)
            self.assertIn('details', sentiment)
            
            # Check that sentiment score is correct
            self.assertEqual(sentiment['sentiment_score'], 0.5)
            
            # Check that sentiment signal is correct (should be 1 for positive)
            self.assertEqual(sentiment['sentiment_signal'], 1)
    
    def test_combine_signals(self):
        """Test combining different signals into a final trading signal"""
        # Create test signals
        technical_signal = {'combined_signal': 0.5}
        sentiment_signal = {'sentiment_signal': 0.8}
        ml_signal = {'ml_signal': -0.3}
        
        # Call the method
        combined = self.strategy.combine_signals(technical_signal, sentiment_signal, ml_signal)
        
        # Calculate expected result
        expected = (0.5 * 0.4) + (0.8 * 0.3) + (-0.3 * 0.3)
        
        # Check that combined signal is correct
        self.assertAlmostEqual(combined, expected)
    
    def test_calculate_position_size(self):
        """Test calculating position size based on signal strength"""
        # Test with equal position sizing
        self.strategy.position_sizing = "equal"
        size = self.strategy.calculate_position_size("AAPL", 0.5, 100000.0)
        
        # Expected: (max_position_pct / num_symbols) * signal_strength * portfolio_value
        expected = (20.0 / 2) * 0.5 * 100000.0 / 100
        self.assertAlmostEqual(size, expected)
        
        # Test with negative signal (short position)
        size = self.strategy.calculate_position_size("AAPL", -0.8, 100000.0)
        expected = -(20.0 / 2) * 0.8 * 100000.0 / 100
        self.assertAlmostEqual(size, expected)
    
    def test_execute_trades(self):
        """Test executing trades to reach target position size"""
        # Test buying (increasing position)
        result = self.strategy.execute_trades("AAPL", 10000.0)
        
        # Check that the order was submitted
        self.mock_api_instance.submit_order.assert_called_once()
        
        # Check the structure of the returned result
        self.assertIn('symbol', result)
        self.assertIn('action', result)
        self.assertIn('quantity', result)
        self.assertIn('amount', result)
        self.assertIn('current_position', result)
        self.assertIn('target_position', result)
        self.assertIn('order_id', result)
        
        # Check that the values are correct
        self.assertEqual(result['symbol'], "AAPL")
        self.assertEqual(result['action'], "buy")
        self.assertEqual(result['order_id'], "test-order-id")
        
        # Reset the mock
        self.mock_api_instance.submit_order.reset_mock()
        
        # Test selling (decreasing position)
        result = self.strategy.execute_trades("AAPL", 2000.0)
        
        # Check that the order was submitted
        self.mock_api_instance.submit_order.assert_called_once()
        
        # Check that the values are correct
        self.assertEqual(result['symbol'], "AAPL")
        self.assertEqual(result['action'], "sell")
    
    def test_run(self):
        """Test running the strategy once"""
        # Mock the other methods to isolate the run method
        self.strategy.get_technical_signals = MagicMock(return_value={
            'individual_signals': {},
            'combined_signal': 0.5,
            'market_regime': MarketRegime.BULL_NORMAL,
            'indicators': {}
        })
        
        self.strategy.get_sentiment_signal = MagicMock(return_value={
            'sentiment_score': 0.3,
            'sentiment_signal': 1
        })
        
        self.strategy.get_ml_prediction = MagicMock(return_value={
            'predicted_price': 155.0,
            'predicted_change': 0.03,
            'ml_signal': 1
        })
        
        self.strategy.calculate_position_size = MagicMock(return_value=7500.0)
        
        self.strategy.execute_trades = MagicMock(return_value={
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 50,
            'amount': 7500.0,
            'current_position': 0,
            'target_position': 7500.0,
            'order_id': 'test-order-id'
        })
        
        # Call the method
        result = self.strategy.run()
        
        # Check the structure of the returned result
        self.assertIn('timestamp', result)
        self.assertIn('portfolio_value', result)
        self.assertIn('cash', result)
        self.assertIn('results', result)
        
        # Check that the methods were called for each symbol
        self.assertEqual(self.strategy.get_technical_signals.call_count, 2)  # Once for each symbol
        self.assertEqual(self.strategy.get_sentiment_signal.call_count, 2)
        self.assertEqual(self.strategy.get_ml_prediction.call_count, 2)
        self.assertEqual(self.strategy.calculate_position_size.call_count, 2)
        self.assertEqual(self.strategy.execute_trades.call_count, 2)
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics for the strategy"""
        # Call the method
        metrics = self.strategy.get_performance_metrics()
        
        # Check the structure of the returned metrics
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('equity', metrics)
        self.assertIn('buying_power', metrics)
        self.assertIn('daily_returns', metrics)
        
        # Check that the values are correct
        self.assertEqual(metrics['total_return'], 10.0)  # (110000 - 100000) / 100000 * 100
        self.assertEqual(metrics['equity'], 110000.0)
        self.assertEqual(metrics['buying_power'], 100000.0)


if __name__ == '__main__':
    unittest.main()
