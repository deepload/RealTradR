"""
Tests for the Risk Management Module
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.ai.risk_management import RiskManager


class TestRiskManagement(unittest.TestCase):
    """Test cases for the Risk Management module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize the risk manager with test parameters
        self.risk_manager = RiskManager(
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            max_drawdown_pct=25.0,
            risk_free_rate=0.02,
            max_correlation=0.7
        )
        
        # Create sample price data for multiple assets
        dates = pd.date_range(start='2023-01-01', periods=60)
        
        # Create correlated price data for testing
        base_prices = np.random.normal(0, 1, 60)
        
        # AAPL and MSFT are highly correlated
        aapl_noise = np.random.normal(0, 0.2, 60)
        msft_noise = np.random.normal(0, 0.2, 60)
        aapl_prices = 150 + 5 * (base_prices + aapl_noise)
        msft_prices = 300 + 10 * (base_prices + msft_noise)
        
        # GOOGL is less correlated
        googl_noise = np.random.normal(0, 1, 60)
        googl_prices = 2000 + 50 * (0.3 * base_prices + googl_noise)
        
        # Create sample price DataFrame
        self.price_data = pd.DataFrame({
            'date': dates,
            'AAPL': aapl_prices,
            'MSFT': msft_prices,
            'GOOGL': googl_prices
        }).set_index('date')
        
        # Create sample equity curve
        self.equity_curve = np.array([
            100000, 102000, 103000, 105000, 104000, 103000, 101000, 
            99000, 97000, 95000, 94000, 96000, 98000, 100000, 102000
        ])
    
    def test_initialization(self):
        """Test risk manager initialization"""
        self.assertEqual(self.risk_manager.stop_loss_pct, 2.0)
        self.assertEqual(self.risk_manager.take_profit_pct, 5.0)
        self.assertEqual(self.risk_manager.max_drawdown_pct, 25.0)
        self.assertEqual(self.risk_manager.risk_free_rate, 0.02)
        self.assertEqual(self.risk_manager.max_correlation, 0.7)
    
    def test_calculate_dynamic_stop_loss(self):
        """Test calculating dynamic stop loss based on volatility and market conditions"""
        # Test with default parameters
        price = 100.0
        volatility = 2.0
        market_regime = 0  # Unknown
        sentiment = 0.0
        
        stop_loss = self.risk_manager.calculate_dynamic_stop_loss(price, volatility, market_regime, sentiment)
        
        # Check that stop loss is below price
        self.assertLess(stop_loss, price)
        
        # Test with bull market (should have tighter stop)
        bull_stop = self.risk_manager.calculate_dynamic_stop_loss(price, volatility, 1, sentiment)
        
        # Test with bear market (should have wider stop)
        bear_stop = self.risk_manager.calculate_dynamic_stop_loss(price, volatility, 2, sentiment)
        
        # Bear market stop should be lower (further from price) than bull market stop
        self.assertLess(bear_stop, bull_stop)
        
        # Test with positive sentiment (should have tighter stop)
        pos_sentiment_stop = self.risk_manager.calculate_dynamic_stop_loss(price, volatility, market_regime, 0.5)
        
        # Test with negative sentiment (should have wider stop)
        neg_sentiment_stop = self.risk_manager.calculate_dynamic_stop_loss(price, volatility, market_regime, -0.5)
        
        # Positive sentiment stop should be higher (closer to price) than negative sentiment stop
        self.assertGreater(pos_sentiment_stop, neg_sentiment_stop)
        
        # Test with high volatility (should have wider stop)
        high_vol_stop = self.risk_manager.calculate_dynamic_stop_loss(price, 4.0, market_regime, sentiment)
        
        # Test with low volatility (should have tighter stop)
        low_vol_stop = self.risk_manager.calculate_dynamic_stop_loss(price, 1.0, market_regime, sentiment)
        
        # High volatility stop should be lower (further from price) than low volatility stop
        self.assertLess(high_vol_stop, low_vol_stop)
    
    def test_calculate_dynamic_take_profit(self):
        """Test calculating dynamic take profit based on volatility and market conditions"""
        # Test with default parameters
        price = 100.0
        volatility = 2.0
        market_regime = 0  # Unknown
        sentiment = 0.0
        
        take_profit = self.risk_manager.calculate_dynamic_take_profit(price, volatility, market_regime, sentiment)
        
        # Check that take profit is above price
        self.assertGreater(take_profit, price)
        
        # Test with bull market (should have higher target)
        bull_tp = self.risk_manager.calculate_dynamic_take_profit(price, volatility, 1, sentiment)
        
        # Test with bear market (should have lower target)
        bear_tp = self.risk_manager.calculate_dynamic_take_profit(price, volatility, 2, sentiment)
        
        # Bull market take profit should be higher than bear market take profit
        self.assertGreater(bull_tp, bear_tp)
        
        # Test with positive sentiment (should have higher target)
        pos_sentiment_tp = self.risk_manager.calculate_dynamic_take_profit(price, volatility, market_regime, 0.5)
        
        # Test with negative sentiment (should have lower target)
        neg_sentiment_tp = self.risk_manager.calculate_dynamic_take_profit(price, volatility, market_regime, -0.5)
        
        # Positive sentiment take profit should be higher than negative sentiment take profit
        self.assertGreater(pos_sentiment_tp, neg_sentiment_tp)
        
        # Test with high volatility (should have wider take profit)
        high_vol_tp = self.risk_manager.calculate_dynamic_take_profit(price, 4.0, market_regime, sentiment)
        
        # Test with low volatility (should have tighter take profit)
        low_vol_tp = self.risk_manager.calculate_dynamic_take_profit(price, 1.0, market_regime, sentiment)
        
        # High volatility take profit should be higher than low volatility take profit
        self.assertGreater(high_vol_tp, low_vol_tp)
    
    def test_calculate_risk_reward_ratio(self):
        """Test calculating risk/reward ratio for a trade"""
        # Test long position
        entry_price = 100.0
        stop_loss = 95.0
        take_profit = 110.0
        
        rr_ratio = self.risk_manager.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        
        # Expected: (take_profit - entry) / (entry - stop_loss) = (110 - 100) / (100 - 95) = 10 / 5 = 2
        self.assertEqual(rr_ratio, 2.0)
        
        # Test short position
        entry_price = 100.0
        stop_loss = 105.0
        take_profit = 90.0
        
        rr_ratio = self.risk_manager.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        
        # Expected: (entry - take_profit) / (stop_loss - entry) = (100 - 90) / (105 - 100) = 10 / 5 = 2
        self.assertEqual(rr_ratio, 2.0)
        
        # Test invalid setup (stop loss and take profit both above entry)
        entry_price = 100.0
        stop_loss = 105.0
        take_profit = 110.0
        
        rr_ratio = self.risk_manager.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        
        # Expected: 0 (invalid setup)
        self.assertEqual(rr_ratio, 0)
    
    def test_calculate_kelly_position_size(self):
        """Test calculating position size using Kelly criterion"""
        # Test with favorable odds
        win_rate = 0.6
        win_loss_ratio = 2.0
        risk_pct = 1.0
        
        kelly_size = self.risk_manager.calculate_kelly_position_size(win_rate, win_loss_ratio, risk_pct)
        
        # Expected: ((p * b - q) / b) * 0.5 * risk_pct
        # = ((0.6 * 2 - 0.4) / 2) * 0.5 * 1.0
        # = (1.2 - 0.4) / 2 * 0.5 * 1.0
        # = 0.8 / 2 * 0.5 * 1.0
        # = 0.4 * 0.5 * 1.0
        # = 0.2
        self.assertAlmostEqual(kelly_size, 0.2)
        
        # Test with unfavorable odds
        win_rate = 0.4
        win_loss_ratio = 1.0
        
        kelly_size = self.risk_manager.calculate_kelly_position_size(win_rate, win_loss_ratio, risk_pct)
        
        # Expected: 0 (no position when odds are unfavorable)
        self.assertEqual(kelly_size, 0)
        
        # Test with very favorable odds (should be capped)
        win_rate = 0.8
        win_loss_ratio = 5.0
        
        kelly_size = self.risk_manager.calculate_kelly_position_size(win_rate, win_loss_ratio, risk_pct)
        
        # Expected: capped at 0.5 * 0.5 * risk_pct = 0.25
        self.assertLessEqual(kelly_size, 0.25)
    
    def test_calculate_volatility_adjusted_position_size(self):
        """Test calculating position size adjusted for volatility"""
        # Test with average volatility
        volatility = 2.0
        avg_volatility = 2.0
        base_position_pct = 10.0
        
        adjusted_size = self.risk_manager.calculate_volatility_adjusted_position_size(
            volatility, avg_volatility, base_position_pct
        )
        
        # Expected: base_position_pct * (avg_volatility / volatility) = 10.0 * (2.0 / 2.0) = 10.0
        self.assertEqual(adjusted_size, 10.0)
        
        # Test with high volatility (should reduce position)
        volatility = 4.0
        
        adjusted_size = self.risk_manager.calculate_volatility_adjusted_position_size(
            volatility, avg_volatility, base_position_pct
        )
        
        # Expected: base_position_pct * (avg_volatility / volatility) = 10.0 * (2.0 / 4.0) = 5.0
        self.assertEqual(adjusted_size, 5.0)
        
        # Test with low volatility (should increase position, but capped at 2x)
        volatility = 1.0
        
        adjusted_size = self.risk_manager.calculate_volatility_adjusted_position_size(
            volatility, avg_volatility, base_position_pct
        )
        
        # Expected: base_position_pct * (avg_volatility / volatility) = 10.0 * (2.0 / 1.0) = 20.0
        self.assertEqual(adjusted_size, 20.0)
        
        # Test with very low volatility (should be capped at 2x)
        volatility = 0.1
        
        adjusted_size = self.risk_manager.calculate_volatility_adjusted_position_size(
            volatility, avg_volatility, base_position_pct
        )
        
        # Expected: capped at 2x base_position_pct = 20.0
        self.assertEqual(adjusted_size, 20.0)
    
    def test_calculate_correlation_matrix(self):
        """Test calculating correlation matrix for a set of assets"""
        # Calculate correlation matrix
        corr_matrix = self.risk_manager.calculate_correlation_matrix(self.price_data)
        
        # Check that the matrix has the right shape
        self.assertEqual(corr_matrix.shape, (3, 3))
        
        # Check that diagonal elements are 1
        for i in range(3):
            self.assertEqual(corr_matrix.iloc[i, i], 1.0)
        
        # Check that AAPL and MSFT are highly correlated
        aapl_msft_corr = corr_matrix.loc['AAPL', 'MSFT']
        self.assertGreater(aapl_msft_corr, 0.7)
        
        # Check that GOOGL is less correlated with AAPL and MSFT
        aapl_googl_corr = corr_matrix.loc['AAPL', 'GOOGL']
        msft_googl_corr = corr_matrix.loc['MSFT', 'GOOGL']
        
        self.assertLess(aapl_googl_corr, aapl_msft_corr)
        self.assertLess(msft_googl_corr, aapl_msft_corr)
    
    def test_check_correlation_risk(self):
        """Test checking if adding a symbol would increase correlation risk"""
        # Test with empty portfolio
        result = self.risk_manager.check_correlation_risk('AAPL', [], self.price_data)
        
        # Expected: no correlation risk with empty portfolio
        self.assertFalse(result['correlation_risk'])
        
        # Test with uncorrelated symbol
        result = self.risk_manager.check_correlation_risk('GOOGL', ['AAPL'], self.price_data)
        
        # Expected: no correlation risk (correlation should be below threshold)
        self.assertFalse(result['correlation_risk'])
        
        # Test with correlated symbol
        result = self.risk_manager.check_correlation_risk('MSFT', ['AAPL'], self.price_data)
        
        # Expected: correlation risk (correlation should be above threshold)
        self.assertTrue(result['correlation_risk'])
        self.assertEqual(result['correlated_with'], 'AAPL')
        self.assertGreater(result['max_correlation'], self.risk_manager.max_correlation)
    
    def test_calculate_max_drawdown(self):
        """Test calculating maximum drawdown from equity curve"""
        # Calculate max drawdown
        max_dd = self.risk_manager.calculate_max_drawdown(self.equity_curve)
        
        # Expected: (105000 - 94000) / 105000 * 100 = 10.48%
        self.assertAlmostEqual(max_dd, 10.48, places=2)
        
        # Test with pandas Series
        max_dd_series = self.risk_manager.calculate_max_drawdown(pd.Series(self.equity_curve))
        
        # Should be the same as with numpy array
        self.assertAlmostEqual(max_dd_series, max_dd)
    
    def test_check_drawdown_limit(self):
        """Test checking if current drawdown exceeds the maximum allowed"""
        # Test with drawdown below limit
        current_equity = 90000
        peak_equity = 100000
        
        # Drawdown: (100000 - 90000) / 100000 * 100 = 10%
        exceeded = self.risk_manager.check_drawdown_limit(current_equity, peak_equity)
        
        # Expected: False (10% < 25%)
        self.assertFalse(exceeded)
        
        # Test with drawdown above limit
        current_equity = 70000
        
        # Drawdown: (100000 - 70000) / 100000 * 100 = 30%
        exceeded = self.risk_manager.check_drawdown_limit(current_equity, peak_equity)
        
        # Expected: True (30% > 25%)
        self.assertTrue(exceeded)
    
    def test_adjust_position_for_drawdown(self):
        """Test adjusting position size based on current drawdown"""
        # Test with small drawdown (no adjustment)
        position_size = 10000
        current_equity = 95000
        peak_equity = 100000
        
        # Drawdown: (100000 - 95000) / 100000 * 100 = 5%
        adjusted_size = self.risk_manager.adjust_position_for_drawdown(
            position_size, current_equity, peak_equity
        )
        
        # Expected: no adjustment (5% < 12.5%)
        self.assertEqual(adjusted_size, position_size)
        
        # Test with moderate drawdown
        current_equity = 85000
        
        # Drawdown: (100000 - 85000) / 100000 * 100 = 15%
        adjusted_size = self.risk_manager.adjust_position_for_drawdown(
            position_size, current_equity, peak_equity
        )
        
        # Expected: reduction by (15% / 25%) = 60% of original
        expected_size = position_size * (1 - 15/25)
        self.assertAlmostEqual(adjusted_size, expected_size)
        
        # Test with severe drawdown
        current_equity = 70000
        
        # Drawdown: (100000 - 70000) / 100000 * 100 = 30%
        adjusted_size = self.risk_manager.adjust_position_for_drawdown(
            position_size, current_equity, peak_equity
        )
        
        # Expected: minimum reduction (10% of original)
        self.assertAlmostEqual(adjusted_size, position_size * 0.1)
    
    def test_calculate_portfolio_var(self):
        """Test calculating portfolio Value at Risk (VaR)"""
        # Create sample positions
        positions = {
            'AAPL': 10000,
            'MSFT': 15000,
            'GOOGL': 20000
        }
        
        # Calculate VaR
        var = self.risk_manager.calculate_portfolio_var(positions, self.price_data)
        
        # Check that VaR is positive
        self.assertGreater(var, 0)
        
        # Test with different confidence level
        var_99 = self.risk_manager.calculate_portfolio_var(positions, self.price_data, confidence=0.99)
        
        # 99% VaR should be greater than 95% VaR
        self.assertGreater(var_99, var)
        
        # Test with different time horizon
        var_5day = self.risk_manager.calculate_portfolio_var(positions, self.price_data, days=5)
        
        # 5-day VaR should be greater than 1-day VaR
        self.assertGreater(var_5day, var)
    
    def test_calculate_expected_shortfall(self):
        """Test calculating Expected Shortfall (Conditional VaR)"""
        # Create sample positions
        positions = {
            'AAPL': 10000,
            'MSFT': 15000,
            'GOOGL': 20000
        }
        
        # Calculate Expected Shortfall
        es = self.risk_manager.calculate_expected_shortfall(positions, self.price_data)
        
        # Check that ES is positive
        self.assertGreater(es, 0)
        
        # Calculate VaR for comparison
        var = self.risk_manager.calculate_portfolio_var(positions, self.price_data)
        
        # ES should be greater than VaR
        self.assertGreater(es, var)


if __name__ == '__main__':
    unittest.main()
