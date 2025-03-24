"""
Risk Management Module for RealTradR

This module provides risk management tools for the advanced trading strategy:
- Dynamic stop-loss and take-profit calculations
- Position sizing using Kelly criterion
- Correlation analysis to avoid overexposure
- Drawdown control mechanisms
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management tools for trading strategies"""
    
    def __init__(self, stop_loss_pct=2.0, take_profit_pct=5.0, max_drawdown_pct=25.0, 
                 risk_free_rate=0.02, max_correlation=0.7):
        """
        Initialize the risk manager
        
        Args:
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            max_drawdown_pct: Maximum allowable drawdown percentage
            risk_free_rate: Annual risk-free rate (used for Kelly criterion)
            max_correlation: Maximum allowable correlation between assets
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_free_rate = risk_free_rate
        self.max_correlation = max_correlation
        
        logger.info(f"Initialized RiskManager with stop_loss={stop_loss_pct}%, "
                    f"take_profit={take_profit_pct}%, max_drawdown={max_drawdown_pct}%")
    
    def calculate_dynamic_stop_loss(self, price, volatility, market_regime, sentiment=0):
        """
        Calculate dynamic stop loss based on volatility and market conditions
        
        Args:
            price: Current price
            volatility: Price volatility (e.g., ATR or standard deviation)
            market_regime: Market regime indicator (0=unknown, 1=bull, 2=bear, etc.)
            sentiment: Sentiment score (-1 to 1)
            
        Returns:
            Stop loss price
        """
        try:
            # Base stop loss as percentage of price
            base_stop = self.stop_loss_pct / 100
            
            # Adjust for volatility (higher volatility = wider stop)
            # Normalize volatility to percentage of price
            vol_pct = volatility / price
            
            # Scale stop loss based on volatility
            # If volatility is high, increase stop loss distance
            vol_factor = 1.0 + (vol_pct * 10)  # Scale volatility impact
            
            # Adjust for market regime
            regime_factor = 1.0
            if market_regime == 1:  # Bull market
                regime_factor = 0.8  # Tighter stops in bull market
            elif market_regime == 2:  # Bear market
                regime_factor = 1.2  # Wider stops in bear market
            
            # Adjust for sentiment
            sentiment_factor = 1.0 - (sentiment * 0.2)  # Positive sentiment = tighter stops
            
            # Calculate final stop loss percentage
            final_stop_pct = base_stop * vol_factor * regime_factor * sentiment_factor
            
            # Cap at reasonable limits
            final_stop_pct = min(max(final_stop_pct, 0.005), 0.15)  # Between 0.5% and 15%
            
            # Calculate stop loss price (for long positions)
            stop_loss_price = price * (1 - final_stop_pct)
            
            logger.debug(f"Dynamic stop loss: {final_stop_pct*100:.2f}% from price {price:.2f} = {stop_loss_price:.2f}")
            
            return stop_loss_price
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {e}")
            # Fall back to default stop loss
            return price * (1 - self.stop_loss_pct/100)
    
    def calculate_dynamic_take_profit(self, price, volatility, market_regime, sentiment=0):
        """
        Calculate dynamic take profit based on volatility and market conditions
        
        Args:
            price: Current price
            volatility: Price volatility (e.g., ATR or standard deviation)
            market_regime: Market regime indicator (0=unknown, 1=bull, 2=bear, etc.)
            sentiment: Sentiment score (-1 to 1)
            
        Returns:
            Take profit price
        """
        try:
            # Base take profit as percentage of price
            base_tp = self.take_profit_pct / 100
            
            # Adjust for volatility (higher volatility = wider take profit)
            vol_pct = volatility / price
            vol_factor = 1.0 + (vol_pct * 15)  # Scale volatility impact
            
            # Adjust for market regime
            regime_factor = 1.0
            if market_regime == 1:  # Bull market
                regime_factor = 1.2  # Higher targets in bull market
            elif market_regime == 2:  # Bear market
                regime_factor = 0.8  # Lower targets in bear market
            
            # Adjust for sentiment
            sentiment_factor = 1.0 + (sentiment * 0.3)  # Positive sentiment = higher targets
            
            # Calculate final take profit percentage
            final_tp_pct = base_tp * vol_factor * regime_factor * sentiment_factor
            
            # Cap at reasonable limits
            final_tp_pct = min(max(final_tp_pct, 0.01), 0.3)  # Between 1% and 30%
            
            # Calculate take profit price (for long positions)
            take_profit_price = price * (1 + final_tp_pct)
            
            logger.debug(f"Dynamic take profit: {final_tp_pct*100:.2f}% from price {price:.2f} = {take_profit_price:.2f}")
            
            return take_profit_price
            
        except Exception as e:
            logger.error(f"Error calculating dynamic take profit: {e}")
            # Fall back to default take profit
            return price * (1 + self.take_profit_pct/100)
    
    def calculate_risk_reward_ratio(self, entry_price, stop_loss, take_profit):
        """
        Calculate risk/reward ratio for a trade
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Risk/reward ratio
        """
        try:
            # For long positions
            if take_profit > entry_price and stop_loss < entry_price:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
                
                if risk <= 0:
                    return float('inf')  # Invalid risk
                
                return reward / risk
            
            # For short positions
            elif take_profit < entry_price and stop_loss > entry_price:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                
                if risk <= 0:
                    return float('inf')  # Invalid risk
                
                return reward / risk
            
            else:
                return 0  # Invalid setup
                
        except Exception as e:
            logger.error(f"Error calculating risk/reward ratio: {e}")
            return 0
    
    def calculate_kelly_position_size(self, win_rate, win_loss_ratio, risk_pct=1.0):
        """
        Calculate position size using Kelly criterion
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss
            risk_pct: Percentage of account to risk per trade
            
        Returns:
            Kelly position size as percentage of account
        """
        try:
            # Kelly formula: f* = (p * b - q) / b
            # where f* is the optimal fraction, p is probability of win,
            # q is probability of loss (1-p), and b is the win/loss ratio
            
            q = 1 - win_rate
            kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
            
            # Limit Kelly to reasonable values
            kelly = max(0, min(kelly, 0.5))  # Cap at 50% of account
            
            # Apply a fraction of Kelly (half-Kelly is common)
            kelly_fraction = kelly * 0.5
            
            # Scale by risk percentage
            position_size = kelly_fraction * risk_pct
            
            logger.debug(f"Kelly position size: {position_size:.2f}% of account")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return risk_pct * 0.1  # Default to 10% of risk percentage
    
    def calculate_volatility_adjusted_position_size(self, volatility, avg_volatility, base_position_pct):
        """
        Calculate position size adjusted for volatility
        
        Args:
            volatility: Current volatility of the asset
            avg_volatility: Average volatility across all assets
            base_position_pct: Base position size percentage
            
        Returns:
            Adjusted position size percentage
        """
        try:
            if volatility <= 0 or avg_volatility <= 0:
                return base_position_pct
            
            # Inverse volatility weighting
            vol_ratio = avg_volatility / volatility
            
            # Adjust position size (higher volatility = smaller position)
            adjusted_position = base_position_pct * vol_ratio
            
            # Cap at reasonable limits
            adjusted_position = min(max(adjusted_position, base_position_pct * 0.2), base_position_pct * 2.0)
            
            logger.debug(f"Volatility-adjusted position: {adjusted_position:.2f}% (base: {base_position_pct:.2f}%)")
            
            return adjusted_position
            
        except Exception as e:
            logger.error(f"Error calculating volatility-adjusted position size: {e}")
            return base_position_pct
    
    def calculate_correlation_matrix(self, price_data):
        """
        Calculate correlation matrix for a set of assets
        
        Args:
            price_data: DataFrame with price data for multiple assets
            
        Returns:
            Correlation matrix
        """
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def check_correlation_risk(self, symbol, portfolio_symbols, price_data):
        """
        Check if adding a symbol would increase correlation risk
        
        Args:
            symbol: Symbol to check
            portfolio_symbols: List of symbols already in portfolio
            price_data: DataFrame with price data for all symbols
            
        Returns:
            Dictionary with correlation risk assessment
        """
        try:
            if symbol in portfolio_symbols or len(portfolio_symbols) == 0:
                return {"correlation_risk": False, "max_correlation": 0, "correlated_with": None}
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Get returns for the symbol and portfolio symbols
            symbol_returns = returns[symbol]
            
            max_corr = 0
            max_corr_symbol = None
            
            # Check correlation with each portfolio symbol
            for port_symbol in portfolio_symbols:
                if port_symbol in returns.columns:
                    port_returns = returns[port_symbol]
                    
                    # Calculate correlation
                    corr, _ = pearsonr(symbol_returns, port_returns)
                    
                    if abs(corr) > max_corr:
                        max_corr = abs(corr)
                        max_corr_symbol = port_symbol
            
            # Check if correlation exceeds threshold
            correlation_risk = max_corr > self.max_correlation
            
            return {
                "correlation_risk": correlation_risk,
                "max_correlation": max_corr,
                "correlated_with": max_corr_symbol
            }
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return {"correlation_risk": False, "max_correlation": 0, "correlated_with": None}
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown from equity curve
        
        Args:
            equity_curve: Series or array of equity values
            
        Returns:
            Maximum drawdown as a percentage
        """
        try:
            # Convert to numpy array if needed
            if isinstance(equity_curve, pd.Series):
                equity_curve = equity_curve.values
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(equity_curve)
            
            # Calculate drawdown
            drawdown = (running_max - equity_curve) / running_max
            
            # Get maximum drawdown
            max_drawdown = np.max(drawdown) * 100  # Convert to percentage
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            return 0
    
    def check_drawdown_limit(self, current_equity, peak_equity):
        """
        Check if current drawdown exceeds the maximum allowed
        
        Args:
            current_equity: Current equity value
            peak_equity: Peak equity value
            
        Returns:
            True if drawdown limit is exceeded, False otherwise
        """
        try:
            if peak_equity <= 0:
                return False
            
            # Calculate current drawdown
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            
            # Check if drawdown exceeds limit
            exceeded = drawdown > self.max_drawdown_pct
            
            if exceeded:
                logger.warning(f"Drawdown limit exceeded: {drawdown:.2f}% > {self.max_drawdown_pct:.2f}%")
            
            return exceeded
            
        except Exception as e:
            logger.error(f"Error checking drawdown limit: {e}")
            return False
    
    def adjust_position_for_drawdown(self, position_size, current_equity, peak_equity):
        """
        Adjust position size based on current drawdown
        
        Args:
            position_size: Original position size
            current_equity: Current equity value
            peak_equity: Peak equity value
            
        Returns:
            Adjusted position size
        """
        try:
            if peak_equity <= 0:
                return position_size
            
            # Calculate current drawdown
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            
            # No adjustment if drawdown is small
            if drawdown < self.max_drawdown_pct * 0.5:
                return position_size
            
            # Linear reduction as drawdown approaches limit
            reduction_factor = 1.0 - (drawdown / self.max_drawdown_pct)
            reduction_factor = max(0.1, min(1.0, reduction_factor))
            
            # Apply reduction
            adjusted_position = position_size * reduction_factor
            
            logger.info(f"Adjusted position for drawdown: {drawdown:.2f}%, "
                        f"reduction: {(1-reduction_factor)*100:.2f}%")
            
            return adjusted_position
            
        except Exception as e:
            logger.error(f"Error adjusting position for drawdown: {e}")
            return position_size * 0.5  # Default to 50% reduction
    
    def calculate_portfolio_var(self, positions, price_data, confidence=0.95, days=1):
        """
        Calculate portfolio Value at Risk (VaR)
        
        Args:
            positions: Dictionary of positions {symbol: position_value}
            price_data: DataFrame with price data for all symbols
            confidence: Confidence level (0-1)
            days: Time horizon in days
            
        Returns:
            Value at Risk in dollars
        """
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Get only the symbols in our portfolio
            portfolio_symbols = list(positions.keys())
            portfolio_returns = returns[portfolio_symbols]
            
            # Calculate portfolio weights
            total_value = sum(positions.values())
            weights = np.array([positions[symbol] / total_value for symbol in portfolio_symbols])
            
            # Calculate portfolio returns
            portfolio_return = portfolio_returns.dot(weights)
            
            # Calculate VaR
            var_percentile = 1 - confidence
            var_daily = np.percentile(portfolio_return, var_percentile * 100) * total_value
            
            # Scale to the specified time horizon
            var = abs(var_daily) * np.sqrt(days)
            
            logger.info(f"Portfolio VaR ({confidence*100:.1f}%, {days} days): ${var:.2f}")
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0
    
    def calculate_expected_shortfall(self, positions, price_data, confidence=0.95, days=1):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            positions: Dictionary of positions {symbol: position_value}
            price_data: DataFrame with price data for all symbols
            confidence: Confidence level (0-1)
            days: Time horizon in days
            
        Returns:
            Expected Shortfall in dollars
        """
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Get only the symbols in our portfolio
            portfolio_symbols = list(positions.keys())
            portfolio_returns = returns[portfolio_symbols]
            
            # Calculate portfolio weights
            total_value = sum(positions.values())
            weights = np.array([positions[symbol] / total_value for symbol in portfolio_symbols])
            
            # Calculate portfolio returns
            portfolio_return = portfolio_returns.dot(weights)
            
            # Calculate VaR percentile
            var_percentile = 1 - confidence
            var_cutoff = np.percentile(portfolio_return, var_percentile * 100)
            
            # Calculate Expected Shortfall
            es_returns = portfolio_return[portfolio_return <= var_cutoff]
            es_daily = es_returns.mean() * total_value
            
            # Scale to the specified time horizon
            es = abs(es_daily) * np.sqrt(days)
            
            logger.info(f"Portfolio Expected Shortfall ({confidence*100:.1f}%, {days} days): ${es:.2f}")
            
            return es
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0


# Example usage
if __name__ == "__main__":
    # Create risk manager
    risk_manager = RiskManager(
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        max_drawdown_pct=25.0
    )
    
    # Example: Calculate dynamic stop loss
    price = 100.0
    volatility = 2.0  # 2% daily volatility
    market_regime = 1  # Bull market
    sentiment = 0.5  # Positive sentiment
    
    stop_loss = risk_manager.calculate_dynamic_stop_loss(price, volatility, market_regime, sentiment)
    take_profit = risk_manager.calculate_dynamic_take_profit(price, volatility, market_regime, sentiment)
    
    print(f"Price: ${price:.2f}")
    print(f"Dynamic Stop Loss: ${stop_loss:.2f} ({(price-stop_loss)/price*100:.2f}% from entry)")
    print(f"Dynamic Take Profit: ${take_profit:.2f} ({(take_profit-price)/price*100:.2f}% from entry)")
    print(f"Risk/Reward Ratio: {risk_manager.calculate_risk_reward_ratio(price, stop_loss, take_profit):.2f}")
