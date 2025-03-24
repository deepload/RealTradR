"""
Technical Indicators for RealTradR

This module provides advanced technical indicators for trading strategies.
"""

import numpy as np
import pandas as pd
import logging
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime enumeration"""
    BULL_STRONG = 5  # Strong bullish trend
    BULL_NORMAL = 4  # Normal bullish trend
    NEUTRAL = 3      # Sideways/neutral market
    BEAR_NORMAL = 2  # Normal bearish trend
    BEAR_STRONG = 1  # Strong bearish trend
    UNKNOWN = 0      # Unknown/insufficient data


class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def sma(data, window):
        """
        Simple Moving Average
        
        Args:
            data: Price data series
            window: Window size
            
        Returns:
            Series with SMA values
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """
        Exponential Moving Average
        
        Args:
            data: Price data series
            window: Window size
            
        Returns:
            Series with EMA values
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """
        Moving Average Convergence Divergence
        
        Args:
            data: Price data series
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line window
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        fast_ema = TechnicalIndicators.ema(data, fast)
        slow_ema = TechnicalIndicators.ema(data, slow)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'hist': histogram
        }
    
    @staticmethod
    def rsi(data, window=14):
        """
        Relative Strength Index
        
        Args:
            data: Price data series
            window: Window size
            
        Returns:
            Series with RSI values
        """
        delta = data.diff()
        
        # Make two series: one for gains and one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA (Exponential Weighted Moving Average)
        roll_up = up.ewm(com=window-1, adjust=False).mean()
        roll_down = down.ewm(com=window-1, adjust=False).mean()
        
        # Calculate the RSI based on EWMA
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """
        Bollinger Bands
        
        Args:
            data: Price data series
            window: Window size for moving average
            num_std: Number of standard deviations
            
        Returns:
            Dictionary with upper band, middle band, and lower band
        """
        middle_band = TechnicalIndicators.sma(data, window)
        std_dev = data.rolling(window=window).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': middle_band,
            'bb_lower': lower_band
        }
    
    @staticmethod
    def atr(high, low, close, window=14):
        """
        Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Window size
            
        Returns:
            Series with ATR values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: %K window
            d_window: %D window
            
        Returns:
            Dictionary with %K and %D values
        """
        # Calculate %K
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k,
            'stoch_d': d
        }
    
    @staticmethod
    def obv(close, volume):
        """
        On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            Series with OBV values
        """
        obv = pd.Series(index=close.index, dtype='float64')
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def adx(high, low, close, window=14):
        """
        Average Directional Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Window size
            
        Returns:
            DataFrame with ADX, +DI, and -DI values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        # When +DM is less than -DM or negative, set to 0
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        
        # When -DM is less than +DM or negative, set to 0
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0).abs()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=window).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    @staticmethod
    def ichimoku_cloud(high, low, close, conversion_window=9, base_window=26, leading_span_b_window=52, lagging_span_shift=26):
        """
        Ichimoku Cloud
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            conversion_window: Conversion line window (Tenkan-sen)
            base_window: Base line window (Kijun-sen)
            leading_span_b_window: Leading span B window (Senkou Span B)
            lagging_span_shift: Lagging span shift (Chikou Span)
            
        Returns:
            DataFrame with Ichimoku Cloud components
        """
        # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
        tenkan_sen = (high.rolling(window=conversion_window).max() + low.rolling(window=conversion_window).min()) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
        kijun_sen = (high.rolling(window=base_window).max() + low.rolling(window=base_window).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2 shifted forward 26 periods
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(lagging_span_shift)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, shifted forward 26 periods
        senkou_span_b = ((high.rolling(window=leading_span_b_window).max() + low.rolling(window=leading_span_b_window).min()) / 2).shift(lagging_span_shift)
        
        # Chikou Span (Lagging Span): Close price shifted backwards 26 periods
        chikou_span = close.shift(-lagging_span_shift)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
    
    @staticmethod
    def fibonacci_retracement(high, low):
        """
        Fibonacci Retracement Levels
        
        Args:
            high: Highest price in the trend
            low: Lowest price in the trend
            
        Returns:
            Dictionary with Fibonacci retracement levels
        """
        diff = high - low
        
        return {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
    
    @staticmethod
    def vwap(high, low, close, volume):
        """
        Volume Weighted Average Price
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            Series with VWAP values
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    @staticmethod
    def pivot_points(high, low, close, method='standard'):
        """
        Pivot Points
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            method: Calculation method ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
            
        Returns:
            Dictionary with pivot points
        """
        if method == 'standard':
            pivot = (high + low + close) / 3
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }
        
        elif method == 'fibonacci':
            pivot = (high + low + close) / 3
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }
        
        elif method == 'woodie':
            pivot = (high + low + 2 * close) / 4
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = s1 - (high - low)
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = r1 + (high - low)
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }
        
        elif method == 'camarilla':
            pivot = (high + low + close) / 3
            s1 = close - 1.1 * (high - low) / 12
            s2 = close - 1.1 * (high - low) / 6
            s3 = close - 1.1 * (high - low) / 4
            s4 = close - 1.1 * (high - low) / 2
            r1 = close + 1.1 * (high - low) / 12
            r2 = close + 1.1 * (high - low) / 6
            r3 = close + 1.1 * (high - low) / 4
            r4 = close + 1.1 * (high - low) / 2
            
            return {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                's4': s4,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                'r4': r4
            }
        
        elif method == 'demark':
            if close < open:
                pivot = high + (2 * low) + close
            elif close > open:
                pivot = (2 * high) + low + close
            else:
                pivot = high + low + (2 * close)
            
            pivot = pivot / 4
            s1 = pivot * 2 - high
            r1 = pivot * 2 - low
            
            return {
                'pivot': pivot,
                's1': s1,
                'r1': r1
            }
        
        else:
            raise ValueError(f"Unknown pivot point method: {method}")
    
    @staticmethod
    def detect_market_regime(close, high, low, volume=None, window=50):
        """
        Detect market regime (bull, bear, sideways)
        
        Args:
            close: Close price series
            high: High price series
            low: Low price series
            volume: Volume series (optional)
            window: Analysis window
            
        Returns:
            MarketRegime enum value
        """
        if len(close) < window:
            return MarketRegime.UNKNOWN
        
        # Calculate indicators
        sma50 = TechnicalIndicators.sma(close, window)
        sma20 = TechnicalIndicators.sma(close, 20)
        adx_data = TechnicalIndicators.adx(high, low, close, window=14)
        adx = adx_data['adx']
        plus_di = adx_data['plus_di']
        minus_di = adx_data['minus_di']
        
        # Get latest values
        current_close = close.iloc[-1]
        current_sma50 = sma50.iloc[-1]
        current_sma20 = sma20.iloc[-1]
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        # Calculate slope of SMA50
        sma50_slope = (sma50.iloc[-1] - sma50.iloc[-10]) / sma50.iloc[-10] * 100
        
        # Check if volume data is available
        volume_trend = 0
        if volume is not None and len(volume) >= window:
            volume_sma = volume.rolling(window=20).mean()
            volume_trend = (volume.iloc[-5:].mean() / volume_sma.iloc[-5:].mean() - 1) * 100
        
        # Determine market regime
        if current_adx >= 25:
            # Strong trend
            if current_plus_di > current_minus_di:
                # Bullish trend
                if current_close > current_sma50 and current_sma20 > current_sma50 and sma50_slope > 0.5:
                    return MarketRegime.BULL_STRONG
                else:
                    return MarketRegime.BULL_NORMAL
            else:
                # Bearish trend
                if current_close < current_sma50 and current_sma20 < current_sma50 and sma50_slope < -0.5:
                    return MarketRegime.BEAR_STRONG
                else:
                    return MarketRegime.BEAR_NORMAL
        else:
            # Weak trend or sideways
            return MarketRegime.NEUTRAL
    
    @staticmethod
    def calculate_all_indicators(df, include_volume=True):
        """
        Calculate all technical indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to include volume-based indicators
            
        Returns:
            DataFrame with all indicators
        """
        result = df.copy()
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if include_volume:
            required_cols.append('volume')
        
        # Convert column names to lowercase
        result.columns = [col.lower() for col in result.columns]
        
        # Check if required columns exist
        for col in required_cols:
            if col not in result.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Moving Averages
        result['sma_10'] = TechnicalIndicators.sma(result['close'], 10)
        result['sma_20'] = TechnicalIndicators.sma(result['close'], 20)
        result['sma_50'] = TechnicalIndicators.sma(result['close'], 50)
        result['sma_200'] = TechnicalIndicators.sma(result['close'], 200)
        
        result['ema_10'] = TechnicalIndicators.ema(result['close'], 10)
        result['ema_20'] = TechnicalIndicators.ema(result['close'], 20)
        result['ema_50'] = TechnicalIndicators.ema(result['close'], 50)
        result['ema_200'] = TechnicalIndicators.ema(result['close'], 200)
        
        # MACD
        macd_data = TechnicalIndicators.macd(result['close'])
        result['macd_line'] = macd_data['macd']
        result['macd_signal'] = macd_data['signal']
        result['macd_histogram'] = macd_data['hist']
        
        # RSI
        result['rsi_14'] = TechnicalIndicators.rsi(result['close'])
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(result['close'])
        result['bb_upper'] = bb_data['bb_upper']
        result['bb_middle'] = bb_data['bb_middle']
        result['bb_lower'] = bb_data['bb_lower']
        
        # ATR
        result['atr_14'] = TechnicalIndicators.atr(result['high'], result['low'], result['close'])
        
        # Stochastic Oscillator
        stoch_data = TechnicalIndicators.stochastic_oscillator(result['high'], result['low'], result['close'])
        result['stoch_k'] = stoch_data['stoch_k']
        result['stoch_d'] = stoch_data['stoch_d']
        
        # ADX
        adx_data = TechnicalIndicators.adx(result['high'], result['low'], result['close'])
        result['adx'] = adx_data['adx']
        result['plus_di'] = adx_data['plus_di']
        result['minus_di'] = adx_data['minus_di']
        
        # Ichimoku Cloud
        ichimoku_data = TechnicalIndicators.ichimoku_cloud(result['high'], result['low'], result['close'])
        result['tenkan_sen'] = ichimoku_data['tenkan_sen']
        result['kijun_sen'] = ichimoku_data['kijun_sen']
        result['senkou_span_a'] = ichimoku_data['senkou_span_a']
        result['senkou_span_b'] = ichimoku_data['senkou_span_b']
        result['chikou_span'] = ichimoku_data['chikou_span']
        
        # Volume-based indicators
        if include_volume and 'volume' in result.columns:
            result['obv'] = TechnicalIndicators.obv(result['close'], result['volume'])
            result['vwap'] = TechnicalIndicators.vwap(result['high'], result['low'], result['close'], result['volume'])
        
        # Detect market regime
        if len(result) >= 50:
            if include_volume and 'volume' in result.columns:
                result['market_regime'] = TechnicalIndicators.detect_market_regime(
                    result['close'], result['high'], result['low'], result['volume']
                ).value
            else:
                result['market_regime'] = TechnicalIndicators.detect_market_regime(
                    result['close'], result['high'], result['low']
                ).value
        
        return result

    @staticmethod
    def calculate_technical_signal(df, market_regime):
        """
        Calculate technical signal based on indicators and market regime
        
        Args:
            df: DataFrame with technical indicators
            market_regime: Current market regime
            
        Returns:
            Signal between -1 and 1
        """
        try:
            # Get latest values
            latest = df.iloc[-1]
            
            # Initialize signals
            trend_signals = []
            momentum_signals = []
            volatility_signals = []
            volume_signals = []
            
            # Trend signals
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                # SMA crossover
                if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] and df['sma_20'].iloc[-2] <= df['sma_50'].iloc[-2]:
                    trend_signals.append(1.0)  # Bullish crossover
                elif df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1] and df['sma_20'].iloc[-2] >= df['sma_50'].iloc[-2]:
                    trend_signals.append(-1.0)  # Bearish crossover
                # SMA position
                if df['close'].iloc[-1] > df['sma_50'].iloc[-1]:
                    trend_signals.append(0.5)  # Price above SMA50
                elif df['close'].iloc[-1] < df['sma_50'].iloc[-1]:
                    trend_signals.append(-0.5)  # Price below SMA50
            
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                # EMA crossover
                if df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1] and df['ema_12'].iloc[-2] <= df['ema_26'].iloc[-2]:
                    trend_signals.append(1.0)  # Bullish crossover
                elif df['ema_12'].iloc[-1] < df['ema_26'].iloc[-1] and df['ema_12'].iloc[-2] >= df['ema_26'].iloc[-2]:
                    trend_signals.append(-1.0)  # Bearish crossover
            
            # MACD signals
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
                    momentum_signals.append(1.0)  # Bullish crossover
                elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
                    momentum_signals.append(-1.0)  # Bearish crossover
                
                # MACD histogram
                if 'macd_histogram' in df.columns:
                    if df['macd_histogram'].iloc[-1] > 0:
                        momentum_signals.append(0.5)  # Positive histogram
                    elif df['macd_histogram'].iloc[-1] < 0:
                        momentum_signals.append(-0.5)  # Negative histogram
            
            # RSI signals
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi < 30:
                    momentum_signals.append(0.8)  # Oversold
                elif rsi > 70:
                    momentum_signals.append(-0.8)  # Overbought
                elif 30 <= rsi < 45:
                    momentum_signals.append(0.4)  # Rising from oversold
                elif 55 < rsi <= 70:
                    momentum_signals.append(-0.4)  # Falling from overbought
            
            # Bollinger Bands signals
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                if df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
                    volatility_signals.append(-0.7)  # Price above upper band (overbought)
                elif df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
                    volatility_signals.append(0.7)  # Price below lower band (oversold)
                
                # Bollinger Band squeeze
                bb_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1]
                bb_width_prev = (df['bb_upper'].iloc[-10] - df['bb_lower'].iloc[-10]) / df['bb_middle'].iloc[-10]
                
                if bb_width < 0.1:
                    volatility_signals.append(0.3)  # Low volatility, potential breakout
                elif bb_width > bb_width_prev * 1.5:
                    volatility_signals.append(0.2)  # Expanding volatility
            
            # Volume signals
            if 'volume' in df.columns and 'obv' in df.columns:
                # Volume trend
                vol_sma = df['volume'].rolling(window=20).mean()
                if df['volume'].iloc[-1] > vol_sma.iloc[-1] * 1.5:
                    volume_signals.append(0.5)  # High volume
                
                # OBV trend
                obv_sma = df['obv'].rolling(window=20).mean()
                if df['obv'].iloc[-1] > obv_sma.iloc[-1] and df['close'].iloc[-1] > df['close'].iloc[-5]:
                    volume_signals.append(0.5)  # Rising OBV with rising price
                elif df['obv'].iloc[-1] < obv_sma.iloc[-1] and df['close'].iloc[-1] < df['close'].iloc[-5]:
                    volume_signals.append(-0.5)  # Falling OBV with falling price
            
            # Combine signals with weights based on market regime
            if market_regime in [MarketRegime.BULL_STRONG, MarketRegime.BULL_NORMAL]:
                # In bull market, emphasize trend and momentum
                weights = {
                    'trend': 0.4,
                    'momentum': 0.3,
                    'volatility': 0.2,
                    'volume': 0.1
                }
            elif market_regime in [MarketRegime.BEAR_STRONG, MarketRegime.BEAR_NORMAL]:
                # In bear market, emphasize volatility and volume
                weights = {
                    'trend': 0.3,
                    'momentum': 0.2,
                    'volatility': 0.3,
                    'volume': 0.2
                }
            else:
                # In neutral market, balanced weights
                weights = {
                    'trend': 0.25,
                    'momentum': 0.25,
                    'volatility': 0.25,
                    'volume': 0.25
                }
            
            # Calculate weighted average of signals
            signal = 0.0
            signal_count = 0
            
            if trend_signals:
                signal += weights['trend'] * sum(trend_signals) / len(trend_signals)
                signal_count += 1
            
            if momentum_signals:
                signal += weights['momentum'] * sum(momentum_signals) / len(momentum_signals)
                signal_count += 1
            
            if volatility_signals:
                signal += weights['volatility'] * sum(volatility_signals) / len(volatility_signals)
                signal_count += 1
            
            if volume_signals:
                signal += weights['volume'] * sum(volume_signals) / len(volume_signals)
                signal_count += 1
            
            # Normalize signal to [-1, 1]
            if signal_count > 0:
                signal = max(min(signal, 1.0), -1.0)
            else:
                signal = 0.0
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating technical signal: {e}")
            return 0.0
    
    @staticmethod
    def add_all_indicators(df, sma_periods=None, ema_periods=None, macd_params=None, rsi_period=14, bb_params=None):
        """
        Add all technical indicators to a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            sma_periods: List of periods for SMA calculation, defaults to [20, 50, 200]
            ema_periods: List of periods for EMA calculation, defaults to [12, 26]
            macd_params: Dictionary with MACD parameters, defaults to {'fast': 12, 'slow': 26, 'signal': 9}
            rsi_period: Period for RSI calculation, defaults to 14
            bb_params: Dictionary with Bollinger Bands parameters, defaults to {'window': 20, 'num_std': 2}
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            result = df.copy()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in result.columns:
                    raise ValueError(f"Required column '{col}' not found in DataFrame")
            
            # Set default parameters if not provided
            if sma_periods is None:
                sma_periods = [20, 50, 200]
            if ema_periods is None:
                ema_periods = [12, 26]
            if macd_params is None:
                macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
            if bb_params is None:
                bb_params = {'window': 20, 'num_std': 2}
            
            # Add SMA indicators
            for period in sma_periods:
                result[f'sma_{period}'] = TechnicalIndicators.sma(result['close'], period)
            
            # Add EMA indicators
            for period in ema_periods:
                result[f'ema_{period}'] = TechnicalIndicators.ema(result['close'], period)
            
            # Add MACD
            macd_data = TechnicalIndicators.macd(
                result['close'],
                fast=macd_params.get('fast', 12),
                slow=macd_params.get('slow', 26),
                signal=macd_params.get('signal', 9)
            )
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_hist'] = macd_data['hist']
            
            # Add RSI
            result[f'rsi_{rsi_period}'] = TechnicalIndicators.rsi(result['close'], rsi_period)
            
            # Add Bollinger Bands
            bb_data = TechnicalIndicators.bollinger_bands(
                result['close'],
                window=bb_params.get('window', 20),
                num_std=bb_params.get('num_std', 2)
            )
            result['bb_upper'] = bb_data['bb_upper']
            result['bb_middle'] = bb_data['bb_middle']
            result['bb_lower'] = bb_data['bb_lower']
            
            # Add ATR if high and low are available
            if 'high' in result.columns and 'low' in result.columns:
                result['atr_14'] = TechnicalIndicators.atr(result['high'], result['low'], result['close'], 14)
            
            # Add Stochastic Oscillator if high and low are available
            if 'high' in result.columns and 'low' in result.columns:
                stoch_data = TechnicalIndicators.stochastic_oscillator(result['high'], result['low'], result['close'])
                result['stoch_k'] = stoch_data['stoch_k']
                result['stoch_d'] = stoch_data['stoch_d']
            
            # Add OBV if volume is available
            if 'volume' in result.columns:
                result['obv'] = TechnicalIndicators.obv(result['close'], result['volume'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise
