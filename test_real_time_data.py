#!/usr/bin/env python
"""
Test script for getting real-time data from Alpaca using the snapshot API
"""
import os
import sys
import logging
import json
import random
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_snapshot_data(symbols):
    """Get real-time snapshot data for a list of symbols"""
    # API credentials
    api_key = "PK88UAXEPBIEQCEAS8YV"
    api_secret = "hu9YIoZSqhLiOLLoTkHt5mh2NK4gTi7fXPQf6X1L"
    data_url = "https://data.alpaca.markets"
    
    logger.info(f"Getting snapshot data for {symbols}")
    
    try:
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, data_url, api_version='v2')
        
        # Get snapshots for each symbol
        results = {}
        for symbol in symbols:
            try:
                logger.info(f"Getting snapshot for {symbol}...")
                snapshot = api.get_snapshot(symbol)
                
                # Extract data from snapshot using dictionary access to avoid attribute errors
                result = {
                    'symbol': symbol,
                    'last_price': None,
                    'bid_price': None,
                    'ask_price': None,
                    'volume': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'close': None,
                    'prev_close': None
                }
                
                # Extract latest trade data
                if hasattr(snapshot, 'latest_trade') and snapshot.latest_trade:
                    result['last_price'] = snapshot.latest_trade.p
                
                # Extract latest quote data
                if hasattr(snapshot, 'latest_quote') and snapshot.latest_quote:
                    result['bid_price'] = snapshot.latest_quote.bp
                    result['ask_price'] = snapshot.latest_quote.ap
                
                # Extract daily bar data
                if hasattr(snapshot, 'daily_bar') and snapshot.daily_bar:
                    result['volume'] = snapshot.daily_bar.v
                    result['open'] = snapshot.daily_bar.o
                    result['high'] = snapshot.daily_bar.h
                    result['low'] = snapshot.daily_bar.l
                    result['close'] = snapshot.daily_bar.c
                
                # Extract previous day data
                if hasattr(snapshot, 'prev_daily_bar') and snapshot.prev_daily_bar:
                    result['prev_close'] = snapshot.prev_daily_bar.c
                
                results[symbol] = result
                logger.info(f"✅ Got data for {symbol}: Last price = ${result['last_price']}")
                
            except Exception as e:
                logger.error(f"❌ Error getting snapshot for {symbol}: {e}")
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Error initializing API: {e}")
        return {}

def generate_synthetic_data(symbol, days, snapshot):
    """
    Generate synthetic historical data based on current snapshot
    
    Args:
        symbol: Symbol to generate data for
        days: Number of days of data to generate
        snapshot: Current market data snapshot
        
    Returns:
        DataFrame with synthetic historical price data
    """
    logger.info(f"Generating synthetic data for {symbol} based on current snapshot")
    
    # Get current price from snapshot
    current_price = snapshot['last_price']
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Oldest to newest
    
    # Generate price data with realistic volatility
    prices = []
    price = current_price
    daily_volatility = 0.015  # 1.5% daily volatility
    
    # Work backwards from current price
    for i in range(days-1, -1, -1):
        if i == 0:  # Current day
            # Use actual data from snapshot
            prices.append({
                'open': snapshot.get('open', price * (1 + random.normalvariate(0, 0.003))),
                'high': snapshot.get('high', price * (1 + random.normalvariate(0, 0.005))),
                'low': snapshot.get('low', price * (1 - random.normalvariate(0, 0.005))),
                'close': current_price,
                'volume': snapshot.get('volume', 5000000)
            })
        else:
            # Random daily return with slight upward bias
            daily_return = random.normalvariate(0.0005, daily_volatility)  # Mean 0.05% daily return
            price = price / (1 + daily_return)  # Work backwards
            
            # Generate OHLC data
            daily_range = price * daily_volatility
            open_price = price * (1 + random.normalvariate(0, 0.003))
            high_price = max(open_price, price) + abs(random.normalvariate(0, daily_range/2))
            low_price = min(open_price, price) - abs(random.normalvariate(0, daily_range/2))
            close_price = price
            
            # Generate volume
            volume = int(random.randint(5000000, 15000000))
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    # Create DataFrame
    df = pd.DataFrame(prices, index=dates)
    
    # Add timestamp column
    df['timestamp'] = df.index
    
    return df

def calculate_technical_indicators(data):
    """Calculate basic technical indicators for the data"""
    try:
        # Add SMA
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Add EMA
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # Add MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Add RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Add Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        std_dev = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (std_dev * 2)
        data['bb_lower'] = data['bb_middle'] - (std_dev * 2)
        
        return data
    
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return data

def analyze_symbol(symbol, snapshot, days=60):
    """Analyze a symbol using real-time data and synthetic historical data"""
    logger.info(f"\nAnalyzing {symbol}...")
    
    # Generate synthetic historical data
    try:
        data = generate_synthetic_data(symbol, days, snapshot)
        
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Get current price
        current_price = snapshot['last_price']
        
        # Calculate signals
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        
        # Determine trend
        if sma_20 > sma_50:
            trend = "BULLISH"
        elif sma_20 < sma_50:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # Determine RSI signal
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"
        
        # Determine MACD signal
        if macd > macd_signal:
            macd_signal_text = "BULLISH"
        else:
            macd_signal_text = "BEARISH"
        
        # Determine Bollinger Bands signal
        if current_price > bb_upper:
            bb_signal = "OVERBOUGHT"
        elif current_price < bb_lower:
            bb_signal = "OVERSOLD"
        else:
            bb_signal = "NEUTRAL"
        
        # Combine signals
        signals = {
            "trend": trend,
            "rsi": rsi_signal,
            "macd": macd_signal_text,
            "bollinger_bands": bb_signal
        }
        
        # Determine overall signal
        bullish_count = sum(1 for signal in signals.values() if signal in ["BULLISH", "OVERSOLD"])
        bearish_count = sum(1 for signal in signals.values() if signal in ["BEARISH", "OVERBOUGHT"])
        
        if bullish_count > bearish_count:
            overall_signal = "BUY"
        elif bearish_count > bullish_count:
            overall_signal = "SELL"
        else:
            overall_signal = "HOLD"
        
        # Print analysis
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Previous Close: ${snapshot.get('prev_close', 'N/A')}")
        logger.info(f"Volume: {snapshot.get('volume', 'N/A')}")
        logger.info(f"SMA(20): ${sma_20:.2f}")
        logger.info(f"SMA(50): ${sma_50:.2f}")
        logger.info(f"RSI(14): {rsi:.2f} - {rsi_signal}")
        logger.info(f"MACD: {macd:.4f} - {macd_signal_text}")
        logger.info(f"Bollinger Bands: ${bb_lower:.2f} - ${bb_upper:.2f} - {bb_signal}")
        logger.info(f"Overall Signal: {overall_signal}")
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "prev_close": snapshot.get('prev_close', None),
            "volume": snapshot.get('volume', None),
            "technical_indicators": {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "bb_lower": bb_lower,
                "bb_upper": bb_upper
            },
            "signals": signals,
            "overall_signal": overall_signal
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("REAL-TIME DATA TEST WITH ALPACA")
    logger.info("=" * 80)
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get real-time data
    snapshots = get_snapshot_data(symbols)
    
    if snapshots:
        logger.info("\nSuccessfully retrieved real-time data")
        
        # Analyze each symbol
        analysis_results = {}
        for symbol, snapshot in snapshots.items():
            result = analyze_symbol(symbol, snapshot)
            analysis_results[symbol] = result
        
        # Save results to file
        with open('real_time_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info("\nAnalysis complete! Results saved to real_time_analysis.json")
    else:
        logger.error("Failed to retrieve real-time data")
    
    logger.info("=" * 80)
