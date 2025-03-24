#!/usr/bin/env python
"""
End-to-End Test for RealTradR Advanced AI Strategy

This script runs a complete end-to-end test of the RealTradR advanced AI trading strategy,
including all components:
- Technical indicators
- Fallback ML models (when TensorFlow is not available)
- Risk management
- Trade execution (in paper trading mode)

Usage:
    python run_advanced_strategy.py --config strategy_config.json
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the monitoring system
try:
    from backend.app.monitoring.logger import LoggerFactory
    logger_factory = LoggerFactory.get_instance()
    logger = logger_factory.get_logger("advanced_strategy_test")
except ImportError:
    # Fallback logging if monitoring module is not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("advanced_strategy_test.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("advanced_strategy_test")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run end-to-end test of RealTradR advanced AI strategy")
    
    # Basic arguments
    parser.add_argument("--config", type=str, default="strategy_config.json",
                      help="Path to strategy configuration file")
    parser.add_argument("--symbols", type=str,
                      help="Comma-separated list of symbols to trade")
    parser.add_argument("--cash", type=float, default=100000.0,
                      help="Starting cash amount")
    parser.add_argument("--paper", action="store_true", default=True,
                      help="Use paper trading (always true for testing)")
    parser.add_argument("--use-mock-data", action="store_true",
                      help="Use mock data instead of real API calls")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.info("Using default configuration")
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "cash_limit": 100000.0,
            "sentiment_weight": 0.0,
            "technical_weight": 0.6,
            "ml_weight": 0.4,
            "position_sizing": "kelly",
            "max_position_pct": 20.0,
            "use_market_regime": True,
            "use_sentiment": False,  # Disable sentiment for testing if API keys aren't set
            "use_ml_models": True,
            "model_dir": "./models",
            "stop_loss_pct": 2.0,
            "take_profit_pct": 5.0,
            "max_drawdown_pct": 25.0,
            "risk_free_rate": 0.02,
            "max_correlation": 0.7
        }

def generate_mock_data(symbols, days=60):
    """Generate mock price data for testing when API calls fail"""
    logger.info(f"Generating mock data for {len(symbols)} symbols")
    
    mock_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for symbol in symbols:
        # Generate random price data with a slight upward trend
        base_price = 100.0 + np.random.rand() * 900.0  # Random starting price between 100 and 1000
        daily_returns = np.random.normal(0.0005, 0.015, size=len(date_range))  # Small positive drift
        prices = base_price * np.cumprod(1 + daily_returns)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'open': prices * (1 - np.random.rand(len(date_range)) * 0.01),
            'high': prices * (1 + np.random.rand(len(date_range)) * 0.01),
            'low': prices * (1 - np.random.rand(len(date_range)) * 0.02),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, size=len(date_range))
        }, index=date_range)
        
        mock_data[symbol] = df
    
    logger.info(f"Generated mock data with {len(date_range)} days of history")
    return mock_data

def setup_monitoring():
    """Set up performance monitoring if available"""
    try:
        from backend.app.monitoring.performance import PerformanceMonitor
        monitor = PerformanceMonitor(interval=5)  # 5-second interval
        monitor.start()
        logger.info("Started performance monitoring")
        return monitor
    except ImportError:
        logger.warning("Performance monitoring not available")
        return None

def setup_trade_logging():
    """Set up trade logging if available"""
    try:
        from backend.app.monitoring.logger import TradeLogger
        trade_logger = TradeLogger("advanced_strategy_test")
        logger.info("Trade logging initialized")
        return trade_logger
    except ImportError:
        logger.warning("Trade logging not available")
        return None

def setup_alert_manager():
    """Set up alert manager if available"""
    try:
        from backend.app.monitoring.alerts import AlertManager
        alert_manager = AlertManager()
        logger.info("Alert manager initialized")
        return alert_manager
    except ImportError:
        logger.warning("Alert manager not available")
        return None

def log_trade(trade_logger, trade_data):
    """Log a trade if trade logger is available"""
    if trade_logger:
        try:
            trade_logger.log_trade(trade_data)
        except Exception as e:
            logger.warning(f"Failed to log trade: {e}")

def send_alert(alert_manager, title, message):
    """Send an alert if alert manager is available"""
    if alert_manager:
        try:
            if hasattr(alert_manager, 'send_alert'):
                alert_manager.send_alert(title, message)
            elif hasattr(alert_manager, 'alert'):
                alert_manager.alert(title, message)
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")

def main():
    """Main function to run the end-to-end test"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up monitoring components
    monitor = setup_monitoring()
    trade_logger = setup_trade_logging()
    alert_manager = setup_alert_manager()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments if provided
        if args.symbols:
            config["symbols"] = args.symbols.split(",")
        if args.cash:
            config["cash_limit"] = args.cash
        
        # Import the advanced strategy (here to avoid circular imports)
        from backend.app.ai.advanced_strategy import AdvancedAIStrategy
        
        # Log start of test
        logger.info("=" * 50)
        logger.info("STARTING END-TO-END TEST OF ADVANCED AI STRATEGY")
        logger.info("=" * 50)
        logger.info(f"Testing with symbols: {config['symbols']}")
        logger.info(f"Cash limit: ${config['cash_limit']}")
        logger.info(f"Signal weights: Technical={config['technical_weight']}, "
                   f"ML={config['ml_weight']}, Sentiment={config['sentiment_weight']}")
        
        # Initialize the strategy
        strategy = AdvancedAIStrategy(config=config)
        
        # If using mock data, patch the get_historical_data method
        if args.use_mock_data:
            logger.info("Using mock data for testing")
            mock_data = generate_mock_data(config['symbols'])
            
            # Monkey patch the get_historical_data method
            original_get_data = strategy.get_historical_data
            
            def mock_get_historical_data(self, symbol, days=60, timeframe="1D"):
                logger.info(f"Using mock data for {symbol}")
                if symbol in mock_data:
                    return mock_data[symbol]
                else:
                    return original_get_data(symbol, days, timeframe)
            
            # Apply the monkey patch
            import types
            strategy.get_historical_data = types.MethodType(mock_get_historical_data, strategy)
        
        # Run the strategy
        logger.info("Running strategy...")
        try:
            results = strategy.run()
            
            # Log results
            logger.info("=" * 50)
            logger.info("STRATEGY RESULTS")
            logger.info("=" * 50)
            
            # Process and log results for each symbol
            buy_count = 0
            sell_count = 0
            no_trade_count = 0
            
            if isinstance(results, dict):
                for symbol, result in results.items():
                    if not isinstance(result, dict):
                        logger.warning(f"Unexpected result type for {symbol}: {type(result)}")
                        continue
                        
                    logger.info(f"Results for {symbol}:")
                    
                    # Extract signals
                    technical_signal = result.get('technical_signal', 0)
                    sentiment_signal = result.get('sentiment_signal', 0)
                    ml_signal = result.get('ml_signal', 0)
                    combined_signal = result.get('combined_signal', 0)
                    
                    logger.info(f"  Technical signal: {technical_signal:.2f}")
                    logger.info(f"  Sentiment signal: {sentiment_signal:.2f}")
                    logger.info(f"  ML signal: {ml_signal:.2f}")
                    logger.info(f"  Combined signal: {combined_signal:.2f}")
                    
                    # Extract trade information
                    if 'trade_result' in result:
                        trade_result = result['trade_result']
                        if isinstance(trade_result, dict):
                            side = trade_result.get('side', 'none')
                            shares = trade_result.get('shares', 0)
                            price = trade_result.get('price', 0)
                            
                            if side == 'buy':
                                buy_count += 1
                                logger.info(f"  Trade: BUY {shares} shares at ${price:.2f}")
                            elif side == 'sell':
                                sell_count += 1
                                logger.info(f"  Trade: SELL {shares} shares at ${price:.2f}")
                            else:
                                no_trade_count += 1
                                logger.info("  No trade executed")
                            
                            # Log the trade
                            trade_data = {
                                "symbol": symbol,
                                "side": side,
                                "quantity": shares,
                                "price": price,
                                "signal": combined_signal
                            }
                            log_trade(trade_logger, trade_data)
                        else:
                            no_trade_count += 1
                            logger.info("  No trade executed (invalid trade result)")
                    else:
                        no_trade_count += 1
                        logger.info("  No trade executed")
            else:
                logger.warning(f"Unexpected results type: {type(results)}")
                logger.info(f"Results: {results}")
            
            # Log summary
            logger.info("=" * 50)
            logger.info("SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Processed {len(results) if isinstance(results, dict) else 0} symbols")
            logger.info(f"Buy trades: {buy_count}")
            logger.info(f"Sell trades: {sell_count}")
            logger.info(f"No trades: {no_trade_count}")
            
            # Get performance metrics if available
            try:
                metrics = strategy.get_performance_metrics()
                logger.info("Performance metrics:")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value}")
            except Exception as e:
                logger.warning(f"Could not get performance metrics: {e}")
            
            # Test alert system
            if buy_count > 0 or sell_count > 0:
                send_alert(alert_manager, 
                    "Trade Alert", 
                    f"Executed {buy_count} buy trades and {sell_count} sell trades"
                )
            
            logger.info("=" * 50)
            logger.info("END-TO-END TEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            logger.error(traceback.format_exc())
            send_alert(alert_manager, "Error Alert", f"Strategy execution failed: {e}")
        
    except Exception as e:
        logger.error(f"Error during end-to-end test: {e}")
        logger.error(traceback.format_exc())
        send_alert(alert_manager, "Error Alert", f"End-to-end test failed: {e}")
    
    finally:
        # Stop performance monitoring
        if monitor:
            try:
                monitor.stop()
                logger.info("Performance monitoring stopped")
            except:
                pass

if __name__ == "__main__":
    main()
