"""
End-to-End Test for RealTradR using Alpaca Paper Trading API

This script runs a complete end-to-end test of the RealTradR trading system
using Alpaca's paper trading API. It tests all components of the system:
- Risk management
- Technical indicators
- ML predictions
- Sentiment analysis
- Trade execution
- Performance monitoring

The test runs with real market data but uses a paper trading account to avoid
actual financial transactions.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from dotenv import load_dotenv

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import RealTradR modules
from backend.app.ai.advanced_strategy import AdvancedAIStrategy
from backend.app.ai.risk_management import RiskManager
from backend.app.ai.technical_indicators import TechnicalIndicators, MarketRegime
from backend.app.ai.ml_models_fallback import ModelManager
from backend.app.utils.performance import PerformanceMonitor
from backend.app.utils.trade_logger import TradeLogger
from backend.app.utils.alert_manager import AlertManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("alpaca_e2e_test")

# Load environment variables
load_dotenv()

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_API_BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

# Check if credentials are available
if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    logger.error("Alpaca API credentials not found in .env file")
    logger.info("Please set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file")
    sys.exit(1)

class AlpacaE2ETest:
    """End-to-End Test for RealTradR using Alpaca Paper Trading"""
    
    def __init__(self, config_file="strategy_config.json", test_days=5):
        """
        Initialize the test
        
        Args:
            config_file: Path to strategy configuration file
            test_days: Number of days to test
        """
        self.config_file = config_file
        self.test_days = test_days
        self.test_results = {}
        self.performance_monitor = None
        self.trade_logger = None
        self.alert_manager = None
        
        # Load strategy configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_file}")
        logger.info(f"Testing with symbols: {self.config['symbols']}")
        logger.info(f"Cash limit: ${self.config['cash_limit']}")
        logger.info(f"Signal weights: Technical={self.config['technical_weight']}, "
                   f"ML={self.config['ml_weight']}, "
                   f"Sentiment={self.config['sentiment_weight']}")
    
    def setup(self):
        """Set up the test environment"""
        logger.info("Setting up test environment...")
        
        # Initialize performance monitoring
        try:
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start()
            logger.info("Performance monitoring initialized")
        except Exception as e:
            logger.warning(f"Performance monitoring not available: {e}")
        
        # Initialize trade logging
        try:
            self.trade_logger = TradeLogger()
            logger.info("Trade logging initialized")
        except Exception as e:
            logger.warning(f"Trade logging not available: {e}")
        
        # Initialize alert manager
        try:
            self.alert_manager = AlertManager()
            logger.info("Alert manager initialized")
        except Exception as e:
            logger.warning(f"Alert manager not available: {e}")
        
        # Initialize strategy
        self.strategy = AdvancedAIStrategy(config=self.config)
        
        logger.info("Test environment setup complete")
    
    def run_test(self):
        """Run the end-to-end test"""
        logger.info("=" * 50)
        logger.info("STARTING END-TO-END TEST WITH ALPACA PAPER TRADING")
        logger.info("=" * 50)
        
        # Run strategy for specified number of days
        for day in range(self.test_days):
            logger.info(f"Running test day {day+1}/{self.test_days}")
            
            # Run strategy once
            result = self.strategy.run()
            
            # Log results
            self.log_results(day, result)
            
            # Wait before next run (if not the last day)
            if day < self.test_days - 1:
                logger.info(f"Waiting 5 seconds before next run...")
                time.sleep(5)
        
        # Get performance metrics
        metrics = self.strategy.get_performance_metrics()
        self.test_results["performance_metrics"] = metrics
        
        logger.info("=" * 50)
        logger.info("END-TO-END TEST COMPLETED")
        logger.info("=" * 50)
        
        return self.test_results
    
    def log_results(self, day, result):
        """Log the results of a strategy run"""
        logger.info(f"Results for day {day+1}:")
        
        # Check for errors
        if "error" in result:
            logger.error(f"Error running strategy: {result['error']}")
            return
        
        # Log portfolio value and cash
        logger.info(f"Portfolio value: ${result.get('portfolio_value', 0):.2f}")
        logger.info(f"Cash: ${result.get('cash', 0):.2f}")
        
        # Log results for each symbol
        for symbol, data in result.get("results", {}).items():
            if "error" in data:
                logger.error(f"Error processing {symbol}: {data['error']}")
                continue
            
            logger.info(f"Symbol: {symbol}")
            logger.info(f"  Technical signal: {data.get('technical_signal', 0):.2f}")
            logger.info(f"  ML signal: {data.get('ml_signal', 0):.2f}")
            logger.info(f"  Sentiment signal: {data.get('sentiment_signal', 0):.2f}")
            logger.info(f"  Combined signal: {data.get('combined_signal', 0):.2f}")
            
            trade_result = data.get("trade_result", {})
            if trade_result:
                if trade_result.get("trade_executed", False):
                    logger.info(f"  Trade executed: {trade_result.get('side')} "
                               f"{trade_result.get('quantity')} shares at "
                               f"${trade_result.get('price', 0):.2f}")
                else:
                    logger.info("  No trade executed")
            else:
                logger.info("  No trade executed")
        
        # Store results
        self.test_results[f"day_{day+1}"] = result
    
    def cleanup(self):
        """Clean up the test environment"""
        logger.info("Cleaning up test environment...")
        
        # Stop performance monitoring
        if self.performance_monitor:
            self.performance_monitor.stop()
        
        logger.info("Test environment cleanup complete")
    
    def generate_report(self, output_file="alpaca_e2e_test_report.json"):
        """Generate a report of the test results"""
        logger.info(f"Generating test report: {output_file}")
        
        # Add summary metrics
        summary = {
            "test_days": self.test_days,
            "symbols": self.config["symbols"],
            "cash_limit": self.config["cash_limit"],
            "technical_weight": self.config["technical_weight"],
            "ml_weight": self.config["ml_weight"],
            "sentiment_weight": self.config["sentiment_weight"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add performance metrics if available
        if "performance_metrics" in self.test_results:
            metrics = self.test_results["performance_metrics"]
            summary["total_return"] = metrics.get("total_return", 0)
            summary["annualized_return"] = metrics.get("annualized_return", 0)
            summary["sharpe_ratio"] = metrics.get("sharpe_ratio", 0)
            summary["max_drawdown"] = metrics.get("max_drawdown", 0)
        
        # Add summary to results
        self.test_results["summary"] = summary
        
        # Write results to file
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Test report generated: {output_file}")
        
        return summary


def main():
    """Main function to run the test"""
    parser = argparse.ArgumentParser(description="Run end-to-end test with Alpaca Paper Trading")
    parser.add_argument("--config", default="strategy_config.json", help="Path to strategy configuration file")
    parser.add_argument("--days", type=int, default=5, help="Number of days to test")
    parser.add_argument("--report", default="alpaca_e2e_test_report.json", help="Output file for test report")
    args = parser.parse_args()
    
    # Run the test
    test = AlpacaE2ETest(config_file=args.config, test_days=args.days)
    
    try:
        test.setup()
        results = test.run_test()
        summary = test.generate_report(output_file=args.report)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Test days: {summary['test_days']}")
        logger.info(f"Symbols: {summary['symbols']}")
        logger.info(f"Total return: {summary.get('total_return', 0):.2f}%")
        logger.info(f"Annualized return: {summary.get('annualized_return', 0):.2f}%")
        logger.info(f"Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max drawdown: {summary.get('max_drawdown', 0):.2f}%")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error running test: {e}")
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
