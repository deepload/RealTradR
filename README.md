# RealTradR - AI-Powered Stock Trading Simulation Bot

A full-stack AI-powered stock trading simulation platform with support for multiple brokers.

## Features

- **AI-Powered Trading**: Uses TensorFlow and Scikit-learn for price prediction, and NLP for sentiment analysis
- **Multi-Broker Support**: Integrates with Alpaca (US stocks) and Interactive Brokers (global markets)
- **Trading Modes**: Paper Trading (simulation) and Live Trading options
- **Real-time Analytics**: Dashboard for monitoring AI trades and performance
- **Market Data Collection**: Fetch and store real-time and historical data
- **Advanced AI Strategies**: LSTM model, sentiment analysis, technical indicators, and reinforcement learning

## Advanced AI Trading Strategy

The RealTradR platform implements a sophisticated AI trading strategy that combines multiple signals:

### Signal Components

1. **Machine Learning Models** (`ml_models.py`)
   - LSTM Networks for time series prediction
   - Ensemble models combining multiple prediction techniques
   - Sentiment-aware models that adjust predictions based on market sentiment

2. **Sentiment Analysis** (`sentiment_analyzer.py`)
   - Twitter/X sentiment analysis for stock symbols
   - Reddit discussions and sentiment tracking
   - Financial news sentiment extraction
   - Weighted sentiment scoring across multiple sources

3. **Technical Indicators** (`technical_indicators.py`)
   - Comprehensive set of technical indicators (SMA, EMA, MACD, RSI, etc.)
   - Market regime detection (bull/bear market identification)
   - Volatility analysis and adaptive parameters

4. **Risk Management** (`risk_management.py`)
   - Dynamic stop-loss and take-profit calculations based on volatility
   - Position sizing using Kelly criterion and volatility-adjusted methods
   - Correlation analysis to prevent overexposure to similar assets
   - Drawdown control mechanisms to reduce risk during downturns

### Strategy Integration

The `advanced_strategy.py` module integrates all these components with configurable weights:
- Technical signals (default: 40% weight)
- Sentiment signals (default: 30% weight)
- Machine learning predictions (default: 30% weight)

The strategy dynamically adjusts its parameters based on detected market regimes and implements sophisticated position sizing to optimize risk-adjusted returns.

### Backtesting

The strategy can be backtested using historical data to evaluate performance before deploying with real money:

```bash
# Run backtest with default parameters
python run_backtest.py

# Run backtest with custom parameters
python run_backtest.py --config custom_strategy_config.json
```

## Tech Stack

- **Backend**: Python (FastAPI)
- **AI Core**: TensorFlow, Scikit-learn, NLTK
- **Database**: PostgreSQL (trade history), Redis (caching)
- **Frontend**: Angular (dashboard)
- **Trading Framework**: Integrated with Hummingbot for execution
- **Security**: JWT authentication

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL
- Redis
- Docker (optional)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/RealTradR.git
cd RealTradR
```

2. Set up the backend
```bash
cd backend
pip install -r requirements.txt
python setup_db.py
```

3. Set up the frontend
```bash
cd ../frontend
npm install
```

4. Configure your broker API keys in `config.yaml`

5. Start the application
```bash
# Start backend
cd backend
uvicorn main:app --reload

# Start frontend
cd ../frontend
ng serve
```

## Configuration

### Broker Configuration

Configure your broker settings in `config.yaml`:

```yaml
broker:
  default: alpaca  # Use Alpaca as default
  alpaca:
    api_key: YOUR_ALPACA_API_KEY
    api_secret: YOUR_ALPACA_API_SECRET
    paper_trading: true  # Set to false for live trading
  ibkr:
    tws_port: 7497
    tws_host: 127.0.0.1
    client_id: 1
    paper_trading: true  # Set to false for live trading

ai:
  use_sentiment: true
  use_technical: true
  use_reinforcement: true
  model_update_frequency: daily  # Options: hourly, daily, weekly
```

### Strategy Configuration

The trading strategy is configured through `strategy_config.json`:

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "cash_limit": 100000.0,
  "sentiment_weight": 0.3,
  "technical_weight": 0.4,
  "ml_weight": 0.3,
  "position_sizing": "kelly",  // Options: "equal", "kelly", "volatility"
  "max_position_pct": 20.0,
  "use_market_regime": true,
  "use_sentiment": true,
  "use_ml_models": true
  // See full configuration options in the file
}
```

## Performance Metrics

The system tracks and reports the following performance metrics:

- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Daily Returns**: Daily percentage returns of the strategy
- **Volatility**: Standard deviation of returns

## Configuration Guide

### API Credentials Setup

RealTradR uses Alpaca for paper and live trading. To set up your API credentials:

1. Create an account at [Alpaca](https://alpaca.markets/)
2. Obtain your API key and secret from the Alpaca dashboard
3. Add them to your `strategy_config.json` file:

```json
"api_credentials": {
  "use_test_credentials": true,
  "api_key": "YOUR_API_KEY",
  "api_secret": "YOUR_API_SECRET",
  "paper_trading": true
}
```

### Market Data Access

Alpaca offers different tiers of market data access:

- **Free tier**: Limited to real-time quotes and snapshot data
- **Paid tier**: Includes historical data and more advanced features

If you're using the free tier, RealTradR will automatically fall back to using mock data for historical analysis while still using real-time quotes for execution.

### Configuration Options

The `strategy_config.json` file contains all configurable parameters:

#### Symbols and Cash Limits
```json
"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
"cash_limit": 100000.0
```

#### Signal Weights
```json
"signal_weights": {
  "sentiment": 0.3,
  "technical": 0.4,
  "ml": 0.3
}
```

#### Position Sizing
```json
"position_sizing": {
  "method": "kelly",
  "max_position_pct": 0.2,
  "base_position_pct": 0.05,
  "min_position_dollars": 1000
}
```

#### Risk Management
```json
"risk_management": {
  "stop_loss": 2.0,
  "take_profit": 5.0,
  "max_drawdown": 25.0,
  "correlation_threshold": 0.7
}
```

#### Market Data Settings
```json
"market_data": {
  "default_days": 60,
  "timeframes": ["1D", "1H", "15Min"],
  "use_mock_data_on_api_failure": true,
  "mock_data_volatility": 0.015,
  "use_mock_data": false
}
```

## Running the Strategy

### Command Line Options

The main script `run_strategy_with_real_data.py` accepts the following command line arguments:

- `--config FILE`: Path to configuration file (default: strategy_config.json)
- `--paper`: Run in paper trading mode (default)
- `--live`: Run in live trading mode (use with caution!)
- `--mock`: Use mock data instead of real market data
- `--continuous`: Run the strategy continuously
- `--interval SECONDS`: Interval between strategy runs in continuous mode (default: 300)
- `--report`: Generate a performance report after execution
- `--verbose`: Enable verbose logging

### Example Commands

```bash
# Run once with paper trading
python run_strategy_with_real_data.py --paper --report

# Run continuously with 5-minute intervals
python run_strategy_with_real_data.py --paper --continuous --interval 300

# Run with mock data for testing
python run_strategy_with_real_data.py --paper --mock --report
```

## Troubleshooting

### API Connection Issues

If you encounter 403 Forbidden errors when connecting to the Alpaca API:

1. **Check your API credentials**: Ensure your API key and secret are correct
2. **Verify account status**: Make sure your Alpaca account is active
3. **Check endpoint permissions**: The free tier has limited access to historical data
4. **Use mock data**: Enable `use_mock_data: true` in the config for testing

### TensorFlow DLL Loading Error

If you see TensorFlow DLL loading errors on Windows:

1. Install the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019
2. Ensure you have the correct version of CUDA and cuDNN installed (if using GPU)
3. Try reinstalling TensorFlow with: `pip uninstall tensorflow && pip install tensorflow`
4. The system will automatically fall back to simplified ML models if TensorFlow fails to load

### Performance Issues

If the strategy is running slowly:

1. Reduce the number of symbols being traded
2. Increase the interval between strategy runs
3. Use a smaller lookback period for historical data
4. Disable sentiment analysis if not needed

## Monitoring and Logging

RealTradR includes comprehensive logging and monitoring:

- **Trade logs**: Stored in `trades.csv` and `trades.json`
- **Performance reports**: Generated as JSON files with timestamp (e.g., `report_20250324_223624.json`)
- **Console logs**: Real-time logging of strategy execution, signals, and trades
- **Alert system**: Configurable alerts for significant events (price movements, execution issues)

## Production Deployment

For production deployment:

1. Set up proper monitoring and alerting
2. Use a dedicated server or cloud instance
3. Configure automatic restart in case of crashes
4. Set up regular backups of configuration and trade data
5. Consider using a process manager like PM2 or supervisord

## API Documentation

### REST API Endpoints

The RealTradR backend exposes the following REST API endpoints:

#### Strategy Management
- `GET /api/strategy/status`: Get current strategy status
- `POST /api/strategy/start`: Start the strategy
- `POST /api/strategy/stop`: Stop the strategy
- `PUT /api/strategy/config`: Update strategy configuration

#### Trading
- `GET /api/trades`: Get list of executed trades
- `GET /api/positions`: Get current positions
- `POST /api/trades/execute`: Manually execute a trade

#### Performance
- `GET /api/performance`: Get performance metrics
- `GET /api/performance/daily`: Get daily performance data
- `GET /api/performance/report`: Generate performance report

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Running Tests

RealTradR includes comprehensive tests for all components:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_sentiment_analyzer.py
pytest tests/test_technical_indicators.py
pytest tests/test_ml_models.py
pytest tests/test_advanced_strategy.py
pytest tests/test_risk_management.py

# Run backtests with performance evaluation
python tests/run_strategy_backtest.py
```
