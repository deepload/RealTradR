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

## Performance Metrics

The advanced strategy tracks the following performance metrics:
- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Value at Risk (VaR)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
