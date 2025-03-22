# RealTradR - AI-Powered Stock Trading Simulation Bot

A full-stack AI-powered stock trading simulation platform with support for multiple brokers.

## Features

- **AI-Powered Trading**: Uses TensorFlow and Scikit-learn for price prediction, and NLP for sentiment analysis
- **Multi-Broker Support**: Integrates with Alpaca (US stocks) and Interactive Brokers (global markets)
- **Trading Modes**: Paper Trading (simulation) and Live Trading options
- **Real-time Analytics**: Dashboard for monitoring AI trades and performance
- **Market Data Collection**: Fetch and store real-time and historical data
- **Advanced AI Strategies**: LSTM model, sentiment analysis, technical indicators, and reinforcement learning

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
