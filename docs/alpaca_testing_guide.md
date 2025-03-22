# RealTradR Alpaca Testing Guide

This guide provides detailed instructions for testing the RealTradR platform with a real Alpaca account. It covers both paper trading and live trading scenarios with appropriate safety measures.

## Prerequisites

Before testing with Alpaca, ensure you have:

1. An Alpaca account (either paper or live) from [app.alpaca.markets](https://app.alpaca.markets/)
2. API keys from your Alpaca account dashboard
3. RealTradR backend running properly
4. Python 3.8+ with all dependencies installed

## Setup Safety Measures

When testing with real accounts, especially live trading accounts, follow these safety precautions:

1. **Start with paper trading** - Always test with paper trading first before moving to live
2. **Use minimal quantities** - When testing with a live account, use minimal quantities (1 share)
3. **Test with liquid stocks** - Use highly liquid stocks like AAPL, MSFT, etc. to ensure easy execution
4. **Set trade limits** - Configure maximum position sizes and order values in your settings
5. **Monitor all activity** - Keep the Alpaca dashboard open to monitor all activity
6. **Have a kill switch** - Know how to quickly cancel all orders if needed

## Connection Setup

### 1. API Credentials Configuration

Store your Alpaca API credentials in a secure .env file:

```bash
# Create .env file (DO NOT commit to repository)
echo "ALPACA_API_KEY=your_api_key_here" > .env
echo "ALPACA_API_SECRET=your_api_secret_here" >> .env
echo "ALPACA_PAPER_TRADING=true" >> .env  # Change to false for live trading
```

### 2. Verify Connection

Use the provided test script to verify your connection:

```bash
# For paper trading (default)
python -m tests.scripts.test_alpaca_connection

# For live trading (view only, no orders)
python -m tests.scripts.test_alpaca_connection --live

# With specific API keys
python -m tests.scripts.test_alpaca_connection --api-key YOUR_KEY --api-secret YOUR_SECRET
```

## Testing Scenarios

### 1. Paper Trading Tests

Paper trading provides a risk-free environment with simulated funds:

```bash
# Test basic connection and account info
python -m tests.scripts.test_alpaca_connection

# Test order flow with paper trading (simulated, no real orders)
python -m tests.scripts.test_alpaca_connection --test-orders

# Test order flow with paper trading (executing orders)
python -m tests.scripts.test_alpaca_connection --test-orders --execute-orders
```

### 2. Live Trading Tests

⚠️ **CAUTION: These tests involve real money** ⚠️

```bash
# Test connection to live account (read-only, no orders)
python -m tests.scripts.test_alpaca_connection --live

# Test order flow simulation with live account (no actual orders)
python -m tests.scripts.test_alpaca_connection --live --test-orders

# Execute real orders with live account (USE WITH EXTREME CAUTION)
python -m tests.scripts.test_alpaca_connection --live --test-orders --execute-orders
```

### 3. API Endpoint Tests

Test the RealTradR API endpoints that integrate with Alpaca:

```bash
# Get a JWT token first
TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}' | jq -r '.access_token')

# Test account information endpoint
curl -X GET "http://localhost:8000/api/trading/account?broker_name=alpaca&paper_trading=true" \
  -H "Authorization: Bearer $TOKEN"

# Test market data endpoint
curl -X GET "http://localhost:8000/api/trading/market-data/AAPL?timeframe=1d&limit=10" \
  -H "Authorization: Bearer $TOKEN"

# Test creating an order (paper trading)
curl -X POST "http://localhost:8000/api/trading/orders" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "symbol_id": 1,
    "broker": "alpaca",
    "order_type": "market",
    "side": "buy",
    "quantity": 1,
    "time_in_force": "day",
    "is_paper": true
  }'
```

## Advanced Testing Scenarios

### 1. Trading Strategy Testing

Test AI-powered trading strategies with Alpaca paper trading:

1. Configure a trading strategy in the database
2. Run the strategy with paper trading
3. Monitor the performance
4. Compare with backtesting results

### 2. Market Data Integration Testing

Test the retrieval and storage of historical market data:

1. Request historical data for multiple timeframes
2. Verify data consistency
3. Test data updating mechanisms
4. Check for proper handling of market hours

### 3. Account Management Testing

Test account management features:

1. Portfolio allocation
2. Risk management rules
3. Position sizing algorithms
4. Account statistics calculations

### 4. WebSocket Feed Testing

Test real-time updates with Alpaca's WebSocket feed:

1. Subscribe to market data streams
2. Listen for order updates
3. Update UI in real-time when events occur
4. Test reconnection logic

## Troubleshooting

### Common Alpaca API Issues

1. **Rate Limiting**: Alpaca has API rate limits. If you hit them, implement exponential backoff.

2. **Market Hours**: Many operations only work during market hours (9:30 AM - 4:00 PM ET). Test accordingly.

3. **Account Restrictions**: New accounts might have restrictions. Check account status first.

4. **Missing Data**: Some symbols may have limited data. Use widely traded stocks for testing.

## Alpaca-Specific Settings

Configure these settings in your environment or config file:

```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_PAPER_TRADING=true|false
ALPACA_API_BASE_URL=https://paper-api.alpaca.markets or https://api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
ALPACA_WEBSOCKET_URL=wss://paper-api.alpaca.markets/stream or wss://api.alpaca.markets/stream
```

## Production Readiness Checklist

Before deploying to production:

- [ ] All paper trading tests pass successfully
- [ ] Order validation logic is thorough
- [ ] Error handling is robust
- [ ] Logging is comprehensive
- [ ] Rate limiting is respected
- [ ] WebSocket reconnection works properly
- [ ] Database correctly stores order and position data
- [ ] Security practices are implemented (secure API key storage)
- [ ] Trading limits are configured properly
- [ ] Notifications are set up for critical events

## Monitoring Live Systems

Once in production:

1. Set up alerts for unusual activity
2. Schedule regular database backups
3. Monitor performance metrics
4. Create a dashboard for real-time account status
5. Implement circuit breakers for extreme market conditions

By following this guide, you can safely test your RealTradR application with both paper and live Alpaca accounts while minimizing risk.
