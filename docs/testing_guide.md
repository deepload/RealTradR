# RealTradR Testing Guide

This document provides detailed instructions for testing the RealTradR platform, with a focus on IBKR broker integration.

## Prerequisites

Before running tests, ensure you have:

1. Python 3.8+ installed
2. All dependencies installed via `pip install -r requirements.txt`
3. IBKR TWS or IB Gateway running (for live tests only)
4. Valid IBKR credentials (for live tests only)
5. PostgreSQL database set up

## Test Environment Setup

### 1. Install Testing Dependencies

```bash
pip install pytest pytest-cov pytest-mock fastapi-security httpx
```

### 2. Configure Test Environment

Create a `.env.test` file in the project root with the following content:

```
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=realtradrtestdb
SECRET_KEY=testsecretkey
ALPACA_API_KEY=testapikey
ALPACA_API_SECRET=testapisecret
ALPACA_PAPER_TRADING=true
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_PAPER_TRADING=true
```

### 3. Prepare Test Database

```bash
# Create test database
createdb realtradrtestdb

# Run migrations
alembic upgrade head
```

## Running Tests

### Automated Tests

Run all tests using pytest:

```bash
pytest -v
```

Run specific test files:

```bash
# Test IBKR broker service
pytest -v tests/test_ibkr_broker.py

# Test trading API
pytest -v tests/test_api_trading.py
```

Generate coverage report:

```bash
pytest --cov=app tests/
```

### Manual Testing Procedures

## 1. IBKR Broker Integration Tests

### 1.1 Connection Test

**Objective**: Verify that the application can connect to Interactive Brokers.

**Steps**:
1. Start TWS or IB Gateway
2. Set up API configuration in `config.yaml`
3. Run the following command:

```bash
python -m scripts.test_ibkr_connection
```

**Expected Result**: Script should output "Successfully connected to IBKR"

### 1.2 Market Data Retrieval

**Objective**: Verify that the application can retrieve market data from IBKR.

**Steps**:
1. Ensure TWS/IB Gateway is running
2. Run the server: `uvicorn app.main:app --reload`
3. Execute the following curl command:

```bash
curl -X GET "http://localhost:8000/api/trading/market-data/AAPL?timeframe=1d&limit=10" -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected Result**: JSON response containing AAPL historical data

### 1.3 Place Paper Trading Order

**Objective**: Verify that the application can place paper trading orders with IBKR.

**Steps**:
1. Ensure TWS/IB Gateway is running in Paper Trading mode
2. Run the server: `uvicorn app.main:app --reload`
3. Execute the following curl command to place a market order:

```bash
curl -X POST "http://localhost:8000/api/trading/orders" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol_id": 1,
    "broker": "ibkr",
    "order_type": "market",
    "side": "buy",
    "quantity": 1,
    "time_in_force": "day",
    "is_paper": true
  }'
```

**Expected Result**: JSON response containing the created order with status "submitted"

### 1.4 Get Account Information

**Objective**: Verify that the application can retrieve account information from IBKR.

**Steps**:
1. Ensure TWS/IB Gateway is running
2. Run the server: `uvicorn app.main:app --reload`
3. Execute the following curl command:

```bash
curl -X GET "http://localhost:8000/api/trading/account?broker_name=ibkr&paper_trading=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected Result**: JSON response containing account details

## 2. End-to-End Trading Workflow Test

**Objective**: Verify the complete trading workflow from market data analysis to order placement and position monitoring.

**Steps**:

1. **Get Market Data**
```bash
curl -X GET "http://localhost:8000/api/trading/market-data/AAPL?timeframe=1d&limit=30" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

2. **Get Current Quote**
```bash
curl -X GET "http://localhost:8000/api/trading/market-data/AAPL/quote?broker_name=ibkr" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

3. **Place a Market Order**
```bash
curl -X POST "http://localhost:8000/api/trading/orders" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol_id": 1,
    "broker": "ibkr",
    "order_type": "market",
    "side": "buy",
    "quantity": 1,
    "time_in_force": "day",
    "is_paper": true
  }'
```

4. **Check Order Status**
```bash
curl -X GET "http://localhost:8000/api/trading/orders/1?update_status=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

5. **Get Positions**
```bash
curl -X GET "http://localhost:8000/api/trading/positions?broker_name=ibkr&paper_trading=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected Results**:
- Market data and quote retrieval should return valid data
- Order placement should succeed with status "submitted" or "filled"
- Position should be visible after order fills

## 3. API Endpoint Testing

The following table outlines all API endpoints for the IBKR integration and how to test them:

| Endpoint | Method | Description | Test Command |
|----------|--------|-------------|-------------|
| /api/trading/symbols | GET | Get available trading symbols | `curl -X GET "http://localhost:8000/api/trading/symbols" -H "Authorization: Bearer YOUR_TOKEN"` |
| /api/trading/orders | POST | Create a new order | See 1.3 above |
| /api/trading/orders | GET | Get user orders | `curl -X GET "http://localhost:8000/api/trading/orders" -H "Authorization: Bearer YOUR_TOKEN"` |
| /api/trading/orders/active | GET | Get active orders | `curl -X GET "http://localhost:8000/api/trading/orders/active" -H "Authorization: Bearer YOUR_TOKEN"` |
| /api/trading/orders/{order_id} | GET | Get order by ID | `curl -X GET "http://localhost:8000/api/trading/orders/1" -H "Authorization: Bearer YOUR_TOKEN"` |
| /api/trading/orders/{order_id} | DELETE | Cancel order | `curl -X DELETE "http://localhost:8000/api/trading/orders/1" -H "Authorization: Bearer YOUR_TOKEN"` |
| /api/trading/positions | GET | Get user positions | See 2.5 above |
| /api/trading/account | GET | Get account info | See 1.4 above |
| /api/trading/market-data/{symbol} | GET | Get market data | See 2.1 above |
| /api/trading/market-data/{symbol}/quote | GET | Get market quote | See 2.2 above |

## Troubleshooting

### Common IBKR Connection Issues

1. **Connection Refused**: 
   - Ensure TWS/IB Gateway is running
   - Verify the port setting matches the API configuration in TWS/IB Gateway
   - Check that API connections are enabled in TWS/IB Gateway

2. **Authentication Failed**:
   - Verify that the client ID is unique and not used by another application
   - Ensure "Allow connections from localhost only" is checked if connecting locally

3. **Market Data Access**:
   - Ensure you have subscriptions to the relevant market data in IBKR
   - Check that your IBKR account has the permissions for the symbols you're requesting

### Debug Mode

Enable debug logging to get more information:

1. Set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

2. Check logs:
```bash
tail -f logs/realtradR.log
```

## Advanced Testing

### Performance Testing

Run the performance test script to measure API response times:

```bash
python -m scripts.performance_test
```

### Security Testing

Run the security test script to check for common vulnerabilities:

```bash
python -m scripts.security_test
```

## Continuous Integration

The repository includes GitHub Actions workflows for automated testing on each pull request and push to main.

To run the CI tests locally:

```bash
docker-compose -f docker-compose.ci.yml up --build
```

This will run all tests in an isolated Docker environment similar to the CI pipeline.
