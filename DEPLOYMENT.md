# RealTradR Deployment Guide

This guide provides detailed instructions for deploying RealTradR in production environments.

## Prerequisites

- Python 3.8+ with pip
- PostgreSQL 12+
- Redis 6+
- Node.js 14+ with npm
- Sufficient RAM (8GB minimum, 16GB recommended)
- API keys for your chosen broker(s)

## Environment Setup

### 1. TensorFlow Installation

TensorFlow can be challenging to set up correctly on Windows. Follow these steps:

```bash
# Install Microsoft Visual C++ Redistributable
# Download from: https://aka.ms/vs/16/release/vc_redist.x64.exe

# Install TensorFlow with specific version
pip install tensorflow==2.10.0

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

If you encounter DLL loading issues:
1. Ensure you have the correct Visual C++ Redistributable installed
2. Try using a CPU-only version of TensorFlow: `pip install tensorflow-cpu`
3. Check compatibility between your Python version and TensorFlow version

### 2. Database Setup

```bash
# Create PostgreSQL database
createdb realtradR

# Initialize database schema
cd backend
python setup_db.py
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```
# API Keys (NEVER commit these to version control)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
IBKR_CLIENT_ID=your_ibkr_client_id

# Database
DATABASE_URL=postgresql://username:password@localhost/realtradR
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET=your_secure_random_string
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60

# Logging
LOG_LEVEL=INFO

# Trading Settings
PAPER_TRADING=true
MAX_POSITION_SIZE_PCT=20
```

## Production Deployment

### Option 1: Docker Deployment (Recommended)

1. Build and start the containers:

```bash
docker-compose up -d
```

2. Monitor the logs:

```bash
docker-compose logs -f
```

### Option 2: Manual Deployment

#### Backend Deployment

1. Install Gunicorn:

```bash
pip install gunicorn
```

2. Start the backend server:

```bash
cd backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

#### Frontend Deployment

1. Build the frontend:

```bash
cd frontend
npm run build
```

2. Serve with Nginx:

```
# Example Nginx configuration
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        root /path/to/realtradR/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Maintenance

### Logging

Logs are stored in the `logs` directory with daily rotation. Monitor them for errors:

```bash
tail -f logs/realtradR.log
```

### Database Backups

Set up automated backups:

```bash
# Create a backup script
mkdir -p backups
pg_dump realtradR > backups/realtradR_$(date +%Y%m%d).sql
```

### Performance Monitoring

1. Use Prometheus and Grafana for monitoring:
   - CPU/Memory usage
   - API response times
   - Trading performance metrics

2. Set up alerting for:
   - Unusual trading activity
   - Large drawdowns
   - System errors
   - API failures

## Security Best Practices

1. **API Keys**: Never store API keys in the code or expose them in client-side code
2. **Rate Limiting**: Implement rate limiting on all API endpoints
3. **Input Validation**: Validate all user inputs
4. **Regular Updates**: Keep all dependencies updated
5. **Firewall**: Restrict access to the server
6. **SSL/TLS**: Use HTTPS for all connections

## Troubleshooting

### Common Issues

#### TensorFlow DLL Loading Error

**Problem**: `ImportError: DLL load failed while importing _pywrap_tensorflow_internal`

**Solution**:
1. Install the Microsoft Visual C++ Redistributable
2. Use a compatible TensorFlow version: `pip install tensorflow==2.10.0`
3. Try the CPU-only version: `pip install tensorflow-cpu`

#### Database Connection Issues

**Problem**: `OperationalError: could not connect to server`

**Solution**:
1. Check PostgreSQL is running: `pg_isready`
2. Verify connection string in `.env` file
3. Check firewall settings

#### API Rate Limiting

**Problem**: Getting rate limited by broker APIs

**Solution**:
1. Implement caching for market data
2. Reduce polling frequency
3. Use websocket connections where available

## Production Checklist

Before going live with real trading:

- [ ] Run extensive backtests with historical data
- [ ] Test in paper trading mode for at least 2-4 weeks
- [ ] Set up proper monitoring and alerting
- [ ] Configure risk limits and circuit breakers
- [ ] Implement disaster recovery procedures
- [ ] Document operational procedures
- [ ] Set up regular database backups
- [ ] Ensure all API keys have appropriate permissions
- [ ] Test system recovery after failures

## Scaling

As your trading volume increases:

1. **Horizontal Scaling**: Add more API servers behind a load balancer
2. **Database Optimization**: Add indexes, consider read replicas
3. **Caching**: Increase Redis capacity for market data caching
4. **Microservices**: Split into separate services for data collection, analysis, and trading

## Support

For production support:
- Email: support@realtradR.com
- Documentation: https://docs.realtradR.com
- GitHub Issues: https://github.com/yourusername/RealTradR/issues
