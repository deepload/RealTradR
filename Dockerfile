FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow separately to ensure compatibility
RUN pip install --no-cache-dir tensorflow==2.12.0

# Copy the project code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data

# Set up environment variables for Alpaca API (these will be overridden by docker-compose)
ENV ALPACA_API_KEY=placeholder
ENV ALPACA_API_SECRET=placeholder
ENV ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
ENV ALPACA_PAPER_TRADING=true

# Default command
CMD ["python", "run_advanced_strategy.py"]
