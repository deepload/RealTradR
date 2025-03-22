#!/usr/bin/env python3
"""
Database setup script for RealTradR

This script creates the necessary PostgreSQL database and tables for the RealTradR application.
"""

import os
import sys
import yaml
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def load_config():
    """Load configuration from config.yaml file"""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def create_database():
    """Create the PostgreSQL database if it doesn't exist"""
    config = load_config()
    db_config = config['database']['postgres']
    
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_config['database'],))
    exists = cursor.fetchone()
    
    if not exists:
        try:
            cursor.execute(f"CREATE DATABASE {db_config['database']}")
            print(f"Database '{db_config['database']}' created successfully")
        except Exception as e:
            print(f"Error creating database: {e}")
            sys.exit(1)
    else:
        print(f"Database '{db_config['database']}' already exists")
    
    cursor.close()
    conn.close()

def create_tables():
    """Create necessary tables in the database"""
    config = load_config()
    db_config = config['database']['postgres']
    
    # Connect to the database
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['database']
    )
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        hashed_password VARCHAR(100) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        is_superuser BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create trading_symbols table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trading_symbols (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) UNIQUE NOT NULL,
        name VARCHAR(100),
        exchange VARCHAR(20),
        is_active BOOLEAN DEFAULT TRUE,
        last_price NUMERIC(12, 4),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create api_keys table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_keys (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        broker VARCHAR(20) NOT NULL,
        api_key VARCHAR(100) NOT NULL,
        api_secret VARCHAR(100) NOT NULL,
        is_paper BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create ai_models table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_models (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        model_type VARCHAR(50) NOT NULL,
        symbol VARCHAR(20) REFERENCES trading_symbols(symbol),
        model_path VARCHAR(255),
        accuracy NUMERIC(5, 2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_trained TIMESTAMP,
        is_active BOOLEAN DEFAULT TRUE
    )
    ''')
    
    # Create trading_strategies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trading_strategies (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        description TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        parameters JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        symbol VARCHAR(20) REFERENCES trading_symbols(symbol),
        broker VARCHAR(20) NOT NULL,
        order_id VARCHAR(100),
        order_type VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        quantity NUMERIC(16, 8) NOT NULL,
        price NUMERIC(16, 8),
        status VARCHAR(20) NOT NULL,
        strategy_id INTEGER REFERENCES trading_strategies(id),
        ai_model_id INTEGER REFERENCES ai_models(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_paper BOOLEAN DEFAULT TRUE
    )
    ''')
    
    # Create trades table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        order_id INTEGER REFERENCES orders(id),
        user_id INTEGER REFERENCES users(id),
        symbol VARCHAR(20) REFERENCES trading_symbols(symbol),
        execution_price NUMERIC(16, 8) NOT NULL,
        quantity NUMERIC(16, 8) NOT NULL,
        side VARCHAR(10) NOT NULL,
        commission NUMERIC(16, 8),
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        profit_loss NUMERIC(16, 8),
        is_paper BOOLEAN DEFAULT TRUE
    )
    ''')
    
    # Create market_data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) REFERENCES trading_symbols(symbol),
        timestamp TIMESTAMP NOT NULL,
        open NUMERIC(16, 8) NOT NULL,
        high NUMERIC(16, 8) NOT NULL,
        low NUMERIC(16, 8) NOT NULL,
        close NUMERIC(16, 8) NOT NULL,
        volume NUMERIC(20, 2) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        UNIQUE(symbol, timestamp, timeframe)
    )
    ''')
    
    # Create news_sentiment table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) REFERENCES trading_symbols(symbol),
        headline TEXT NOT NULL,
        source VARCHAR(50),
        url TEXT,
        published_at TIMESTAMP,
        sentiment_score NUMERIC(5, 4),
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create backtests table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS backtests (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        strategy_id INTEGER REFERENCES trading_strategies(id),
        symbol VARCHAR(20) REFERENCES trading_symbols(symbol),
        start_date TIMESTAMP NOT NULL,
        end_date TIMESTAMP NOT NULL,
        initial_capital NUMERIC(16, 2) NOT NULL,
        final_capital NUMERIC(16, 2) NOT NULL,
        total_trades INTEGER NOT NULL,
        winning_trades INTEGER NOT NULL,
        profit_loss NUMERIC(16, 2) NOT NULL,
        sharpe_ratio NUMERIC(8, 4),
        max_drawdown NUMERIC(8, 4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        parameters JSONB
    )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("Database tables created successfully")

if __name__ == "__main__":
    create_database()
    create_tables()
    print("Database setup complete!")
