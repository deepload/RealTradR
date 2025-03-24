"""
Trade Logger Module for RealTradR

This module provides functionality to log trade executions and maintain
a history of all trades for analysis and reporting.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Logger for trade executions
    
    This class provides methods to log trades to a database and file for
    record-keeping and analysis.
    """
    
    def __init__(self, db_path="trades.db", log_file="trades.json"):
        """
        Initialize the trade logger
        
        Args:
            db_path: Path to SQLite database file
            log_file: Path to JSON log file
        """
        self.db_path = db_path
        self.log_file = log_file
        
        # Create database if it doesn't exist
        self._init_db()
        
        logger.info(f"Initialized TradeLogger with db: {db_path}, log: {log_file}")
    
    def _init_db(self):
        """Initialize the SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                order_id TEXT,
                strategy TEXT,
                signal REAL,
                technical_signal REAL,
                ml_signal REAL,
                sentiment_signal REAL,
                market_regime TEXT,
                notes TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing trade database: {e}")
    
    def log_trade(self, trade_data):
        """
        Log a trade to database and file
        
        Args:
            trade_data: Dictionary with trade information
                Required keys: symbol, side, quantity, price
                Optional keys: order_id, strategy, signal, notes, etc.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure required fields are present
            required_fields = ["symbol", "side", "quantity", "price"]
            for field in required_fields:
                if field not in trade_data:
                    logger.error(f"Missing required field in trade data: {field}")
                    return False
            
            # Add timestamp if not present
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = datetime.now().isoformat()
            
            # Log to database
            self._log_to_db(trade_data)
            
            # Log to file
            self._log_to_file(trade_data)
            
            logger.info(f"Logged trade: {trade_data['side']} {trade_data['quantity']} {trade_data['symbol']} @ ${trade_data['price']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return False
    
    def _log_to_db(self, trade_data):
        """Log trade to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare fields and values
            fields = []
            values = []
            placeholders = []
            
            for key, value in trade_data.items():
                fields.append(key)
                values.append(value)
                placeholders.append("?")
            
            # Build SQL query
            sql = f"INSERT INTO trades ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
            
            # Execute query
            cursor.execute(sql, values)
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging trade to database: {e}")
            raise
    
    def _log_to_file(self, trade_data):
        """Log trade to JSON file"""
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Load existing trades
            trades = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    try:
                        trades = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Error reading trades from {self.log_file}, starting new file")
                        trades = []
            
            # Add new trade
            trades.append(trade_data)
            
            # Write to file
            with open(self.log_file, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error logging trade to file: {e}")
            raise
    
    def get_trades(self, symbol=None, start_date=None, end_date=None, strategy=None):
        """
        Get trades from database
        
        Args:
            symbol: Filter by symbol
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            strategy: Filter by strategy name
        
        Returns:
            List of trade dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            # Build query
            sql = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                sql += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                sql += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                sql += " AND timestamp <= ?"
                params.append(end_date)
            
            if strategy:
                sql += " AND strategy = ?"
                params.append(strategy)
            
            sql += " ORDER BY timestamp DESC"
            
            # Execute query
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            trades = [dict(row) for row in rows]
            
            conn.close()
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def get_trade_summary(self, symbol=None, start_date=None, end_date=None, strategy=None):
        """
        Get summary of trades
        
        Args:
            symbol: Filter by symbol
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            strategy: Filter by strategy name
        
        Returns:
            Dictionary with trade summary
        """
        trades = self.get_trades(symbol, start_date, end_date, strategy)
        
        if not trades:
            return {"error": "No trades found"}
        
        # Calculate summary statistics
        buy_trades = [t for t in trades if t["side"].lower() == "buy"]
        sell_trades = [t for t in trades if t["side"].lower() == "sell"]
        
        total_buys = len(buy_trades)
        total_sells = len(sell_trades)
        
        buy_volume = sum(t["quantity"] for t in buy_trades)
        sell_volume = sum(t["quantity"] for t in sell_trades)
        
        buy_value = sum(t["quantity"] * t["price"] for t in buy_trades)
        sell_value = sum(t["quantity"] * t["price"] for t in sell_trades)
        
        avg_buy_price = buy_value / buy_volume if buy_volume > 0 else 0
        avg_sell_price = sell_value / sell_volume if sell_volume > 0 else 0
        
        return {
            "total_trades": len(trades),
            "total_buys": total_buys,
            "total_sells": total_sells,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "avg_buy_price": avg_buy_price,
            "avg_sell_price": avg_sell_price,
            "net_volume": buy_volume - sell_volume,
            "net_value": buy_value - sell_value,
            "first_trade_date": trades[-1]["timestamp"],
            "last_trade_date": trades[0]["timestamp"]
        }
