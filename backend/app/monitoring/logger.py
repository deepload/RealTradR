"""
Logging and Monitoring Module for RealTradR

This module provides a centralized logging system with different log levels,
file and console handlers, and integration with monitoring services.
"""

import os
import sys
import logging
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
import socket
import platform

# Optional imports for monitoring services
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class LoggerFactory:
    """Factory class to create and configure loggers."""
    
    _instance = None
    _lock = threading.Lock()
    _loggers = {}
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of LoggerFactory."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the logger factory."""
        self.default_level = logging.INFO
        self.log_dir = os.environ.get("LOG_DIR", "logs")
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.max_bytes = 10 * 1024 * 1024  # 10MB
        self.backup_count = 10
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Set up the root logger with console handler."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name, level=None, log_to_file=True, rotate_when='midnight'):
        """
        Get a logger with the specified name and configuration.
        
        Args:
            name: Logger name
            level: Log level (default: INFO)
            log_to_file: Whether to log to file
            rotate_when: When to rotate logs ('size' or 'midnight')
            
        Returns:
            Configured logger
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        
        # Set level
        if level is None:
            level = self.default_level
        logger.setLevel(level)
        
        # Add file handler if requested
        if log_to_file:
            log_file = os.path.join(self.log_dir, f"{name}.log")
            
            if rotate_when == 'size':
                # Size-based rotation
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count
                )
            else:
                # Time-based rotation
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when=rotate_when,
                    backupCount=self.backup_count
                )
            
            file_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
            logger.addHandler(file_handler)
        
        # Store logger
        self._loggers[name] = logger
        
        return logger


class TradeLogger:
    """Logger specifically for trade-related events."""
    
    def __init__(self, strategy_name):
        """
        Initialize the trade logger.
        
        Args:
            strategy_name: Name of the trading strategy
        """
        factory = LoggerFactory.get_instance()
        self.logger = factory.get_logger(f"trade_{strategy_name}")
        self.strategy_name = strategy_name
        
        # Create trade log directory
        self.trade_log_dir = os.path.join(factory.log_dir, "trades")
        os.makedirs(self.trade_log_dir, exist_ok=True)
        
        # Create trade log file
        self.trade_log_file = os.path.join(
            self.trade_log_dir,
            f"{strategy_name}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        # Initialize trade metrics
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
    
    def log_trade(self, trade_data):
        """
        Log a trade to both the logger and the trade log file.
        
        Args:
            trade_data: Dictionary with trade information
        """
        # Add timestamp and strategy name
        trade_data["timestamp"] = datetime.now().isoformat()
        trade_data["strategy"] = self.strategy_name
        
        # Log to logger
        self.logger.info(f"Trade: {json.dumps(trade_data)}")
        
        # Log to trade log file
        with open(self.trade_log_file, "a") as f:
            f.write(json.dumps(trade_data) + "\n")
        
        # Update metrics
        self.trade_count += 1
        
        # Update profit/loss metrics if available
        if "profit_loss" in trade_data:
            pl = trade_data["profit_loss"]
            if pl > 0:
                self.win_count += 1
                self.total_profit += pl
            elif pl < 0:
                self.loss_count += 1
                self.total_loss += abs(pl)
    
    def log_signal(self, symbol, signal_data):
        """
        Log a trading signal.
        
        Args:
            symbol: Symbol the signal is for
            signal_data: Dictionary with signal information
        """
        # Add timestamp, strategy name, and symbol
        signal_data["timestamp"] = datetime.now().isoformat()
        signal_data["strategy"] = self.strategy_name
        signal_data["symbol"] = symbol
        
        # Log to logger
        self.logger.info(f"Signal for {symbol}: {json.dumps(signal_data)}")
    
    def get_metrics(self):
        """
        Get trade metrics.
        
        Returns:
            Dictionary with trade metrics
        """
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        return {
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": win_rate,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "profit_factor": profit_factor
        }


class PerformanceMonitor:
    """Monitor and log system performance metrics."""
    
    def __init__(self, interval=60):
        """
        Initialize the performance monitor.
        
        Args:
            interval: Interval in seconds to collect metrics
        """
        factory = LoggerFactory.get_instance()
        self.logger = factory.get_logger("performance")
        self.interval = interval
        self.running = False
        self.thread = None
        
        # Initialize Prometheus metrics if available
        self.prometheus_port = int(os.environ.get("PROMETHEUS_PORT", 8000))
        self.setup_prometheus()
    
    def setup_prometheus(self):
        """Set up Prometheus metrics if available."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available. Skipping metrics setup.")
            return
        
        try:
            # Start Prometheus HTTP server
            start_http_server(self.prometheus_port)
            self.logger.info(f"Started Prometheus metrics server on port {self.prometheus_port}")
            
            # Define metrics
            self.cpu_gauge = Gauge('realtrad_cpu_usage', 'CPU usage percentage')
            self.memory_gauge = Gauge('realtrad_memory_usage', 'Memory usage in MB')
            self.trade_counter = Counter('realtrad_trades_total', 'Total number of trades', ['strategy', 'symbol', 'side'])
            self.signal_histogram = Histogram('realtrad_signal_strength', 'Trading signal strength', ['strategy', 'symbol', 'type'])
            
            self.logger.info("Prometheus metrics initialized")
        except Exception as e:
            self.logger.error(f"Error setting up Prometheus metrics: {e}")
    
    def start(self):
        """Start the performance monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.logger.info(f"Started performance monitoring with interval {self.interval}s")
    
    def stop(self):
        """Stop the performance monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self.logger.info("Stopped performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect and log metrics
                metrics = self._collect_metrics()
                self.logger.info(f"Performance metrics: {json.dumps(metrics)}")
                
                # Update Prometheus metrics if available
                if PROMETHEUS_AVAILABLE:
                    self.cpu_gauge.set(metrics["cpu_percent"])
                    self.memory_gauge.set(metrics["memory_mb"])
            except Exception as e:
                self.logger.error(f"Error collecting performance metrics: {e}")
            
            # Sleep for the specified interval
            time.sleep(self.interval)
    
    def _collect_metrics(self):
        """
        Collect system performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform()
        }
        
        try:
            import psutil
            
            # CPU metrics
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            metrics["cpu_count"] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["memory_total_mb"] = memory.total / (1024 * 1024)
            metrics["memory_available_mb"] = memory.available / (1024 * 1024)
            metrics["memory_used_mb"] = memory.used / (1024 * 1024)
            metrics["memory_mb"] = memory.used / (1024 * 1024)
            metrics["memory_percent"] = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics["disk_total_gb"] = disk.total / (1024 * 1024 * 1024)
            metrics["disk_used_gb"] = disk.used / (1024 * 1024 * 1024)
            metrics["disk_free_gb"] = disk.free / (1024 * 1024 * 1024)
            metrics["disk_percent"] = disk.percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics["net_bytes_sent"] = net_io.bytes_sent
            metrics["net_bytes_recv"] = net_io.bytes_recv
            
        except ImportError:
            metrics["error"] = "psutil not available"
        
        return metrics
    
    def record_trade(self, strategy, symbol, side, quantity, price):
        """
        Record a trade in Prometheus metrics.
        
        Args:
            strategy: Strategy name
            symbol: Symbol traded
            side: Trade side (buy/sell)
            quantity: Trade quantity
            price: Trade price
        """
        if PROMETHEUS_AVAILABLE:
            self.trade_counter.labels(strategy=strategy, symbol=symbol, side=side).inc()
    
    def record_signal(self, strategy, symbol, signal_type, signal_value):
        """
        Record a signal in Prometheus metrics.
        
        Args:
            strategy: Strategy name
            symbol: Symbol
            signal_type: Signal type (technical, sentiment, ml, combined)
            signal_value: Signal value
        """
        if PROMETHEUS_AVAILABLE:
            self.signal_histogram.labels(
                strategy=strategy, symbol=symbol, type=signal_type
            ).observe(signal_value)


class AlertManager:
    """Manager for system and trading alerts."""
    
    def __init__(self, config=None):
        """
        Initialize the alert manager.
        
        Args:
            config: Alert configuration
        """
        factory = LoggerFactory.get_instance()
        self.logger = factory.get_logger("alerts")
        
        # Default configuration
        self.config = {
            "enabled": True,
            "log_alerts": True,
            "email_alerts": False,
            "slack_alerts": False,
            "email_config": {
                "smtp_server": os.environ.get("SMTP_SERVER", ""),
                "smtp_port": int(os.environ.get("SMTP_PORT", 587)),
                "smtp_username": os.environ.get("SMTP_USERNAME", ""),
                "smtp_password": os.environ.get("SMTP_PASSWORD", ""),
                "from_email": os.environ.get("ALERT_FROM_EMAIL", ""),
                "to_emails": os.environ.get("ALERT_TO_EMAILS", "").split(",")
            },
            "slack_config": {
                "webhook_url": os.environ.get("SLACK_WEBHOOK_URL", ""),
                "channel": os.environ.get("SLACK_CHANNEL", "#alerts")
            },
            "thresholds": {
                "drawdown_pct": float(os.environ.get("ALERT_DRAWDOWN_PCT", 10.0)),
                "cpu_pct": float(os.environ.get("ALERT_CPU_PCT", 90.0)),
                "memory_pct": float(os.environ.get("ALERT_MEMORY_PCT", 90.0)),
                "disk_pct": float(os.environ.get("ALERT_DISK_PCT", 90.0))
            }
        }
        
        # Update with provided configuration
        if config:
            self._update_config(config)
        
        self.logger.info("Alert manager initialized")
    
    def _update_config(self, config):
        """
        Update the configuration with provided values.
        
        Args:
            config: New configuration values
        """
        def update_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    update_dict(target[key], value)
                else:
                    target[key] = value
        
        update_dict(self.config, config)
    
    def alert(self, level, message, details=None, alert_type=None):
        """
        Send an alert.
        
        Args:
            level: Alert level (info, warning, error, critical)
            message: Alert message
            details: Additional details
            alert_type: Type of alert (system, trading, performance)
        """
        if not self.config["enabled"]:
            return
        
        # Create alert data
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "details": details or {},
            "type": alert_type or "general"
        }
        
        # Log alert
        if self.config["log_alerts"]:
            log_method = getattr(self.logger, level.lower(), self.logger.warning)
            log_method(f"ALERT: {message} - {json.dumps(details or {})}")
        
        # Send email alert
        if self.config["email_alerts"]:
            self._send_email_alert(alert_data)
        
        # Send Slack alert
        if self.config["slack_alerts"]:
            self._send_slack_alert(alert_data)
    
    def _send_email_alert(self, alert_data):
        """
        Send an email alert.
        
        Args:
            alert_data: Alert data
        """
        if not REQUESTS_AVAILABLE:
            self.logger.warning("Requests package not available. Cannot send email alert.")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            config = self.config["email_config"]
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = config["from_email"]
            msg["To"] = ", ".join(config["to_emails"])
            msg["Subject"] = f"RealTradR Alert: {alert_data['level'].upper()} - {alert_data['message']}"
            
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>RealTradR Alert</h2>
                <p><strong>Level:</strong> {alert_data['level'].upper()}</p>
                <p><strong>Type:</strong> {alert_data['type']}</p>
                <p><strong>Time:</strong> {alert_data['timestamp']}</p>
                <p><strong>Message:</strong> {alert_data['message']}</p>
                <h3>Details:</h3>
                <pre>{json.dumps(alert_data['details'], indent=2)}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
                server.starttls()
                server.login(config["smtp_username"], config["smtp_password"])
                server.send_message(msg)
            
            self.logger.info(f"Sent email alert to {', '.join(config['to_emails'])}")
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, alert_data):
        """
        Send a Slack alert.
        
        Args:
            alert_data: Alert data
        """
        if not REQUESTS_AVAILABLE:
            self.logger.warning("Requests package not available. Cannot send Slack alert.")
            return
        
        try:
            config = self.config["slack_config"]
            webhook_url = config["webhook_url"]
            
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return
            
            # Create message
            color = {
                "info": "#36a64f",
                "warning": "#ffcc00",
                "error": "#ff0000",
                "critical": "#9b0000"
            }.get(alert_data["level"].lower(), "#36a64f")
            
            payload = {
                "channel": config["channel"],
                "username": "RealTradR Alert",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "fallback": f"{alert_data['level'].upper()}: {alert_data['message']}",
                        "color": color,
                        "title": f"{alert_data['level'].upper()}: {alert_data['message']}",
                        "fields": [
                            {
                                "title": "Type",
                                "value": alert_data["type"],
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert_data["timestamp"],
                                "short": True
                            }
                        ],
                        "text": f"```{json.dumps(alert_data['details'], indent=2)}```"
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code != 200:
                self.logger.warning(f"Error sending Slack alert: {response.status_code} - {response.text}")
            else:
                self.logger.info(f"Sent Slack alert to {config['channel']}")
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
    
    def check_drawdown(self, portfolio_value, peak_value):
        """
        Check if drawdown exceeds threshold and send alert if needed.
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            
        Returns:
            True if alert was sent, False otherwise
        """
        if peak_value <= 0:
            return False
        
        drawdown_pct = (peak_value - portfolio_value) / peak_value * 100
        threshold = self.config["thresholds"]["drawdown_pct"]
        
        if drawdown_pct >= threshold:
            self.alert(
                "warning",
                f"Portfolio drawdown of {drawdown_pct:.2f}% exceeds threshold of {threshold:.2f}%",
                {
                    "current_value": portfolio_value,
                    "peak_value": peak_value,
                    "drawdown_pct": drawdown_pct,
                    "threshold": threshold
                },
                "trading"
            )
            return True
        
        return False
    
    def check_system_resources(self, metrics):
        """
        Check if system resource usage exceeds thresholds and send alerts if needed.
        
        Args:
            metrics: Dictionary with system metrics
            
        Returns:
            List of alert types sent
        """
        alerts_sent = []
        thresholds = self.config["thresholds"]
        
        # Check CPU usage
        if "cpu_percent" in metrics and metrics["cpu_percent"] >= thresholds["cpu_pct"]:
            self.alert(
                "warning",
                f"CPU usage of {metrics['cpu_percent']:.2f}% exceeds threshold of {thresholds['cpu_pct']:.2f}%",
                {
                    "cpu_percent": metrics["cpu_percent"],
                    "threshold": thresholds["cpu_pct"]
                },
                "system"
            )
            alerts_sent.append("cpu")
        
        # Check memory usage
        if "memory_percent" in metrics and metrics["memory_percent"] >= thresholds["memory_pct"]:
            self.alert(
                "warning",
                f"Memory usage of {metrics['memory_percent']:.2f}% exceeds threshold of {thresholds['memory_pct']:.2f}%",
                {
                    "memory_percent": metrics["memory_percent"],
                    "memory_used_mb": metrics.get("memory_used_mb", 0),
                    "memory_total_mb": metrics.get("memory_total_mb", 0),
                    "threshold": thresholds["memory_pct"]
                },
                "system"
            )
            alerts_sent.append("memory")
        
        # Check disk usage
        if "disk_percent" in metrics and metrics["disk_percent"] >= thresholds["disk_pct"]:
            self.alert(
                "warning",
                f"Disk usage of {metrics['disk_percent']:.2f}% exceeds threshold of {thresholds['disk_pct']:.2f}%",
                {
                    "disk_percent": metrics["disk_percent"],
                    "disk_used_gb": metrics.get("disk_used_gb", 0),
                    "disk_total_gb": metrics.get("disk_total_gb", 0),
                    "threshold": thresholds["disk_pct"]
                },
                "system"
            )
            alerts_sent.append("disk")
        
        return alerts_sent


# Example usage
if __name__ == "__main__":
    # Initialize logger
    factory = LoggerFactory.get_instance()
    logger = factory.get_logger("example")
    
    # Log some messages
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Initialize trade logger
    trade_logger = TradeLogger("example_strategy")
    
    # Log a trade
    trade_logger.log_trade({
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 150.0,
        "profit_loss": 0.0
    })
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(interval=5)
    monitor.start()
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Send an alert
    alert_manager.alert(
        "warning",
        "Test alert",
        {"test": True},
        "test"
    )
    
    # Sleep for a few seconds to allow performance monitoring
    time.sleep(10)
    
    # Stop performance monitor
    monitor.stop()
