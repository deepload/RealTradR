"""
Alert Manager Module for RealTradR

This module provides functionality to generate and manage alerts for
significant trading events, market conditions, and system status.
"""

import os
import json
import logging
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manager for trading alerts and notifications
    
    This class provides methods to generate alerts for significant events
    and send notifications via email, SMS, or other channels.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the alert manager
        
        Args:
            config_file: Path to alert configuration file
        """
        self.config = self._load_config(config_file)
        self.alerts = []
        
        # Email settings
        self.email_enabled = self.config.get("email", {}).get("enabled", False)
        self.email_from = self.config.get("email", {}).get("from", os.getenv("EMAIL_FROM", ""))
        self.email_to = self.config.get("email", {}).get("to", os.getenv("EMAIL_TO", ""))
        self.email_server = self.config.get("email", {}).get("server", os.getenv("EMAIL_SERVER", ""))
        self.email_port = self.config.get("email", {}).get("port", int(os.getenv("EMAIL_PORT", "587")))
        self.email_username = self.config.get("email", {}).get("username", os.getenv("EMAIL_USERNAME", ""))
        self.email_password = self.config.get("email", {}).get("password", os.getenv("EMAIL_PASSWORD", ""))
        
        # Webhook settings
        self.webhook_enabled = self.config.get("webhook", {}).get("enabled", False)
        self.webhook_url = self.config.get("webhook", {}).get("url", os.getenv("WEBHOOK_URL", ""))
        
        logger.info("Initialized AlertManager")
    
    def _load_config(self, config_file):
        """Load alert configuration from file"""
        if not config_file or not os.path.exists(config_file):
            logger.warning(f"Alert config file not found: {config_file}")
            return {
                "email": {"enabled": False},
                "webhook": {"enabled": False},
                "alert_levels": {
                    "critical": {"notify": True, "channels": ["email", "webhook"]},
                    "warning": {"notify": True, "channels": ["webhook"]},
                    "info": {"notify": False}
                }
            }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded alert configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading alert config from {config_file}: {e}")
            return {"email": {"enabled": False}, "webhook": {"enabled": False}}
    
    def add_alert(self, message, level="info", symbol=None, data=None, notify=None):
        """
        Add an alert
        
        Args:
            message: Alert message
            level: Alert level (critical, warning, info)
            symbol: Stock symbol (if applicable)
            data: Additional data for the alert
            notify: Override notification setting for this alert
            
        Returns:
            Alert ID
        """
        alert_id = len(self.alerts) + 1
        
        # Create alert
        alert = {
            "id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
            "symbol": symbol,
            "data": data or {}
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Log alert
        log_method = getattr(logger, level, logger.info)
        log_method(f"Alert {alert_id}: {message}")
        
        # Determine if notification should be sent
        should_notify = notify
        if should_notify is None:
            level_config = self.config.get("alert_levels", {}).get(level, {})
            should_notify = level_config.get("notify", False)
        
        # Send notification if needed
        if should_notify:
            channels = self.config.get("alert_levels", {}).get(level, {}).get("channels", [])
            self._send_notification(alert, channels)
        
        return alert_id
    
    def _send_notification(self, alert, channels):
        """
        Send notification for an alert
        
        Args:
            alert: Alert dictionary
            channels: List of notification channels
        """
        for channel in channels:
            if channel == "email" and self.email_enabled:
                self._send_email_notification(alert)
            elif channel == "webhook" and self.webhook_enabled:
                self._send_webhook_notification(alert)
    
    def _send_email_notification(self, alert):
        """Send email notification"""
        if not self.email_enabled or not self.email_from or not self.email_to:
            logger.warning("Email notifications not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"RealTradR Alert: {alert['level'].upper()} - {alert['message']}"
            
            # Create message body
            body = f"""
            <html>
            <body>
                <h2>RealTradR Alert</h2>
                <p><strong>Level:</strong> {alert['level'].upper()}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <p><strong>Message:</strong> {alert['message']}</p>
            """
            
            if alert['symbol']:
                body += f"<p><strong>Symbol:</strong> {alert['symbol']}</p>"
            
            if alert['data']:
                body += "<h3>Additional Data:</h3><ul>"
                for key, value in alert['data'].items():
                    body += f"<li><strong>{key}:</strong> {value}</li>"
                body += "</ul>"
            
            body += """
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to server and send
            if self.email_server and self.email_port:
                server = smtplib.SMTP(self.email_server, self.email_port)
                server.starttls()
                
                if self.email_username and self.email_password:
                    server.login(self.email_username, self.email_password)
                
                server.send_message(msg)
                server.quit()
                
                logger.info(f"Sent email notification for alert {alert['id']}")
                return True
            else:
                logger.warning("Email server not configured")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _send_webhook_notification(self, alert):
        """Send webhook notification"""
        if not self.webhook_enabled or not self.webhook_url:
            logger.warning("Webhook notifications not configured")
            return False
        
        try:
            # Prepare payload
            payload = {
                "alert_id": alert['id'],
                "timestamp": alert['timestamp'],
                "level": alert['level'],
                "message": alert['message'],
                "symbol": alert['symbol'],
                "data": alert['data']
            }
            
            # Send POST request
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Sent webhook notification for alert {alert['id']}")
                return True
            else:
                logger.warning(f"Webhook notification failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    def get_alerts(self, level=None, symbol=None, start_time=None, end_time=None):
        """
        Get alerts with optional filtering
        
        Args:
            level: Filter by alert level
            symbol: Filter by symbol
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            
        Returns:
            List of filtered alerts
        """
        filtered = self.alerts
        
        if level:
            filtered = [a for a in filtered if a['level'] == level]
        
        if symbol:
            filtered = [a for a in filtered if a['symbol'] == symbol]
        
        if start_time:
            start = datetime.fromisoformat(start_time)
            filtered = [a for a in filtered if datetime.fromisoformat(a['timestamp']) >= start]
        
        if end_time:
            end = datetime.fromisoformat(end_time)
            filtered = [a for a in filtered if datetime.fromisoformat(a['timestamp']) <= end]
        
        return filtered
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        logger.info("Cleared all alerts")
        
    def get_alert_summary(self):
        """
        Get summary of alerts
        
        Returns:
            Dictionary with alert summary
        """
        if not self.alerts:
            return {"total": 0}
        
        levels = {}
        symbols = {}
        
        for alert in self.alerts:
            level = alert['level']
            symbol = alert['symbol'] or "none"
            
            levels[level] = levels.get(level, 0) + 1
            symbols[symbol] = symbols.get(symbol, 0) + 1
        
        return {
            "total": len(self.alerts),
            "by_level": levels,
            "by_symbol": symbols,
            "first_alert": self.alerts[0]['timestamp'],
            "last_alert": self.alerts[-1]['timestamp']
        }
