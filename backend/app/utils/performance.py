"""
Performance Monitoring Module for RealTradR

This module provides functionality to monitor system performance metrics
such as CPU usage, memory usage, and disk I/O during strategy execution.
"""

import os
import sys
import time
import logging
import threading
import json
from datetime import datetime
import platform

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor system performance metrics during strategy execution
    
    This class provides methods to track CPU usage, memory usage, and other
    system metrics to ensure the trading strategy runs efficiently.
    """
    
    def __init__(self, interval=60):
        """
        Initialize the performance monitor
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics = []
        
        # Try to import psutil for detailed metrics
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            logger.warning("psutil not available, performance monitoring will be limited")
    
    def start(self):
        """Start performance monitoring"""
        if self.running:
            logger.warning("Performance monitoring already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started performance monitoring")
    
    def stop(self):
        """Stop performance monitoring"""
        if not self.running:
            logger.warning("Performance monitoring not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Stopped performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Log metrics
                logger.info(f"Performance metrics: {json.dumps(metrics, default=str)}")
                
                # Sleep until next collection
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
                time.sleep(self.interval)
    
    def _collect_metrics(self):
        """
        Collect system performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform()
        }
        
        # Collect detailed metrics if psutil is available
        if self.psutil_available:
            try:
                import psutil
                
                # CPU metrics
                metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
                metrics["cpu_count"] = psutil.cpu_count()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                metrics["memory_total"] = memory.total
                metrics["memory_available"] = memory.available
                metrics["memory_used"] = memory.used
                metrics["memory_percent"] = memory.percent
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                metrics["disk_total"] = disk.total
                metrics["disk_used"] = disk.used
                metrics["disk_free"] = disk.free
                metrics["disk_percent"] = disk.percent
                
                # Process metrics
                process = psutil.Process(os.getpid())
                metrics["process_cpu_percent"] = process.cpu_percent(interval=1)
                metrics["process_memory_percent"] = process.memory_percent()
                metrics["process_memory_rss"] = process.memory_info().rss
                metrics["process_threads"] = process.num_threads()
                
            except Exception as e:
                metrics["error"] = str(e)
        else:
            metrics["error"] = "psutil not available"
        
        return metrics
    
    def get_metrics(self):
        """
        Get collected performance metrics
        
        Returns:
            List of metrics dictionaries
        """
        return self.metrics
    
    def get_summary(self):
        """
        Get summary of performance metrics
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Calculate summary statistics
        summary = {
            "start_time": self.metrics[0]["timestamp"],
            "end_time": self.metrics[-1]["timestamp"],
            "samples": len(self.metrics)
        }
        
        # Calculate averages if psutil is available
        if self.psutil_available and "cpu_percent" in self.metrics[0]:
            cpu_values = [m.get("cpu_percent", 0) for m in self.metrics]
            memory_values = [m.get("memory_percent", 0) for m in self.metrics]
            process_cpu_values = [m.get("process_cpu_percent", 0) for m in self.metrics]
            
            summary["avg_cpu_percent"] = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            summary["max_cpu_percent"] = max(cpu_values) if cpu_values else 0
            summary["avg_memory_percent"] = sum(memory_values) / len(memory_values) if memory_values else 0
            summary["max_memory_percent"] = max(memory_values) if memory_values else 0
            summary["avg_process_cpu_percent"] = sum(process_cpu_values) / len(process_cpu_values) if process_cpu_values else 0
            summary["max_process_cpu_percent"] = max(process_cpu_values) if process_cpu_values else 0
        
        return summary

class PerformanceTracker:
    """
    Track trading performance metrics
    
    This class provides methods to track trading performance metrics
    such as returns, Sharpe ratio, drawdown, and win rate.
    """
    
    def __init__(self):
        """Initialize the performance tracker"""
        self.equity_history = []
        self.trade_history = []
        self.daily_returns = []
        self.metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    def update(self, result):
        """
        Update performance metrics with new results
        
        Args:
            result: Dictionary with strategy execution results
        """
        # Update equity history
        if 'account' in result and 'equity' in result['account']:
            equity = result['account']['equity']
            timestamp = datetime.now()
            self.equity_history.append({
                'timestamp': timestamp,
                'equity': equity
            })
            
            # Calculate daily return if we have at least two data points
            if len(self.equity_history) >= 2:
                prev_equity = self.equity_history[-2]['equity']
                daily_return = (equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)
        
        # Update trade history
        if 'trades' in result:
            for trade in result['trades']:
                self.trade_history.append(trade)
                
                # Update win/loss counts
                if trade.get('pnl', 0) > 0:
                    self.metrics['winning_trades'] += 1
                elif trade.get('pnl', 0) < 0:
                    self.metrics['losing_trades'] += 1
                    
                self.metrics['total_trades'] += 1
        
        # Recalculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 0:
            import numpy as np
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            if std_return > 0:
                self.metrics['sharpe_ratio'] = mean_return / std_return * (252 ** 0.5)  # Annualized
        
        # Calculate max drawdown
        if len(self.equity_history) > 0:
            equity_values = [entry['equity'] for entry in self.equity_history]
            max_drawdown = 0
            peak = equity_values[0]
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            self.metrics['max_drawdown'] = max_drawdown
        
        # Calculate win rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Calculate profit factor
        total_profit = sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) > 0)
        total_loss = sum(abs(trade.get('pnl', 0)) for trade in self.trade_history if trade.get('pnl', 0) < 0)
        
        if total_loss > 0:
            self.metrics['profit_factor'] = total_profit / total_loss
    
    def get_metrics(self):
        """Get current performance metrics"""
        return self.metrics
    
    def get_equity_history(self):
        """Get equity history"""
        return self.equity_history
    
    def get_trade_history(self):
        """Get trade history"""
        return self.trade_history
    
    def reset(self):
        """Reset performance tracker"""
        self.equity_history = []
        self.trade_history = []
        self.daily_returns = []
        self.metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
