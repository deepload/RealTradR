"""
Configuration Settings for RealTradR

This module loads settings from config.yaml and provides them as a Pydantic settings object.
"""

import os
import yaml
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseSettings, validator
from pydantic_settings import BaseSettings

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load the configuration file
config_path = os.path.join(BASE_DIR, "config.yaml")
try:
    with open(config_path, "r") as file:
        yaml_config = yaml.safe_load(file)
except FileNotFoundError:
    yaml_config = {}

class Settings(BaseSettings):
    # Server settings
    host: str = yaml_config.get("server", {}).get("host", "127.0.0.1")
    port: int = yaml_config.get("server", {}).get("port", 8000)
    debug: bool = yaml_config.get("server", {}).get("debug", False)
    
    # API and security
    api_prefix: str = "/api"
    secret_key: str = yaml_config.get("security", {}).get("jwt_secret", "CHANGE_THIS_TO_A_STRONG_SECRET")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = yaml_config.get("security", {}).get("jwt_expire_minutes", 60)
    cors_origins: List[str] = ["http://localhost:4200", "http://localhost:3000"]
    
    # Database settings
    postgres_host: str = yaml_config.get("database", {}).get("postgres", {}).get("host", "localhost")
    postgres_port: int = yaml_config.get("database", {}).get("postgres", {}).get("port", 5432)
    postgres_user: str = yaml_config.get("database", {}).get("postgres", {}).get("username", "postgres")
    postgres_password: str = yaml_config.get("database", {}).get("postgres", {}).get("password", "postgres")
    postgres_db: str = yaml_config.get("database", {}).get("postgres", {}).get("database", "realtrad_db")
    sqlalchemy_database_uri: Optional[str] = None
    
    redis_host: str = yaml_config.get("database", {}).get("redis", {}).get("host", "localhost")
    redis_port: int = yaml_config.get("database", {}).get("redis", {}).get("port", 6379)
    redis_db: int = yaml_config.get("database", {}).get("redis", {}).get("db", 0)
    redis_password: Optional[str] = yaml_config.get("database", {}).get("redis", {}).get("password")
    
    # Broker settings
    default_broker: str = yaml_config.get("broker", {}).get("default", "alpaca")
    available_brokers: List[str] = ["alpaca", "ibkr"]
    
    # Alpaca settings
    alpaca_api_key: str = yaml_config.get("broker", {}).get("alpaca", {}).get("api_key", "")
    alpaca_api_secret: str = yaml_config.get("broker", {}).get("alpaca", {}).get("api_secret", "")
    alpaca_base_url: str = yaml_config.get("broker", {}).get("alpaca", {}).get("base_url", "https://paper-api.alpaca.markets")
    alpaca_data_url: str = yaml_config.get("broker", {}).get("alpaca", {}).get("data_url", "https://data.alpaca.markets")
    alpaca_paper_trading: bool = yaml_config.get("broker", {}).get("alpaca", {}).get("paper_trading", True)
    
    # IBKR settings
    ibkr_port: int = yaml_config.get("broker", {}).get("ibkr", {}).get("tws_port", 7497)
    ibkr_host: str = yaml_config.get("broker", {}).get("ibkr", {}).get("tws_host", "127.0.0.1")
    ibkr_client_id: int = yaml_config.get("broker", {}).get("ibkr", {}).get("client_id", 1)
    ibkr_paper_trading: bool = yaml_config.get("broker", {}).get("ibkr", {}).get("paper_trading", True)
    
    # AI model settings
    enabled_ai_models: List[str] = yaml_config.get("ai", {}).get("enabled_models", ["lstm", "sentiment", "technical"])
    model_path: str = yaml_config.get("ai", {}).get("model_path", "./models")
    model_update_frequency: str = yaml_config.get("ai", {}).get("model_update_frequency", "daily")
    use_sentiment: bool = yaml_config.get("ai", {}).get("use_sentiment", True)
    use_technical: bool = yaml_config.get("ai", {}).get("use_technical", True)
    use_lstm: bool = yaml_config.get("ai", {}).get("use_lstm", True)
    use_reinforcement: bool = yaml_config.get("ai", {}).get("use_reinforcement", True)
    
    # Trading settings
    default_symbols: List[str] = yaml_config.get("trading", {}).get("default_symbols", ["AAPL", "MSFT", "AMZN"])
    risk_per_trade: float = yaml_config.get("trading", {}).get("risk_per_trade", 0.02)
    stop_loss_pct: float = yaml_config.get("trading", {}).get("stop_loss_pct", 0.05)
    take_profit_pct: float = yaml_config.get("trading", {}).get("take_profit_pct", 0.10)
    max_positions: int = yaml_config.get("trading", {}).get("max_positions", 10)
    trading_hours: Dict[str, str] = yaml_config.get("trading", {}).get("trading_hours", {"start": "09:30", "end": "16:00"})
    
    # Logging settings
    log_level: str = yaml_config.get("logging", {}).get("level", "INFO")
    log_file: str = os.path.join(BASE_DIR, yaml_config.get("logging", {}).get("file", "logs/realtrad.log"))
    
    @validator("sqlalchemy_database_uri", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        
        user = values.get("postgres_user")
        password = values.get("postgres_password")
        host = values.get("postgres_host")
        port = values.get("postgres_port")
        db = values.get("postgres_db")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
