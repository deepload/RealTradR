#!/usr/bin/env python
"""
Setup Environment Script

This script creates a .env file with your Alpaca API credentials.
"""

import os
import json
from pathlib import Path
import getpass

def main():
    """Create .env file with user-provided credentials"""
    print("\n=== RealTradR Environment Setup ===\n")
    
    # Default values
    default_api_key = "PKFK3EAPXA9D9CXZ33JD"  # The key you provided
    default_api_secret = "nqGS1mnSKFYfkE6pLvZhYNZizurzAMCGcWEDnLR3"  # The secret you provided
    default_api_base_url = "https://paper-api.alpaca.markets"
    
    # Get user input with defaults
    print("Press Enter to accept the default values or input new ones.")
    api_key = input(f"Alpaca API Key [{default_api_key[:4]}...{default_api_key[-4:]}]: ") or default_api_key
    api_secret = input(f"Alpaca API Secret [{default_api_secret[:4]}...{default_api_secret[-4:]}]: ") or default_api_secret
    
    # Ask about paper trading vs live trading
    while True:
        trading_mode = input("Trading Mode (paper/live) [paper]: ").lower() or "paper"
        if trading_mode in ["paper", "live"]:
            break
        print("Please enter either 'paper' or 'live'")
    
    # Set the API base URL based on trading mode
    if trading_mode == "paper":
        api_base_url = "https://paper-api.alpaca.markets"
        is_paper = "true"
        print("\n✓ Using PAPER trading (no real money at risk)")
    else:
        api_base_url = "https://api.alpaca.markets"
        is_paper = "false"
        print("\n⚠️ Using LIVE trading (REAL MONEY will be used)")
        confirm = input("Are you sure you want to use LIVE trading? Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Defaulting to paper trading for safety.")
            api_base_url = "https://paper-api.alpaca.markets"
            is_paper = "true"
            trading_mode = "paper"
    
    # Create .env file
    env_path = Path(".env")
    
    # Format API credentials
    env_content = f"""# Alpaca API Credentials
ALPACA_API_KEY={api_key}
ALPACA_API_SECRET={api_secret}
ALPACA_API_BASE_URL={api_base_url}
ALPACA_PAPER_TRADING={is_paper}

# Database Configuration
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=realtradR

# Application Settings
SECRET_KEY=realtradrsecretkey
"""
    
    # Write to .env file
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"\n✓ Created .env file at {env_path.absolute()}")
    print(f"✓ Trading mode: {trading_mode.upper()}")
    
    # Create a configuration for the AI trading strategy
    strategy_config = {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "cash_limit": 10000,
        "short_window": 10,
        "long_window": 30,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 5.0,
    }
    
    # Allow customization of the strategy
    print("\n=== AI Trading Strategy Configuration ===\n")
    print("Default symbols: AAPL, MSFT, GOOGL, AMZN, META")
    custom_symbols = input("Enter custom symbols separated by commas, or press Enter for default: ")
    if custom_symbols:
        strategy_config["symbols"] = [s.strip().upper() for s in custom_symbols.split(",")]
    
    cash_input = input(f"Maximum cash to use per trade (${strategy_config['cash_limit']}): ")
    if cash_input:
        strategy_config["cash_limit"] = float(cash_input)
    
    # Write strategy config
    config_path = Path("strategy_config.json")
    with open(config_path, "w") as f:
        json.dump(strategy_config, f, indent=2)
    
    print(f"\n✓ Created strategy config at {config_path.absolute()}")
    print(f"✓ Trading symbols: {', '.join(strategy_config['symbols'])}")
    print(f"✓ Cash limit per trade: ${strategy_config['cash_limit']}")
    
    print("\n=== Setup Complete ===")
    print("\nTo run the trading strategy:")
    print("1. Make sure your virtual environment is activated")
    print("2. Run: python -m backend.app.ai.simple_strategy")
    print("\nHappy trading!")

if __name__ == "__main__":
    main()
