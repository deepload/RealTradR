"""
Broker Factory Service for RealTradR

This module provides a factory for creating broker instances.
"""

import logging
from typing import Optional, Dict, Any

from app.services.broker_base import BrokerBase
from app.services.alpaca_broker import AlpacaBroker
from app.services.ibkr_broker import IBKRBroker
from app.core.config import settings
from app.models.api_key import APIKey
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

class BrokerFactory:
    """Factory for creating broker instances"""
    
    @staticmethod
    def create_broker(
        broker_name: str, 
        user_id: Optional[int] = None,
        api_key_id: Optional[int] = None,
        paper_trading: Optional[bool] = None,
        **kwargs
    ) -> BrokerBase:
        """
        Create a broker instance
        
        Args:
            broker_name: Name of the broker ('alpaca' or 'ibkr')
            user_id: Optional user ID to load API keys from database
            api_key_id: Optional specific API key ID to use
            paper_trading: Whether to use paper trading (defaults to config)
            **kwargs: Additional broker-specific arguments
            
        Returns:
            BrokerBase: Broker instance
            
        Raises:
            ValueError: If broker is invalid or configuration is missing
        """
        broker_name = broker_name.lower()
        
        # If user_id is provided, load API keys from database
        api_credentials = {}
        if user_id is not None:
            api_credentials = BrokerFactory._get_api_credentials(
                user_id, broker_name, api_key_id
            )
            
        # Override with kwargs if provided
        api_credentials.update(kwargs)
        
        # Set paper trading to default if not specified
        if paper_trading is None:
            if broker_name == 'alpaca':
                paper_trading = settings.alpaca_paper_trading
            elif broker_name == 'ibkr':
                paper_trading = settings.ibkr_paper_trading
            else:
                paper_trading = True
        
        # Create the broker instance
        if broker_name == 'alpaca':
            return AlpacaBroker(
                api_key=api_credentials.get('api_key'),
                api_secret=api_credentials.get('api_secret'),
                base_url=api_credentials.get('base_url'),
                data_url=api_credentials.get('data_url'),
                paper_trading=paper_trading
            )
        elif broker_name == 'ibkr':
            return IBKRBroker(
                host=api_credentials.get('host'),
                port=api_credentials.get('port'),
                client_id=api_credentials.get('client_id'),
                paper_trading=paper_trading
            )
        else:
            raise ValueError(f"Invalid broker name: {broker_name}")
    
    @staticmethod
    def _get_api_credentials(
        user_id: int, 
        broker_name: str, 
        api_key_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get API credentials from database
        
        Args:
            user_id: User ID
            broker_name: Broker name
            api_key_id: Optional specific API key ID
            
        Returns:
            Dict[str, Any]: API credentials
        """
        try:
            db = SessionLocal()
            
            # Build query
            query = db.query(APIKey).filter(
                APIKey.user_id == user_id,
                APIKey.broker_name == broker_name
            )
            
            # Add API key ID filter if provided
            if api_key_id is not None:
                query = query.filter(APIKey.id == api_key_id)
                
            # Get the most recently created API key if multiple exist
            api_key = query.order_by(APIKey.created_at.desc()).first()
            
            if api_key is None:
                logger.warning(
                    f"No API key found for user {user_id} and broker {broker_name}"
                )
                return {}
                
            # Return credentials based on broker type
            if broker_name == 'alpaca':
                return {
                    'api_key': api_key.api_key,
                    'api_secret': api_key.api_secret,
                    'base_url': api_key.extra_data.get('base_url'),
                    'data_url': api_key.extra_data.get('data_url')
                }
            elif broker_name == 'ibkr':
                return {
                    'host': api_key.extra_data.get('host'),
                    'port': api_key.extra_data.get('port'),
                    'client_id': api_key.extra_data.get('client_id')
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting API credentials: {str(e)}")
            return {}
        finally:
            db.close()
