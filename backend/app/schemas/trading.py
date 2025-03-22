"""
Trading schemas for RealTradR API

This module defines Pydantic models for trading-related API interactions.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class BrokerEnum(str, Enum):
    """Supported brokers"""
    ALPACA = "alpaca"
    IBKR = "ibkr"


class OrderTypeEnum(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSideEnum(str, Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatusEnum(str, Enum):
    """Order statuses"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForceEnum(str, Enum):
    """Time in force options"""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class TradingSymbolBase(BaseModel):
    """Base schema for trading symbol"""
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    asset_class: Optional[str] = None
    is_tradable: bool = True
    is_shortable: bool = False
    is_marginable: bool = False
    enabled_for_trading: bool = True
    enabled_for_fractional: bool = False


class TradingSymbolCreate(TradingSymbolBase):
    """Schema for creating a trading symbol"""
    pass


class TradingSymbolUpdate(BaseModel):
    """Schema for updating a trading symbol"""
    name: Optional[str] = None
    is_tradable: Optional[bool] = None
    is_shortable: Optional[bool] = None
    is_marginable: Optional[bool] = None
    enabled_for_trading: Optional[bool] = None
    enabled_for_fractional: Optional[bool] = None


class TradingSymbol(TradingSymbolBase):
    """Schema for trading symbol from database"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class OrderBase(BaseModel):
    """Base schema for order"""
    symbol_id: int
    broker: BrokerEnum = BrokerEnum.ALPACA
    order_type: OrderTypeEnum = OrderTypeEnum.MARKET
    side: OrderSideEnum
    quantity: float
    time_in_force: TimeInForceEnum = TimeInForceEnum.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    is_paper: bool = True
    client_order_id: Optional[str] = None
    notes: Optional[str] = None
    
    @validator('limit_price')
    def limit_price_required_for_limit_orders(cls, v, values):
        if values.get('order_type') in [OrderTypeEnum.LIMIT, OrderTypeEnum.STOP_LIMIT] and v is None:
            raise ValueError('Limit price is required for limit orders')
        return v
        
    @validator('stop_price')
    def stop_price_required_for_stop_orders(cls, v, values):
        if values.get('order_type') in [OrderTypeEnum.STOP, OrderTypeEnum.STOP_LIMIT] and v is None:
            raise ValueError('Stop price is required for stop orders')
        return v


class OrderCreate(OrderBase):
    """Schema for creating an order"""
    user_id: Optional[int] = None
    strategy_id: Optional[int] = None
    

class OrderUpdate(BaseModel):
    """Schema for updating an order"""
    quantity: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[TimeInForceEnum] = None
    notes: Optional[str] = None


class Order(OrderBase):
    """Schema for order from database"""
    id: int
    user_id: int
    status: OrderStatusEnum
    broker_order_id: Optional[str] = None
    filled_quantity: float = 0
    filled_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    strategy_id: Optional[int] = None
    symbol: TradingSymbol

    class Config:
        orm_mode = True


class TradeBase(BaseModel):
    """Base schema for trade"""
    order_id: int
    symbol_id: int
    execution_price: float
    quantity: float
    side: OrderSideEnum
    commission: Optional[float] = 0
    is_paper: bool = True
    broker_trade_id: Optional[str] = None


class TradeCreate(TradeBase):
    """Schema for creating a trade"""
    user_id: Optional[int] = None
    profit_loss: Optional[float] = None


class Trade(TradeBase):
    """Schema for trade from database"""
    id: int
    user_id: int
    executed_at: datetime
    profit_loss: Optional[float] = None
    symbol: TradingSymbol
    
    class Config:
        orm_mode = True


class PortfolioPosition(BaseModel):
    """Schema for portfolio position"""
    symbol: str
    symbol_id: int
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    side: str
    broker: str
    is_paper: bool

    class Config:
        orm_mode = True


class AccountInfo(BaseModel):
    """Schema for account information"""
    broker: str
    account_id: str
    status: str
    cash: float
    portfolio_value: float
    equity: float
    buying_power: float
    initial_margin: Optional[float] = None
    maintenance_margin: Optional[float] = None
    is_paper: bool
    currency: str = "USD"
    
    class Config:
        orm_mode = True


class MarketDataBase(BaseModel):
    """Base schema for market data"""
    symbol_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


class MarketDataCreate(MarketDataBase):
    """Schema for creating market data"""
    pass


class MarketData(MarketDataBase):
    """Schema for market data from database"""
    id: int
    symbol: Optional[TradingSymbol] = None
    
    class Config:
        orm_mode = True


class MarketSnapshot(BaseModel):
    """Schema for market snapshot"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: int
    timestamp: datetime
    
    class Config:
        orm_mode = True
