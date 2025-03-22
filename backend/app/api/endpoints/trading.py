"""
Trading API endpoints for RealTradR

This module provides API endpoints for trading operations.
"""

from typing import Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.api_key import APIKey
from app.repositories.trading import (
    TradingSymbolRepository,
    OrderRepository,
    TradeRepository,
    MarketDataRepository
)
from app.schemas.trading import (
    TradingSymbol,
    Order,
    OrderCreate,
    Trade,
    PortfolioPosition,
    AccountInfo,
    MarketData,
    MarketSnapshot
)
from app.services.broker_factory import BrokerFactory

router = APIRouter()


@router.get("/symbols", response_model=List[TradingSymbol])
def get_trading_symbols(
    db: Session = Depends(get_db),
    q: Optional[str] = None,
    active_only: bool = True,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get trading symbols.
    """
    # Initialize repository
    repository = TradingSymbolRepository(db)
    
    # Search symbols if query provided
    if q:
        symbols = repository.search_symbols(query=q)
        return symbols
    
    # Get active symbols only
    if active_only:
        symbols = repository.get_active_symbols()
        # Apply skip and limit manually
        return symbols[skip:skip+limit]
    
    # Get all symbols with pagination
    filters = None
    if active_only:
        filters = {"enabled_for_trading": True}
        
    symbols = repository.get_multi(skip=skip, limit=limit, filters=filters)
    return symbols


@router.post("/orders", response_model=Order, status_code=status.HTTP_201_CREATED)
def create_order(
    *,
    db: Session = Depends(get_db),
    order_in: OrderCreate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Create a new order.
    """
    # Ensure user_id is set
    if not order_in.user_id:
        order_in.user_id = current_user.id
    
    # Initialize repositories
    symbol_repo = TradingSymbolRepository(db)
    order_repo = OrderRepository(db)
    
    # Check if symbol exists
    symbol = symbol_repo.get(id=order_in.symbol_id)
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Symbol not found"
        )
    
    # Initialize broker
    try:
        broker = BrokerFactory.create_broker(
            broker_name=order_in.broker.value,
            user_id=current_user.id,
            paper_trading=order_in.is_paper
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize broker: {str(e)}"
        )
    
    # Connect to broker
    try:
        broker.connect()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to broker: {str(e)}"
        )
    
    # Create order in database with pending status
    order_data = order_in.dict()
    order_data["status"] = "pending"
    order = order_repo.create_from_dict(order_data)
    
    # Place order with broker
    try:
        if order_in.order_type == "market":
            broker_order = broker.place_market_order(
                symbol=symbol.symbol,
                qty=order_in.quantity,
                side=order_in.side.value
            )
        elif order_in.order_type == "limit":
            broker_order = broker.place_limit_order(
                symbol=symbol.symbol,
                qty=order_in.quantity,
                side=order_in.side.value,
                limit_price=order_in.limit_price,
                time_in_force=order_in.time_in_force.value
            )
        elif order_in.order_type == "stop":
            broker_order = broker.place_stop_order(
                symbol=symbol.symbol,
                qty=order_in.quantity,
                side=order_in.side.value,
                stop_price=order_in.stop_price,
                time_in_force=order_in.time_in_force.value
            )
        elif order_in.order_type == "stop_limit":
            broker_order = broker.place_stop_limit_order(
                symbol=symbol.symbol,
                qty=order_in.quantity,
                side=order_in.side.value,
                limit_price=order_in.limit_price,
                stop_price=order_in.stop_price,
                time_in_force=order_in.time_in_force.value
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported order type: {order_in.order_type}"
            )
        
        # Update order with broker information
        order = order_repo.update_order_status(
            order_id=order.id,
            status="submitted",
            broker_order_id=broker_order.get("id")
        )
    except Exception as e:
        # Update order status to rejected
        order = order_repo.update_order_status(
            order_id=order.id,
            status="rejected"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to place order: {str(e)}"
        )
    finally:
        # Disconnect from broker
        broker.disconnect()
    
    return order


@router.get("/orders", response_model=List[Order])
def get_orders(
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    symbol_id: Optional[int] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get user orders.
    """
    # Initialize repository
    repository = OrderRepository(db)
    
    # Get orders
    orders = repository.get_user_orders(
        user_id=current_user.id,
        status=status,
        symbol_id=symbol_id,
        limit=limit
    )
    
    return orders


@router.get("/orders/active", response_model=List[Order])
def get_active_orders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get active orders.
    """
    # Initialize repository
    repository = OrderRepository(db)
    
    # Get active orders
    orders = repository.get_active_orders(user_id=current_user.id)
    
    return orders


@router.get("/orders/{order_id}", response_model=Order)
def get_order(
    *,
    db: Session = Depends(get_db),
    order_id: int,
    update_status: bool = False,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get order by ID.
    """
    # Initialize repository
    repository = OrderRepository(db)
    
    # Get order
    order = repository.get(id=order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    # Check if order belongs to user
    if order.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Update order status if requested
    if update_status and order.status not in ["filled", "cancelled", "rejected", "expired"]:
        try:
            # Initialize broker
            broker = BrokerFactory.create_broker(
                broker_name=order.broker,
                user_id=current_user.id,
                paper_trading=order.is_paper
            )
            
            # Connect to broker
            broker.connect()
            
            # Get order status from broker
            broker_order = broker.get_order(order_id=order.broker_order_id)
            
            # Update order status
            if broker_order:
                order = repository.update_order_status(
                    order_id=order.id,
                    status=broker_order.get("status"),
                    filled_quantity=broker_order.get("filled_qty"),
                    filled_price=broker_order.get("filled_avg_price")
                )
        except Exception as e:
            pass
        finally:
            # Disconnect from broker
            broker.disconnect()
    
    return order


@router.delete("/orders/{order_id}", response_model=Order)
def cancel_order(
    *,
    db: Session = Depends(get_db),
    order_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Cancel order.
    """
    # Initialize repository
    repository = OrderRepository(db)
    
    # Get order
    order = repository.get(id=order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    # Check if order belongs to user
    if order.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check if order can be cancelled
    if order.status not in ["pending", "submitted", "accepted", "partial"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order with status: {order.status}"
        )
    
    # Initialize broker
    try:
        broker = BrokerFactory.create_broker(
            broker_name=order.broker,
            user_id=current_user.id,
            paper_trading=order.is_paper
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize broker: {str(e)}"
        )
    
    # Connect to broker
    try:
        broker.connect()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to broker: {str(e)}"
        )
    
    # Cancel order
    try:
        broker.cancel_order(order_id=order.broker_order_id)
        
        # Update order status
        order = repository.update_order_status(
            order_id=order.id,
            status="cancelled"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )
    finally:
        # Disconnect from broker
        broker.disconnect()
    
    return order


@router.get("/trades", response_model=List[Trade])
def get_trades(
    db: Session = Depends(get_db),
    symbol_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get user trades.
    """
    # Initialize repository
    repository = TradeRepository(db)
    
    # Get trades
    trades = repository.get_user_trades(
        user_id=current_user.id,
        symbol_id=symbol_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    return trades


@router.get("/positions", response_model=List[PortfolioPosition])
def get_positions(
    db: Session = Depends(get_db),
    broker_name: str = "ibkr",
    paper_trading: bool = True,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get user positions.
    """
    # Initialize broker
    try:
        broker = BrokerFactory.create_broker(
            broker_name=broker_name,
            user_id=current_user.id,
            paper_trading=paper_trading
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize broker: {str(e)}"
        )
    
    # Connect to broker
    try:
        broker.connect()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to broker: {str(e)}"
        )
    
    # Get positions
    try:
        positions = broker.get_positions()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}"
        )
    finally:
        # Disconnect from broker
        broker.disconnect()
    
    # Get symbol IDs for positions
    symbol_repo = TradingSymbolRepository(db)
    for position in positions:
        symbol = symbol_repo.get_by_symbol(symbol=position.get("symbol"))
        if symbol:
            position["symbol_id"] = symbol.id
        else:
            # Create symbol if it doesn't exist
            new_symbol = symbol_repo.create(
                obj_in={
                    "symbol": position.get("symbol"),
                    "name": position.get("symbol"),
                    "enabled_for_trading": True
                }
            )
            position["symbol_id"] = new_symbol.id
        
        # Add broker name and paper trading flag
        position["broker"] = broker_name
        position["is_paper"] = paper_trading
    
    return positions


@router.get("/account", response_model=AccountInfo)
def get_account_info(
    broker_name: str = "ibkr",
    paper_trading: bool = True,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get account information.
    """
    # Initialize broker
    try:
        broker = BrokerFactory.create_broker(
            broker_name=broker_name,
            user_id=current_user.id,
            paper_trading=paper_trading
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize broker: {str(e)}"
        )
    
    # Connect to broker
    try:
        broker.connect()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to broker: {str(e)}"
        )
    
    # Get account information
    try:
        account_info = broker.get_account()
        
        # Add broker name and paper trading flag
        account_info["broker"] = broker_name
        account_info["is_paper"] = paper_trading
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get account information: {str(e)}"
        )
    finally:
        # Disconnect from broker
        broker.disconnect()
    
    return account_info


@router.get("/market-data/{symbol}", response_model=List[MarketData])
def get_market_data(
    *,
    db: Session = Depends(get_db),
    symbol: str,
    timeframe: str = "1d",
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get market data for a symbol.
    """
    # Initialize repositories
    symbol_repo = TradingSymbolRepository(db)
    market_data_repo = MarketDataRepository(db)
    
    # Get symbol
    symbol_obj = symbol_repo.get_by_symbol(symbol=symbol)
    if not symbol_obj:
        # Try to create symbol
        try:
            symbol_obj = symbol_repo.create(
                obj_in={
                    "symbol": symbol,
                    "name": symbol,
                    "enabled_for_trading": True
                }
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Symbol not found"
            )
    
    # Calculate end date (now) and start date (based on limit and timeframe)
    end_date = datetime.utcnow()
    
    # Map timeframe to timedelta
    timeframe_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1)
    }
    
    # Calculate start date
    start_date = end_date - (timeframe_map.get(timeframe, timedelta(days=1)) * limit)
    
    # Get market data from repository
    market_data = market_data_repo.get_history(
        symbol_id=symbol_obj.id,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    # If no data or data is stale, fetch from broker
    latest_data = market_data_repo.get_latest(
        symbol_id=symbol_obj.id,
        timeframe=timeframe
    )
    
    if not market_data or (latest_data and latest_data.timestamp < end_date - timeframe_map.get(timeframe, timedelta(days=1))):
        # Initialize broker (use IBKR as default)
        try:
            broker = BrokerFactory.create_broker(
                broker_name="ibkr",
                user_id=current_user.id
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to initialize broker: {str(e)}"
            )
        
        # Connect to broker
        try:
            broker.connect()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to broker: {str(e)}"
            )
        
        # Get market data from broker
        try:
            broker_data = broker.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            # Format data for batch insert
            data_list = []
            for bar in broker_data:
                data_list.append({
                    "symbol_id": symbol_obj.id,
                    "timestamp": bar.get("timestamp"),
                    "open": bar.get("open"),
                    "high": bar.get("high"),
                    "low": bar.get("low"),
                    "close": bar.get("close"),
                    "volume": bar.get("volume"),
                    "timeframe": timeframe
                })
            
            # Batch insert data
            if data_list:
                market_data_repo.batch_insert(data_list)
            
            # Get updated market data
            market_data = market_data_repo.get_history(
                symbol_id=symbol_obj.id,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
        except Exception as e:
            # If we already have some data, return it
            if market_data:
                return market_data
                
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get market data: {str(e)}"
            )
        finally:
            # Disconnect from broker
            broker.disconnect()
    
    return market_data


@router.get("/market-data/{symbol}/quote", response_model=MarketSnapshot)
def get_market_quote(
    *,
    db: Session = Depends(get_db),
    symbol: str,
    broker_name: str = "ibkr",
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get market quote for a symbol.
    """
    # Initialize repositories
    symbol_repo = TradingSymbolRepository(db)
    
    # Get symbol
    symbol_obj = symbol_repo.get_by_symbol(symbol=symbol)
    if not symbol_obj:
        # Try to create symbol
        try:
            symbol_obj = symbol_repo.create(
                obj_in={
                    "symbol": symbol,
                    "name": symbol,
                    "enabled_for_trading": True
                }
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Symbol not found"
            )
    
    # Initialize broker
    try:
        broker = BrokerFactory.create_broker(
            broker_name=broker_name,
            user_id=current_user.id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to initialize broker: {str(e)}"
        )
    
    # Connect to broker
    try:
        broker.connect()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to broker: {str(e)}"
        )
    
    # Get market quote
    try:
        quote = broker.get_quote(symbol=symbol)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market quote: {str(e)}"
        )
    finally:
        # Disconnect from broker
        broker.disconnect()
    
    return quote
