"""
User repository for RealTradR

This module provides a repository for user operations.
"""

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.repositories.base import BaseRepository
from app.core.security import get_password_hash, verify_password


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Repository for user operations"""
    
    def __init__(self, db: Session):
        """
        Initialize the repository
        
        Args:
            db: Database session
        """
        super().__init__(User, db)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email
        
        Args:
            email: User's email
            
        Returns:
            Optional[User]: User or None if not found
        """
        return self.db.query(User).filter(User.email == email).first()
    
    def get_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by username
        
        Args:
            username: User's username
            
        Returns:
            Optional[User]: User or None if not found
        """
        return self.db.query(User).filter(User.username == username).first()
    
    def create(self, *, obj_in: UserCreate) -> User:
        """
        Create a new user
        
        Args:
            obj_in: User create schema
            
        Returns:
            User: Created user
        """
        db_obj = User(
            email=obj_in.email,
            username=obj_in.username,
            full_name=obj_in.full_name,
            hashed_password=get_password_hash(obj_in.password),
            is_active=obj_in.is_active,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def update(
        self, 
        *,
        db_obj: User,
        obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        """
        Update a user
        
        Args:
            db_obj: User to update
            obj_in: User update schema or dictionary
            
        Returns:
            User: Updated user
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
            
        # Handle password separately
        if 'password' in update_data and update_data['password']:
            hashed_password = get_password_hash(update_data['password'])
            del update_data['password']
            update_data['hashed_password'] = hashed_password
            
        # Update timestamp
        update_data['updated_at'] = datetime.utcnow()
        
        return super().update(db_obj=db_obj, obj_in=update_data)
    
    def authenticate(self, *, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Optional[User]: User or None if authentication fails
        """
        user = self.get_by_email(email=email)
        
        if not user:
            return None
            
        if not verify_password(password, user.hashed_password):
            return None
            
        return user
    
    def is_active(self, user: User) -> bool:
        """
        Check if a user is active
        
        Args:
            user: User to check
            
        Returns:
            bool: True if user is active, False otherwise
        """
        return user.is_active
    
    def is_superuser(self, user: User) -> bool:
        """
        Check if a user is a superuser
        
        Args:
            user: User to check
            
        Returns:
            bool: True if user is a superuser, False otherwise
        """
        return user.is_superuser
    
    def get_user_with_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get a user with trading statistics
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, Any]: User with trading statistics
        """
        user = self.get(id=user_id)
        
        if not user:
            return None
            
        # Get trading statistics
        trades_query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as successful_trades
            FROM trades
            WHERE user_id = :user_id
        """
        
        result = self.db.execute(trades_query, {"user_id": user_id}).fetchone()
        
        total_trades = result.total_trades if result and result.total_trades else 0
        successful_trades = result.successful_trades if result and result.successful_trades else 0
        win_rate = (successful_trades / total_trades) if total_trades > 0 else 0
        
        # Get account information - this would normally come from the broker
        # For simplicity, we're using placeholder values
        portfolio_value = 0
        account_balance = 0
        
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "win_rate": win_rate,
            "portfolio_value": portfolio_value,
            "account_balance": account_balance
        }
