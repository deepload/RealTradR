"""
Authentication API endpoints for RealTradR

This module provides API endpoints for user authentication and management.
"""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.core.config import settings
from app.core.security import create_access_token
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.token import Token
from app.schemas.user import User as UserSchema, UserCreate, UserUpdate, UserWithStats

router = APIRouter()


@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
def create_user(*, db: Session = Depends(get_db), user_in: UserCreate) -> Any:
    """
    Create a new user.
    """
    # Initialize repository
    repository = UserRepository(db)
    
    # Check if user with the given email already exists
    user = repository.get_by_email(email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )
        
    # Check if user with the given username already exists
    user = repository.get_by_username(username=user_in.username)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this username already exists."
        )
        
    # Create new user
    user = repository.create(obj_in=user_in)
    
    return user


@router.post("/login", response_model=Token)
def login_for_access_token(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # Initialize repository
    repository = UserRepository(db)
    
    # Authenticate user
    user = repository.authenticate(email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Check if user is active
    if not repository.is_active(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
        
    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        subject=str(user.id), expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserSchema)
def read_users_me(current_user: User = Depends(get_current_user)) -> Any:
    """
    Get current user.
    """
    return current_user


@router.get("/me/stats", response_model=UserWithStats)
def read_user_stats(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get current user with trading statistics.
    """
    # Initialize repository
    repository = UserRepository(db)
    
    # Get user with statistics
    user_stats = repository.get_user_with_stats(user_id=current_user.id)
    
    if not user_stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        
    return user_stats


@router.put("/me", response_model=UserSchema)
def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Update current user.
    """
    # Initialize repository
    repository = UserRepository(db)
    
    # Update user
    user = repository.update(db_obj=current_user, obj_in=user_in)
    
    return user
