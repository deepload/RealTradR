"""
Database base model for RealTradR

This module provides the base SQLAlchemy model with common methods that other models will inherit from.
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from sqlalchemy.ext.declarative import as_declarative, declared_attr
import sqlalchemy as sa
from sqlalchemy.orm import Session

# Type variable for ID
ModelType = TypeVar("ModelType")

@as_declarative()
class Base:
    """Base class for all database models"""
    id: Any
    __name__: str
    
    # Generate tablename automatically based on class name
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
    
    # Common columns for all models
    created_at = sa.Column(sa.DateTime, default=sa.func.now())
    updated_at = sa.Column(sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    
    def dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    @classmethod
    def get_by_id(cls, db: Session, id: Any) -> Optional[ModelType]:
        """Get a record by ID"""
        return db.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def get_all(cls, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination"""
        return db.query(cls).offset(skip).limit(limit).all()
    
    @classmethod
    def create(cls, db: Session, obj_in: Dict[str, Any]) -> ModelType:
        """Create a new record"""
        db_obj = cls(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    @classmethod
    def update(cls, db: Session, db_obj: ModelType, obj_in: Union[Dict[str, Any], ModelType]) -> ModelType:
        """Update a record"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
            
        for field in update_data:
            if hasattr(db_obj, field):
                setattr(db_obj, field, update_data[field])
                
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    @classmethod
    def delete(cls, db: Session, id: Any) -> ModelType:
        """Delete a record"""
        obj = db.query(cls).get(id)
        db.delete(obj)
        db.commit()
        return obj
