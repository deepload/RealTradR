"""
Base repository for RealTradR

This module provides a base repository class for database operations.
"""

from typing import Generic, TypeVar, Type, Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository for CRUD operations"""
    
    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize the repository
        
        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
    
    def get(self, id: Any) -> Optional[ModelType]:
        """
        Get a record by ID
        
        Args:
            id: Record ID
            
        Returns:
            Optional[ModelType]: Record or None if not found
        """
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_by_attribute(self, attr_name: str, attr_value: Any) -> Optional[ModelType]:
        """
        Get a record by attribute
        
        Args:
            attr_name: Attribute name
            attr_value: Attribute value
            
        Returns:
            Optional[ModelType]: Record or None if not found
        """
        return self.db.query(self.model).filter(getattr(self.model, attr_name) == attr_value).first()
    
    def get_multi(
        self, 
        *,
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        Get multiple records
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters as key-value pairs
            
        Returns:
            List[ModelType]: List of records
        """
        query = self.db.query(self.model)
        
        if filters:
            for attr_name, attr_value in filters.items():
                query = query.filter(getattr(self.model, attr_name) == attr_value)
                
        return query.offset(skip).limit(limit).all()
    
    def create(self, *, obj_in: Union[CreateSchemaType, Dict[str, Any]]) -> ModelType:
        """
        Create a new record
        
        Args:
            obj_in: Create schema or dictionary
            
        Returns:
            ModelType: Created record
        """
        if isinstance(obj_in, dict):
            obj_in_data = obj_in
        else:
            obj_in_data = obj_in.dict(exclude_unset=True)
            
        db_obj = self.model(**obj_in_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def update(
        self, 
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """
        Update a record
        
        Args:
            db_obj: Database object to update
            obj_in: Update schema or dictionary
            
        Returns:
            ModelType: Updated record
        """
        obj_data = db_obj.to_dict()
        
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
            
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
                
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def delete(self, *, id: int) -> ModelType:
        """
        Delete a record
        
        Args:
            id: Record ID
            
        Returns:
            ModelType: Deleted record
        """
        obj = self.db.query(self.model).get(id)
        self.db.delete(obj)
        self.db.commit()
        
        return obj
