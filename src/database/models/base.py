"""
Base database model classes and mixins
"""

from sqlalchemy import Column, Integer, DateTime, func, String
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, Any
import uuid

Base = declarative_base()


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps"""
    
    created_at = Column(
        DateTime(timezone=True), 
        default=func.now(), 
        nullable=False,
        comment="Record creation timestamp"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        default=func.now(), 
        onupdate=func.now(), 
        nullable=False,
        comment="Record last update timestamp"
    )


class BaseModel(Base, TimestampMixin):
    """Base model class with common fields"""
    
    __abstract__ = True
    
    id = Column(
        Integer, 
        primary_key=True, 
        index=True,
        comment="Primary key"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class UUIDMixin:
    """Mixin to add UUID primary key"""
    
    id = Column(
        String(36),  # Use String for SQLite compatibility
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        comment="UUID primary key"
    )