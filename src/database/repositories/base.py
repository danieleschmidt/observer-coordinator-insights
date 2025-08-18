"""Base repository class with common CRUD operations
"""

from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy import and_, asc, desc, or_
from sqlalchemy.orm import Session

from ..models.base import Base


T = TypeVar('T', bound=Base)


class BaseRepository:
    """Base repository class with common database operations"""

    def __init__(self, db: Session, model: Type[T]):
        self.db = db
        self.model = model

    def create(self, obj_data: Dict[str, Any]) -> T:
        """Create a new record"""
        db_obj = self.model(**obj_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def get(self, id: Any) -> Optional[T]:
        """Get record by ID"""
        return self.db.query(self.model).filter(self.model.id == id).first()

    def get_by_field(self, field_name: str, value: Any) -> Optional[T]:
        """Get record by field value"""
        field = getattr(self.model, field_name)
        return self.db.query(self.model).filter(field == value).first()

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = None,
        desc_order: bool = False
    ) -> List[T]:
        """Get all records with pagination"""
        query = self.db.query(self.model)

        if order_by:
            field = getattr(self.model, order_by, None)
            if field:
                if desc_order:
                    query = query.order_by(desc(field))
                else:
                    query = query.order_by(asc(field))

        return query.offset(skip).limit(limit).all()

    def get_multi_by_ids(self, ids: List[Any]) -> List[T]:
        """Get multiple records by IDs"""
        return self.db.query(self.model).filter(self.model.id.in_(ids)).all()

    def update(self, id: Any, obj_data: Dict[str, Any]) -> Optional[T]:
        """Update record by ID"""
        db_obj = self.get(id)
        if db_obj:
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            self.db.commit()
            self.db.refresh(db_obj)
        return db_obj

    def delete(self, id: Any) -> bool:
        """Delete record by ID"""
        db_obj = self.get(id)
        if db_obj:
            self.db.delete(db_obj)
            self.db.commit()
            return True
        return False

    def count(self, filters: Dict[str, Any] = None) -> int:
        """Count records with optional filters"""
        query = self.db.query(self.model)
        if filters:
            query = self._apply_filters(query, filters)
        return query.count()

    def exists(self, id: Any) -> bool:
        """Check if record exists by ID"""
        return self.db.query(self.model.id).filter(self.model.id == id).first() is not None

    def search(
        self,
        filters: Dict[str, Any] = None,
        search_term: str = None,
        search_fields: List[str] = None,
        skip: int = 0,
        limit: int = 100,
        order_by: str = None,
        desc_order: bool = False
    ) -> List[T]:
        """Search records with filters and text search"""
        query = self.db.query(self.model)

        # Apply filters
        if filters:
            query = self._apply_filters(query, filters)

        # Apply text search
        if search_term and search_fields:
            search_conditions = []
            for field_name in search_fields:
                field = getattr(self.model, field_name, None)
                if field and hasattr(field.type, 'python_type') and field.type.python_type == str:
                    search_conditions.append(field.ilike(f'%{search_term}%'))

            if search_conditions:
                query = query.filter(or_(*search_conditions))

        # Apply ordering
        if order_by:
            field = getattr(self.model, order_by, None)
            if field:
                if desc_order:
                    query = query.order_by(desc(field))
                else:
                    query = query.order_by(asc(field))

        return query.offset(skip).limit(limit).all()

    def bulk_create(self, obj_data_list: List[Dict[str, Any]]) -> List[T]:
        """Create multiple records in bulk"""
        db_objs = [self.model(**obj_data) for obj_data in obj_data_list]
        self.db.add_all(db_objs)
        self.db.commit()
        for obj in db_objs:
            self.db.refresh(obj)
        return db_objs

    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """Update multiple records in bulk"""
        if not updates:
            return 0

        count = 0
        for update_data in updates:
            if 'id' in update_data:
                obj_id = update_data.pop('id')
                result = self.db.query(self.model).filter(self.model.id == obj_id).update(update_data)
                count += result

        self.db.commit()
        return count

    def bulk_delete(self, ids: List[Any]) -> int:
        """Delete multiple records in bulk"""
        count = self.db.query(self.model).filter(self.model.id.in_(ids)).delete(synchronize_session=False)
        self.db.commit()
        return count

    def get_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        desc_order: bool = False
    ) -> Dict[str, Any]:
        """Get paginated results with metadata"""
        skip = (page - 1) * page_size

        query = self.db.query(self.model)
        if filters:
            query = self._apply_filters(query, filters)

        total = query.count()

        if order_by:
            field = getattr(self.model, order_by, None)
            if field:
                if desc_order:
                    query = query.order_by(desc(field))
                else:
                    query = query.order_by(asc(field))

        items = query.offset(skip).limit(page_size).all()

        return {
            'items': items,
            'total': total,
            'page': page,
            'page_size': page_size,
            'pages': (total + page_size - 1) // page_size,
            'has_next': page * page_size < total,
            'has_prev': page > 1
        }

    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to query"""
        conditions = []

        for field_name, value in filters.items():
            if hasattr(self.model, field_name):
                field = getattr(self.model, field_name)

                if isinstance(value, dict):
                    # Handle operators like {'gte': 10, 'lte': 20}
                    for op, op_value in value.items():
                        if op == 'gte':
                            conditions.append(field >= op_value)
                        elif op == 'lte':
                            conditions.append(field <= op_value)
                        elif op == 'gt':
                            conditions.append(field > op_value)
                        elif op == 'lt':
                            conditions.append(field < op_value)
                        elif op == 'ne':
                            conditions.append(field != op_value)
                        elif op == 'in':
                            conditions.append(field.in_(op_value))
                        elif op == 'like':
                            conditions.append(field.like(op_value))
                        elif op == 'ilike':
                            conditions.append(field.ilike(op_value))
                elif isinstance(value, list):
                    # Handle IN operator
                    conditions.append(field.in_(value))
                else:
                    # Handle equality
                    conditions.append(field == value)

        if conditions:
            query = query.filter(and_(*conditions))

        return query
