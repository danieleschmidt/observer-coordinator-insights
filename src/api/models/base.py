"""
Base Pydantic models for common API responses
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the operation was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseResponse):
    """Success response model"""
    success: bool = Field(default=True)
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=50, ge=1, le=1000, description="Items per page")


class PaginatedResponse(BaseResponse):
    """Paginated response model"""
    success: bool = Field(default=True)
    data: List[Any] = Field(..., description="Page data")
    pagination: Dict[str, Any] = Field(..., description="Pagination info")
    
    @classmethod
    def create(cls, data: List[Any], page: int, size: int, total: int):
        """Create paginated response"""
        return cls(
            data=data,
            pagination={
                "page": page,
                "size": size,
                "total": total,
                "pages": (total + size - 1) // size,
                "has_next": page * size < total,
                "has_prev": page > 1
            }
        )