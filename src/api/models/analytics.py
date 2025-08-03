"""
Analytics API models for clustering and validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


class EmployeeData(BaseModel):
    """Employee Insights Discovery data model"""
    employee_id: str = Field(..., description="Unique employee identifier")
    red_energy: float = Field(..., ge=0, le=100, description="Red energy percentage")
    blue_energy: float = Field(..., ge=0, le=100, description="Blue energy percentage") 
    green_energy: float = Field(..., ge=0, le=100, description="Green energy percentage")
    yellow_energy: float = Field(..., ge=0, le=100, description="Yellow energy percentage")
    
    @validator('red_energy', 'blue_energy', 'green_energy', 'yellow_energy')
    def validate_energy_sum(cls, v, values):
        """Validate that energy values are reasonable"""
        if v < 0 or v > 100:
            raise ValueError('Energy values must be between 0 and 100')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "employee_id": "EMP001",
                "red_energy": 25.5,
                "blue_energy": 30.2,
                "green_energy": 22.8,
                "yellow_energy": 21.5
            }
        }


class ClusteringRequest(BaseModel):
    """Request model for clustering operations"""
    n_clusters: int = Field(default=4, ge=2, le=10, description="Number of clusters")
    optimize_clusters: bool = Field(default=False, description="Auto-optimize cluster count")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    class Config:
        schema_extra = {
            "example": {
                "n_clusters": 4,
                "optimize_clusters": False,
                "random_state": 42
            }
        }


class ClusterInfo(BaseModel):
    """Cluster information model"""
    cluster_id: int = Field(..., description="Cluster identifier")
    size: int = Field(..., description="Number of employees in cluster")
    centroid: Dict[str, float] = Field(..., description="Cluster centroid coordinates")
    dominant_energy: str = Field(..., description="Dominant energy type")
    characteristics: List[str] = Field(..., description="Cluster characteristics")


class ClusteringResponse(BaseModel):
    """Response model for clustering operations"""
    success: bool = Field(default=True)
    employee_count: int = Field(..., description="Total employees clustered")
    cluster_count: int = Field(..., description="Number of clusters created")
    quality_metrics: Dict[str, float] = Field(..., description="Clustering quality metrics")
    clusters: List[ClusterInfo] = Field(..., description="Cluster information")
    recommendations: List[str] = Field(..., description="Analysis recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "employee_count": 100,
                "cluster_count": 4,
                "quality_metrics": {
                    "silhouette_score": 0.72,
                    "calinski_harabasz_score": 156.8,
                    "inertia": 425.6
                },
                "clusters": [
                    {
                        "cluster_id": 0,
                        "size": 25,
                        "centroid": {"red_energy": 35.2, "blue_energy": 25.1, "green_energy": 20.3, "yellow_energy": 19.4},
                        "dominant_energy": "red",
                        "characteristics": ["High assertiveness", "Results-oriented", "Competitive"]
                    }
                ],
                "recommendations": ["Consider team diversity when forming project groups"],
                "processing_time": 1.23
            }
        }


class ValidationRequest(BaseModel):
    """Request model for data validation"""
    strict_validation: bool = Field(default=False, description="Enable strict validation rules")
    auto_fix: bool = Field(default=True, description="Automatically fix common issues")
    
    class Config:
        schema_extra = {
            "example": {
                "strict_validation": False,
                "auto_fix": True
            }
        }


class ValidationIssue(BaseModel):
    """Data validation issue model"""
    level: str = Field(..., description="Issue severity: error, warning, info")
    message: str = Field(..., description="Issue description")
    field: Optional[str] = Field(None, description="Affected field name")
    count: Optional[int] = Field(None, description="Number of affected records")
    suggestion: Optional[str] = Field(None, description="Suggested fix")


class ValidationResponse(BaseModel):
    """Response model for data validation"""
    success: bool = Field(default=True)
    is_valid: bool = Field(..., description="Whether data passed validation")
    quality_score: float = Field(..., ge=0, le=100, description="Data quality score")
    total_records: int = Field(..., description="Total number of records")
    valid_records: int = Field(..., description="Number of valid records")
    issues: List[ValidationIssue] = Field(..., description="Validation issues found")
    metrics: Dict[str, Any] = Field(..., description="Data quality metrics")
    suggestions: List[str] = Field(..., description="Improvement suggestions")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "is_valid": True,
                "quality_score": 92.5,
                "total_records": 100,
                "valid_records": 98,
                "issues": [
                    {
                        "level": "warning",
                        "message": "2 records have missing department information",
                        "field": "department",
                        "count": 2,
                        "suggestion": "Fill missing department values"
                    }
                ],
                "metrics": {
                    "completeness": 98.0,
                    "consistency": 95.5,
                    "accuracy": 96.8
                },
                "suggestions": ["Review and fill missing department information"]
            }
        }


class AnalyticsJobRequest(BaseModel):
    """Request model for async analytics jobs"""
    job_type: str = Field(..., description="Type of analysis job")
    priority: str = Field(default="normal", description="Job priority: low, normal, high")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    callback_url: Optional[str] = Field(None, description="Webhook URL for job completion")


class AnalyticsJobResponse(BaseModel):
    """Response model for analytics job status"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: pending, running, completed, failed")
    progress: float = Field(..., ge=0, le=100, description="Job progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if failed")