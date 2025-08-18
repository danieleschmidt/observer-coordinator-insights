"""Analytics API routes for clustering and data validation
"""

import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.analytics import (
    AnalyticsJobRequest,
    AnalyticsJobResponse,
    ClusteringRequest,
    ClusteringResponse,
    EmployeeData,
    ValidationRequest,
    ValidationResponse,
)
from api.models.base import SuccessResponse
from api.services.analytics import AnalyticsService


router = APIRouter()
analytics_service = AnalyticsService()


@router.post("/upload", response_model=ClusteringResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    clustering_params: ClusteringRequest = Depends()
):
    """Upload CSV file and perform clustering analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)

        # Perform analysis
        result = await analytics_service.analyze_csv_file(
            tmp_file_path,
            clustering_params
        )

        # Clean up temporary file
        tmp_file_path.unlink()

        return result

    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            tmp_file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}")


@router.post("/validate", response_model=ValidationResponse)
async def validate_data(
    file: UploadFile = File(...),
    validation_params: ValidationRequest = Depends()
):
    """Validate uploaded Insights Discovery data"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)

        # Perform validation
        result = await analytics_service.validate_csv_file(
            tmp_file_path,
            validation_params
        )

        # Clean up temporary file
        tmp_file_path.unlink()

        return result

    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            tmp_file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {e!s}")


@router.post("/cluster", response_model=ClusteringResponse)
async def cluster_employees(
    employees: List[EmployeeData],
    clustering_params: ClusteringRequest = Depends()
):
    """Perform clustering on provided employee data"""
    try:
        result = await analytics_service.cluster_employee_data(
            employees,
            clustering_params
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e!s}")


@router.get("/optimize-clusters/{job_id}", response_model=ClusteringResponse)
async def optimize_clusters(
    job_id: str,
    max_clusters: int = 10
):
    """Find optimal number of clusters for given dataset"""
    try:
        result = await analytics_service.optimize_cluster_count(job_id, max_clusters)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e!s}")


@router.post("/jobs", response_model=AnalyticsJobResponse)
async def create_analytics_job(
    job_request: AnalyticsJobRequest,
    background_tasks: BackgroundTasks
):
    """Create an asynchronous analytics job"""
    try:
        job_id = str(uuid.uuid4())

        # Add job to background processing
        background_tasks.add_task(
            analytics_service.process_analytics_job,
            job_id,
            job_request
        )

        return AnalyticsJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            result=None,
            error=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job creation failed: {e!s}")


@router.get("/jobs/{job_id}", response_model=AnalyticsJobResponse)
async def get_job_status(job_id: str):
    """Get status of analytics job"""
    try:
        job_status = await analytics_service.get_job_status(job_id)

        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")

        return job_status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e!s}")


@router.get("/insights/{analysis_id}")
async def get_analysis_insights(analysis_id: str):
    """Get detailed insights from a completed analysis"""
    try:
        insights = await analytics_service.get_analysis_insights(analysis_id)

        if not insights:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return SuccessResponse(
            message="Insights retrieved successfully",
            data=insights
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {e!s}")


@router.get("/export/{analysis_id}")
async def export_analysis_results(
    analysis_id: str,
    format: str = "json"  # json, csv, xlsx
):
    """Export analysis results in specified format"""
    if format not in ["json", "csv", "xlsx"]:
        raise HTTPException(status_code=400, detail="Format must be json, csv, or xlsx")

    try:
        export_data = await analytics_service.export_analysis(analysis_id, format)

        if format == "json":
            return export_data
        else:
            # For CSV/Excel, return file download
            from fastapi.responses import FileResponse
            return FileResponse(
                path=export_data["file_path"],
                filename=export_data["filename"],
                media_type=export_data["media_type"]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e!s}")


@router.get("/history")
async def get_analysis_history(
    limit: int = 50,
    offset: int = 0
):
    """Get history of analytics operations"""
    try:
        history = await analytics_service.get_analysis_history(limit, offset)

        return SuccessResponse(
            message="Analysis history retrieved",
            data=history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {e!s}")


@router.delete("/cleanup")
async def cleanup_old_analyses(
    days_old: int = 30
):
    """Clean up old analysis data"""
    try:
        cleaned_count = await analytics_service.cleanup_old_analyses(days_old)

        return SuccessResponse(
            message=f"Cleaned up {cleaned_count} old analyses",
            data={"cleaned_count": cleaned_count}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e!s}")
