"""Admin API routes for system management
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.base import SuccessResponse
from api.services.admin import AdminService


router = APIRouter()
admin_service = AdminService()


@router.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status = await admin_service.get_system_status()

        return SuccessResponse(
            message="System status retrieved",
            data=status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e!s}")


@router.post("/cache/clear")
async def clear_cache():
    """Clear system cache"""
    try:
        result = await admin_service.clear_cache()

        return SuccessResponse(
            message="Cache cleared successfully",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e!s}")


@router.get("/metrics")
async def get_system_metrics():
    """Get detailed system metrics"""
    try:
        metrics = await admin_service.get_detailed_metrics()

        return SuccessResponse(
            message="System metrics retrieved",
            data=metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e!s}")


@router.post("/maintenance/start")
async def start_maintenance_mode():
    """Enable maintenance mode"""
    try:
        result = await admin_service.enable_maintenance_mode()

        return SuccessResponse(
            message="Maintenance mode enabled",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable maintenance mode: {e!s}")


@router.post("/maintenance/stop")
async def stop_maintenance_mode():
    """Disable maintenance mode"""
    try:
        result = await admin_service.disable_maintenance_mode()

        return SuccessResponse(
            message="Maintenance mode disabled",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable maintenance mode: {e!s}")
