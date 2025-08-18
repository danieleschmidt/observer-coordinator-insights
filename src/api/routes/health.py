"""Health check and system status routes
"""

import sys
import time
from pathlib import Path

from fastapi import APIRouter


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.base import SuccessResponse
from api.services.health import HealthService


router = APIRouter()
health_service = HealthService()


@router.get("/", response_model=SuccessResponse)
async def health_check():
    """Basic health check endpoint"""
    return SuccessResponse(
        message="Service is healthy",
        data={
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
    )


@router.get("/detailed", response_model=SuccessResponse)
async def detailed_health_check():
    """Detailed health check with system metrics"""
    system_info = await health_service.get_system_health()

    return SuccessResponse(
        message="Detailed health check completed",
        data=system_info
    )


@router.get("/ready", response_model=SuccessResponse)
async def readiness_check():
    """Readiness check for container orchestration"""
    ready = await health_service.check_readiness()

    if not ready["ready"]:
        return SuccessResponse(
            message="Service not ready",
            data=ready
        )

    return SuccessResponse(
        message="Service is ready",
        data=ready
    )


@router.get("/live", response_model=SuccessResponse)
async def liveness_check():
    """Liveness check for container orchestration"""
    alive = await health_service.check_liveness()

    return SuccessResponse(
        message="Service is alive",
        data=alive
    )


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics endpoint"""
    metrics = await health_service.get_prometheus_metrics()

    # Return plain text metrics for Prometheus
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=metrics,
        media_type="text/plain"
    )
