#!/usr/bin/env python3
"""Health Check Endpoints for Generation 2 Robustness
Provides comprehensive health monitoring via HTTP endpoints
"""

import time
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gen2_robustness import HealthStatus, health_checker, performance_monitor


app = FastAPI(
    title="Observer Coordinator Insights Health API",
    description="Generation 2 Robustness Health Monitoring",
    version="2.0.0"
)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    components: Dict[str, Any]
    system_metrics: Dict[str, Any]


class LivenessResponse(BaseModel):
    """Liveness probe response"""
    status: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness probe response"""
    status: str
    timestamp: datetime
    ready: bool
    checks: Dict[str, str]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    # Run all health checks
    health_results = await health_checker.run_all_checks()

    # Get system metrics
    system_metrics = health_checker.get_system_metrics()

    # Determine overall status
    overall_status = HealthStatus.HEALTHY
    for result in health_results.values():
        if result.status == HealthStatus.CRITICAL:
            overall_status = HealthStatus.CRITICAL
            break
        elif result.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED

    # Format response
    response_data = {
        "status": overall_status.value,
        "timestamp": datetime.now(),
        "components": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
                "details": result.details
            }
            for name, result in health_results.items()
        },
        "system_metrics": {
            "cpu_percent": system_metrics.cpu_percent,
            "memory_percent": system_metrics.memory_percent,
            "memory_used_mb": system_metrics.memory_used_mb,
            "memory_available_mb": system_metrics.memory_available_mb,
            "disk_usage_percent": system_metrics.disk_usage_percent,
            "disk_free_gb": system_metrics.disk_free_gb,
            "load_avg": system_metrics.load_avg,
            "active_connections": system_metrics.active_connections
        }
    }

    # Set appropriate HTTP status code
    status_code = status.HTTP_200_OK
    if overall_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif overall_status == HealthStatus.DEGRADED:
        status_code = status.HTTP_200_OK  # Still available but degraded

    return JSONResponse(
        content=response_data,
        status_code=status_code,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/health/live", response_model=LivenessResponse)
async def liveness_probe():
    """Kubernetes liveness probe - simple alive check"""
    return {
        "status": "alive",
        "timestamp": datetime.now()
    }


@app.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe():
    """Kubernetes readiness probe - ready to serve traffic"""
    # Run critical health checks only
    health_results = await health_checker.run_all_checks()

    # Check if critical components are healthy
    critical_components = ['memory', 'disk']  # Define critical components
    ready = True
    check_statuses = {}

    for name, result in health_results.items():
        check_statuses[name] = result.status.value
        if name in critical_components and result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
            ready = False

    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        content={
            "status": "ready" if ready else "not_ready",
            "timestamp": datetime.now(),
            "ready": ready,
            "checks": check_statuses
        },
        status_code=status_code,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/health/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    recent_metrics = performance_monitor.get_recent_metrics(minutes=5)
    system_metrics = health_checker.get_system_metrics()

    # Generate Prometheus format metrics
    metrics_output = []

    # System metrics
    metrics_output.append("# TYPE system_cpu_percent gauge")
    metrics_output.append(f"system_cpu_percent {system_metrics.cpu_percent}")

    metrics_output.append("# TYPE system_memory_percent gauge")
    metrics_output.append(f"system_memory_percent {system_metrics.memory_percent}")

    metrics_output.append("# TYPE system_disk_usage_percent gauge")
    metrics_output.append(f"system_disk_usage_percent {system_metrics.disk_usage_percent}")

    # Application metrics
    for metric in recent_metrics[-10:]:  # Last 10 metrics
        metric_name = metric['name'].replace('.', '_')
        metrics_output.append(f"# TYPE app_{metric_name} gauge")
        metrics_output.append(f"app_{metric_name} {metric['value']}")

    return Response(
        content="\n".join(metrics_output),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/health/startup")
async def startup_probe():
    """Kubernetes startup probe - application started successfully"""
    # Simple startup check - could be enhanced with actual startup validation
    return JSONResponse(
        content={
            "status": "started",
            "timestamp": datetime.now(),
            "uptime_seconds": time.time() - start_time
        },
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# Track startup time
start_time = time.time()


if __name__ == "__main__":
    import uvicorn

    # Run health service on a different port
    uvicorn.run(
        app,
        host="127.0.0.1",  # Secure localhost binding
        port=8001,
        log_level="info",
        access_log=True
    )
