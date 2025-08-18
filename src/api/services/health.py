"""Health check service for system monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict

import psutil


logger = logging.getLogger(__name__)


class HealthService:
    """Service for health checks and system monitoring"""

    def __init__(self):
        self.start_time = time.time()
        self.health_checks_count = 0
        self.last_health_check = None

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        self.health_checks_count += 1
        self.last_health_check = datetime.utcnow()

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Application metrics
        uptime = time.time() - self.start_time

        health_data = {
            "status": "healthy",
            "timestamp": self.last_health_check.isoformat(),
            "uptime_seconds": round(uptime, 2),
            "uptime_human": str(timedelta(seconds=int(uptime))),
            "health_checks_count": self.health_checks_count,
            "system": {
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_mb": round(memory.total / 1024 / 1024, 1),
                    "available_mb": round(memory.available / 1024 / 1024, 1),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 1),
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
                    "used_percent": round((disk.used / disk.total) * 100, 1)
                }
            },
            "components": await self._check_components()
        }

        # Determine overall health status
        if cpu_percent > 90:
            health_data["status"] = "degraded"
            health_data["warnings"] = health_data.get("warnings", []) + ["High CPU usage"]

        if memory.percent > 85:
            health_data["status"] = "degraded"
            health_data["warnings"] = health_data.get("warnings", []) + ["High memory usage"]

        if any(not comp["healthy"] for comp in health_data["components"].values()):
            health_data["status"] = "unhealthy"

        return health_data

    async def check_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to handle requests"""
        ready_checks = {
            "analytics_engine": await self._check_analytics_engine(),
            "clustering_service": await self._check_clustering_service(),
            "team_simulator": await self._check_team_simulator(),
        }

        all_ready = all(check["ready"] for check in ready_checks.values())

        return {
            "ready": all_ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": ready_checks
        }

    async def check_liveness(self) -> Dict[str, Any]:
        """Check if service is alive and responding"""
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "pid": psutil.Process().pid
        }

    async def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics"""
        health_data = await self.get_system_health()

        metrics = []

        # Application metrics
        metrics.append(f"app_uptime_seconds {health_data['uptime_seconds']}")
        metrics.append(f"app_health_checks_total {health_data['health_checks_count']}")
        metrics.append(f"app_status{{status=\"{health_data['status']}\"}} 1")

        # System metrics
        metrics.append(f"system_cpu_usage_percent {health_data['system']['cpu_usage_percent']}")
        metrics.append(f"system_memory_usage_percent {health_data['system']['memory']['used_percent']}")
        metrics.append(f"system_disk_usage_percent {health_data['system']['disk']['used_percent']}")

        # Component health
        for component, status in health_data["components"].items():
            healthy_value = 1 if status["healthy"] else 0
            metrics.append(f"component_healthy{{component=\"{component}\"}} {healthy_value}")
            if "response_time" in status:
                metrics.append(f"component_response_time_seconds{{component=\"{component}\"}} {status['response_time']}")

        return "\\n".join(metrics)

    async def _check_components(self) -> Dict[str, Dict[str, Any]]:
        """Check health of individual components"""
        components = {}

        # Check analytics components
        components["insights_parser"] = await self._check_insights_parser()
        components["clustering_engine"] = await self._check_clustering_engine()
        components["team_simulator"] = await self._check_team_simulator()
        components["data_validator"] = await self._check_data_validator()

        return components

    async def _check_insights_parser(self) -> Dict[str, Any]:
        """Check Insights Discovery parser health"""
        try:
            start_time = time.time()

            # Try importing the parser
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from insights_clustering import InsightsDataParser

            # Quick functionality test
            parser = InsightsDataParser(validate_data=False)

            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": round(response_time, 3),
                "message": "Parser is functional"
            }
        except Exception as e:
            logger.error(f"Insights parser health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "message": "Parser is not functional"
            }

    async def _check_clustering_engine(self) -> Dict[str, Any]:
        """Check clustering engine health"""
        try:
            start_time = time.time()

            # Try importing clustering
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from insights_clustering import KMeansClusterer

            # Quick functionality test
            clusterer = KMeansClusterer(n_clusters=2)

            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": round(response_time, 3),
                "message": "Clustering engine is functional"
            }
        except Exception as e:
            logger.error(f"Clustering engine health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "message": "Clustering engine is not functional"
            }

    async def _check_team_simulator(self) -> Dict[str, Any]:
        """Check team simulator health"""
        try:
            start_time = time.time()

            # Try importing team simulator
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from team_simulator import TeamCompositionSimulator

            # Quick functionality test
            simulator = TeamCompositionSimulator()

            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": round(response_time, 3),
                "message": "Team simulator is functional"
            }
        except Exception as e:
            logger.error(f"Team simulator health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "message": "Team simulator is not functional"
            }

    async def _check_data_validator(self) -> Dict[str, Any]:
        """Check data validator health"""
        try:
            start_time = time.time()

            # Try importing data validator
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from insights_clustering import DataValidator

            # Quick functionality test
            validator = DataValidator()

            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": round(response_time, 3),
                "message": "Data validator is functional"
            }
        except Exception as e:
            logger.error(f"Data validator health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "message": "Data validator is not functional"
            }

    async def _check_analytics_engine(self) -> Dict[str, Any]:
        """Check if analytics engine is ready"""
        try:
            # This would check database connections, cache, etc.
            await asyncio.sleep(0.1)  # Simulate check
            return {
                "ready": True,
                "message": "Analytics engine is ready"
            }
        except Exception as e:
            return {
                "ready": False,
                "error": str(e),
                "message": "Analytics engine is not ready"
            }
