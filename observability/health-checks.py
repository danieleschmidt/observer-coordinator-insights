#!/usr/bin/env python3
"""
Health check endpoints for Observer Coordinator Insights.

This module provides comprehensive health checks for all system components
including application health, database connectivity, cache systems, and
external dependencies.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
import redis
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checking for all system components."""
    
    def __init__(self):
        self.checks = {
            'basic': self._basic_health,
            'database': self._database_health,
            'cache': self._cache_health,
            'storage': self._storage_health,
            'external': self._external_dependencies,
            'resources': self._resource_health,
            'clustering': self._clustering_health,
        }
        
    async def check_health(self, check_type: str = 'basic') -> Dict[str, Any]:
        """Perform health check of specified type."""
        start_time = time.time()
        
        try:
            if check_type == 'all':
                results = {}
                for name, check_func in self.checks.items():
                    try:
                        results[name] = await check_func()
                    except Exception as e:
                        results[name] = {
                            'status': 'unhealthy',
                            'error': str(e),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        
                overall_status = 'healthy' if all(
                    r.get('status') == 'healthy' for r in results.values()
                ) else 'unhealthy'
                
                return {
                    'status': overall_status,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'duration_ms': round((time.time() - start_time) * 1000, 2),
                    'checks': results
                }
            else:
                check_func = self.checks.get(check_type)
                if not check_func:
                    raise ValueError(f"Unknown check type: {check_type}")
                    
                result = await check_func()
                result['duration_ms'] = round((time.time() - start_time) * 1000, 2)
                return result
                
        except Exception as e:
            logger.exception("Health check failed")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_ms': round((time.time() - start_time) * 1000, 2)
            }
    
    async def _basic_health(self) -> Dict[str, Any]:
        """Basic application health check."""
        try:
            # Check if core modules can be imported
            import src.insights_clustering.clustering
            import src.insights_clustering.parser
            import src.team_simulator.simulator
            
            # Check Python environment
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'application': {
                    'name': 'observer-coordinator-insights',
                    'version': '0.1.0',  # Should be read from package
                    'python_version': python_version,
                    'modules_loaded': True
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f"Basic health check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _database_health(self) -> Dict[str, Any]:
        """Database connectivity and health check."""
        try:
            # Mock database check - replace with actual database logic
            # For file-based storage, check if data directory is accessible
            import os
            from pathlib import Path
            
            data_dir = Path('./data')
            if not data_dir.exists():
                data_dir.mkdir(exist_ok=True)
            
            # Test file I/O
            test_file = data_dir / '.health_check'
            test_file.write_text('health_check')
            content = test_file.read_text()
            test_file.unlink()
            
            if content != 'health_check':
                raise Exception("File I/O test failed")
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'database': {
                    'type': 'file_system',
                    'writable': True,
                    'accessible': True
                }
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': f"Database check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _cache_health(self) -> Dict[str, Any]:
        """Cache system health check."""
        try:
            # Check if cache directory is accessible
            import os
            from pathlib import Path
            
            cache_dir = Path('./cache')
            if not cache_dir.exists():
                cache_dir.mkdir(exist_ok=True)
            
            # Test cache write/read
            cache_file = cache_dir / 'health_check.cache'
            test_data = {'test': True, 'timestamp': time.time()}
            
            with open(cache_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            cache_file.unlink()
            
            if cached_data.get('test') is not True:
                raise Exception("Cache read/write test failed")
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cache': {
                    'type': 'file_system',
                    'writable': True,
                    'readable': True
                }
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': f"Cache check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _storage_health(self) -> Dict[str, Any]:
        """Storage system health check."""
        try:
            import shutil
            from pathlib import Path
            
            # Check disk space
            disk_usage = shutil.disk_usage('.')
            free_space_gb = disk_usage.free / (1024**3)
            total_space_gb = disk_usage.total / (1024**3)
            usage_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
            
            # Check critical directories
            directories = ['./data', './logs', './output', './cache']
            dir_status = {}
            
            for dir_path in directories:
                path = Path(dir_path)
                dir_status[dir_path] = {
                    'exists': path.exists(),
                    'writable': path.is_dir() and os.access(path, os.W_OK) if path.exists() else False
                }
            
            storage_healthy = (
                free_space_gb > 1.0 and  # At least 1GB free
                usage_percent < 95.0 and  # Less than 95% usage
                all(d['exists'] and d['writable'] for d in dir_status.values())
            )
            
            return {
                'status': 'healthy' if storage_healthy else 'degraded',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'storage': {
                    'free_space_gb': round(free_space_gb, 2),
                    'total_space_gb': round(total_space_gb, 2),
                    'usage_percent': round(usage_percent, 1),
                    'directories': dir_status
                }
            }
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': f"Storage check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _external_dependencies(self) -> Dict[str, Any]:
        """External dependencies health check."""
        try:
            dependencies = {}
            
            # Check if optional dependencies are available
            optional_deps = [
                ('redis', 'redis'),
                ('psycopg2', 'psycopg2'),
                ('requests', 'requests'),
            ]
            
            for package_name, import_name in optional_deps:
                try:
                    __import__(import_name)
                    dependencies[package_name] = {'status': 'available', 'error': None}
                except ImportError as e:
                    dependencies[package_name] = {'status': 'unavailable', 'error': str(e)}
            
            return {
                'status': 'healthy',  # External deps are optional
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'external_dependencies': dependencies
            }
        except Exception as e:
            logger.error(f"External dependencies check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': f"External dependencies check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _resource_health(self) -> Dict[str, Any]:
        """System resource health check."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_received': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_received': net_io.packets_recv
                }
            except:
                network_stats = None
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            try:
                load_avg = psutil.getloadavg()
                load_avg_1min = load_avg[0]
            except:
                load_avg_1min = None
            
            resource_healthy = (
                cpu_percent < 90.0 and
                memory_percent < 90.0 and
                memory_available_gb > 0.5
            )
            
            return {
                'status': 'healthy' if resource_healthy else 'degraded',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'resources': {
                    'cpu_percent': round(cpu_percent, 1),
                    'memory_percent': round(memory_percent, 1),
                    'memory_available_gb': round(memory_available_gb, 2),
                    'process_count': process_count,
                    'load_average_1min': round(load_avg_1min, 2) if load_avg_1min else None,
                    'disk_io': {
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    } if disk_io else None,
                    'network_io': network_stats
                }
            }
        except Exception as e:
            logger.error(f"Resource health check failed: {e}")
            return {
                'status': 'unhealthy', 
                'error': f"Resource check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _clustering_health(self) -> Dict[str, Any]:
        """Clustering algorithm health check."""
        try:
            # Test clustering functionality with small dataset
            import numpy as np
            from src.insights_clustering.clustering import ClusteringEngine
            
            # Generate small test dataset
            test_data = np.random.rand(50, 4)  # 50 samples, 4 features
            
            # Test clustering
            start_time = time.time()
            engine = ClusteringEngine({'n_clusters': 3, 'random_state': 42})
            clusters = engine.fit_predict(test_data)
            clustering_time = time.time() - start_time
            
            # Validate results
            unique_clusters = len(set(clusters))
            all_points_clustered = len(clusters) == len(test_data)
            
            clustering_healthy = (
                unique_clusters > 0 and
                all_points_clustered and
                clustering_time < 10.0  # Should complete in under 10 seconds
            )
            
            return {
                'status': 'healthy' if clustering_healthy else 'degraded',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'clustering': {
                    'algorithm': 'kmeans',
                    'test_dataset_size': len(test_data),
                    'clusters_generated': unique_clusters,
                    'processing_time_ms': round(clustering_time * 1000, 2),
                    'all_points_clustered': all_points_clustered
                }
            }
        except Exception as e:
            logger.error(f"Clustering health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': f"Clustering check failed: {str(e)}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


# FastAPI integration
health_checker = HealthChecker()

def create_health_endpoints(app: FastAPI):
    """Add health check endpoints to FastAPI app."""
    
    @app.get("/health", tags=["Health"])
    async def basic_health():
        """Basic health check endpoint."""
        result = await health_checker.check_health('basic')
        status_code = status.HTTP_200_OK if result['status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)
    
    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health():
        """Detailed health check with all components."""
        result = await health_checker.check_health('all')
        status_code = status.HTTP_200_OK if result['status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=result, status_code=status_code)
    
    @app.get("/health/ready", tags=["Health"])
    async def readiness_probe():
        """Kubernetes readiness probe."""
        result = await health_checker.check_health('basic')
        if result['status'] == 'healthy':
            return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    
    @app.get("/health/live", tags=["Health"])
    async def liveness_probe():
        """Kubernetes liveness probe."""
        return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    @app.get("/health/{check_type}", tags=["Health"])
    async def specific_health_check(check_type: str):
        """Specific health check endpoint."""
        try:
            result = await health_checker.check_health(check_type)
            status_code = status.HTTP_200_OK if result['status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
            return JSONResponse(content=result, status_code=status_code)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    # CLI interface for health checks
    import sys
    import asyncio
    
    async def main():
        checker = HealthChecker()
        check_type = sys.argv[1] if len(sys.argv) > 1 else 'basic'
        
        result = await checker.check_health(check_type)
        print(json.dumps(result, indent=2))
        
        # Exit with error code if unhealthy
        sys.exit(0 if result['status'] == 'healthy' else 1)
    
    asyncio.run(main())