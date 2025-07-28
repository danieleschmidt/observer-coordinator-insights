"""
Health check endpoints and monitoring for Observer Coordinator Insights.
Provides comprehensive system health validation.
"""

import os
import psutil
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status information."""
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: str
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checking for the application."""
    
    def __init__(self):
        self.start_time = time.time()
        self.version = os.getenv('APP_VERSION', '0.1.0')
        self.checks = {}
        
    def register_check(self, name: str, check_func, critical: bool = True):
        """Register a health check function."""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'status': 'healthy' if memory.percent < 85 else 'degraded' if memory.percent < 95 else 'unhealthy',
                'usage_percent': memory.percent,
                'available_gb': round(memory.available / (1024**3), 2),
                'total_gb': round(memory.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            percent = (disk.used / disk.total) * 100
            return {
                'status': 'healthy' if percent < 80 else 'degraded' if percent < 90 else 'unhealthy',
                'usage_percent': round(percent, 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'total_gb': round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return {
                'status': 'healthy' if cpu_percent < 80 else 'degraded' if cpu_percent < 95 else 'unhealthy',
                'usage_percent': cpu_percent,
                'count': psutil.cpu_count()
            }
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_python_dependencies(self) -> Dict[str, Any]:
        """Check if critical Python dependencies are available."""
        try:
            required_modules = [
                'pandas',
                'numpy', 
                'scikit-learn',
                'matplotlib',
                'seaborn',
                'yaml'
            ]
            
            missing = []
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing.append(module)
            
            if missing:
                return {
                    'status': 'unhealthy',
                    'missing_modules': missing
                }
            
            return {
                'status': 'healthy',
                'modules_checked': len(required_modules)
            }
            
        except Exception as e:
            logger.error(f"Dependencies check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_data_directories(self) -> Dict[str, Any]:
        """Check if required data directories exist and are writable."""
        try:
            required_dirs = [
                'data',
                'output', 
                'logs',
                'cache'
            ]
            
            status_info = {}
            all_healthy = True
            
            for dir_name in required_dirs:
                dir_path = os.path.join(os.getcwd(), dir_name)
                
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                        status_info[dir_name] = {'status': 'created', 'writable': True}
                    except Exception as e:
                        status_info[dir_name] = {'status': 'missing', 'error': str(e)}
                        all_healthy = False
                        continue
                
                # Check if writable
                try:
                    test_file = os.path.join(dir_path, '.health_check')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    status_info[dir_name] = {'status': 'healthy', 'writable': True}
                except Exception as e:
                    status_info[dir_name] = {'status': 'not_writable', 'error': str(e)}
                    all_healthy = False
            
            return {
                'status': 'healthy' if all_healthy else 'unhealthy',
                'directories': status_info
            }
            
        except Exception as e:
            logger.error(f"Data directories check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_clustering_algorithm(self) -> Dict[str, Any]:
        """Check if clustering algorithms are functional."""
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Quick test of clustering functionality
            test_data = np.random.rand(10, 4)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(test_data)
            
            return {
                'status': 'healthy',
                'algorithm': 'kmeans',
                'test_clusters': 2
            }
            
        except Exception as e:
            logger.error(f"Clustering check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def run_all_checks(self) -> HealthStatus:
        """Run all registered health checks."""
        timestamp = datetime.now(timezone.utc).isoformat()
        uptime = self.get_uptime()
        
        # Default system checks
        default_checks = {
            'memory': self.check_memory_usage(),
            'disk': self.check_disk_usage(),
            'cpu': self.check_cpu_usage(),
            'dependencies': self.check_python_dependencies(),
            'data_directories': self.check_data_directories(),
            'clustering': self.check_clustering_algorithm()
        }
        
        # Run custom registered checks
        custom_checks = {}
        for name, check_info in self.checks.items():
            try:
                custom_checks[name] = check_info['func']()
            except Exception as e:
                logger.error(f"Custom check '{name}' failed: {e}")
                custom_checks[name] = {'status': 'unhealthy', 'error': str(e)}
        
        all_checks = {**default_checks, **custom_checks}
        
        # Determine overall status
        critical_unhealthy = any(
            check.get('status') == 'unhealthy' 
            for check in all_checks.values()
        )
        
        any_degraded = any(
            check.get('status') == 'degraded'
            for check in all_checks.values()
        )
        
        if critical_unhealthy:
            overall_status = 'unhealthy'
        elif any_degraded:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return HealthStatus(
            status=overall_status,
            timestamp=timestamp,
            version=self.version,
            uptime_seconds=uptime,
            checks=all_checks
        )


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker


# Flask blueprint for health endpoints (if using Flask)
try:
    from flask import Blueprint, jsonify
    
    health_blueprint = Blueprint('health', __name__)
    
    @health_blueprint.route('/health')
    def health_check():
        """Main health check endpoint."""
        health_status = _health_checker.run_all_checks()
        status_code = 200 if health_status.status == 'healthy' else 503
        return jsonify(asdict(health_status)), status_code
    
    @health_blueprint.route('/health/live')
    def liveness_check():
        """Kubernetes liveness probe endpoint."""
        return jsonify({
            'status': 'alive',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    
    @health_blueprint.route('/health/ready')
    def readiness_check():
        """Kubernetes readiness probe endpoint."""
        # Quick readiness check - only essential services
        ready_checks = {
            'dependencies': _health_checker.check_python_dependencies(),
            'data_directories': _health_checker.check_data_directories()
        }
        
        all_ready = all(
            check.get('status') == 'healthy'
            for check in ready_checks.values()
        )
        
        status_code = 200 if all_ready else 503
        return jsonify({
            'status': 'ready' if all_ready else 'not_ready',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': ready_checks
        }), status_code

except ImportError:
    # Flask not available, define dummy blueprint
    health_blueprint = None
    logger.info("Flask not available, health endpoints disabled")