"""
Generation 3 Distributed Clustering Coordinator
High-level coordinator for managing distributed neuromorphic clustering
with microservice architecture and API gateway integration
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from aiohttp import ClientSession, web
except ImportError:
    ClientSession = None
    web = None  # Will use simplified coordinator without aiohttp
import threading
import numpy as np
import pandas as pd

try:
    import aiohttp
    from aiohttp import web, ClientSession
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import celery
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClusteringJob:
    """High-level clustering job specification"""
    job_id: str
    job_type: str  # 'single', 'ensemble', 'incremental', 'streaming'
    features_data: Union[pd.DataFrame, Dict[str, Any]]
    parameters: Dict[str, Any]
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = 'pending'
    subtasks: List[str] = field(default_factory=list)
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def processing_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    service_id: str
    name: str
    host: str
    port: int
    capabilities: List[str]
    health_status: str = 'unknown'
    last_check: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        return (self.health_status == 'healthy' and 
                time.time() - self.last_check < 120)  # 2 minute timeout


class ServiceRegistry:
    """Registry for microservices in the distributed system"""
    
    def __init__(self):
        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_types: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
        self.health_check_interval = 60  # seconds
        self.health_check_thread = None
        self.running = False
    
    def register_service(self, service: ServiceEndpoint):
        """Register a new service endpoint"""
        with self.lock:
            self.services[service.service_id] = service
            
            # Group by capabilities
            for capability in service.capabilities:
                if capability not in self.service_types:
                    self.service_types[capability] = []
                if service.service_id not in self.service_types[capability]:
                    self.service_types[capability].append(service.service_id)
            
            logger.info(f"Registered service {service.name} ({service.service_id})")
    
    def unregister_service(self, service_id: str):
        """Unregister a service endpoint"""
        with self.lock:
            if service_id in self.services:
                service = self.services.pop(service_id)
                
                # Remove from capability groups
                for capability in service.capabilities:
                    if capability in self.service_types:
                        self.service_types[capability] = [
                            sid for sid in self.service_types[capability] 
                            if sid != service_id
                        ]
                
                logger.info(f"Unregistered service {service.name} ({service_id})")
    
    def get_services_by_capability(self, capability: str) -> List[ServiceEndpoint]:
        """Get healthy services that have a specific capability"""
        with self.lock:
            if capability not in self.service_types:
                return []
            
            service_ids = self.service_types[capability]
            return [
                self.services[sid] for sid in service_ids 
                if sid in self.services and self.services[sid].is_healthy
            ]
    
    def get_service(self, service_id: str) -> Optional[ServiceEndpoint]:
        """Get service by ID"""
        with self.lock:
            return self.services.get(service_id)
    
    def get_all_services(self) -> List[ServiceEndpoint]:
        """Get all registered services"""
        with self.lock:
            return list(self.services.values())
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Started service health monitoring")
    
    def stop_health_monitoring(self):
        """Stop background health monitoring"""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Stopped service health monitoring")
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                self._check_all_services_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_all_services_health(self):
        """Check health of all registered services"""
        if not AIOHTTP_AVAILABLE:
            return
        
        async def check_health():
            services = self.get_all_services()
            
            async with ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                tasks = []
                for service in services:
                    task = self._check_service_health(session, service)
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run health checks
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(check_health())
            loop.close()
        except Exception as e:
            logger.warning(f"Health check execution failed: {e}")
    
    async def _check_service_health(self, session: ClientSession, service: ServiceEndpoint):
        """Check health of a single service"""
        try:
            start_time = time.time()
            
            async with session.get(f"{service.url}/health") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    service.health_status = 'healthy'
                    service.response_time_ms = response_time
                else:
                    service.health_status = 'unhealthy'
                
                service.last_check = time.time()
                
        except Exception as e:
            service.health_status = 'unreachable'
            service.last_check = time.time()
            logger.debug(f"Health check failed for {service.name}: {e}")


class LoadBalancingStrategy:
    """Load balancing strategies for service selection"""
    
    @staticmethod
    def round_robin(services: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """Simple round-robin selection"""
        if not services:
            return None
        return services[int(time.time()) % len(services)]
    
    @staticmethod
    def least_response_time(services: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """Select service with lowest response time"""
        if not services:
            return None
        return min(services, key=lambda s: s.response_time_ms)
    
    @staticmethod
    def weighted_random(services: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """Weighted random selection based on inverse response time"""
        if not services:
            return None
        
        # Calculate weights (higher for faster services)
        weights = []
        for service in services:
            weight = 1000.0 / max(service.response_time_ms, 1.0)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return services[0]
        
        weights = [w / total_weight for w in weights]
        
        # Random selection
        import random
        return random.choices(services, weights=weights)[0]


class CeleryTaskManager:
    """Celery-based task management for distributed processing"""
    
    def __init__(self, broker_url: str = 'redis://localhost:6379/0',
                 result_backend: str = 'redis://localhost:6379/0'):
        
        if not CELERY_AVAILABLE:
            logger.warning("Celery not available, task management will be limited")
            self.celery_app = None
            return
        
        self.celery_app = Celery(
            'neuromorphic_clustering',
            broker=broker_url,
            backend=result_backend,
            include=['src.distributed.clustering_coordinator']
        )
        
        # Configure Celery
        self.celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_routes={
                'clustering.*': {'queue': 'clustering'},
                'feature_extraction.*': {'queue': 'features'},
                'ensemble.*': {'queue': 'ensemble'}
            },
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_disable_rate_limits=False,
            task_compression='gzip',
            result_compression='gzip'
        )
        
        logger.info("Initialized Celery task manager")
    
    def submit_task(self, task_name: str, args: List[Any], 
                   kwargs: Dict[str, Any], priority: int = 5) -> Optional[str]:
        """Submit task to Celery"""
        if not self.celery_app:
            logger.warning("Celery not available, cannot submit task")
            return None
        
        try:
            result = self.celery_app.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                priority=priority,
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 0,
                    'interval_step': 0.2,
                    'interval_max': 0.2,
                }
            )
            
            logger.info(f"Submitted Celery task {task_name}: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit Celery task {task_name}: {e}")
            return None
    
    def get_task_result(self, task_id: str, timeout: float = 300) -> Any:
        """Get result of a Celery task"""
        if not self.celery_app:
            raise RuntimeError("Celery not available")
        
        result = self.celery_app.AsyncResult(task_id)
        return result.get(timeout=timeout)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a Celery task"""
        if not self.celery_app:
            return {'status': 'unknown'}
        
        result = self.celery_app.AsyncResult(task_id)
        return {
            'status': result.status,
            'result': result.result if result.ready() else None,
            'traceback': result.traceback
        }


class APIGateway:
    """API Gateway for routing requests to appropriate services"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for API Gateway")
        
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[self._error_middleware])
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancingStrategy()
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Initialized API Gateway on {host}:{port}")
    
    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/health', self._health_check)
        self.app.router.add_get('/services', self._list_services)
        self.app.router.add_post('/services/register', self._register_service)
        self.app.router.add_delete('/services/{service_id}', self._unregister_service)
        
        # Clustering routes
        self.app.router.add_post('/clustering/neuromorphic', self._clustering_neuromorphic)
        self.app.router.add_post('/clustering/ensemble', self._clustering_ensemble)
        self.app.router.add_post('/clustering/incremental', self._clustering_incremental)
        
        # Job management routes
        self.app.router.add_get('/jobs/{job_id}', self._get_job_status)
        self.app.router.add_get('/jobs/{job_id}/result', self._get_job_result)
        
        # System routes
        self.app.router.add_get('/metrics', self._get_metrics)
        self.app.router.add_get('/status', self._get_system_status)
    
    @web.middleware
    async def _error_middleware(self, request, handler):
        """Error handling middleware"""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return web.json_response(
                {'error': str(e)}, 
                status=500
            )
    
    async def _health_check(self, request):
        """API Gateway health check"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': time.time(),
            'services_registered': len(self.service_registry.services)
        })
    
    async def _list_services(self, request):
        """List all registered services"""
        services = self.service_registry.get_all_services()
        
        services_data = []
        for service in services:
            services_data.append({
                'service_id': service.service_id,
                'name': service.name,
                'url': service.url,
                'capabilities': service.capabilities,
                'health_status': service.health_status,
                'response_time_ms': service.response_time_ms
            })
        
        return web.json_response({'services': services_data})
    
    async def _register_service(self, request):
        """Register a new service"""
        try:
            data = await request.json()
            
            service = ServiceEndpoint(
                service_id=data.get('service_id') or str(uuid.uuid4()),
                name=data['name'],
                host=data['host'],
                port=data['port'],
                capabilities=data.get('capabilities', [])
            )
            
            self.service_registry.register_service(service)
            
            return web.json_response({
                'service_id': service.service_id,
                'status': 'registered'
            })
            
        except Exception as e:
            return web.json_response(
                {'error': f'Registration failed: {str(e)}'}, 
                status=400
            )
    
    async def _unregister_service(self, request):
        """Unregister a service"""
        service_id = request.match_info['service_id']
        self.service_registry.unregister_service(service_id)
        
        return web.json_response({'status': 'unregistered'})
    
    async def _clustering_neuromorphic(self, request):
        """Handle neuromorphic clustering request"""
        try:
            data = await request.json()
            
            # Find suitable service
            services = self.service_registry.get_services_by_capability('neuromorphic_clustering')
            if not services:
                return web.json_response(
                    {'error': 'No neuromorphic clustering services available'}, 
                    status=503
                )
            
            # Select service using load balancing
            selected_service = self.load_balancer.least_response_time(services)
            
            # Forward request to selected service
            async with ClientSession() as session:
                async with session.post(
                    f"{selected_service.url}/clustering/neuromorphic",
                    json=data
                ) as response:
                    result = await response.json()
                    
                    return web.json_response(result, status=response.status)
                    
        except Exception as e:
            return web.json_response(
                {'error': f'Clustering failed: {str(e)}'}, 
                status=500
            )
    
    async def _clustering_ensemble(self, request):
        """Handle ensemble clustering request"""
        try:
            data = await request.json()
            
            # Get multiple clustering services for ensemble
            services = self.service_registry.get_services_by_capability('neuromorphic_clustering')
            if len(services) < 2:
                return web.json_response(
                    {'error': 'Insufficient services for ensemble clustering'}, 
                    status=503
                )
            
            # Submit to multiple services
            tasks = []
            async with ClientSession() as session:
                for service in services[:data.get('ensemble_size', 3)]:
                    task = session.post(
                        f"{service.url}/clustering/neuromorphic",
                        json=data
                    )
                    tasks.append(task)
                
                # Collect results
                results = []
                for task in asyncio.as_completed(tasks):
                    try:
                        response = await task
                        result = await response.json()
                        if response.status == 200:
                            results.append(result)
                    except Exception as e:
                        logger.warning(f"Ensemble member failed: {e}")
                
                if not results:
                    return web.json_response(
                        {'error': 'All ensemble members failed'}, 
                        status=500
                    )
                
                # Combine ensemble results
                ensemble_result = self._combine_clustering_results(results)
                
                return web.json_response(ensemble_result)
                
        except Exception as e:
            return web.json_response(
                {'error': f'Ensemble clustering failed: {str(e)}'}, 
                status=500
            )
    
    async def _clustering_incremental(self, request):
        """Handle incremental clustering request"""
        # Implementation for incremental clustering
        return web.json_response(
            {'error': 'Incremental clustering not implemented'}, 
            status=501
        )
    
    async def _get_job_status(self, request):
        """Get job status"""
        job_id = request.match_info['job_id']
        
        # Implementation depends on job storage backend
        return web.json_response({
            'job_id': job_id,
            'status': 'pending'
        })
    
    async def _get_job_result(self, request):
        """Get job result"""
        job_id = request.match_info['job_id']
        
        # Implementation depends on job storage backend
        return web.json_response({
            'job_id': job_id,
            'result': None
        })
    
    async def _get_metrics(self, request):
        """Get system metrics"""
        services = self.service_registry.get_all_services()
        
        healthy_services = [s for s in services if s.is_healthy]
        total_response_time = sum(s.response_time_ms for s in healthy_services)
        avg_response_time = total_response_time / len(healthy_services) if healthy_services else 0
        
        return web.json_response({
            'total_services': len(services),
            'healthy_services': len(healthy_services),
            'avg_response_time_ms': avg_response_time,
            'capabilities': list(self.service_registry.service_types.keys())
        })
    
    async def _get_system_status(self, request):
        """Get comprehensive system status"""
        return web.json_response({
            'gateway_status': 'healthy',
            'service_registry': {
                'total_services': len(self.service_registry.services),
                'service_types': {
                    capability: len(service_ids)
                    for capability, service_ids in self.service_registry.service_types.items()
                }
            },
            'timestamp': time.time()
        })
    
    def _combine_clustering_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple clustering results into ensemble result"""
        if not results:
            return {'error': 'No results to combine'}
        
        # Simple ensemble: majority voting on cluster assignments
        all_assignments = [result.get('cluster_assignments', []) for result in results]
        
        if not all_assignments or not all_assignments[0]:
            return results[0]  # Return first result if no assignments
        
        n_samples = len(all_assignments[0])
        ensemble_assignments = []
        
        for i in range(n_samples):
            sample_votes = [assignments[i] for assignments in all_assignments if i < len(assignments)]
            if sample_votes:
                # Majority vote
                ensemble_label = max(set(sample_votes), key=sample_votes.count)
                ensemble_assignments.append(ensemble_label)
            else:
                ensemble_assignments.append(0)  # Default
        
        # Calculate consensus strength
        consensus_scores = []
        for i in range(n_samples):
            sample_votes = [assignments[i] for assignments in all_assignments if i < len(assignments)]
            if sample_votes:
                most_common_count = sample_votes.count(max(set(sample_votes), key=sample_votes.count))
                consensus_scores.append(most_common_count / len(sample_votes))
            else:
                consensus_scores.append(0.0)
        
        return {
            'cluster_assignments': ensemble_assignments,
            'ensemble_size': len(results),
            'consensus_strength': sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0,
            'individual_results': results
        }
    
    def start(self):
        """Start the API Gateway"""
        self.service_registry.start_health_monitoring()
        
        logger.info(f"Starting API Gateway on {self.host}:{self.port}")
        
        web.run_app(self.app, host=self.host, port=self.port)
    
    def stop(self):
        """Stop the API Gateway"""
        self.service_registry.stop_health_monitoring()


class DistributedClusteringCoordinator:
    """Main coordinator for distributed neuromorphic clustering system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.service_registry = ServiceRegistry()
        self.api_gateway = None
        self.celery_manager = None
        
        # Job management
        self.active_jobs: Dict[str, ClusteringJob] = {}
        self.job_executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize optional components
        if self.config.get('enable_api_gateway', True):
            try:
                self.api_gateway = APIGateway(
                    host=self.config.get('gateway_host', '0.0.0.0'),
                    port=self.config.get('gateway_port', 8080)
                )
                self.api_gateway.service_registry = self.service_registry
            except ImportError:
                logger.warning("API Gateway disabled - aiohttp not available")
        
        if self.config.get('enable_celery', False):
            try:
                self.celery_manager = CeleryTaskManager(
                    broker_url=self.config.get('celery_broker', 'redis://localhost:6379/0'),
                    result_backend=self.config.get('celery_backend', 'redis://localhost:6379/0')
                )
            except ImportError:
                logger.warning("Celery task manager disabled - celery not available")
        
        logger.info("Initialized Distributed Clustering Coordinator")
    
    def start(self):
        """Start the distributed clustering coordinator"""
        logger.info("Starting Distributed Clustering Coordinator")
        
        # Start service monitoring
        self.service_registry.start_health_monitoring()
        
        # Start API gateway if available
        if self.api_gateway:
            # Run in separate thread to avoid blocking
            gateway_thread = threading.Thread(
                target=self.api_gateway.start, 
                daemon=True
            )
            gateway_thread.start()
        
        logger.info("Distributed Clustering Coordinator started")
    
    def stop(self):
        """Stop the distributed clustering coordinator"""
        logger.info("Stopping Distributed Clustering Coordinator")
        
        # Stop service monitoring
        self.service_registry.stop_health_monitoring()
        
        # Shutdown job executor
        self.job_executor.shutdown(wait=True)
        
        if self.api_gateway:
            self.api_gateway.stop()
        
        logger.info("Distributed Clustering Coordinator stopped")
    
    def submit_clustering_job(self, features: pd.DataFrame,
                            job_type: str = 'single',
                            parameters: Optional[Dict[str, Any]] = None,
                            priority: int = 5) -> str:
        """Submit a high-level clustering job"""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        job = ClusteringJob(
            job_id=job_id,
            job_type=job_type,
            features_data=features.to_dict(),
            parameters=parameters or {},
            priority=priority
        )
        
        self.active_jobs[job_id] = job
        
        # Submit job for processing
        future = self.job_executor.submit(self._process_clustering_job, job)
        
        logger.info(f"Submitted clustering job {job_id} of type {job_type}")
        return job_id
    
    def _process_clustering_job(self, job: ClusteringJob):
        """Process a clustering job"""
        try:
            job.status = 'processing'
            job.started_at = time.time()
            
            if job.job_type == 'single':
                result = self._process_single_clustering(job)
            elif job.job_type == 'ensemble':
                result = self._process_ensemble_clustering(job)
            elif job.job_type == 'incremental':
                result = self._process_incremental_clustering(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            job.results = result
            job.status = 'completed'
            job.completed_at = time.time()
            
        except Exception as e:
            job.error = str(e)
            job.status = 'failed'
            job.completed_at = time.time()
            logger.error(f"Job {job.job_id} failed: {e}")
    
    def _process_single_clustering(self, job: ClusteringJob) -> Dict[str, Any]:
        """Process single clustering job"""
        # Find available clustering service
        services = self.service_registry.get_services_by_capability('neuromorphic_clustering')
        if not services:
            raise RuntimeError("No clustering services available")
        
        # For now, use a simple approach - this would be enhanced with actual service calls
        from ..insights_clustering.neuromorphic_clustering import NeuromorphicClusterer
        
        features = pd.DataFrame.from_dict(job.features_data)
        
        clusterer = NeuromorphicClusterer(
            n_clusters=job.parameters.get('n_clusters', 4),
            method=job.parameters.get('method', 'hybrid_reservoir')
        )
        
        clusterer.fit(features)
        
        return {
            'cluster_assignments': clusterer.get_cluster_assignments().tolist(),
            'cluster_interpretations': clusterer.get_cluster_interpretation(),
            'metrics': clusterer.get_clustering_metrics().__dict__
        }
    
    def _process_ensemble_clustering(self, job: ClusteringJob) -> Dict[str, Any]:
        """Process ensemble clustering job"""
        # This would coordinate multiple clustering services
        # For now, return a placeholder
        return {'error': 'Ensemble clustering not fully implemented'}
    
    def _process_incremental_clustering(self, job: ClusteringJob) -> Dict[str, Any]:
        """Process incremental clustering job"""
        # This would handle streaming/incremental updates
        # For now, return a placeholder
        return {'error': 'Incremental clustering not fully implemented'}
    
    def get_job_status(self, job_id: str) -> Optional[ClusteringJob]:
        """Get status of a clustering job"""
        return self.active_jobs.get(job_id)
    
    def get_job_result(self, job_id: str, timeout: float = 300) -> Optional[Dict[str, Any]]:
        """Wait for and get job result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job = self.active_jobs.get(job_id)
            if not job:
                return None
            
            if job.status == 'completed':
                return job.results
            elif job.status == 'failed':
                raise RuntimeError(f"Job {job_id} failed: {job.error}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Job {job_id} timed out")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        services = self.service_registry.get_all_services()
        healthy_services = [s for s in services if s.is_healthy]
        
        return {
            'coordinator_status': 'running',
            'services': {
                'total': len(services),
                'healthy': len(healthy_services),
                'by_capability': {
                    capability: len([s for s in healthy_services if capability in s.capabilities])
                    for capability in set(cap for s in services for cap in s.capabilities)
                }
            },
            'jobs': {
                'total': len(self.active_jobs),
                'by_status': {
                    status: len([j for j in self.active_jobs.values() if j.status == status])
                    for status in ['pending', 'processing', 'completed', 'failed']
                }
            },
            'api_gateway_enabled': self.api_gateway is not None,
            'celery_enabled': self.celery_manager is not None
        }


# Global coordinator instance
distributed_coordinator: Optional[DistributedClusteringCoordinator] = None


def initialize_coordinator(config: Optional[Dict[str, Any]] = None) -> DistributedClusteringCoordinator:
    """Initialize global distributed clustering coordinator"""
    global distributed_coordinator
    
    distributed_coordinator = DistributedClusteringCoordinator(config)
    return distributed_coordinator


def get_coordinator() -> Optional[DistributedClusteringCoordinator]:
    """Get current distributed clustering coordinator"""
    return distributed_coordinator