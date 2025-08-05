"""
Scalability enhancements for Observer Coordinator Insights
Auto-scaling, load balancing, and distributed processing capabilities
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue, Empty
import threading
import psutil
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Different processing modes for scalability"""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED = "distributed"
    AUTO_SCALE = "auto_scale"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions"""
    cpu_usage: float
    memory_usage: float
    queue_size: int
    processing_rate: float
    response_time: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Represents a worker node in distributed processing"""
    node_id: str
    endpoint: str
    capacity: int
    current_load: int = 0
    health_status: str = "healthy"
    last_heartbeat: float = field(default_factory=time.time)


class AutoScaler:
    """Automatic scaling controller"""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 10,
                 scale_up_threshold: float = 80.0, scale_down_threshold: float = 30.0):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.current_workers = min_workers
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_lock = threading.Lock()
        self.last_scaling_action = 0
        self.cooling_period = 60  # seconds
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add new metrics for scaling decisions"""
        self.metrics_history.append(metrics)
        # Keep only recent metrics (last 10 minutes)
        cutoff_time = time.time() - 600
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed"""
        if not self.metrics_history or self.current_workers >= self.max_workers:
            return False
        
        recent_metrics = self.metrics_history[-5:]  # Last 5 data points
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_queue = np.mean([m.queue_size for m in recent_metrics])
        avg_response_time = np.mean([m.response_time for m in recent_metrics])
        
        return (avg_cpu > self.scale_up_threshold or 
                avg_queue > 10 or 
                avg_response_time > 5.0)
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is possible"""
        if not self.metrics_history or self.current_workers <= self.min_workers:
            return False
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 data points for stability
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_queue = np.mean([m.queue_size for m in recent_metrics])
        
        return avg_cpu < self.scale_down_threshold and avg_queue < 2
    
    def scale_workers(self) -> Optional[str]:
        """Execute scaling action if needed"""
        with self.scaling_lock:
            current_time = time.time()
            if current_time - self.last_scaling_action < self.cooling_period:
                return None
            
            if self.should_scale_up():
                new_workers = min(self.current_workers + 1, self.max_workers)
                if new_workers > self.current_workers:
                    self.current_workers = new_workers
                    self.last_scaling_action = current_time
                    logger.info(f"Scaled up to {self.current_workers} workers")
                    return f"scaled_up_to_{self.current_workers}"
            
            elif self.should_scale_down():
                new_workers = max(self.current_workers - 1, self.min_workers)
                if new_workers < self.current_workers:
                    self.current_workers = new_workers
                    self.last_scaling_action = current_time
                    logger.info(f"Scaled down to {self.current_workers} workers")
                    return f"scaled_down_to_{self.current_workers}"
        
        return None


class LoadBalancer:
    """Load balancer for distributing work across workers"""
    
    def __init__(self):
        self.workers: List[WorkerNode] = []
        self.round_robin_index = 0
        self.health_check_interval = 30
        self.last_health_check = 0
    
    def add_worker(self, node: WorkerNode):
        """Add a worker node"""
        self.workers.append(node)
        logger.info(f"Added worker node: {node.node_id}")
    
    def remove_worker(self, node_id: str):
        """Remove a worker node"""
        self.workers = [w for w in self.workers if w.node_id != node_id]
        logger.info(f"Removed worker node: {node_id}")
    
    def get_least_loaded_worker(self) -> Optional[WorkerNode]:
        """Get worker with least current load"""
        healthy_workers = [w for w in self.workers if w.health_status == "healthy"]
        if not healthy_workers:
            return None
        
        return min(healthy_workers, key=lambda w: w.current_load / w.capacity)
    
    def get_next_worker_round_robin(self) -> Optional[WorkerNode]:
        """Get next worker using round-robin strategy"""
        healthy_workers = [w for w in self.workers if w.health_status == "healthy"]
        if not healthy_workers:
            return None
        
        worker = healthy_workers[self.round_robin_index % len(healthy_workers)]
        self.round_robin_index += 1
        return worker
    
    def assign_work(self, work_item: Dict[str, Any], strategy: str = "least_loaded") -> Optional[WorkerNode]:
        """Assign work to a worker using specified strategy"""
        if strategy == "least_loaded":
            worker = self.get_least_loaded_worker()
        else:  # round_robin
            worker = self.get_next_worker_round_robin()
        
        if worker:
            worker.current_load += 1
            logger.debug(f"Assigned work to {worker.node_id}, load: {worker.current_load}/{worker.capacity}")
        
        return worker
    
    def complete_work(self, node_id: str):
        """Mark work as completed for a worker"""
        for worker in self.workers:
            if worker.node_id == node_id and worker.current_load > 0:
                worker.current_load -= 1
                break
    
    async def health_check_workers(self):
        """Perform health checks on all workers"""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        
        async with aiohttp.ClientSession() as session:
            for worker in self.workers:
                try:
                    async with session.get(f"{worker.endpoint}/health", timeout=5) as response:
                        if response.status == 200:
                            worker.health_status = "healthy"
                            worker.last_heartbeat = current_time
                        else:
                            worker.health_status = "unhealthy"
                except Exception as e:
                    logger.warning(f"Health check failed for {worker.node_id}: {e}")
                    worker.health_status = "unhealthy"


class DistributedTaskQueue:
    """Distributed task queue with persistence and reliability"""
    
    def __init__(self, queue_dir: Optional[Path] = None):
        self.queue_dir = queue_dir or Path.home() / '.observer_coordinator_queue'
        self.queue_dir.mkdir(exist_ok=True)
        self.pending_queue = Queue()
        self.processing_queue = Queue()
        self.completed_queue = Queue()
        self.failed_queue = Queue()
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def generate_task_id(self, task_data: Dict[str, Any]) -> str:
        """Generate unique task ID"""
        task_str = json.dumps(task_data, sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()[:16]
    
    def submit_task(self, task_type: str, task_data: Dict[str, Any], 
                   priority: int = 5) -> str:
        """Submit a task to the queue"""
        task_id = self.generate_task_id(task_data)
        
        task = {
            'task_id': task_id,
            'task_type': task_type,
            'task_data': task_data,
            'priority': priority,
            'submitted_at': time.time(),
            'status': 'pending',
            'retries': 0,
            'max_retries': 3
        }
        
        with self.lock:
            self.task_registry[task_id] = task
            self.pending_queue.put(task)
        
        # Persist task to disk
        self._persist_task(task)
        
        logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id
    
    def get_next_task(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get next task from the queue"""
        try:
            task = self.pending_queue.get(timeout=timeout)
            task['status'] = 'processing'
            task['started_at'] = time.time()
            
            with self.lock:
                self.task_registry[task['task_id']] = task
                self.processing_queue.put(task)
            
            return task
        except Empty:
            return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed"""
        with self.lock:
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                task['status'] = 'completed'
                task['completed_at'] = time.time()
                task['result'] = result
                self.completed_queue.put(task)
                
                # Remove from processing
                self._remove_from_processing_queue(task_id)
                
                logger.info(f"Completed task {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        with self.lock:
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                task['retries'] += 1
                task['last_error'] = error
                task['failed_at'] = time.time()
                
                if task['retries'] < task['max_retries']:
                    # Retry the task
                    task['status'] = 'pending'
                    self.pending_queue.put(task)
                    logger.warning(f"Retrying task {task_id} (attempt {task['retries']})")
                else:
                    # Mark as permanently failed
                    task['status'] = 'failed'
                    self.failed_queue.put(task)
                    logger.error(f"Task {task_id} permanently failed: {error}")
                
                self._remove_from_processing_queue(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        with self.lock:
            return self.task_registry.get(task_id)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                'pending': self.pending_queue.qsize(),
                'processing': self.processing_queue.qsize(),
                'completed': self.completed_queue.qsize(),
                'failed': self.failed_queue.qsize(),
                'total_tasks': len(self.task_registry)
            }
    
    def _persist_task(self, task: Dict[str, Any]):
        """Persist task to disk for reliability"""
        task_file = self.queue_dir / f"{task['task_id']}.json"
        try:
            with open(task_file, 'w') as f:
                json.dump(task, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to persist task {task['task_id']}: {e}")
    
    def _remove_from_processing_queue(self, task_id: str):
        """Remove task from processing queue"""
        # Note: Queue doesn't support direct removal, so we use a registry approach
        pass
    
    def restore_tasks_from_disk(self):
        """Restore tasks from disk on startup"""
        for task_file in self.queue_dir.glob('*.json'):
            try:
                with open(task_file, 'r') as f:
                    task = json.load(f)
                
                if task['status'] in ['pending', 'processing']:
                    task['status'] = 'pending'  # Reset processing tasks to pending
                    self.task_registry[task['task_id']] = task
                    self.pending_queue.put(task)
                    logger.info(f"Restored task {task['task_id']} from disk")
                
            except Exception as e:
                logger.warning(f"Failed to restore task from {task_file}: {e}")


class ScalableClusteringEngine:
    """Scalable clustering engine with auto-scaling and load balancing"""
    
    def __init__(self):
        self.autoscaler = AutoScaler()
        self.load_balancer = LoadBalancer()
        self.task_queue = DistributedTaskQueue()
        self.worker_pool = None
        self.processing_active = False
        
        # Restore any persisted tasks
        self.task_queue.restore_tasks_from_disk()
    
    def add_worker_node(self, endpoint: str, capacity: int = 10) -> str:
        """Add a distributed worker node"""
        node_id = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        worker = WorkerNode(node_id, endpoint, capacity)
        self.load_balancer.add_worker(worker)
        return node_id
    
    def submit_clustering_job(self, features: pd.DataFrame, n_clusters: int,
                            priority: int = 5) -> str:
        """Submit a clustering job"""
        # Convert DataFrame to serializable format
        features_dict = {
            'data': features.to_dict(),
            'shape': features.shape,
            'columns': features.columns.tolist()
        }
        
        task_data = {
            'features': features_dict,
            'n_clusters': n_clusters,
            'algorithm': 'kmeans'
        }
        
        return self.task_queue.submit_task('clustering', task_data, priority)
    
    def submit_team_generation_job(self, employee_data: pd.DataFrame,
                                 cluster_assignments: np.ndarray,
                                 num_teams: int, priority: int = 5) -> str:
        """Submit a team generation job"""
        task_data = {
            'employee_data': employee_data.to_dict(),
            'cluster_assignments': cluster_assignments.tolist(),
            'num_teams': num_teams
        }
        
        return self.task_queue.submit_task('team_generation', task_data, priority)
    
    def start_processing(self):
        """Start the processing workers"""
        self.processing_active = True
        
        # Start worker threads
        num_workers = self.autoscaler.current_workers
        self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        # Start worker threads
        for i in range(num_workers):
            self.worker_pool.submit(self._worker_loop, f"worker_{i}")
        
        # Start monitoring thread
        self.worker_pool.submit(self._monitoring_loop)
        
        logger.info(f"Started processing with {num_workers} workers")
    
    def stop_processing(self):
        """Stop the processing workers"""
        self.processing_active = False
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        logger.info("Stopped processing")
    
    def get_job_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a submitted job"""
        return self.task_queue.get_task_status(task_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'autoscaler': {
                'current_workers': self.autoscaler.current_workers,
                'min_workers': self.autoscaler.min_workers,
                'max_workers': self.autoscaler.max_workers
            },
            'load_balancer': {
                'total_workers': len(self.load_balancer.workers),
                'healthy_workers': len([w for w in self.load_balancer.workers if w.health_status == "healthy"])
            },
            'task_queue': self.task_queue.get_queue_stats(),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        }
    
    def _worker_loop(self, worker_name: str):
        """Main worker loop"""
        logger.info(f"Started worker: {worker_name}")
        
        while self.processing_active:
            try:
                task = self.task_queue.get_next_task(timeout=1.0)
                if task is None:
                    continue
                
                start_time = time.time()
                result = self._process_task(task)
                processing_time = time.time() - start_time
                
                if result['success']:
                    self.task_queue.complete_task(task['task_id'], result)
                else:
                    self.task_queue.fail_task(task['task_id'], result['error'])
                
                # Update metrics
                metrics = ScalingMetrics(
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    queue_size=self.task_queue.pending_queue.qsize(),
                    processing_rate=1.0 / processing_time if processing_time > 0 else 0,
                    response_time=processing_time,
                    error_rate=0 if result['success'] else 1
                )
                self.autoscaler.add_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Stopped worker: {worker_name}")
    
    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task"""
        task_type = task['task_type']
        task_data = task['task_data']
        
        try:
            if task_type == 'clustering':
                return self._process_clustering_task(task_data)
            elif task_type == 'team_generation':
                return self._process_team_generation_task(task_data)
            else:
                return {'success': False, 'error': f'Unknown task type: {task_type}'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_clustering_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process clustering task"""
        from .insights_clustering import KMeansClusterer
        
        # Reconstruct DataFrame from serialized data
        features_dict = task_data['features']
        features = pd.DataFrame.from_dict(features_dict['data'])
        
        clusterer = KMeansClusterer(n_clusters=task_data['n_clusters'])
        clusterer.fit(features)
        
        return {
            'success': True,
            'assignments': clusterer.get_cluster_assignments().tolist(),
            'centroids': clusterer.get_cluster_centroids().to_dict(),
            'metrics': clusterer.get_cluster_quality_metrics()
        }
    
    def _process_team_generation_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process team generation task"""
        from .team_simulator import TeamCompositionSimulator
        
        employee_data = pd.DataFrame.from_dict(task_data['employee_data'])
        cluster_assignments = np.array(task_data['cluster_assignments'])
        
        simulator = TeamCompositionSimulator()
        simulator.load_employee_data(employee_data, cluster_assignments)
        compositions = simulator.recommend_optimal_teams(task_data['num_teams'], iterations=3)
        
        return {
            'success': True,
            'compositions': compositions
        }
    
    def _monitoring_loop(self):
        """Monitoring and auto-scaling loop"""
        while self.processing_active:
            try:
                # Perform auto-scaling
                scaling_action = self.autoscaler.scale_workers()
                if scaling_action:
                    # Restart worker pool with new worker count
                    # In a real implementation, this would be more sophisticated
                    logger.info(f"Auto-scaling triggered: {scaling_action}")
                
                # Health check workers (if distributed)
                asyncio.run(self.load_balancer.health_check_workers())
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
        
        logger.info("Stopped monitoring loop")


# Global scalable engine instance
scalable_engine = ScalableClusteringEngine()