"""
Generation 3 Horizontal Scaling Coordination
Manages distributed clustering across multiple nodes with Redis-based coordination,
load balancing, and automatic scaling
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import psutil
import numpy as np
import pandas as pd

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Distributed task status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a cluster node"""
    node_id: str
    host: str
    port: int
    status: NodeStatus
    capacity: int
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_available: bool = False
    capabilities: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    @property
    def load_percentage(self) -> float:
        return (self.current_load / self.capacity * 100) if self.capacity > 0 else 0
    
    @property
    def is_healthy(self) -> bool:
        return (self.status in [NodeStatus.READY, NodeStatus.BUSY] and
                time.time() - self.last_heartbeat < 60)  # 60 second timeout
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class DistributedTask:
    """Distributed clustering task"""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    priority: int = 5
    estimated_duration: float = 60.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    @property
    def processing_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def wait_time(self) -> Optional[float]:
        if self.started_at:
            return self.started_at - self.created_at
        return None


class RedisCoordinator:
    """Redis-based coordination for distributed clustering"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379,
                 redis_db: int = 0, key_prefix: str = 'neuromorphic_cluster:'):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install redis-py package.")
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.key_prefix = key_prefix
        
        # Redis clients
        self.sync_client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db,
            decode_responses=True, socket_keepalive=True
        )
        
        # Test connection
        try:
            self.sync_client.ping()
            logger.info(f"Connected to Redis coordinator at {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis coordinator: {e}")
            raise
    
    def _make_key(self, suffix: str) -> str:
        """Create Redis key with prefix"""
        return f"{self.key_prefix}{suffix}"
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a node in the cluster"""
        try:
            key = self._make_key(f"nodes:{node_info.node_id}")
            data = {
                'node_id': node_info.node_id,
                'host': node_info.host,
                'port': node_info.port,
                'status': node_info.status.value,
                'capacity': node_info.capacity,
                'current_load': node_info.current_load,
                'last_heartbeat': node_info.last_heartbeat,
                'cpu_percent': node_info.cpu_percent,
                'memory_percent': node_info.memory_percent,
                'gpu_available': node_info.gpu_available,
                'capabilities': json.dumps(node_info.capabilities),
                'version': node_info.version
            }
            
            self.sync_client.hset(key, mapping=data)
            self.sync_client.expire(key, 300)  # 5 minute expiration
            
            # Add to active nodes set
            self.sync_client.sadd(self._make_key("active_nodes"), node_info.node_id)
            
            logger.info(f"Registered node {node_info.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_info.node_id}: {e}")
            return False
    
    def update_node_heartbeat(self, node_id: str, status: NodeStatus,
                            cpu_percent: float, memory_percent: float,
                            current_load: int) -> bool:
        """Update node heartbeat and status"""
        try:
            key = self._make_key(f"nodes:{node_id}")
            
            updates = {
                'status': status.value,
                'last_heartbeat': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'current_load': current_load
            }
            
            self.sync_client.hset(key, mapping=updates)
            self.sync_client.expire(key, 300)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for node {node_id}: {e}")
            return False
    
    def get_active_nodes(self) -> List[NodeInfo]:
        """Get list of active nodes"""
        try:
            node_ids = self.sync_client.smembers(self._make_key("active_nodes"))
            active_nodes = []
            
            for node_id in node_ids:
                key = self._make_key(f"nodes:{node_id}")
                node_data = self.sync_client.hgetall(key)
                
                if node_data:
                    try:
                        node_info = NodeInfo(
                            node_id=node_data['node_id'],
                            host=node_data['host'],
                            port=int(node_data['port']),
                            status=NodeStatus(node_data['status']),
                            capacity=int(node_data['capacity']),
                            current_load=int(node_data['current_load']),
                            last_heartbeat=float(node_data['last_heartbeat']),
                            cpu_percent=float(node_data['cpu_percent']),
                            memory_percent=float(node_data['memory_percent']),
                            gpu_available=bool(node_data.get('gpu_available', False)),
                            capabilities=json.loads(node_data.get('capabilities', '[]')),
                            version=node_data.get('version', '1.0.0')
                        )
                        
                        if node_info.is_healthy:
                            active_nodes.append(node_info)
                        else:
                            # Remove unhealthy node
                            self.sync_client.srem(self._make_key("active_nodes"), node_id)
                            self.sync_client.delete(key)
                            
                    except Exception as e:
                        logger.warning(f"Invalid node data for {node_id}: {e}")
            
            return active_nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    def submit_task(self, task: DistributedTask) -> bool:
        """Submit a task to the distributed queue"""
        try:
            task_key = self._make_key(f"tasks:{task.task_id}")
            task_data = {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'data': json.dumps(task.data),
                'priority': task.priority,
                'estimated_duration': task.estimated_duration,
                'resource_requirements': json.dumps(task.resource_requirements),
                'status': task.status.value,
                'created_at': task.created_at,
                'retries': task.retries,
                'max_retries': task.max_retries
            }
            
            # Store task data
            self.sync_client.hset(task_key, mapping=task_data)
            self.sync_client.expire(task_key, 3600)  # 1 hour expiration
            
            # Add to priority queue
            priority_score = -task.priority  # Higher priority = lower score
            self.sync_client.zadd(
                self._make_key("task_queue"), 
                {task.task_id: priority_score}
            )
            
            logger.info(f"Submitted task {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def get_next_task(self, node_id: str, 
                     capabilities: Optional[List[str]] = None) -> Optional[DistributedTask]:
        """Get next available task for a node"""
        try:
            # Get highest priority task
            task_ids = self.sync_client.zrange(self._make_key("task_queue"), 0, 0)
            
            if not task_ids:
                return None
            
            task_id = task_ids[0]
            task_key = self._make_key(f"tasks:{task_id}")
            task_data = self.sync_client.hgetall(task_key)
            
            if not task_data:
                # Task no longer exists, remove from queue
                self.sync_client.zrem(self._make_key("task_queue"), task_id)
                return None
            
            # Create task object
            task = DistributedTask(
                task_id=task_data['task_id'],
                task_type=task_data['task_type'],
                data=json.loads(task_data['data']),
                priority=int(task_data['priority']),
                estimated_duration=float(task_data['estimated_duration']),
                resource_requirements=json.loads(task_data['resource_requirements']),
                status=TaskStatus(task_data['status']),
                retries=int(task_data['retries']),
                max_retries=int(task_data['max_retries']),
                created_at=float(task_data['created_at'])
            )
            
            # Check if node can handle this task
            if capabilities:
                required_caps = task.resource_requirements.get('capabilities', [])
                if not all(cap in capabilities for cap in required_caps):
                    return None  # Node doesn't have required capabilities
            
            # Assign task to node
            task.assigned_node = node_id
            task.status = TaskStatus.ASSIGNED
            task.started_at = time.time()
            
            # Update task in Redis
            updates = {
                'assigned_node': node_id,
                'status': task.status.value,
                'started_at': task.started_at
            }
            self.sync_client.hset(task_key, mapping=updates)
            
            # Remove from queue and add to assigned tasks
            self.sync_client.zrem(self._make_key("task_queue"), task_id)
            self.sync_client.sadd(self._make_key(f"assigned_tasks:{node_id}"), task_id)
            
            logger.info(f"Assigned task {task_id} to node {node_id}")
            return task
            
        except Exception as e:
            logger.error(f"Failed to get next task for node {node_id}: {e}")
            return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark task as completed"""
        try:
            task_key = self._make_key(f"tasks:{task_id}")
            
            updates = {
                'status': TaskStatus.COMPLETED.value,
                'completed_at': time.time(),
                'result': json.dumps(result)
            }
            
            self.sync_client.hset(task_key, mapping=updates)
            
            # Remove from assigned tasks
            task_data = self.sync_client.hgetall(task_key)
            if task_data and task_data.get('assigned_node'):
                self.sync_client.srem(
                    self._make_key(f"assigned_tasks:{task_data['assigned_node']}"), 
                    task_id
                )
            
            logger.info(f"Completed task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed and potentially retry"""
        try:
            task_key = self._make_key(f"tasks:{task_id}")
            task_data = self.sync_client.hgetall(task_key)
            
            if not task_data:
                return False
            
            retries = int(task_data.get('retries', 0))
            max_retries = int(task_data.get('max_retries', 3))
            
            if retries < max_retries:
                # Retry task
                updates = {
                    'status': TaskStatus.PENDING.value,
                    'assigned_node': '',
                    'retries': retries + 1,
                    'error': error
                }
                
                self.sync_client.hset(task_key, mapping=updates)
                
                # Add back to queue with lower priority
                priority = int(task_data.get('priority', 5))
                self.sync_client.zadd(
                    self._make_key("task_queue"), 
                    {task_id: -(priority - 1)}  # Lower priority for retries
                )
                
                logger.warning(f"Retrying task {task_id} (attempt {retries + 1})")
            else:
                # Mark as permanently failed
                updates = {
                    'status': TaskStatus.FAILED.value,
                    'completed_at': time.time(),
                    'error': error
                }
                
                self.sync_client.hset(task_key, mapping=updates)
                logger.error(f"Task {task_id} permanently failed: {error}")
            
            # Remove from assigned tasks
            if task_data.get('assigned_node'):
                self.sync_client.srem(
                    self._make_key(f"assigned_tasks:{task_data['assigned_node']}"), 
                    task_id
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle task failure {task_id}: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get current status of a task"""
        try:
            task_key = self._make_key(f"tasks:{task_id}")
            task_data = self.sync_client.hgetall(task_key)
            
            if not task_data:
                return None
            
            task = DistributedTask(
                task_id=task_data['task_id'],
                task_type=task_data['task_type'],
                data=json.loads(task_data['data']),
                priority=int(task_data['priority']),
                estimated_duration=float(task_data['estimated_duration']),
                resource_requirements=json.loads(task_data['resource_requirements']),
                status=TaskStatus(task_data['status']),
                assigned_node=task_data.get('assigned_node') or None,
                created_at=float(task_data['created_at']),
                retries=int(task_data['retries']),
                max_retries=int(task_data['max_retries'])
            )
            
            if task_data.get('started_at'):
                task.started_at = float(task_data['started_at'])
            
            if task_data.get('completed_at'):
                task.completed_at = float(task_data['completed_at'])
            
            if task_data.get('result'):
                task.result = json.loads(task_data['result'])
            
            if task_data.get('error'):
                task.error = task_data['error']
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to get task status {task_id}: {e}")
            return None


class LoadBalancer:
    """Intelligent load balancer for distributed clustering"""
    
    def __init__(self, coordinator: RedisCoordinator):
        self.coordinator = coordinator
        self.balancing_strategy = "least_loaded"  # or "round_robin", "weighted"
    
    def select_node(self, task: DistributedTask) -> Optional[NodeInfo]:
        """Select best node for a task"""
        active_nodes = self.coordinator.get_active_nodes()
        
        if not active_nodes:
            return None
        
        # Filter nodes by capabilities
        required_capabilities = task.resource_requirements.get('capabilities', [])
        suitable_nodes = [
            node for node in active_nodes
            if all(cap in node.capabilities for cap in required_capabilities)
            and node.status in [NodeStatus.READY, NodeStatus.BUSY]
            and node.load_percentage < 90  # Don't overload nodes
        ]
        
        if not suitable_nodes:
            return None
        
        # Apply balancing strategy
        if self.balancing_strategy == "least_loaded":
            return min(suitable_nodes, key=lambda n: n.load_percentage)
        elif self.balancing_strategy == "round_robin":
            # Simple round-robin (not persistent across restarts)
            return suitable_nodes[hash(task.task_id) % len(suitable_nodes)]
        else:
            # Default to least loaded
            return min(suitable_nodes, key=lambda n: n.load_percentage)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        active_nodes = self.coordinator.get_active_nodes()
        
        if not active_nodes:
            return {
                'total_nodes': 0,
                'healthy_nodes': 0,
                'total_capacity': 0,
                'total_load': 0,
                'avg_cpu_percent': 0.0,
                'avg_memory_percent': 0.0,
                'gpu_nodes': 0
            }
        
        total_capacity = sum(node.capacity for node in active_nodes)
        total_load = sum(node.current_load for node in active_nodes)
        avg_cpu = sum(node.cpu_percent for node in active_nodes) / len(active_nodes)
        avg_memory = sum(node.memory_percent for node in active_nodes) / len(active_nodes)
        gpu_nodes = sum(1 for node in active_nodes if node.gpu_available)
        
        return {
            'total_nodes': len(active_nodes),
            'healthy_nodes': len(active_nodes),  # Already filtered
            'total_capacity': total_capacity,
            'total_load': total_load,
            'load_percentage': (total_load / total_capacity * 100) if total_capacity > 0 else 0,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'gpu_nodes': gpu_nodes,
            'nodes_by_status': {
                status.value: len([n for n in active_nodes if n.status == status])
                for status in NodeStatus
            }
        }


class DistributedClusteringManager:
    """Main manager for distributed neuromorphic clustering"""
    
    def __init__(self, node_id: Optional[str] = None,
                 host: str = 'localhost',
                 port: int = 8000,
                 capacity: int = 10,
                 redis_config: Optional[Dict] = None):
        
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.host = host
        self.port = port
        self.capacity = capacity
        
        # Initialize Redis coordinator
        redis_config = redis_config or {}
        self.coordinator = RedisCoordinator(**redis_config)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(self.coordinator)
        
        # Node information
        self.node_info = NodeInfo(
            node_id=self.node_id,
            host=host,
            port=port,
            status=NodeStatus.INITIALIZING,
            capacity=capacity,
            gpu_available=self._detect_gpu(),
            capabilities=self._get_capabilities()
        )
        
        # Task management
        self.current_tasks = {}
        self.task_executor = ThreadPoolExecutor(max_workers=capacity)
        self.running = False
        
        # Heartbeat thread
        self.heartbeat_thread = None
        self.heartbeat_interval = 30  # seconds
        
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available"""
        try:
            from .gpu_acceleration import gpu_ops
            return gpu_ops.gpu_manager.gpu_available
        except:
            return False
    
    def _get_capabilities(self) -> List[str]:
        """Get node capabilities"""
        capabilities = ['clustering', 'neuromorphic']
        
        if self._detect_gpu():
            capabilities.extend(['gpu_acceleration', 'cuda'])
        
        # Add other capabilities based on available packages
        try:
            import cupy
            capabilities.append('cupy')
        except ImportError:
            pass
        
        try:
            import redis
            capabilities.append('redis')
        except ImportError:
            pass
        
        return capabilities
    
    def start(self):
        """Start the distributed clustering manager"""
        logger.info(f"Starting distributed clustering manager: {self.node_id}")
        
        # Register node
        self.node_info.status = NodeStatus.READY
        if not self.coordinator.register_node(self.node_info):
            logger.error("Failed to register node with coordinator")
            return False
        
        # Start heartbeat thread
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        # Start task processing
        self.task_executor.submit(self._task_processing_loop)
        
        logger.info(f"Node {self.node_id} started successfully")
        return True
    
    def stop(self):
        """Stop the distributed clustering manager"""
        logger.info(f"Stopping node {self.node_id}")
        
        self.running = False
        
        # Update status to maintenance
        self.node_info.status = NodeStatus.MAINTENANCE
        self.coordinator.update_node_heartbeat(
            self.node_id, self.node_info.status,
            0.0, 0.0, 0
        )
        
        # Shutdown executor
        self.task_executor.shutdown(wait=True)
        
        logger.info(f"Node {self.node_id} stopped")
    
    def _heartbeat_loop(self):
        """Heartbeat loop to maintain node presence"""
        while self.running:
            try:
                # Update system metrics
                self.node_info.cpu_percent = psutil.cpu_percent()
                self.node_info.memory_percent = psutil.virtual_memory().percent
                self.node_info.current_load = len(self.current_tasks)
                self.node_info.last_heartbeat = time.time()
                
                # Determine status based on load
                if self.node_info.load_percentage >= 90:
                    self.node_info.status = NodeStatus.OVERLOADED
                elif self.node_info.load_percentage >= 70:
                    self.node_info.status = NodeStatus.BUSY
                else:
                    self.node_info.status = NodeStatus.READY
                
                # Send heartbeat
                self.coordinator.update_node_heartbeat(
                    self.node_id,
                    self.node_info.status,
                    self.node_info.cpu_percent,
                    self.node_info.memory_percent,
                    self.node_info.current_load
                )
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _task_processing_loop(self):
        """Main task processing loop"""
        while self.running:
            try:
                # Get next task
                task = self.coordinator.get_next_task(
                    self.node_id, 
                    self.node_info.capabilities
                )
                
                if task:
                    # Process task asynchronously
                    future = self.task_executor.submit(self._process_task, task)
                    self.current_tasks[task.task_id] = future
                    
                    # Clean up completed tasks
                    self._cleanup_completed_tasks()
                else:
                    # No tasks available, wait a bit
                    time.sleep(5)
                    
            except Exception as e:
                logger.error(f"Task processing loop error: {e}")
                time.sleep(10)
    
    def _process_task(self, task: DistributedTask) -> bool:
        """Process a single distributed task"""
        logger.info(f"Processing task {task.task_id} of type {task.task_type}")
        
        try:
            task.status = TaskStatus.PROCESSING
            
            # Process based on task type
            if task.task_type == 'neuromorphic_clustering':
                result = self._process_neuromorphic_clustering(task.data)
            elif task.task_type == 'feature_extraction':
                result = self._process_feature_extraction(task.data)
            elif task.task_type == 'ensemble_clustering':
                result = self._process_ensemble_clustering(task.data)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Mark task as completed
            self.coordinator.complete_task(task.task_id, result)
            
            logger.info(f"Completed task {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            self.coordinator.fail_task(task.task_id, str(e))
            return False
        
        finally:
            # Remove from current tasks
            self.current_tasks.pop(task.task_id, None)
    
    def _process_neuromorphic_clustering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process neuromorphic clustering task"""
        from .neuromorphic_clustering import NeuromorphicClusterer
        
        # Reconstruct DataFrame from data
        features_data = data['features']
        if isinstance(features_data, dict) and 'data' in features_data:
            features = pd.DataFrame.from_dict(features_data['data'])
        else:
            features = pd.DataFrame(features_data)
        
        # Create and fit clusterer
        clusterer = NeuromorphicClusterer(
            n_clusters=data.get('n_clusters', 4),
            method=data.get('method', 'hybrid_reservoir')
        )
        
        clusterer.fit(features)
        
        # Return results
        return {
            'cluster_assignments': clusterer.get_cluster_assignments().tolist(),
            'cluster_interpretations': clusterer.get_cluster_interpretation(),
            'metrics': clusterer.get_clustering_metrics().__dict__,
            'fallback_used': clusterer.fallback_used
        }
    
    def _process_feature_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feature extraction task"""
        from .gpu_acceleration import gpu_ops
        
        # Extract features using GPU acceleration if available
        states = np.array(data['states'])
        extraction_type = data.get('extraction_type', 'statistical')
        
        features = gpu_ops.accelerated_feature_extraction(states, extraction_type)
        
        return {
            'features': features.tolist(),
            'feature_count': len(features),
            'extraction_type': extraction_type
        }
    
    def _process_ensemble_clustering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ensemble clustering task"""
        # Combine multiple clustering results
        clustering_results = data['clustering_results']
        
        # Simple ensemble: majority voting
        n_samples = len(clustering_results[0])
        ensemble_labels = []
        
        for i in range(n_samples):
            sample_votes = [result[i] for result in clustering_results]
            # Take most common label
            ensemble_label = max(set(sample_votes), key=sample_votes.count)
            ensemble_labels.append(ensemble_label)
        
        return {
            'ensemble_labels': ensemble_labels,
            'n_clusterers': len(clustering_results),
            'consensus_strength': sum(
                sample_votes.count(max(set(sample_votes), key=sample_votes.count)) / len(sample_votes)
                for sample_votes in [
                    [result[i] for result in clustering_results]
                    for i in range(n_samples)
                ]
            ) / n_samples
        }
    
    def _cleanup_completed_tasks(self):
        """Clean up completed task futures"""
        completed_tasks = []
        for task_id, future in self.current_tasks.items():
            if future.done():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            self.current_tasks.pop(task_id, None)
    
    def submit_clustering_task(self, features: pd.DataFrame,
                             n_clusters: int = 4,
                             method: str = 'hybrid_reservoir',
                             priority: int = 5) -> str:
        """Submit a neuromorphic clustering task"""
        task_id = f"clustering_{uuid.uuid4().hex[:12]}"
        
        # Prepare data
        task_data = {
            'features': features.to_dict(),
            'n_clusters': n_clusters,
            'method': method
        }
        
        # Create task
        task = DistributedTask(
            task_id=task_id,
            task_type='neuromorphic_clustering',
            data=task_data,
            priority=priority,
            estimated_duration=60.0,  # Estimate based on data size
            resource_requirements={
                'capabilities': ['clustering', 'neuromorphic']
            }
        )
        
        # Submit to coordinator
        if self.coordinator.submit_task(task):
            logger.info(f"Submitted clustering task {task_id}")
            return task_id
        else:
            raise RuntimeError(f"Failed to submit task {task_id}")
    
    def get_task_result(self, task_id: str, timeout: float = 300) -> Optional[Dict[str, Any]]:
        """Wait for and get task result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.coordinator.get_task_status(task_id)
            
            if not task:
                logger.warning(f"Task {task_id} not found")
                return None
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task {task_id} failed: {task.error}")
            
            time.sleep(2)  # Poll every 2 seconds
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        return {
            'node_info': {
                'node_id': self.node_id,
                'status': self.node_info.status.value,
                'load_percentage': self.node_info.load_percentage,
                'capabilities': self.node_info.capabilities,
                'current_tasks': len(self.current_tasks)
            },
            'cluster_stats': self.load_balancer.get_cluster_stats()
        }


# Global distributed manager instance (initialized when needed)
distributed_manager: Optional[DistributedClusteringManager] = None


def initialize_distributed_clustering(node_id: Optional[str] = None,
                                     host: str = 'localhost',
                                     port: int = 8000,
                                     capacity: int = 10,
                                     redis_config: Optional[Dict] = None) -> DistributedClusteringManager:
    """Initialize global distributed clustering manager"""
    global distributed_manager
    
    distributed_manager = DistributedClusteringManager(
        node_id=node_id,
        host=host,
        port=port,
        capacity=capacity,
        redis_config=redis_config
    )
    
    return distributed_manager


def get_distributed_manager() -> Optional[DistributedClusteringManager]:
    """Get current distributed clustering manager"""
    return distributed_manager