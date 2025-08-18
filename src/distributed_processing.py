#!/usr/bin/env python3
"""Distributed Processing Engine
High-performance distributed computation for large-scale organizational analytics
"""

import hashlib
import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Distributed processing task"""
    task_id: str
    task_type: str
    data: Any
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: float = 0

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()


@dataclass
class ProcessingResult:
    """Processing result with metadata"""
    task_id: str
    result: Any
    processing_time: float
    worker_id: str
    success: bool
    error: Optional[str] = None


class DistributedProcessor:
    """High-performance distributed processing engine"""

    def __init__(self, max_workers: int = None, use_processes: bool = True):
        self.max_workers = max_workers or min(32, (mp.cpu_count() * 2))
        self.use_processes = use_processes
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        self.workers = []
        self.running = False
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "peak_queue_size": 0,
            "workers_active": 0
        }
        self._lock = threading.Lock()

    def start(self):
        """Start the distributed processing engine"""
        if self.running:
            return

        self.running = True

        # Start worker processes/threads
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        self.executor = executor_class(max_workers=self.max_workers)

        # Start result collector thread
        self.result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.result_thread.start()

        logger.info(f"Started distributed processor with {self.max_workers} {'processes' if self.use_processes else 'threads'}")

    def stop(self):
        """Stop the distributed processing engine"""
        if not self.running:
            return

        self.running = False

        # Shutdown executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        logger.info("Stopped distributed processing engine")

    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for processing"""
        self.task_queue.put(task)

        with self._lock:
            current_size = self.task_queue.qsize()
            self.stats["peak_queue_size"] = max(self.stats["peak_queue_size"], current_size)

        return task.task_id

    def submit_clustering_task(self, data: pd.DataFrame, n_clusters: int,
                              algorithm: str = "kmeans", **kwargs) -> str:
        """Submit clustering task"""
        task_id = f"cluster_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        task = ProcessingTask(
            task_id=task_id,
            task_type="clustering",
            data=data,
            parameters={
                "n_clusters": n_clusters,
                "algorithm": algorithm,
                **kwargs
            }
        )

        return self.submit_task(task)

    def submit_batch_clustering(self, datasets: List[pd.DataFrame],
                               n_clusters_list: List[int]) -> List[str]:
        """Submit multiple clustering tasks"""
        task_ids = []

        for i, (data, n_clusters) in enumerate(zip(datasets, n_clusters_list)):
            task_id = self.submit_clustering_task(data, n_clusters)
            task_ids.append(task_id)

        return task_ids

    def get_result(self, task_id: str, timeout: float = None) -> Optional[ProcessingResult]:
        """Get result for a specific task"""
        start_time = time.time()

        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result.task_id == task_id:
                    return result
                else:
                    # Put back if not the one we want
                    self.result_queue.put(result)

            except Empty:
                if timeout and (time.time() - start_time) > timeout:
                    break

        return None

    def wait_for_completion(self, task_ids: List[str],
                           timeout: float = None) -> Dict[str, ProcessingResult]:
        """Wait for multiple tasks to complete"""
        results = {}
        start_time = time.time()

        while len(results) < len(task_ids) and self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result.task_id in task_ids:
                    results[result.task_id] = result
                else:
                    # Put back if not one we're waiting for
                    self.result_queue.put(result)

            except Empty:
                if timeout and (time.time() - start_time) > timeout:
                    break

        return results

    def _collect_results(self):
        """Background thread to collect results from executor"""
        futures = {}

        while self.running:
            try:
                # Submit new tasks
                while not self.task_queue.empty():
                    try:
                        task = self.task_queue.get_nowait()
                        future = self.executor.submit(self._process_task, task)
                        futures[future] = task.task_id

                        with self._lock:
                            self.stats["workers_active"] = len(futures)

                    except Empty:
                        break

                # Check completed tasks
                completed = []
                for future in futures:
                    if future.done():
                        completed.append(future)

                for future in completed:
                    task_id = futures.pop(future)
                    try:
                        result = future.result()
                        self.result_queue.put(result)

                        with self._lock:
                            if result.success:
                                self.stats["tasks_completed"] += 1
                            else:
                                self.stats["tasks_failed"] += 1
                            self.stats["total_processing_time"] += result.processing_time

                    except Exception as e:
                        error_result = ProcessingResult(
                            task_id=task_id,
                            result=None,
                            processing_time=0,
                            worker_id="unknown",
                            success=False,
                            error=str(e)
                        )
                        self.result_queue.put(error_result)

                        with self._lock:
                            self.stats["tasks_failed"] += 1

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in result collector: {e}")

    @staticmethod
    def _process_task(task: ProcessingTask) -> ProcessingResult:
        """Process a single task (runs in worker process/thread)"""
        start_time = time.time()
        worker_id = f"{mp.current_process().pid}_{threading.current_thread().ident}"

        try:
            if task.task_type == "clustering":
                result = DistributedProcessor._process_clustering_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            processing_time = time.time() - start_time

            return ProcessingResult(
                task_id=task.task_id,
                result=result,
                processing_time=processing_time,
                worker_id=worker_id,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time

            return ProcessingResult(
                task_id=task.task_id,
                result=None,
                processing_time=processing_time,
                worker_id=worker_id,
                success=False,
                error=str(e)
            )

    @staticmethod
    def _process_clustering_task(task: ProcessingTask) -> Dict[str, Any]:
        """Process clustering task"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import calinski_harabasz_score, silhouette_score

        data = task.data
        params = task.parameters

        # Extract clustering features if needed
        if isinstance(data, pd.DataFrame):
            # Look for energy columns
            energy_cols = [col for col in data.columns if 'energy' in col.lower()]
            if energy_cols:
                features = data[energy_cols].values
            else:
                # Use numeric columns
                features = data.select_dtypes(include=[np.number]).values
        else:
            features = data

        # Perform clustering
        algorithm = params.get("algorithm", "kmeans")
        n_clusters = params["n_clusters"]

        if algorithm == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=params.get("random_state", 42),
                max_iter=params.get("max_iter", 300),
                n_init=params.get("n_init", 10)
            )

            cluster_labels = clusterer.fit_predict(features)
            centroids = clusterer.cluster_centers_

            # Calculate quality metrics
            silhouette = silhouette_score(features, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(features, cluster_labels)

            return {
                "cluster_labels": cluster_labels.tolist(),
                "centroids": centroids.tolist(),
                "silhouette_score": float(silhouette),
                "calinski_harabasz_score": float(calinski_harabasz),
                "inertia": float(clusterer.inertia_)
            }
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self._lock:
            stats = self.stats.copy()

        stats.update({
            "queue_size": self.task_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "running": self.running
        })

        if stats["tasks_completed"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["tasks_completed"]
        else:
            stats["avg_processing_time"] = 0

        return stats


class LoadBalancer:
    """Intelligent load balancer for distributed processing"""

    def __init__(self):
        self.processors: List[DistributedProcessor] = []
        self.current_index = 0
        self._lock = threading.Lock()

    def add_processor(self, processor: DistributedProcessor):
        """Add a processor to the load balancer"""
        with self._lock:
            self.processors.append(processor)

    def get_least_loaded_processor(self) -> Optional[DistributedProcessor]:
        """Get processor with least load"""
        if not self.processors:
            return None

        with self._lock:
            # Simple queue size based load balancing
            min_load = float('inf')
            best_processor = None

            for processor in self.processors:
                if not processor.running:
                    continue

                load = processor.task_queue.qsize()
                if load < min_load:
                    min_load = load
                    best_processor = processor

            return best_processor

    def submit_task(self, task: ProcessingTask) -> Optional[str]:
        """Submit task to least loaded processor"""
        processor = self.get_least_loaded_processor()
        if processor:
            return processor.submit_task(task)
        return None


# Global distributed processing instance
distributed_processor = DistributedProcessor()
load_balancer = LoadBalancer()


def initialize_distributed_processing(max_workers: int = None, use_processes: bool = True):
    """Initialize distributed processing"""
    try:
        global distributed_processor
        distributed_processor = DistributedProcessor(max_workers, use_processes)
        distributed_processor.start()

        # Add to load balancer
        load_balancer.add_processor(distributed_processor)

        logger.info("🚀 Distributed processing engine initialized")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize distributed processing: {e}")
        return False


def shutdown_distributed_processing():
    """Shutdown distributed processing"""
    try:
        distributed_processor.stop()
        logger.info("Distributed processing engine shutdown")
    except Exception as e:
        logger.warning(f"Error shutting down distributed processing: {e}")


# Convenience functions
def process_clustering_distributed(data: pd.DataFrame, n_clusters: int,
                                  **kwargs) -> str:
    """Submit clustering task for distributed processing"""
    return distributed_processor.submit_clustering_task(data, n_clusters, **kwargs)


def get_processing_stats() -> Dict[str, Any]:
    """Get distributed processing statistics"""
    return distributed_processor.get_statistics()
