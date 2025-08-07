"""
Generation 3 GPU Acceleration for Neuromorphic Clustering
Provides CUDA-accelerated operations for Echo State Networks, Spiking Neural Networks,
and Liquid State Machines with automatic fallback to CPU
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from contextlib import contextmanager

try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    from cupyx.scipy.sparse.linalg import spsolve
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import jit, cuda, vectorize, guvectorize
    import numba.types
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_percent: float
    cuda_cores_used: int
    kernel_execution_time_ms: float
    data_transfer_time_ms: float
    speedup_factor: float
    
    @property
    def memory_utilization(self) -> float:
        return self.gpu_memory_used_mb / self.gpu_memory_total_mb if self.gpu_memory_total_mb > 0 else 0.0


class GPUResourceManager:
    """Manages GPU resources and memory"""
    
    def __init__(self):
        self.gpu_available = CUPY_AVAILABLE
        self.device_id = 0
        self.memory_pool = None
        
        if self.gpu_available:
            try:
                cp.cuda.Device(self.device_id).use()
                self.memory_pool = cp.get_default_memory_pool()
                logger.info(f"GPU acceleration enabled on device {self.device_id}")
                
                # Get GPU info
                device = cp.cuda.Device()
                memory_info = cp.cuda.MemoryInfo()
                logger.info(f"GPU: {device.compute_capability}, Memory: {memory_info.total / 1e9:.1f}GB")
                
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.gpu_available = False
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics"""
        if not self.gpu_available:
            return None
        
        try:
            memory_info = cp.cuda.MemoryInfo()
            device = cp.cuda.Device()
            
            return GPUMetrics(
                gpu_memory_used_mb=memory_info.used / (1024**2),
                gpu_memory_total_mb=memory_info.total / (1024**2),
                gpu_utilization_percent=0.0,  # CuPy doesn't provide utilization directly
                cuda_cores_used=0,
                kernel_execution_time_ms=0.0,
                data_transfer_time_ms=0.0,
                speedup_factor=1.0
            )
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return None
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager for GPU memory management"""
        if self.gpu_available and self.memory_pool:
            initial_memory = self.memory_pool.used_bytes()
            
            try:
                yield
            finally:
                # Free unused memory
                self.memory_pool.free_all_blocks()
                final_memory = self.memory_pool.used_bytes()
                if final_memory > initial_memory * 1.5:  # 50% increase threshold
                    logger.debug("Forcing GPU garbage collection")
                    cp._default_memory_pool.free_all_blocks()
        else:
            yield
    
    def transfer_to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Transfer numpy array to GPU"""
        if not self.gpu_available:
            return array
        
        try:
            return cp.asarray(array)
        except Exception as e:
            logger.warning(f"GPU transfer failed: {e}")
            return array
    
    def transfer_to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Transfer array from GPU to CPU"""
        if self.gpu_available and hasattr(array, 'get'):
            try:
                return array.get()
            except:
                pass
        
        return np.asarray(array)


class GPUAcceleratedOperations:
    """GPU-accelerated mathematical operations for neuromorphic computing"""
    
    def __init__(self):
        self.gpu_manager = GPUResourceManager()
        self.performance_cache = {}
        
        # Compile CUDA kernels if available
        if NUMBA_AVAILABLE and self.gpu_manager.gpu_available:
            self._compile_cuda_kernels()
    
    def _compile_cuda_kernels(self):
        """Compile CUDA kernels for common operations"""
        if not NUMBA_AVAILABLE:
            return
        
        try:
            # Vectorized operations
            @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], 
                      target='cuda')
            def gpu_tanh(x, scale):
                return np.tanh(x * scale)
            
            @vectorize(['float32(float32)', 'float64(float64)'], target='cuda')
            def gpu_sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            
            @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], 
                      target='cuda')
            def gpu_relu(x, threshold):
                return max(0.0, x - threshold)
            
            # Matrix operations
            @cuda.jit
            def gpu_matrix_multiply(A, B, C):
                """GPU-accelerated matrix multiplication"""
                i, j = cuda.grid(2)
                if i < C.shape[0] and j < C.shape[1]:
                    tmp = 0.0
                    for k in range(A.shape[1]):
                        tmp += A[i, k] * B[k, j]
                    C[i, j] = tmp
            
            @cuda.jit
            def gpu_update_reservoir_states(inputs, W_in, W_res, states, leak_rate):
                """Update reservoir states on GPU"""
                i = cuda.grid(1)
                if i < states.shape[0]:
                    # Compute input drive
                    input_drive = 0.0
                    for j in range(inputs.shape[0]):
                        input_drive += W_in[i, j] * inputs[j]
                    
                    # Compute recurrent drive
                    recurrent_drive = 0.0
                    for j in range(states.shape[0]):
                        recurrent_drive += W_res[i, j] * states[j]
                    
                    # Update state with leaky integration
                    new_state = np.tanh(input_drive + recurrent_drive)
                    states[i] = (1 - leak_rate) * states[i] + leak_rate * new_state
            
            # Store compiled kernels
            self.gpu_tanh = gpu_tanh
            self.gpu_sigmoid = gpu_sigmoid
            self.gpu_relu = gpu_relu
            self.gpu_matrix_multiply = gpu_matrix_multiply
            self.gpu_update_reservoir_states = gpu_update_reservoir_states
            
            logger.info("CUDA kernels compiled successfully")
            
        except Exception as e:
            logger.warning(f"CUDA kernel compilation failed: {e}")
    
    def accelerated_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication with automatic fallback"""
        if not self.gpu_manager.gpu_available:
            return np.dot(A, B)
        
        with self.gpu_manager.gpu_memory_context():
            try:
                start_time = time.time()
                
                # Transfer to GPU
                A_gpu = self.gpu_manager.transfer_to_gpu(A)
                B_gpu = self.gpu_manager.transfer_to_gpu(B)
                
                # Perform multiplication
                C_gpu = cp.dot(A_gpu, B_gpu)
                
                # Transfer back
                result = self.gpu_manager.transfer_to_cpu(C_gpu)
                
                execution_time = time.time() - start_time
                logger.debug(f"GPU matrix multiply: {execution_time:.4f}s")
                
                return result
                
            except Exception as e:
                logger.warning(f"GPU matrix multiply failed, falling back to CPU: {e}")
                return np.dot(A, B)
    
    def accelerated_activation_function(self, x: np.ndarray, 
                                      activation: str = 'tanh',
                                      **kwargs) -> np.ndarray:
        """GPU-accelerated activation functions"""
        if not self.gpu_manager.gpu_available or not NUMBA_AVAILABLE:
            # CPU fallback
            if activation == 'tanh':
                return np.tanh(x)
            elif activation == 'sigmoid':
                return 1.0 / (1.0 + np.exp(-x))
            elif activation == 'relu':
                return np.maximum(0, x)
            else:
                return np.tanh(x)
        
        with self.gpu_manager.gpu_memory_context():
            try:
                x_gpu = self.gpu_manager.transfer_to_gpu(x.astype(np.float32))
                
                if activation == 'tanh':
                    scale = kwargs.get('scale', 1.0)
                    result_gpu = self.gpu_tanh(x_gpu, scale)
                elif activation == 'sigmoid':
                    result_gpu = self.gpu_sigmoid(x_gpu)
                elif activation == 'relu':
                    threshold = kwargs.get('threshold', 0.0)
                    result_gpu = self.gpu_relu(x_gpu, threshold)
                else:
                    result_gpu = self.gpu_tanh(x_gpu, 1.0)
                
                return self.gpu_manager.transfer_to_cpu(result_gpu)
                
            except Exception as e:
                logger.warning(f"GPU activation function failed: {e}")
                # Fallback to CPU
                if activation == 'tanh':
                    return np.tanh(x)
                elif activation == 'sigmoid':
                    return 1.0 / (1.0 + np.exp(-x))
                elif activation == 'relu':
                    return np.maximum(0, x)
                else:
                    return np.tanh(x)
    
    def accelerated_reservoir_update(self, inputs: np.ndarray, 
                                   W_in: np.ndarray, 
                                   W_res: np.ndarray,
                                   states: np.ndarray, 
                                   leak_rate: float) -> np.ndarray:
        """GPU-accelerated reservoir state update"""
        if not self.gpu_manager.gpu_available or not NUMBA_AVAILABLE:
            # CPU fallback
            input_drive = np.dot(W_in, inputs)
            recurrent_drive = np.dot(W_res, states)
            new_states = np.tanh(input_drive + recurrent_drive)
            return (1 - leak_rate) * states + leak_rate * new_states
        
        with self.gpu_manager.gpu_memory_context():
            try:
                # Transfer to GPU
                inputs_gpu = cuda.to_device(inputs.astype(np.float32))
                W_in_gpu = cuda.to_device(W_in.astype(np.float32))
                W_res_gpu = cuda.to_device(W_res.astype(np.float32))
                states_gpu = cuda.to_device(states.astype(np.float32))
                
                # Configure CUDA kernel launch
                threads_per_block = 256
                blocks = (states.shape[0] + threads_per_block - 1) // threads_per_block
                
                # Launch kernel
                self.gpu_update_reservoir_states[blocks, threads_per_block](
                    inputs_gpu, W_in_gpu, W_res_gpu, states_gpu, leak_rate
                )
                
                # Copy result back
                result = states_gpu.copy_to_host()
                return result
                
            except Exception as e:
                logger.warning(f"GPU reservoir update failed: {e}")
                # CPU fallback
                input_drive = np.dot(W_in, inputs)
                recurrent_drive = np.dot(W_res, states)
                new_states = np.tanh(input_drive + recurrent_drive)
                return (1 - leak_rate) * states + leak_rate * new_states
    
    def accelerated_spike_simulation(self, spike_trains: List[List[float]], 
                                   weights: np.ndarray,
                                   duration: float, dt: float) -> np.ndarray:
        """GPU-accelerated spiking neural network simulation"""
        n_neurons = weights.shape[1] if len(weights.shape) > 1 else weights.shape[0]
        n_steps = int(duration / dt)
        
        if not self.gpu_manager.gpu_available:
            # CPU fallback - simplified simulation
            spike_response = np.zeros((n_neurons, n_steps))
            membrane_potential = np.zeros(n_neurons)
            
            for step in range(n_steps):
                t = step * dt
                
                # Calculate input currents
                input_current = np.zeros(n_neurons)
                for input_idx, spikes in enumerate(spike_trains):
                    recent_spikes = [s for s in spikes if t - 10 <= s <= t]
                    if recent_spikes:
                        last_spike = max(recent_spikes)
                        decay = np.exp(-(t - last_spike) / 5.0)  # tau_synapse = 5.0
                        if len(weights.shape) > 1:
                            input_current += weights[input_idx] * decay
                        else:
                            input_current[input_idx % n_neurons] += decay
                
                # Update membrane potentials
                tau_membrane = 20.0
                membrane_potential += dt * (-membrane_potential / tau_membrane + input_current)
                
                # Check for spikes
                threshold = 1.0
                spiking = membrane_potential >= threshold
                spike_response[spiking, step] = 1
                membrane_potential[spiking] = 0  # Reset
            
            return spike_response
        
        with self.gpu_manager.gpu_memory_context():
            try:
                # Simplified GPU implementation using CuPy
                spike_response_gpu = cp.zeros((n_neurons, n_steps), dtype=cp.float32)
                membrane_potential_gpu = cp.zeros(n_neurons, dtype=cp.float32)
                weights_gpu = self.gpu_manager.transfer_to_gpu(weights.astype(np.float32))
                
                for step in range(n_steps):
                    t = step * dt
                    
                    # Calculate input currents (simplified)
                    input_current_gpu = cp.zeros(n_neurons, dtype=cp.float32)
                    
                    for input_idx, spikes in enumerate(spike_trains):
                        if spikes:
                            # Find most recent spike
                            recent_spike_times = [s for s in spikes if t - 10 <= s <= t]
                            if recent_spike_times:
                                last_spike = max(recent_spike_times)
                                decay = cp.exp(-(t - last_spike) / 5.0)
                                
                                if len(weights_gpu.shape) > 1:
                                    input_current_gpu += weights_gpu[input_idx] * decay
                                else:
                                    target_idx = input_idx % n_neurons
                                    input_current_gpu[target_idx] += decay
                    
                    # Update membrane potentials
                    tau_membrane = 20.0
                    membrane_potential_gpu += dt * (-membrane_potential_gpu / tau_membrane + input_current_gpu)
                    
                    # Check for spikes
                    threshold = 1.0
                    spiking_mask = membrane_potential_gpu >= threshold
                    spike_response_gpu[spiking_mask, step] = 1
                    membrane_potential_gpu[spiking_mask] = 0  # Reset
                
                return self.gpu_manager.transfer_to_cpu(spike_response_gpu)
                
            except Exception as e:
                logger.warning(f"GPU spike simulation failed: {e}")
                # Fallback to CPU implementation above
                spike_response = np.zeros((n_neurons, n_steps))
                membrane_potential = np.zeros(n_neurons)
                
                for step in range(n_steps):
                    t = step * dt
                    
                    input_current = np.zeros(n_neurons)
                    for input_idx, spikes in enumerate(spike_trains):
                        recent_spikes = [s for s in spikes if t - 10 <= s <= t]
                        if recent_spikes:
                            last_spike = max(recent_spikes)
                            decay = np.exp(-(t - last_spike) / 5.0)
                            if len(weights.shape) > 1:
                                input_current += weights[input_idx] * decay
                            else:
                                input_current[input_idx % n_neurons] += decay
                    
                    tau_membrane = 20.0
                    membrane_potential += dt * (-membrane_potential / tau_membrane + input_current)
                    
                    threshold = 1.0
                    spiking = membrane_potential >= threshold
                    spike_response[spiking, step] = 1
                    membrane_potential[spiking] = 0
                
                return spike_response
    
    def accelerated_feature_extraction(self, states: np.ndarray,
                                     extraction_type: str = 'statistical') -> np.ndarray:
        """GPU-accelerated feature extraction from neural states"""
        if not self.gpu_manager.gpu_available:
            # CPU fallback
            if extraction_type == 'statistical':
                features = []
                features.extend([np.mean(states, axis=0)])
                features.extend([np.std(states, axis=0)])
                features.extend([np.max(states, axis=0)])
                features.extend([np.min(states, axis=0)])
                return np.concatenate(features)
            else:
                return np.mean(states, axis=0)
        
        with self.gpu_manager.gpu_memory_context():
            try:
                states_gpu = self.gpu_manager.transfer_to_gpu(states)
                
                if extraction_type == 'statistical':
                    # Calculate statistical features on GPU
                    mean_feat = cp.mean(states_gpu, axis=0)
                    std_feat = cp.std(states_gpu, axis=0)
                    max_feat = cp.max(states_gpu, axis=0)
                    min_feat = cp.min(states_gpu, axis=0)
                    
                    features_gpu = cp.concatenate([mean_feat, std_feat, max_feat, min_feat])
                    return self.gpu_manager.transfer_to_cpu(features_gpu)
                else:
                    # Simple mean extraction
                    mean_feat = cp.mean(states_gpu, axis=0)
                    return self.gpu_manager.transfer_to_cpu(mean_feat)
                    
            except Exception as e:
                logger.warning(f"GPU feature extraction failed: {e}")
                # CPU fallback
                if extraction_type == 'statistical':
                    features = []
                    features.extend([np.mean(states, axis=0)])
                    features.extend([np.std(states, axis=0)])
                    features.extend([np.max(states, axis=0)])
                    features.extend([np.min(states, axis=0)])
                    return np.concatenate(features)
                else:
                    return np.mean(states, axis=0)
    
    def accelerated_clustering(self, features: np.ndarray, 
                             n_clusters: int,
                             algorithm: str = 'kmeans',
                             max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated clustering with fallback"""
        if not self.gpu_manager.gpu_available or algorithm != 'kmeans':
            # CPU fallback
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iters, random_state=42)
            labels = kmeans.fit_predict(features)
            centroids = kmeans.cluster_centers_
            return labels, centroids
        
        with self.gpu_manager.gpu_memory_context():
            try:
                # Simple GPU k-means implementation
                features_gpu = self.gpu_manager.transfer_to_gpu(features.astype(np.float32))
                n_samples, n_features = features_gpu.shape
                
                # Initialize centroids randomly
                centroid_indices = cp.random.choice(n_samples, n_clusters, replace=False)
                centroids_gpu = features_gpu[centroid_indices].copy()
                
                labels_gpu = cp.zeros(n_samples, dtype=cp.int32)
                
                for iteration in range(max_iters):
                    # Assign points to closest centroids
                    distances = cp.linalg.norm(
                        features_gpu[:, cp.newaxis, :] - centroids_gpu[cp.newaxis, :, :], 
                        axis=2
                    )
                    new_labels = cp.argmin(distances, axis=1)
                    
                    # Check for convergence
                    if cp.array_equal(labels_gpu, new_labels):
                        break
                    
                    labels_gpu = new_labels
                    
                    # Update centroids
                    for k in range(n_clusters):
                        mask = labels_gpu == k
                        if cp.sum(mask) > 0:
                            centroids_gpu[k] = cp.mean(features_gpu[mask], axis=0)
                
                labels = self.gpu_manager.transfer_to_cpu(labels_gpu)
                centroids = self.gpu_manager.transfer_to_cpu(centroids_gpu)
                
                return labels, centroids
                
            except Exception as e:
                logger.warning(f"GPU clustering failed: {e}")
                # CPU fallback
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iters, random_state=42)
                labels = kmeans.fit_predict(features)
                centroids = kmeans.cluster_centers_
                return labels, centroids
    
    def benchmark_operations(self) -> Dict[str, Dict[str, float]]:
        """Benchmark GPU vs CPU performance"""
        results = {}
        test_sizes = [100, 1000, 10000]
        
        for size in test_sizes:
            # Matrix multiplication benchmark
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # CPU timing
            start_time = time.time()
            cpu_result = np.dot(A, B)
            cpu_time = time.time() - start_time
            
            # GPU timing
            start_time = time.time()
            gpu_result = self.accelerated_matrix_multiply(A, B)
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            
            results[f'matmul_{size}'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'gpu_available': self.gpu_manager.gpu_available
            }
        
        return results


# Global GPU operations instance
gpu_ops = GPUAcceleratedOperations()


def gpu_accelerated(fallback_to_cpu: bool = True):
    """Decorator for GPU acceleration with automatic fallback"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not gpu_ops.gpu_manager.gpu_available and not fallback_to_cpu:
                raise RuntimeError("GPU not available and fallback disabled")
            
            # Check if there's a GPU-accelerated version
            gpu_func_name = f"accelerated_{func.__name__}"
            if hasattr(gpu_ops, gpu_func_name):
                gpu_func = getattr(gpu_ops, gpu_func_name)
                return gpu_func(*args, **kwargs)
            else:
                # No GPU version, run original function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator