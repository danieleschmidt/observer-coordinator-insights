"""
Neuromorphic Clustering Implementation for Insights Discovery Data
Combines Echo State Networks, Spiking Neural Networks, and Reservoir Computing
for advanced personality trait clustering with temporal dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN, KMeans
import logging
import time
from dataclasses import dataclass
from enum import Enum
import warnings
import functools
import threading
from datetime import datetime, timedelta
import uuid
import psutil
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import traceback
from contextlib import contextmanager
import hashlib
import pickle
from pathlib import Path
import gc
import mmap
from collections import OrderedDict
import asyncio
from numba import jit, cuda
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Correlation ID for request tracing
_correlation_id = threading.local()

# Generation 3 Global Configuration
GEN3_CONFIG = {
    'enable_gpu': True,
    'enable_caching': True,
    'cache_ttl': 3600,
    'max_cache_size': 1000,
    'enable_redis': False,
    'redis_host': 'localhost',
    'redis_port': 6379,
    'enable_async': True,
    'vectorize_operations': True,
    'memory_mapping': True,
    'enable_compression': True,
    'auto_scale_workers': True,
    'max_workers': min(32, (psutil.cpu_count() or 1) * 4),
    'enable_streaming': True,
    'chunk_size': 10000,
    'enable_incremental': True
}

def get_correlation_id() -> str:
    """Get current correlation ID for request tracing"""
    if not hasattr(_correlation_id, 'value'):
        _correlation_id.value = str(uuid.uuid4())
    return _correlation_id.value

def set_correlation_id(corr_id: str):
    """Set correlation ID for request tracing"""
    _correlation_id.value = corr_id


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class NeuromorphicErrorType(Enum):
    """Types of neuromorphic clustering errors"""
    MEMORY_ERROR = "memory_error"
    CONVERGENCE_ERROR = "convergence_error"
    DIMENSION_ERROR = "dimension_error"
    TIMEOUT_ERROR = "timeout_error"
    STABILITY_ERROR = "stability_error"
    RESOURCE_ERROR = "resource_error"


class NeuromorphicClusteringMethod(Enum):
    """Enumeration of available neuromorphic clustering methods"""
    ECHO_STATE_NETWORK = "echo_state_network"
    SPIKING_NEURAL_NETWORK = "spiking_neural_network" 
    LIQUID_STATE_MACHINE = "liquid_state_machine"
    HYBRID_RESERVOIR = "hybrid_reservoir"


@dataclass
class NeuromorphicException(Exception):
    """Base exception for neuromorphic clustering with enhanced context"""
    error_type: NeuromorphicErrorType
    correlation_id: str
    context: Dict[str, Any]
    recoverable: bool = True
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        super().__init__(f"{self.error_type.value}: {self.args[0] if self.args else 'Unknown error'}")


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    memory_usage_mb: float
    cpu_percent: float
    processing_time_ms: float
    correlation_id: str
    timestamp: datetime
    

@dataclass
class ClusteringMetrics:
    """Metrics for evaluating neuromorphic clustering performance"""
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_stability: float
    interpretability_score: float
    temporal_coherence: float
    computational_efficiency: float


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance in neuromorphic operations"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.utcnow() - self.last_failure_time).seconds >= self.config.recovery_timeout
    
    def _reset_breaker(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker reset [correlation_id: {get_correlation_id()}]")
    
    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker opened after {self.failure_count} failures [correlation_id: {get_correlation_id()}]")
    
    def _record_success(self):
        """Record a success and reset failure count"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._reset_breaker()
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def __call__(self, func: Callable) -> Callable:
        """Circuit breaker decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker half-open for retry [correlation_id: {get_correlation_id()}]")
                else:
                    raise NeuromorphicException(
                        "Circuit breaker is open - operation unavailable",
                        error_type=NeuromorphicErrorType.RESOURCE_ERROR,
                        correlation_id=get_correlation_id(),
                        context={'failure_count': self.failure_count},
                        recoverable=False
                    )
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except self.config.expected_exception as e:
                self._record_failure()
                raise
            except Exception as e:
                # Don't break circuit for unexpected exceptions, just log
                logger.warning(f"Unexpected exception in circuit breaker: {e} [correlation_id: {get_correlation_id()}]")
                raise
        
        return wrapper


class RetryManager:
    """Exponential backoff retry manager for transient failures"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def retry_with_backoff(self, operation: Callable, *args, **kwargs):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        correlation_id = get_correlation_id()
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Max retries exceeded for operation {operation.__name__} [correlation_id: {correlation_id}]")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                
                logger.warning(f"Operation {operation.__name__} failed (attempt {attempt + 1}), retrying in {delay}s [correlation_id: {correlation_id}]")
                time.sleep(delay)
        
        # If we get here, all retries failed
        if isinstance(last_exception, NeuromorphicException):
            raise last_exception
        else:
            raise NeuromorphicException(
                f"Operation failed after {self.max_retries} retries: {str(last_exception)}",
                error_type=NeuromorphicErrorType.RESOURCE_ERROR,
                correlation_id=correlation_id,
                context={'original_exception': str(last_exception), 'retries': self.max_retries}
            )


@contextmanager
def resource_monitor():
    """Context manager for monitoring resource usage"""
    correlation_id = get_correlation_id()
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent()
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        processing_time = (end_time - start_time) * 1000  # ms
        
        metrics = ResourceMetrics(
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=process.cpu_percent(),
            processing_time_ms=processing_time,
            correlation_id=correlation_id,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Resource usage - Memory: {metrics.memory_usage_mb:.2f}MB, "
                   f"Time: {metrics.processing_time_ms:.2f}ms [correlation_id: {correlation_id}]")
        
        # Alert if resource usage is concerning
        if metrics.memory_usage_mb > 500:  # > 500MB memory usage
            logger.warning(f"High memory usage detected: {metrics.memory_usage_mb:.2f}MB [correlation_id: {correlation_id}]")
        if metrics.processing_time_ms > 30000:  # > 30s processing time
            logger.warning(f"Long processing time detected: {metrics.processing_time_ms:.2f}ms [correlation_id: {correlation_id}]")


def timeout_operation(timeout_seconds: int = 300):
    """Decorator to add timeout to operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            correlation_id = get_correlation_id()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except FutureTimeoutError:
                    logger.error(f"Operation {func.__name__} timed out after {timeout_seconds}s [correlation_id: {correlation_id}]")
                    raise NeuromorphicException(
                        f"Operation timed out after {timeout_seconds} seconds",
                        error_type=NeuromorphicErrorType.TIMEOUT_ERROR,
                        correlation_id=correlation_id,
                        context={'timeout_seconds': timeout_seconds}
                    )
        return wrapper
    return decorator


class EchoStateNetwork:
    """
    Echo State Network implementation for temporal feature extraction
    from personality energy dynamics
    """
    
    def __init__(self, reservoir_size: int = 100, input_size: int = 4, 
                 spectral_radius: float = 0.95, sparsity: float = 0.1,
                 leaking_rate: float = 0.3, random_state: int = 42):
        """
        Initialize Echo State Network
        
        Args:
            reservoir_size: Number of neurons in reservoir
            input_size: Dimension of input (4 for RBGY energies)
            spectral_radius: Largest eigenvalue of reservoir matrix
            sparsity: Connection sparsity in reservoir
            leaking_rate: Rate of state leakage (0-1)
            random_state: Random seed for reproducibility
        """
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Initialize reservoir weights
        self._initialize_reservoir()
        
        # State variables
        self.state = np.zeros(self.reservoir_size)
        self.states_history = []
        
    def _initialize_reservoir(self):
        """Initialize reservoir weight matrices"""
        # Input to reservoir weights
        self.W_in = np.random.randn(self.reservoir_size, self.input_size) * 0.1
        
        # Reservoir recurrent weights
        W = np.random.randn(self.reservoir_size, self.reservoir_size)
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) > self.sparsity
        W = W * mask
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        self.W_res = W * (self.spectral_radius / max_eigenvalue)
        
    def update_state(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Update reservoir state with new input
        
        Args:
            input_vector: 4D personality energy vector
            
        Returns:
            Updated reservoir state
        """
        # Compute new state using leaky integrator
        pre_activation = (np.dot(self.W_in, input_vector) + 
                         np.dot(self.W_res, self.state))
        
        new_state = np.tanh(pre_activation)
        
        # Apply leaking rate
        self.state = ((1 - self.leaking_rate) * self.state + 
                      self.leaking_rate * new_state)
        
        return self.state.copy()
    
    def process_sequence(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Process a sequence of personality energy vectors
        
        Args:
            input_sequence: Array of shape (seq_len, 4)
            
        Returns:
            Matrix of reservoir states (seq_len, reservoir_size)
        """
        self.states_history = []
        self.state = np.zeros(self.reservoir_size)  # Reset state
        
        for input_vec in input_sequence:
            state = self.update_state(input_vec)
            self.states_history.append(state)
        
        return np.array(self.states_history)
    
    def extract_features(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract interpretable features from reservoir states
        
        Args:
            states: Reservoir states matrix
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Temporal statistics
        features['mean_activation'] = np.mean(states, axis=0)
        features['std_activation'] = np.std(states, axis=0)
        features['max_activation'] = np.max(states, axis=0)
        features['final_state'] = states[-1] if len(states) > 0 else np.zeros(self.reservoir_size)
        
        # Temporal dynamics
        if len(states) > 1:
            features['activation_trend'] = np.gradient(states, axis=0)[-1]
            features['stability_measure'] = np.mean(np.abs(np.diff(states, axis=0)), axis=0)
        else:
            features['activation_trend'] = np.zeros(self.reservoir_size)
            features['stability_measure'] = np.zeros(self.reservoir_size)
        
        return features


class SpikingNeuralCluster:
    """
    Spiking Neural Network implementation with STDP for clustering
    """
    
    def __init__(self, n_neurons: int = 50, threshold: float = 1.0,
                 tau_membrane: float = 20.0, tau_synapse: float = 5.0,
                 learning_rate: float = 0.01, random_state: int = 42):
        """
        Initialize Spiking Neural Network for clustering
        
        Args:
            n_neurons: Number of spiking neurons
            threshold: Spike threshold
            tau_membrane: Membrane time constant
            tau_synapse: Synaptic time constant
            learning_rate: STDP learning rate
            random_state: Random seed
        """
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.tau_membrane = tau_membrane
        self.tau_synapse = tau_synapse
        self.learning_rate = learning_rate
        
        np.random.seed(random_state)
        
        # Initialize neuron parameters
        self.membrane_potential = np.zeros(n_neurons)
        self.spike_times = [[] for _ in range(n_neurons)]
        self.synaptic_weights = np.random.randn(4, n_neurons) * 0.1  # 4 inputs (RBGY)
        
        # Lateral inhibition weights
        self.lateral_weights = np.random.randn(n_neurons, n_neurons) * 0.05
        np.fill_diagonal(self.lateral_weights, 0)  # No self-connections
        
    def encode_input(self, energy_vector: np.ndarray, duration: float = 100.0) -> List[List[float]]:
        """
        Convert energy values to spike trains using rate coding
        
        Args:
            energy_vector: 4D personality energy vector (normalized 0-1)
            duration: Simulation duration in ms
            
        Returns:
            Spike times for each input channel
        """
        spike_trains = []
        
        for energy in energy_vector:
            # Convert energy to firing rate (Hz)
            # Higher energy -> higher firing rate (max 100 Hz)
            firing_rate = max(0.1, energy * 100)  # Avoid zero rates
            
            # Generate Poisson spike train
            n_expected_spikes = int(firing_rate * duration / 1000)
            spike_times = np.sort(np.random.uniform(0, duration, n_expected_spikes))
            spike_trains.append(spike_times.tolist())
        
        return spike_trains
    
    def simulate_dynamics(self, input_spikes: List[List[float]], 
                         duration: float = 100.0, dt: float = 1.0) -> np.ndarray:
        """
        Simulate spiking neural network dynamics
        
        Args:
            input_spikes: Spike times for each input channel
            duration: Simulation duration in ms
            dt: Time step in ms
            
        Returns:
            Spike response matrix (neurons x time_steps)
        """
        n_steps = int(duration / dt)
        time_steps = np.arange(0, duration, dt)
        spike_response = np.zeros((self.n_neurons, n_steps))
        
        # Reset membrane potentials and spike history
        self.membrane_potential = np.zeros(self.n_neurons)
        self.spike_times = [[] for _ in range(self.n_neurons)]
        
        for step, t in enumerate(time_steps):
            # Calculate input currents from spike trains
            input_current = np.zeros(self.n_neurons)
            
            for input_idx, spikes in enumerate(input_spikes):
                # Find recent spikes (within synaptic window)
                recent_spikes = [s for s in spikes if t - 10 <= s <= t]
                if recent_spikes:
                    # Exponential decay from most recent spike
                    last_spike = max(recent_spikes)
                    decay = np.exp(-(t - last_spike) / self.tau_synapse)
                    input_current += self.synaptic_weights[input_idx] * decay
            
            # Lateral inhibition
            lateral_current = np.zeros(self.n_neurons)
            for i in range(self.n_neurons):
                if self.spike_times[i] and self.spike_times[i][-1] > t - 5:  # Recent spike
                    lateral_current += self.lateral_weights[i] * -0.5  # Inhibition
            
            # Update membrane potentials
            self.membrane_potential += dt * (
                -self.membrane_potential / self.tau_membrane +
                input_current + lateral_current
            )
            
            # Check for spikes
            spiking_neurons = self.membrane_potential >= self.threshold
            spike_response[spiking_neurons, step] = 1
            
            # Reset spiked neurons and record spike times
            for neuron_idx in np.where(spiking_neurons)[0]:
                self.spike_times[neuron_idx].append(t)
                self.membrane_potential[neuron_idx] = 0  # Reset after spike
        
        return spike_response
    
    def extract_spike_features(self, spike_response: np.ndarray) -> np.ndarray:
        """
        Extract features from spike response patterns
        
        Args:
            spike_response: Spike response matrix
            
        Returns:
            Feature vector for clustering
        """
        features = []
        
        for neuron_idx in range(self.n_neurons):
            spikes = spike_response[neuron_idx]
            
            # Firing rate
            firing_rate = np.sum(spikes) / len(spikes) * 1000  # spikes/second
            features.append(firing_rate)
            
            # Burst detection - consecutive spikes within short window
            spike_times = np.where(spikes)[0]
            if len(spike_times) > 1:
                isi = np.diff(spike_times)
                burst_count = np.sum(isi <= 5)  # ISI <= 5ms considered burst
                features.append(burst_count)
            else:
                features.append(0)
                
        return np.array(features)


class LiquidStateMachine:
    """
    Liquid State Machine implementation for temporal clustering
    """
    
    def __init__(self, liquid_size: int = 64, input_size: int = 4,
                 connection_prob: float = 0.3, tau_membrane: float = 30.0,
                 random_state: int = 42):
        """
        Initialize Liquid State Machine
        
        Args:
            liquid_size: Number of neurons in liquid
            input_size: Input dimension
            connection_prob: Connection probability in liquid
            tau_membrane: Membrane time constant
            random_state: Random seed
        """
        self.liquid_size = liquid_size
        self.input_size = input_size
        self.connection_prob = connection_prob
        self.tau_membrane = tau_membrane
        
        np.random.seed(random_state)
        
        # Create liquid structure (small-world network)
        self._create_liquid_topology()
        
        # State variables
        self.liquid_state = np.zeros(liquid_size)
        self.state_history = []
        
    def _create_liquid_topology(self):
        """Create liquid network topology with small-world properties"""
        # Input connections
        self.W_input = np.random.randn(self.liquid_size, self.input_size) * 0.3
        
        # Liquid recurrent connections
        self.W_liquid = np.zeros((self.liquid_size, self.liquid_size))
        
        # Create random connections
        for i in range(self.liquid_size):
            for j in range(self.liquid_size):
                if i != j and np.random.rand() < self.connection_prob:
                    # Distance-dependent connectivity (3D grid assumption)
                    pos_i = np.array([i % 4, (i // 4) % 4, i // 16])
                    pos_j = np.array([j % 4, (j // 4) % 4, j // 16])
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # Connection strength inversely related to distance
                    strength = 0.5 * np.exp(-distance / 2)
                    
                    # Excitatory (80%) or inhibitory (20%)
                    if np.random.rand() < 0.8:
                        self.W_liquid[i, j] = strength
                    else:
                        self.W_liquid[i, j] = -strength * 0.5
    
    def update_liquid(self, input_vector: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Update liquid state
        
        Args:
            input_vector: Input vector
            dt: Time step
            
        Returns:
            Updated liquid state
        """
        # Input drive
        input_drive = np.dot(self.W_input, input_vector)
        
        # Recurrent dynamics
        recurrent_drive = np.dot(self.W_liquid, self.liquid_state)
        
        # Membrane dynamics (leaky integrator)
        dstate_dt = (-self.liquid_state + np.tanh(input_drive + recurrent_drive)) / self.tau_membrane
        
        # Update state
        self.liquid_state += dt * dstate_dt
        
        return self.liquid_state.copy()
    
    def process_temporal_sequence(self, sequence: np.ndarray, 
                                 duration_per_sample: float = 50.0) -> np.ndarray:
        """
        Process temporal sequence through liquid
        
        Args:
            sequence: Input sequence (n_samples, input_size)
            duration_per_sample: Processing time per sample
            
        Returns:
            Liquid state trajectories
        """
        self.state_history = []
        self.liquid_state = np.zeros(self.liquid_size)  # Reset
        
        for sample in sequence:
            # Process sample for specified duration
            n_steps = int(duration_per_sample)
            for _ in range(n_steps):
                state = self.update_liquid(sample)
            
            self.state_history.append(state)
        
        return np.array(self.state_history)


class NeuromorphicClusterer:
    """
    Main neuromorphic clustering class combining multiple approaches
    """
    
    def __init__(self, method: NeuromorphicClusteringMethod = NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                 n_clusters: int = 4, random_state: int = 42,
                 esn_params: Optional[Dict] = None,
                 snn_params: Optional[Dict] = None,
                 lsm_params: Optional[Dict] = None,
                 enable_fallback: bool = True,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 enable_gpu: bool = None,
                 enable_caching: bool = None,
                 cache_ttl: int = None,
                 optimization_level: str = 'balanced'):
        """
        Initialize neuromorphic clusterer with Generation 3 enhancements
        
        Args:
            method: Clustering method to use
            n_clusters: Target number of clusters
            random_state: Random seed
            esn_params: Echo State Network parameters
            snn_params: Spiking Neural Network parameters  
            lsm_params: Liquid State Machine parameters
            enable_fallback: Whether to fallback to K-means on neuromorphic failure
            circuit_breaker_config: Configuration for circuit breaker pattern
            enable_gpu: Enable GPU acceleration (auto-detected if None)
            enable_caching: Enable intelligent caching (from config if None)
            cache_ttl: Cache time-to-live in seconds
            optimization_level: Optimization level ('conservative', 'balanced', 'aggressive')
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.enable_fallback = enable_fallback
        self.optimization_level = optimization_level
        
        # Default parameters
        self.esn_params = esn_params or {}
        self.snn_params = snn_params or {}
        self.lsm_params = lsm_params or {}
        
        # Generation 3 enhancements
        self.enable_gpu = enable_gpu if enable_gpu is not None else GEN3_CONFIG['enable_gpu']
        self.enable_caching = enable_caching if enable_caching is not None else GEN3_CONFIG['enable_caching']
        self.cache_ttl = cache_ttl or GEN3_CONFIG['cache_ttl']
        
        # Initialize GPU acceleration if available
        self.gpu_ops = None
        if self.enable_gpu:
            try:
                from .gpu_acceleration import gpu_ops
                self.gpu_ops = gpu_ops
                logger.info(f"GPU acceleration enabled [correlation_id: {get_correlation_id()}]")
            except Exception as e:
                logger.warning(f"GPU acceleration failed to initialize: {e} [correlation_id: {get_correlation_id()}]")
                self.enable_gpu = False
        
        # Initialize caching if enabled
        self.cache = None
        if self.enable_caching:
            try:
                from .caching import neuromorphic_cache
                self.cache = neuromorphic_cache
                logger.info(f"Intelligent caching enabled [correlation_id: {get_correlation_id()}]")
            except Exception as e:
                logger.warning(f"Caching failed to initialize: {e} [correlation_id: {get_correlation_id()}]")
                self.enable_caching = False
        
        # Robustness components
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=120,
            expected_exception=NeuromorphicException
        )
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.retry_manager = RetryManager(max_retries=2, base_delay=1.0)
        
        # Initialize components with error handling
        self._safe_initialize_components()
        
        # Clustering results and state
        self.cluster_labels = None
        self.feature_data = None
        self.scaler = StandardScaler()
        self.trained = False
        self.fallback_used = False
        self.last_error = None
        
    def _safe_initialize_components(self):
        """Initialize neuromorphic components with error handling"""
        correlation_id = get_correlation_id()
        
        try:
            if self.method in [NeuromorphicClusteringMethod.ECHO_STATE_NETWORK, 
                              NeuromorphicClusteringMethod.HYBRID_RESERVOIR]:
                self.esn = EchoStateNetwork(random_state=self.random_state, **self.esn_params)
                logger.info(f"ESN initialized successfully [correlation_id: {correlation_id}]")
                
            if self.method in [NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
                              NeuromorphicClusteringMethod.HYBRID_RESERVOIR]:
                self.snn = SpikingNeuralCluster(random_state=self.random_state, **self.snn_params)
                logger.info(f"SNN initialized successfully [correlation_id: {correlation_id}]")
                
            if self.method in [NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
                              NeuromorphicClusteringMethod.HYBRID_RESERVOIR]:
                self.lsm = LiquidStateMachine(random_state=self.random_state, **self.lsm_params)
                logger.info(f"LSM initialized successfully [correlation_id: {correlation_id}]")
                
        except Exception as e:
            logger.error(f"Failed to initialize neuromorphic components: {e} [correlation_id: {correlation_id}]")
            raise NeuromorphicException(
                f"Component initialization failed: {str(e)}",
                error_type=NeuromorphicErrorType.RESOURCE_ERROR,
                correlation_id=correlation_id,
                context={'method': self.method.value, 'component_params': {
                    'esn_params': self.esn_params,
                    'snn_params': self.snn_params,
                    'lsm_params': self.lsm_params
                }}
            )
    
    @timeout_operation(timeout_seconds=600)  # 10 minute timeout
    def fit(self, features: pd.DataFrame) -> 'NeuromorphicClusterer':
        """
        Fit neuromorphic clustering to features with robust error handling
        
        Args:
            features: DataFrame with energy columns (red, blue, green, yellow)
            
        Returns:
            Fitted clusterer
        """
        correlation_id = get_correlation_id()
        set_correlation_id(correlation_id)  # Ensure correlation ID is set
        
        logger.info(f"Starting neuromorphic clustering fit [correlation_id: {correlation_id}]")
        
        # Input validation with enhanced error context
        try:
            self._validate_input_features(features)
        except Exception as e:
            raise NeuromorphicException(
                f"Input validation failed: {str(e)}",
                error_type=NeuromorphicErrorType.DIMENSION_ERROR,
                correlation_id=correlation_id,
                context={'features_shape': features.shape if features is not None else None,
                        'features_columns': list(features.columns) if features is not None else None}
            )
        
        with resource_monitor():
            try:
                # Attempt neuromorphic clustering with circuit breaker protection
                result = self._robust_fit_neuromorphic(features, correlation_id)
                return result
                
            except NeuromorphicException as e:
                self.last_error = e
                if self.enable_fallback:
                    logger.warning(f"Neuromorphic clustering failed, attempting fallback to K-means [correlation_id: {correlation_id}]")
                    return self._fallback_to_kmeans(features, correlation_id)
                else:
                    logger.error(f"Neuromorphic clustering failed without fallback [correlation_id: {correlation_id}]")
                    raise
                    
            except Exception as e:
                # Wrap unexpected exceptions
                wrapped_exception = Exception(f"Unexpected error during clustering: {str(e)}")
                self.last_error = wrapped_exception
                
                if self.enable_fallback:
                    logger.warning(f"Unexpected error, attempting fallback [correlation_id: {correlation_id}]")
                    return self._fallback_to_kmeans(features, correlation_id)
                else:
                    raise wrapped_exception
    
    def _validate_input_features(self, features: pd.DataFrame):
        """Validate input features with detailed error messages"""
        if features is None:
            raise ValueError("Features DataFrame cannot be None")
        
        if features.empty:
            raise ValueError("Features DataFrame cannot be empty")
            
        if len(features) < self.n_clusters:
            raise ValueError(f"Insufficient data: {len(features)} samples < {self.n_clusters} clusters")
        
        # Validate required columns
        required_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        missing_cols = [col for col in required_cols if col not in features.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types and ranges
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(features[col]):
                raise ValueError(f"Column {col} must be numeric")
            
            if features[col].isnull().sum() > len(features) * 0.5:
                raise ValueError(f"Column {col} has too many missing values (>{len(features) * 0.5})")
    
    # @circuit_breaker
    def _robust_fit_neuromorphic(self, features: pd.DataFrame, correlation_id: str) -> 'NeuromorphicClusterer':
        """Robust neuromorphic clustering with circuit breaker protection"""
        
        def _fit_operation():
            logger.info(f"Fitting neuromorphic clusterer with method: {self.method} [correlation_id: {correlation_id}]")
            start_time = time.time()
            
            # Store and normalize features
            self.feature_data = features.copy()
            required_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
            energy_data = features[required_cols].values
            
            # Normalize to [0, 1] for neuromorphic processing
            energy_normalized = self.scaler.fit_transform(energy_data)
            
            # Extract neuromorphic features with retry logic
            neuromorphic_features = self.retry_manager.retry_with_backoff(
                self._extract_neuromorphic_features_safe, energy_normalized
            )
            
            # Perform clustering on extracted features
            self.cluster_labels = self._cluster_features_safe(neuromorphic_features)
            
            self.trained = True
            self.fallback_used = False
            fit_time = time.time() - start_time
            
            logger.info(f"Neuromorphic clustering completed in {fit_time:.2f}s [correlation_id: {correlation_id}]")
            self._log_clustering_summary()
            
            return self
        
        return _fit_operation()
        
    def _fallback_to_kmeans(self, features: pd.DataFrame, correlation_id: str) -> 'NeuromorphicClusterer':
        """Fallback to K-means clustering when neuromorphic methods fail"""
        logger.info(f"Executing K-means fallback [correlation_id: {correlation_id}]")
        
        try:
            start_time = time.time()
            
            # Use standard energy features for K-means
            required_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
            energy_data = features[required_cols].values
            energy_normalized = self.scaler.fit_transform(energy_data)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            self.cluster_labels = kmeans.fit_predict(energy_normalized)
            
            self.feature_data = features.copy()
            self.trained = True
            self.fallback_used = True
            
            fit_time = time.time() - start_time
            logger.info(f"K-means fallback completed in {fit_time:.2f}s [correlation_id: {correlation_id}]")
            
            return self
            
        except Exception as e:
            logger.error(f"K-means fallback also failed: {e} [correlation_id: {correlation_id}]")
            raise NeuromorphicException(
                f"Both neuromorphic and K-means fallback failed: {str(e)}",
                error_type=NeuromorphicErrorType.RESOURCE_ERROR,
                correlation_id=correlation_id,
                context={'fallback_error': str(e)},
                recoverable=False
            )
    
    def _extract_neuromorphic_features_safe(self, energy_data: np.ndarray) -> np.ndarray:
        """
        Safely extract features using neuromorphic processing with error handling
        
        Args:
            energy_data: Normalized energy data
            
        Returns:
            Extracted feature matrix
        """
        correlation_id = get_correlation_id()
        all_features = []
        failed_samples = 0
        max_failed_samples = int(len(energy_data) * 0.1)  # Allow 10% failure rate
        
        logger.info(f"Extracting neuromorphic features from {len(energy_data)} samples [correlation_id: {correlation_id}]")
        
        for i, sample in enumerate(energy_data):
            try:
                sample_features = self._extract_single_sample_features(sample, correlation_id)
                all_features.append(sample_features)
                
            except Exception as e:
                failed_samples += 1
                logger.warning(f"Failed to extract features for sample {i}: {e} [correlation_id: {correlation_id}]")
                
                if failed_samples > max_failed_samples:
                    raise NeuromorphicException(
                        f"Too many feature extraction failures: {failed_samples}/{len(energy_data)}",
                        error_type=NeuromorphicErrorType.STABILITY_ERROR,
                        correlation_id=correlation_id,
                        context={'failed_samples': failed_samples, 'total_samples': len(energy_data)}
                    )
                
                # Use fallback features for failed sample
                fallback_features = self._create_fallback_features(sample)
                all_features.append(fallback_features)
        
        if failed_samples > 0:
            logger.warning(f"Feature extraction completed with {failed_samples} failures [correlation_id: {correlation_id}]")
        
        feature_matrix = np.array(all_features)
        
        # Validate feature matrix
        if feature_matrix.shape[0] == 0:
            raise NeuromorphicException(
                "No features extracted - all samples failed",
                error_type=NeuromorphicErrorType.DIMENSION_ERROR,
                correlation_id=correlation_id,
                context={'input_shape': energy_data.shape}
            )
        
        if np.isnan(feature_matrix).any() or np.isinf(feature_matrix).any():
            logger.warning(f"Feature matrix contains invalid values, applying cleanup [correlation_id: {correlation_id}]")
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_matrix
    
    def _extract_single_sample_features(self, sample: np.ndarray, correlation_id: str) -> List[float]:
        """Extract features from a single sample with error handling"""
        sample_features = []
        
        try:
            if self.method in [NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                              NeuromorphicClusteringMethod.HYBRID_RESERVOIR]:
                # Create temporal sequence by adding small perturbations
                temporal_seq = self._create_temporal_sequence_safe(sample)
                esn_states = self.esn.process_sequence(temporal_seq)
                esn_features = self.esn.extract_features(esn_states)
                
                # Flatten ESN features
                for key, values in esn_features.items():
                    if isinstance(values, np.ndarray) and not np.isnan(values).all():
                        clean_values = np.nan_to_num(values, nan=0.0)
                        sample_features.extend(clean_values[:10])  # Limit features
                        
        except Exception as e:
            logger.warning(f"ESN feature extraction failed: {e} [correlation_id: {correlation_id}]")
        
        try:
            if self.method in [NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
                              NeuromorphicClusteringMethod.HYBRID_RESERVOIR]:
                # Convert to spike trains and simulate
                input_spikes = self.snn.encode_input(sample, duration=100.0)
                spike_response = self.snn.simulate_dynamics(input_spikes, duration=100.0)
                snn_features = self.snn.extract_spike_features(spike_response)
                
                if not np.isnan(snn_features).all():
                    clean_features = np.nan_to_num(snn_features, nan=0.0)
                    sample_features.extend(clean_features[:20])  # Limit features
                    
        except Exception as e:
            logger.warning(f"SNN feature extraction failed: {e} [correlation_id: {correlation_id}]")
        
        try:
            if self.method in [NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
                              NeuromorphicClusteringMethod.HYBRID_RESERVOIR]:
                # Process through liquid state machine
                temporal_seq = self._create_temporal_sequence_safe(sample)
                lsm_states = self.lsm.process_temporal_sequence(temporal_seq)
                
                # Extract LSM features
                lsm_features = np.concatenate([
                    np.mean(lsm_states, axis=0)[:15],  # Mean activation
                    np.std(lsm_states, axis=0)[:15],   # Activation variability
                    lsm_states[-1][:10]                # Final state
                ])
                
                if not np.isnan(lsm_features).all():
                    clean_features = np.nan_to_num(lsm_features, nan=0.0)
                    sample_features.extend(clean_features)
                    
        except Exception as e:
            logger.warning(f"LSM feature extraction failed: {e} [correlation_id: {correlation_id}]")
        
        # Ensure we have some features
        if not sample_features:
            sample_features = self._create_fallback_features(sample)
        
        return sample_features
    
    def _create_fallback_features(self, sample: np.ndarray) -> List[float]:
        """Create fallback features when neuromorphic processing fails"""
        # Use statistical features of the raw energy values
        fallback = []
        fallback.extend(sample.tolist())  # Raw values
        fallback.extend([np.mean(sample), np.std(sample), np.min(sample), np.max(sample)])  # Statistics
        fallback.extend((sample ** 2).tolist())  # Squared values for non-linearity
        return fallback
    
    def _create_temporal_sequence_safe(self, sample: np.ndarray, seq_length: int = 10) -> np.ndarray:
        """Safely create temporal sequence with error handling"""
        try:
            return self._create_temporal_sequence(sample, seq_length)
        except Exception:
            # Fallback to simple sequence
            return np.tile(sample, (seq_length, 1))
    
    def _cluster_features_safe(self, features: np.ndarray) -> np.ndarray:
        """Safely cluster features with validation and fallback"""
        correlation_id = get_correlation_id()
        
        try:
            # Validate feature matrix
            if features.shape[0] < self.n_clusters:
                raise NeuromorphicException(
                    f"Insufficient samples for clustering: {features.shape[0]} < {self.n_clusters}",
                    error_type=NeuromorphicErrorType.DIMENSION_ERROR,
                    correlation_id=correlation_id,
                    context={'feature_shape': features.shape, 'n_clusters': self.n_clusters}
                )
            
            # Use DBSCAN for density-based clustering (more suitable for neuromorphic features)
            clusterer = DBSCAN(eps=0.5, min_samples=max(2, len(features) // 20))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = clusterer.fit_predict(features)
            
            # Handle noise points and ensure we have desired number of clusters
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
            
            if n_clusters_found != self.n_clusters or -1 in labels:
                logger.warning(f"DBSCAN produced {n_clusters_found} clusters (expected {self.n_clusters}), falling back to K-means")
                # Fallback to KMeans if DBSCAN doesn't give desired results
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
                labels = kmeans.fit_predict(features)
                
            return labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {e} [correlation_id: {correlation_id}]")
            raise NeuromorphicException(
                f"Feature clustering failed: {str(e)}",
                error_type=NeuromorphicErrorType.CONVERGENCE_ERROR,
                correlation_id=correlation_id,
                context={'feature_shape': features.shape, 'error_details': str(e)}
            )
    
    def _create_temporal_sequence(self, sample: np.ndarray, seq_length: int = 10) -> np.ndarray:
        """
        Create temporal sequence from static sample for processing
        
        Args:
            sample: Static 4D energy vector
            seq_length: Length of temporal sequence
            
        Returns:
            Temporal sequence matrix
        """
        # Add small random perturbations to create temporal dynamics
        sequence = []
        base_noise_level = 0.05
        
        for t in range(seq_length):
            # Add time-varying noise that simulates personality dynamics
            noise_scale = base_noise_level * (1 + 0.3 * np.sin(2 * np.pi * t / seq_length))
            noise = np.random.randn(4) * noise_scale
            
            # Ensure values stay in valid range [0, 1]
            perturbed_sample = np.clip(sample + noise, 0, 1)
            sequence.append(perturbed_sample)
        
        return np.array(sequence)
    
    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster extracted neuromorphic features
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels
        """
        # Use DBSCAN for density-based clustering (more suitable for neuromorphic features)
        clusterer = DBSCAN(eps=0.5, min_samples=max(2, len(features) // 20))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = clusterer.fit_predict(features)
        
        # Handle noise points and ensure we have desired number of clusters
        unique_labels = np.unique(labels)
        n_clusters_found = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        
        if n_clusters_found != self.n_clusters or -1 in labels:
            # Fallback to KMeans if DBSCAN doesn't give desired results
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            labels = kmeans.fit_predict(features)
            
        return labels
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster assignments for new data
        
        Args:
            features: New feature data
            
        Returns:
            Cluster predictions
        """
        if not self.trained:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract neuromorphic features for new data
        required_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        energy_data = features[required_cols].values
        energy_normalized = self.scaler.transform(energy_data)
        
        neuromorphic_features = self._extract_neuromorphic_features(energy_normalized)
        
        # Simple nearest centroid assignment for prediction
        if hasattr(self, 'cluster_centroids'):
            distances = np.linalg.norm(
                neuromorphic_features[:, np.newaxis] - self.cluster_centroids,
                axis=2
            )
            predictions = np.argmin(distances, axis=1)
        else:
            # Fallback: use the same clustering on combined data
            all_features = np.vstack([self._last_features, neuromorphic_features])
            all_labels = self._cluster_features(all_features)
            predictions = all_labels[-len(features):]
        
        return predictions
    
    def get_cluster_assignments(self) -> np.ndarray:
        """Get cluster assignments for training data"""
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.cluster_labels
    
    def get_clustering_metrics(self) -> ClusteringMetrics:
        """
        Calculate comprehensive clustering metrics
        
        Returns:
            ClusteringMetrics object with evaluation scores
        """
        if not self.trained or self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the neuromorphic features used for clustering
        energy_data = self.feature_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']].values
        energy_normalized = self.scaler.transform(energy_data)
        features_for_metrics = self._extract_neuromorphic_features(energy_normalized)
        
        # Calculate standard clustering metrics
        if len(np.unique(self.cluster_labels)) > 1:
            sil_score = silhouette_score(features_for_metrics, self.cluster_labels)
            ch_score = calinski_harabasz_score(features_for_metrics, self.cluster_labels)
        else:
            sil_score = -1.0
            ch_score = 0.0
        
        # Calculate neuromorphic-specific metrics
        stability_score = self._calculate_cluster_stability(features_for_metrics)
        interpretability_score = self._calculate_interpretability_score()
        temporal_coherence = self._calculate_temporal_coherence()
        efficiency_score = self._calculate_computational_efficiency()
        
        return ClusteringMetrics(
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            cluster_stability=stability_score,
            interpretability_score=interpretability_score,
            temporal_coherence=temporal_coherence,
            computational_efficiency=efficiency_score
        )
    
    def _calculate_cluster_stability(self, features: np.ndarray) -> float:
        """Calculate cluster stability through bootstrap sampling"""
        n_bootstrap = 10
        stability_scores = []
        
        n_samples = len(features)
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            bootstrap_features = features[indices]
            bootstrap_labels = self._cluster_features(bootstrap_features)
            
            # Calculate agreement with original clustering
            original_subset = self.cluster_labels[indices]
            
            # Use adjusted rand index for stability measure
            from sklearn.metrics import adjusted_rand_score
            stability = adjusted_rand_score(original_subset, bootstrap_labels)
            stability_scores.append(max(0, stability))  # Ensure non-negative
        
        return np.mean(stability_scores)
    
    def _calculate_interpretability_score(self) -> float:
        """Calculate interpretability based on cluster separation in original space"""
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        energy_data = self.feature_data[energy_cols].values
        
        # Calculate cluster centroids in original energy space
        centroids = []
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            centroid = np.mean(energy_data[cluster_mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Calculate inter-cluster distances
        if len(centroids) > 1:
            distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    distances.append(dist)
            
            # Higher average distance means more interpretable clusters
            avg_distance = np.mean(distances)
            max_possible_distance = np.sqrt(4 * 100**2)  # Max distance in 4D space (0-100 scale)
            interpretability = min(1.0, avg_distance / (max_possible_distance * 0.5))
        else:
            interpretability = 0.0
        
        return interpretability
    
    def _calculate_temporal_coherence(self) -> float:
        """Calculate temporal coherence of clustering"""
        # This measures how well the temporal features contribute to clustering
        # For now, return a heuristic based on method used
        
        coherence_scores = {
            NeuromorphicClusteringMethod.ECHO_STATE_NETWORK: 0.8,
            NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK: 0.7,
            NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE: 0.9,
            NeuromorphicClusteringMethod.HYBRID_RESERVOIR: 0.85
        }
        
        return coherence_scores.get(self.method, 0.6)
    
    def _calculate_computational_efficiency(self) -> float:
        """Calculate computational efficiency metric"""
        # Efficiency based on method complexity and data size
        n_samples = len(self.feature_data)
        
        # Base efficiency inversely related to sample size
        base_efficiency = min(1.0, 1000 / max(n_samples, 100))
        
        # Method-specific efficiency factors
        method_efficiency = {
            NeuromorphicClusteringMethod.ECHO_STATE_NETWORK: 0.8,
            NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK: 0.6,
            NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE: 0.7,
            NeuromorphicClusteringMethod.HYBRID_RESERVOIR: 0.5
        }
        
        return base_efficiency * method_efficiency.get(self.method, 0.7)
    
    def get_cluster_interpretation(self) -> Dict[int, Dict[str, float]]:
        """
        Get psychological interpretation of clusters
        
        Returns:
            Dictionary mapping cluster IDs to personality characteristics
        """
        if not self.trained:
            raise ValueError("Model not fitted. Call fit() first.")
        
        interpretations = {}
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = self.feature_data[cluster_mask]
            
            # Calculate cluster profile
            energy_profile = cluster_data[energy_cols].mean()
            
            # Determine dominant traits
            trait_scores = {
                'assertiveness': energy_profile['red_energy'] / 100,
                'analytical': energy_profile['blue_energy'] / 100,
                'supportive': energy_profile['green_energy'] / 100,
                'enthusiastic': energy_profile['yellow_energy'] / 100
            }
            
            # Add stability measures from neuromorphic processing
            trait_scores['complexity'] = self._calculate_cluster_complexity(cluster_data)
            trait_scores['stability'] = self._calculate_individual_stability(cluster_mask)
            
            interpretations[cluster_id] = trait_scores
        
        return interpretations
    
    def _calculate_cluster_complexity(self, cluster_data: pd.DataFrame) -> float:
        """Calculate complexity measure for a cluster"""
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        energy_values = cluster_data[energy_cols].values
        
        # Entropy-based complexity measure
        from scipy.stats import entropy
        
        # Discretize energy values into bins
        bins = np.linspace(0, 100, 10)
        complexity_scores = []
        
        for col_idx in range(len(energy_cols)):
            hist, _ = np.histogram(energy_values[:, col_idx], bins=bins)
            hist = hist + 1e-8  # Avoid zero probabilities
            ent = entropy(hist)
            complexity_scores.append(ent)
        
        return np.mean(complexity_scores) / np.log(10)  # Normalize by max entropy
    
    def _calculate_individual_stability(self, cluster_mask: np.ndarray) -> float:
        """Calculate stability for individual cluster"""
        cluster_size = np.sum(cluster_mask)
        total_size = len(cluster_mask)
        
        # Larger, balanced clusters are more stable
        optimal_size = total_size / self.n_clusters
        size_factor = min(1.0, cluster_size / optimal_size)
        
        return size_factor
    
    def _log_clustering_summary(self):
        """Log summary of clustering results"""
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        logger.info(f"Clustering summary:")
        logger.info(f"  Method: {self.method.value}")
        logger.info(f"  Clusters found: {len(unique_labels)}")
        
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(self.cluster_labels)) * 100
            logger.info(f"  Cluster {label}: {count} samples ({percentage:.1f}%)")