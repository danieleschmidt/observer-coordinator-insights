"""Generation 3 Configuration Management for Neuromorphic Clustering
Centralized configuration with environment-based overrides and validation
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU acceleration configuration"""
    enabled: bool = True
    fallback_to_cpu: bool = True
    memory_fraction: float = 0.8
    preferred_device_id: int = 0
    enable_mixed_precision: bool = False

    def __post_init__(self):
        if not (0.0 <= self.memory_fraction <= 1.0):
            raise ValueError("memory_fraction must be between 0.0 and 1.0")


@dataclass
class CachingConfig:
    """Caching system configuration"""
    enabled: bool = True
    l1_size: int = 500
    l1_ttl_seconds: int = 1800
    l2_enabled: bool = True
    l2_max_size_gb: float = 5.0
    l3_redis_enabled: bool = False
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    compression_enabled: bool = True
    compression_algorithm: str = 'auto'  # 'auto', 'zlib', 'lz4', 'none'


@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    enabled: bool = True
    min_workers: int = 1
    max_workers: int = 20
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    kubernetes_enabled: bool = False
    k8s_namespace: str = 'default'
    k8s_deployment: str = 'neuromorphic-clustering'
    min_k8s_replicas: int = 1
    max_k8s_replicas: int = 10
    streaming_enabled: bool = True
    streaming_chunk_size: int = 1000
    drift_detection_threshold: float = 0.3


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    memory_mapping_enabled: bool = True
    memory_mapping_threshold_mb: int = 50
    vectorized_operations: bool = True
    async_processing: bool = True
    max_concurrent_tasks: int = 10
    profiling_enabled: bool = True
    detailed_profiling: bool = False
    optimization_level: str = 'balanced'  # 'conservative', 'balanced', 'aggressive'


@dataclass
class DistributedConfig:
    """Distributed clustering configuration"""
    enabled: bool = False
    coordinator_enabled: bool = False
    api_gateway_enabled: bool = False
    gateway_host: str = '0.0.0.0'
    gateway_port: int = 8080
    celery_enabled: bool = False
    celery_broker: str = 'redis://localhost:6379/0'
    celery_backend: str = 'redis://localhost:6379/0'
    service_discovery_enabled: bool = True
    health_check_interval: int = 60


@dataclass
class NeuromorphicConfig:
    """Neuromorphic clustering specific configuration"""
    default_method: str = 'hybrid_reservoir'
    enable_fallback: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: int = 120
    max_retries: int = 2
    timeout_seconds: int = 600

    # ESN parameters
    esn_reservoir_size: int = 100
    esn_spectral_radius: float = 0.95
    esn_sparsity: float = 0.1
    esn_leaking_rate: float = 0.3

    # SNN parameters
    snn_n_neurons: int = 50
    snn_threshold: float = 1.0
    snn_tau_membrane: float = 20.0
    snn_learning_rate: float = 0.01

    # LSM parameters
    lsm_liquid_size: int = 64
    lsm_connection_prob: float = 0.3
    lsm_tau_membrane: float = 30.0


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    enabled: bool = True
    log_level: str = 'INFO'
    metrics_collection_enabled: bool = True
    performance_tracking_enabled: bool = True
    resource_monitoring_enabled: bool = True
    export_metrics: bool = False
    metrics_export_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 90.0,
        'memory_percent': 85.0,
        'gpu_memory_percent': 90.0,
        'processing_time_seconds': 300.0,
        'error_rate_percent': 5.0
    })


@dataclass
class Gen3Config:
    """Comprehensive Generation 3 configuration"""
    # Core settings
    version: str = '3.0.0'
    environment: str = 'development'  # 'development', 'staging', 'production'
    debug_mode: bool = False

    # Component configurations
    gpu: GPUConfig = field(default_factory=GPUConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    neuromorphic: NeuromorphicConfig = field(default_factory=NeuromorphicConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Additional settings
    data_directory: Optional[str] = None
    temp_directory: Optional[str] = None
    max_memory_gb: Optional[float] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._apply_environment_overrides()

    def _validate_config(self):
        """Validate configuration values"""
        # Environment validation
        valid_environments = ['development', 'staging', 'production']
        if self.environment not in valid_environments:
            raise ValueError(f"environment must be one of {valid_environments}")

        # Performance level validation
        valid_optimization_levels = ['conservative', 'balanced', 'aggressive']
        if self.performance.optimization_level not in valid_optimization_levels:
            raise ValueError(f"optimization_level must be one of {valid_optimization_levels}")

        # Resource limits validation
        if self.max_memory_gb and self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")

        # Network ports validation
        if not (1 <= self.caching.redis_port <= 65535):
            raise ValueError("redis_port must be between 1 and 65535")

        if not (1 <= self.distributed.gateway_port <= 65535):
            raise ValueError("gateway_port must be between 1 and 65535")

    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # GPU settings
        if 'NEUROMORPHIC_GPU_ENABLED' in os.environ:
            self.gpu.enabled = os.environ['NEUROMORPHIC_GPU_ENABLED'].lower() == 'true'

        # Caching settings
        if 'NEUROMORPHIC_CACHE_ENABLED' in os.environ:
            self.caching.enabled = os.environ['NEUROMORPHIC_CACHE_ENABLED'].lower() == 'true'

        if 'REDIS_HOST' in os.environ:
            self.caching.redis_host = os.environ['REDIS_HOST']

        if 'REDIS_PORT' in os.environ:
            self.caching.redis_port = int(os.environ['REDIS_PORT'])

        # Scaling settings
        if 'NEUROMORPHIC_MAX_WORKERS' in os.environ:
            self.scaling.max_workers = int(os.environ['NEUROMORPHIC_MAX_WORKERS'])

        if 'KUBERNETES_NAMESPACE' in os.environ:
            self.scaling.k8s_namespace = os.environ['KUBERNETES_NAMESPACE']

        # Distributed settings
        if 'NEUROMORPHIC_DISTRIBUTED_ENABLED' in os.environ:
            self.distributed.enabled = os.environ['NEUROMORPHIC_DISTRIBUTED_ENABLED'].lower() == 'true'

        if 'CELERY_BROKER_URL' in os.environ:
            self.distributed.celery_broker = os.environ['CELERY_BROKER_URL']

        if 'CELERY_RESULT_BACKEND' in os.environ:
            self.distributed.celery_backend = os.environ['CELERY_RESULT_BACKEND']

        # Performance settings
        if 'NEUROMORPHIC_OPTIMIZATION_LEVEL' in os.environ:
            self.performance.optimization_level = os.environ['NEUROMORPHIC_OPTIMIZATION_LEVEL']

        # Monitoring settings
        if 'LOG_LEVEL' in os.environ:
            self.monitoring.log_level = os.environ['LOG_LEVEL']

        # Environment-specific defaults
        if self.environment == 'production':
            self.debug_mode = False
            self.monitoring.log_level = 'WARNING'
            self.performance.detailed_profiling = False
        elif self.environment == 'development':
            self.debug_mode = True
            self.monitoring.log_level = 'DEBUG'
            self.performance.detailed_profiling = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to file"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'Gen3Config':
        """Load configuration from file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        if file_path.suffix.lower() == '.json':
            with open(file_path) as f:
                data = json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Reconstruct nested dataclass objects
        config = cls()
        config._update_from_dict(data)

        logger.info(f"Configuration loaded from {file_path}")
        return config

    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary data"""
        for key, value in data.items():
            if hasattr(self, key):
                current_attr = getattr(self, key)

                if isinstance(current_attr, (GPUConfig, CachingConfig, ScalingConfig,
                                          PerformanceConfig, DistributedConfig,
                                          NeuromorphicConfig, MonitoringConfig)):
                    # Update nested dataclass
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            if hasattr(current_attr, nested_key):
                                setattr(current_attr, nested_key, nested_value)
                else:
                    setattr(self, key, value)

    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration including computed values"""
        config_dict = self.to_dict()

        # Add computed values
        config_dict['computed'] = {
            'cache_enabled': self.caching.enabled,
            'gpu_acceleration_available': self._check_gpu_availability(),
            'distributed_mode': self.distributed.enabled,
            'kubernetes_mode': self.scaling.kubernetes_enabled,
            'production_mode': self.environment == 'production',
            'total_worker_capacity': self.scaling.max_workers,
            'memory_optimization_enabled': self.performance.memory_mapping_enabled
        }

        return config_dict

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is actually available"""
        if not self.gpu.enabled:
            return False

        try:
            import cupy
            cupy.cuda.Device(self.gpu.preferred_device_id).use()
            return True
        except:
            return False

    def apply_optimization_profile(self, profile: str):
        """Apply predefined optimization profile"""
        if profile == 'high_throughput':
            self.performance.optimization_level = 'aggressive'
            self.scaling.max_workers = min(32, self.scaling.max_workers * 2)
            self.caching.l1_size = 1000
            self.performance.async_processing = True
            self.performance.max_concurrent_tasks = 20

        elif profile == 'low_latency':
            self.performance.optimization_level = 'aggressive'
            self.caching.l1_ttl_seconds = 300  # 5 minutes
            self.gpu.enabled = True
            self.performance.vectorized_operations = True
            self.neuromorphic.timeout_seconds = 60

        elif profile == 'memory_optimized':
            self.performance.memory_mapping_enabled = True
            self.performance.memory_mapping_threshold_mb = 10
            self.caching.l1_size = 100
            self.caching.compression_enabled = True
            self.scaling.streaming_chunk_size = 500

        elif profile == 'distributed':
            self.distributed.enabled = True
            self.distributed.coordinator_enabled = True
            self.distributed.api_gateway_enabled = True
            self.scaling.kubernetes_enabled = True
            self.caching.l3_redis_enabled = True

        else:
            raise ValueError(f"Unknown optimization profile: {profile}")

        logger.info(f"Applied optimization profile: {profile}")


class ConfigManager:
    """Global configuration manager"""

    def __init__(self):
        self._config: Optional[Gen3Config] = None
        self._config_file: Optional[Path] = None
        self._auto_reload: bool = False

    def initialize(self, config: Optional[Gen3Config] = None,
                  config_file: Optional[Union[str, Path]] = None,
                  auto_reload: bool = False) -> Gen3Config:
        """Initialize configuration"""
        if config:
            self._config = config
        elif config_file:
            self._config_file = Path(config_file)
            self._config = Gen3Config.load_from_file(self._config_file)
        else:
            self._config = Gen3Config()

        self._auto_reload = auto_reload

        # Setup logging based on config
        self._setup_logging()

        logger.info(f"Configuration initialized for environment: {self._config.environment}")
        return self._config

    def get_config(self) -> Gen3Config:
        """Get current configuration"""
        if self._config is None:
            return self.initialize()

        # Auto-reload if enabled and file exists
        if (self._auto_reload and self._config_file and
            self._config_file.exists()):
            try:
                self._config = Gen3Config.load_from_file(self._config_file)
            except Exception as e:
                logger.warning(f"Failed to reload configuration: {e}")

        return self._config

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        if self._config is None:
            self.initialize()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        logger.info("Configuration updated")

    def save_config(self, file_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file"""
        if self._config is None:
            raise RuntimeError("No configuration to save")

        target_path = Path(file_path) if file_path else self._config_file
        if target_path is None:
            raise ValueError("No file path specified for saving configuration")

        self._config.save_to_file(target_path)

    def _setup_logging(self):
        """Setup logging based on configuration"""
        if self._config:
            log_level = getattr(logging, self._config.monitoring.log_level.upper())
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Gen3Config:
    """Get global configuration"""
    return config_manager.get_config()


def initialize_config(config: Optional[Gen3Config] = None,
                     config_file: Optional[Union[str, Path]] = None,
                     auto_reload: bool = False) -> Gen3Config:
    """Initialize global configuration"""
    return config_manager.initialize(config, config_file, auto_reload)


def update_config(**kwargs):
    """Update global configuration"""
    config_manager.update_config(**kwargs)


# Default configuration profiles
OPTIMIZATION_PROFILES = {
    'development': {
        'environment': 'development',
        'debug_mode': True,
        'monitoring': {'log_level': 'DEBUG'},
        'performance': {'detailed_profiling': True}
    },
    'production': {
        'environment': 'production',
        'debug_mode': False,
        'monitoring': {'log_level': 'WARNING'},
        'performance': {'optimization_level': 'aggressive'}
    },
    'high_performance': {
        'gpu': {'enabled': True},
        'performance': {
            'optimization_level': 'aggressive',
            'vectorized_operations': True,
            'async_processing': True
        },
        'scaling': {'max_workers': 32}
    },
    'distributed': {
        'distributed': {
            'enabled': True,
            'coordinator_enabled': True,
            'api_gateway_enabled': True
        },
        'caching': {'l3_redis_enabled': True},
        'scaling': {'kubernetes_enabled': True}
    }
}
