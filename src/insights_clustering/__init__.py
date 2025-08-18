"""Insights Discovery Data Clustering Module - Generation 4 Enhanced
Advanced quantum neuromorphic clustering with adaptive AI optimization
"""

from .clustering import KMeansClusterer
from .parser import InsightsDataParser
from .validator import DataValidator


# Import neuromorphic clustering conditionally
try:
    from .neuromorphic_clustering import NeuromorphicClusterer
    _NEUROMORPHIC_AVAILABLE = True
except ImportError:
    _NEUROMORPHIC_AVAILABLE = False
    NeuromorphicClusterer = None

# Generation 4 Quantum Enhanced Components
try:
    from .adaptive_ai_engine import (
        AdaptiveAIEngine,
        HyperparameterOptimizer,
        LearningStrategy,
        ModelPerformance,
    )
    from .gen4_integration import (
        Gen4ClusteringPipeline,
        Gen4Config,
        quantum_neuromorphic_clustering,
    )
    from .quantum_neuromorphic import (
        QuantumNeuromorphicClusterer,
        QuantumNeuron,
        QuantumReservoir,
        QuantumState,
        create_quantum_ensemble,
    )

    GENERATION_4_AVAILABLE = True

except ImportError as e:
    # Graceful fallback for environments without Generation 4 dependencies
    GENERATION_4_AVAILABLE = False
    import warnings
    warnings.warn(f"Generation 4 components not available: {e}")

# Export all available components
__all__ = [
    # Core components
    'InsightsDataParser',
    'KMeansClusterer',
    'DataValidator',
    *(['NeuromorphicClusterer'] if _NEUROMORPHIC_AVAILABLE else [])
]

# Add Generation 4 components if available
if GENERATION_4_AVAILABLE:
    __all__.extend([
        # Quantum Neuromorphic
        'QuantumNeuromorphicClusterer',
        'create_quantum_ensemble',
        'QuantumState',
        'QuantumNeuron',
        'QuantumReservoir',

        # Adaptive AI
        'AdaptiveAIEngine',
        'LearningStrategy',
        'HyperparameterOptimizer',
        'ModelPerformance',

        # Integration
        'Gen4ClusteringPipeline',
        'Gen4Config',
        'quantum_neuromorphic_clustering'
    ])

# Version and capability information
VERSION = "4.0.0"
CAPABILITIES = {
    'basic_clustering': True,
    'neuromorphic_clustering': True,
    'quantum_enhanced': GENERATION_4_AVAILABLE,
    'adaptive_ai': GENERATION_4_AVAILABLE,
    'ensemble_learning': GENERATION_4_AVAILABLE,
    'continuous_optimization': GENERATION_4_AVAILABLE
}
