"""
Insights Discovery Data Clustering Module - Generation 4 Enhanced
Advanced quantum neuromorphic clustering with adaptive AI optimization
"""

from .parser import InsightsDataParser
from .clustering import KMeansClusterer
from .validator import DataValidator
from .neuromorphic_clustering import NeuromorphicClusterer

# Generation 4 Quantum Enhanced Components
try:
    from .quantum_neuromorphic import (
        QuantumNeuromorphicClusterer, 
        create_quantum_ensemble,
        QuantumState,
        QuantumNeuron,
        QuantumReservoir
    )
    from .adaptive_ai_engine import (
        AdaptiveAIEngine,
        LearningStrategy,
        HyperparameterOptimizer,
        ModelPerformance
    )
    from .gen4_integration import (
        Gen4ClusteringPipeline,
        Gen4Config,
        quantum_neuromorphic_clustering
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
    'NeuromorphicClusterer'
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