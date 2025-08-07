"""
Distributed processing module for Generation 3 neuromorphic clustering
"""

from .clustering_coordinator import (
    DistributedClusteringCoordinator,
    initialize_coordinator,
    get_coordinator
)

__all__ = [
    'DistributedClusteringCoordinator',
    'initialize_coordinator',
    'get_coordinator'
]