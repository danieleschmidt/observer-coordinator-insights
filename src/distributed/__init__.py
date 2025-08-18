"""Distributed processing module for Generation 3 neuromorphic clustering
"""

from .clustering_coordinator import (
    DistributedClusteringCoordinator,
    get_coordinator,
    initialize_coordinator,
)


__all__ = [
    'DistributedClusteringCoordinator',
    'get_coordinator',
    'initialize_coordinator'
]
