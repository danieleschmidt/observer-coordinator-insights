"""
API routes package
"""

from .analytics import router as analytics_router
from .teams import router as teams_router
from .health import router as health_router
from .admin import router as admin_router

__all__ = [
    'analytics_router',
    'teams_router', 
    'health_router',
    'admin_router'
]