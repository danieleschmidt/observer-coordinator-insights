"""API routes package
"""

from .admin import router as admin_router
from .analytics import router as analytics_router
from .health import router as health_router
from .teams import router as teams_router


__all__ = [
    'admin_router',
    'analytics_router',
    'health_router',
    'teams_router'
]
