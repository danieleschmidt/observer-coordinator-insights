"""
REST API package for Observer Coordinator Insights
"""

from .main import app
from .models import *
from .routes import *

__all__ = ['app']