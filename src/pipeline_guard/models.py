"""
Core data models for Pipeline Guard
"""

import time
from typing import List, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class PipelineState(Enum):
    """Pipeline operational states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    OFFLINE = "offline"


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RESTART = "restart"
    ROLLBACK = "rollback"
    FAILOVER = "failover"
    SCALE_UP = "scale_up"
    CONFIGURATION_RESET = "config_reset"
    DEPENDENCY_REFRESH = "dep_refresh"
    CUSTOM = "custom"


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    strategy: RecoveryStrategy
    action: Callable[[], bool]
    timeout: int = 60
    prerequisites: List[str] = field(default_factory=list)
    rollback_action: Optional[Callable[[], bool]] = None


@dataclass
class PipelineComponent:
    """Represents a monitored pipeline component"""
    name: str
    component_type: str
    health_check: Callable[[], bool]
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    critical: bool = False
    max_failures: int = 3
    failure_count: int = 0
    last_failure: Optional[float] = None
    state: PipelineState = PipelineState.HEALTHY


@dataclass
class HealthMetric:
    """Health metric data point"""
    timestamp: float
    value: float
    component: str
    metric_type: str


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: dict
    open_files: int
    timestamp: float


@dataclass
class FailureEvent:
    """Failure event record"""
    timestamp: float
    component_name: str
    failure_type: str
    error_message: str
    stack_trace: Optional[str] = None
    system_state: dict = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False