"""
Audit logging data model
"""

from sqlalchemy import Column, String, Text, DateTime, Index
from .base import BaseModel, UUIDMixin


class AuditLog(BaseModel, UUIDMixin):
    """Audit log for tracking system activities"""
    
    __tablename__ = 'audit_logs'
    __table_args__ = (
        Index('idx_action_type', 'action_type'),
        Index('idx_user_id', 'user_id'),
        Index('idx_resource_type', 'resource_type'),
        Index('idx_timestamp', 'created_at'),
        Index('idx_severity', 'severity'),
    )
    
    # Action details
    action_type = Column(
        String(50),
        nullable=False,
        comment="Type of action performed (create, read, update, delete, login, etc.)"
    )
    action_description = Column(
        String(500),
        comment="Human-readable description of the action"
    )
    
    # User and session information
    user_id = Column(
        String(100),
        comment="User who performed the action"
    )
    session_id = Column(
        String(100),
        comment="Session identifier"
    )
    user_agent = Column(
        String(500),
        comment="User agent string"
    )
    ip_address = Column(
        String(45),  # IPv6 compatible
        comment="IP address of the user"
    )
    
    # Resource information
    resource_type = Column(
        String(50),
        comment="Type of resource affected (employee, clustering_result, team, etc.)"
    )
    resource_id = Column(
        String(100),
        comment="ID of the affected resource"
    )
    resource_name = Column(
        String(200),
        comment="Human-readable name of the resource"
    )
    
    # Change tracking
    old_values = Column(
        Text,
        comment="JSON object with previous values (for updates)"
    )
    new_values = Column(
        Text,
        comment="JSON object with new values (for creates/updates)"
    )
    changes_summary = Column(
        Text,
        comment="Summary of changes made"
    )
    
    # Context and metadata
    context = Column(
        Text,
        comment="JSON object with additional context"
    )
    request_id = Column(
        String(100),
        comment="Unique request identifier for tracing"
    )
    correlation_id = Column(
        String(100),
        comment="Correlation ID for related actions"
    )
    
    # Outcome and impact
    outcome = Column(
        String(20),
        nullable=False,
        default='success',
        comment="Action outcome (success, failure, partial)"
    )
    error_message = Column(
        Text,
        comment="Error message if action failed"
    )
    
    # Security and compliance
    severity = Column(
        String(10),
        default='info',
        comment="Log severity (debug, info, warning, error, critical)"
    )
    risk_level = Column(
        String(10),
        default='low',
        comment="Risk level of the action (low, medium, high, critical)"
    )
    compliance_flags = Column(
        Text,
        comment="JSON array of compliance requirements this action relates to"
    )
    
    # Performance metrics
    duration_ms = Column(
        Integer,
        comment="Action duration in milliseconds"
    )
    
    # Data sensitivity
    contains_pii = Column(
        Integer,  # Boolean for SQLite compatibility
        default=0,
        comment="Whether this log entry contains PII (0/1)"
    )
    retention_period = Column(
        Integer,
        comment="Retention period in days for this log entry"
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action_type='{self.action_type}', user_id='{self.user_id}', outcome='{self.outcome}')>"
    
    @classmethod
    def create_entry(
        cls,
        action_type: str,
        action_description: str = None,
        user_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        **kwargs
    ):
        """Create a new audit log entry"""
        return cls(
            action_type=action_type,
            action_description=action_description,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs
        )