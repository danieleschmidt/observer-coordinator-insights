"""Analysis job and result data models
"""

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text

from .base import BaseModel, UUIDMixin


class AnalysisJob(BaseModel, UUIDMixin):
    """Background analysis job tracking"""

    __tablename__ = 'analysis_jobs'
    __table_args__ = (
        Index('idx_job_type', 'job_type'),
        Index('idx_status', 'status'),
        Index('idx_priority', 'priority'),
        Index('idx_created_at', 'created_at'),
    )

    # Job identification
    job_type = Column(
        String(50),
        nullable=False,
        comment="Type of analysis job (clustering, team_simulation, optimization, etc.)"
    )
    title = Column(
        String(200),
        comment="Human-readable job title"
    )
    description = Column(
        Text,
        comment="Job description"
    )

    # Job parameters
    parameters = Column(
        Text,
        nullable=False,
        comment="JSON object with job parameters"
    )
    input_data = Column(
        Text,
        comment="JSON object with input data or reference to data source"
    )

    # Job status and progress
    status = Column(
        String(20),
        nullable=False,
        default='pending',
        comment="Job status (pending, running, completed, failed, cancelled)"
    )
    progress = Column(
        Float,
        default=0.0,
        comment="Job progress percentage (0-100)"
    )
    priority = Column(
        String(10),
        default='normal',
        comment="Job priority (low, normal, high, urgent)"
    )

    # Timing information
    started_at = Column(
        DateTime(timezone=True),
        comment="Job start timestamp"
    )
    completed_at = Column(
        DateTime(timezone=True),
        comment="Job completion timestamp"
    )
    estimated_duration = Column(
        Integer,
        comment="Estimated duration in seconds"
    )
    actual_duration = Column(
        Integer,
        comment="Actual duration in seconds"
    )

    # Results and errors
    result_data = Column(
        Text,
        comment="JSON object with job results"
    )
    error_message = Column(
        Text,
        comment="Error message if job failed"
    )
    error_details = Column(
        Text,
        comment="Detailed error information and stack trace"
    )

    # Resource usage
    cpu_time = Column(
        Float,
        comment="CPU time used in seconds"
    )
    memory_peak = Column(
        Integer,
        comment="Peak memory usage in bytes"
    )

    # Callback and notification
    callback_url = Column(
        String(500),
        comment="Webhook URL to notify on completion"
    )
    notification_sent = Column(
        Integer,  # Boolean for SQLite compatibility
        default=0,
        comment="Whether completion notification was sent (0/1)"
    )

    # Retry and recovery
    retry_count = Column(
        Integer,
        default=0,
        comment="Number of retry attempts"
    )
    max_retries = Column(
        Integer,
        default=3,
        comment="Maximum number of retries allowed"
    )

    # User and session information
    user_id = Column(
        String(100),
        comment="User who submitted the job"
    )
    session_id = Column(
        String(100),
        comment="Session identifier"
    )

    def __repr__(self):
        return f"<AnalysisJob(id={self.id}, job_type='{self.job_type}', status='{self.status}')>"

    @property
    def is_completed(self):
        """Check if job is completed (successfully or with error)"""
        return self.status in ['completed', 'failed', 'cancelled']

    @property
    def is_running(self):
        """Check if job is currently running"""
        return self.status == 'running'

    @property
    def duration(self):
        """Get job duration if completed"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AnalysisResult(BaseModel, UUIDMixin):
    """Stored analysis results for caching and history"""

    __tablename__ = 'analysis_results'
    __table_args__ = (
        Index('idx_result_type', 'result_type'),
        Index('idx_data_hash', 'data_hash'),
        Index('idx_expires_at', 'expires_at'),
    )

    # Result identification
    result_type = Column(
        String(50),
        nullable=False,
        comment="Type of analysis result"
    )
    result_name = Column(
        String(200),
        comment="Human-readable result name"
    )

    # Data and caching
    data_hash = Column(
        String(64),
        comment="SHA-256 hash of input data for cache validation"
    )
    parameters_hash = Column(
        String(64),
        comment="SHA-256 hash of parameters for cache validation"
    )

    # Result content
    result_data = Column(
        Text,
        nullable=False,
        comment="JSON object with analysis results"
    )
    metadata = Column(
        Text,
        comment="JSON object with result metadata"
    )

    # Quality and validation
    quality_score = Column(
        Float,
        comment="Result quality score (0-100)"
    )
    validation_status = Column(
        String(20),
        default='validated',
        comment="Validation status (validated, pending, failed)"
    )

    # Performance metrics
    computation_time = Column(
        Float,
        comment="Computation time in seconds"
    )
    memory_used = Column(
        Integer,
        comment="Memory used in bytes"
    )

    # Caching and lifecycle
    expires_at = Column(
        DateTime(timezone=True),
        comment="Result expiration timestamp"
    )
    access_count = Column(
        Integer,
        default=0,
        comment="Number of times result was accessed"
    )
    last_accessed = Column(
        DateTime(timezone=True),
        comment="Last access timestamp"
    )

    # Size and storage
    data_size = Column(
        Integer,
        comment="Size of result data in bytes"
    )
    compressed = Column(
        Integer,  # Boolean for SQLite compatibility
        default=0,
        comment="Whether result data is compressed (0/1)"
    )

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, result_type='{self.result_type}', quality_score={self.quality_score})>"

    @property
    def is_expired(self):
        """Check if result has expired"""
        if self.expires_at:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc) > self.expires_at
        return False
