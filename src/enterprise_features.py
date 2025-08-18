#!/usr/bin/env python3
"""Enterprise Features Implementation
Multi-tenancy, audit logging, compliance, and enterprise integrations
"""

import json
import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Audit event for compliance tracking"""
    event_id: str
    timestamp: datetime
    user_id: str
    tenant_id: str
    event_type: str
    resource: str
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ComplianceReport:
    """Compliance report for regulatory requirements"""
    report_id: str
    tenant_id: str
    report_type: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    data_summary: Dict[str, Any]
    compliance_status: str
    recommendations: List[str]


class EnterpriseAuditLogger:
    """Enterprise-grade audit logging system"""

    def __init__(self, log_file_path: Optional[Path] = None):
        self.events = []
        self.log_file_path = log_file_path or Path("audit.jsonl")
        self.buffer_size = 100
        self._lock = threading.Lock()

    def log_event(self, user_id: str, tenant_id: str, event_type: str,
                  resource: str, action: str, details: Dict[str, Any] = None,
                  ip_address: str = None, user_agent: str = None,
                  session_id: str = None):
        """Log an audit event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            tenant_id=tenant_id,
            event_type=event_type,
            resource=resource,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )

        with self._lock:
            self.events.append(event)

            # Write to file immediately for compliance
            self._write_event_to_file(event)

            # Keep in-memory buffer for performance
            if len(self.events) > self.buffer_size:
                self.events = self.events[-self.buffer_size:]

        logger.info(f"Audit event logged: {event_type}:{action} by {user_id}")

    def _write_event_to_file(self, event: AuditEvent):
        """Write audit event to file"""
        try:
            # Create directory if it doesn't exist
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and handle datetime serialization
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()

            # Append to JSONL file
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")

    def get_events(self, tenant_id: str = None, user_id: str = None,
                   event_type: str = None, hours: int = 24) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_events = []
        for event in self.events:
            if event.timestamp < cutoff_time:
                continue

            if tenant_id and event.tenant_id != tenant_id:
                continue

            if user_id and event.user_id != user_id:
                continue

            if event_type and event.event_type != event_type:
                continue

            filtered_events.append(event)

        return filtered_events

    def generate_compliance_report(self, tenant_id: str,
                                   period_days: int = 30) -> ComplianceReport:
        """Generate compliance report for a tenant"""
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()

        events = self.get_events(tenant_id=tenant_id, hours=period_days * 24)

        # Analyze events for compliance
        event_types = defaultdict(int)
        actions = defaultdict(int)
        users = set()

        for event in events:
            event_types[event.event_type] += 1
            actions[event.action] += 1
            users.add(event.user_id)

        # Generate summary
        data_summary = {
            "total_events": len(events),
            "unique_users": len(users),
            "event_types": dict(event_types),
            "actions": dict(actions),
            "period_days": period_days
        }

        # Determine compliance status
        compliance_status = "compliant"
        recommendations = []

        if len(events) == 0:
            compliance_status = "warning"
            recommendations.append("No audit events found for the period")

        # Check for suspicious patterns
        if event_types.get("data_access", 0) > 1000:
            compliance_status = "review_required"
            recommendations.append("High volume of data access events detected")

        return ComplianceReport(
            report_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            report_type="audit_summary",
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            data_summary=data_summary,
            compliance_status=compliance_status,
            recommendations=recommendations
        )


class MultiTenantManager:
    """Multi-tenancy management system"""

    def __init__(self):
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self.tenant_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register_tenant(self, tenant_id: str, tenant_info: Dict[str, Any]):
        """Register a new tenant"""
        with self._lock:
            self.tenants[tenant_id] = {
                "tenant_id": tenant_id,
                "created_at": datetime.now().isoformat(),
                "status": "active",
                **tenant_info
            }

            # Default tenant configuration
            self.tenant_configs[tenant_id] = {
                "max_employees": 10000,
                "max_clusters": 50,
                "retention_days": 365,
                "features_enabled": [
                    "clustering",
                    "team_simulation",
                    "analytics",
                    "reporting"
                ],
                "compliance_mode": "standard"
            }

        logger.info(f"Registered tenant: {tenant_id}")

    def get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant configuration"""
        return self.tenant_configs.get(tenant_id, {})

    def validate_tenant_operation(self, tenant_id: str, operation: str,
                                  resource_count: int = 1) -> bool:
        """Validate if tenant can perform operation"""
        config = self.get_tenant_config(tenant_id)

        if not config:
            logger.warning(f"Unknown tenant: {tenant_id}")
            return False

        # Check feature availability
        if operation not in config.get("features_enabled", []):
            logger.warning(f"Feature {operation} not enabled for tenant {tenant_id}")
            return False

        # Check resource limits
        if operation == "clustering":
            max_clusters = config.get("max_clusters", 10)
            if resource_count > max_clusters:
                logger.warning(f"Cluster limit exceeded for tenant {tenant_id}: {resource_count} > {max_clusters}")
                return False

        return True

    def get_tenant_usage_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage statistics for tenant"""
        # In a real implementation, this would query actual usage data
        return {
            "tenant_id": tenant_id,
            "current_employees": 0,
            "clusters_created": 0,
            "analyses_run": 0,
            "storage_used_mb": 0,
            "api_calls_today": 0
        }


class DataRetentionManager:
    """Enterprise data retention and lifecycle management"""

    def __init__(self):
        self.retention_policies: Dict[str, Dict[str, Any]] = {}

    def set_retention_policy(self, tenant_id: str, data_type: str,
                             retention_days: int, archive_enabled: bool = True):
        """Set data retention policy"""
        policy_key = f"{tenant_id}:{data_type}"
        self.retention_policies[policy_key] = {
            "tenant_id": tenant_id,
            "data_type": data_type,
            "retention_days": retention_days,
            "archive_enabled": archive_enabled,
            "created_at": datetime.now().isoformat()
        }

        logger.info(f"Set retention policy for {tenant_id}:{data_type} = {retention_days} days")

    def should_retain_data(self, tenant_id: str, data_type: str,
                           created_at: datetime) -> bool:
        """Check if data should be retained based on policy"""
        policy_key = f"{tenant_id}:{data_type}"
        policy = self.retention_policies.get(policy_key)

        if not policy:
            # Default retention: 1 year
            retention_days = 365
        else:
            retention_days = policy["retention_days"]

        cutoff_date = datetime.now() - timedelta(days=retention_days)
        return created_at >= cutoff_date

    def get_data_for_cleanup(self, tenant_id: str = None) -> List[Dict[str, Any]]:
        """Get data that can be cleaned up based on retention policies"""
        # In a real implementation, this would scan actual data stores
        cleanup_candidates = []

        for policy_key, policy in self.retention_policies.items():
            if tenant_id and not policy_key.startswith(f"{tenant_id}:"):
                continue

            cutoff_date = datetime.now() - timedelta(days=policy["retention_days"])

            # Placeholder for actual data scanning
            cleanup_candidates.append({
                "tenant_id": policy["tenant_id"],
                "data_type": policy["data_type"],
                "cutoff_date": cutoff_date.isoformat(),
                "action": "archive" if policy["archive_enabled"] else "delete"
            })

        return cleanup_candidates


# Global instances for enterprise features
audit_logger = EnterpriseAuditLogger()
multi_tenant_manager = MultiTenantManager()
retention_manager = DataRetentionManager()


def initialize_enterprise_features():
    """Initialize enterprise features"""
    try:
        # Set up default configurations
        logger.info("🏢 Initializing enterprise features")

        # Register default tenant for single-tenant deployments
        multi_tenant_manager.register_tenant("default", {
            "name": "Default Organization",
            "subscription_type": "enterprise"
        })

        # Set default retention policies
        retention_manager.set_retention_policy("default", "clustering_results", 365)
        retention_manager.set_retention_policy("default", "team_compositions", 180)
        retention_manager.set_retention_policy("default", "audit_logs", 2555)  # 7 years

        logger.info("✅ Enterprise features initialized")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize enterprise features: {e}")
        return False


# Convenience functions
def log_audit_event(user_id: str, action: str, resource: str,
                    details: Dict[str, Any] = None, tenant_id: str = "default"):
    """Log an audit event (convenience function)"""
    audit_logger.log_event(
        user_id=user_id,
        tenant_id=tenant_id,
        event_type="system",
        resource=resource,
        action=action,
        details=details
    )


def check_tenant_permission(tenant_id: str, operation: str,
                            resource_count: int = 1) -> bool:
    """Check if tenant has permission for operation"""
    return multi_tenant_manager.validate_tenant_operation(
        tenant_id, operation, resource_count
    )
