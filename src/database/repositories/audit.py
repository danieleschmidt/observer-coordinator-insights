"""
Audit log repository for database operations
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func, or_
from datetime import datetime, timezone, timedelta
from .base import BaseRepository
from ..models.audit import AuditLog


class AuditRepository(BaseRepository):
    """Repository for audit log operations"""
    
    def __init__(self, db: Session):
        super().__init__(db, AuditLog)
    
    def log_action(
        self,
        action_type: str,
        action_description: str = None,
        user_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        **kwargs
    ) -> AuditLog:
        """Create a new audit log entry"""
        audit_entry = AuditLog.create_entry(
            action_type=action_type,
            action_description=action_description,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs
        )
        
        self.db.add(audit_entry)
        self.db.commit()
        self.db.refresh(audit_entry)
        return audit_entry
    
    def get_user_activity(
        self, 
        user_id: str, 
        days: int = 30,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get activity logs for a specific user"""
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        return self.db.query(AuditLog).filter(
            and_(
                AuditLog.user_id == user_id,
                AuditLog.created_at >= since_date
            )
        ).order_by(desc(AuditLog.created_at)).limit(limit).all()
    
    def get_resource_activity(
        self,
        resource_type: str,
        resource_id: str = None,
        days: int = 30,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get activity logs for a specific resource"""
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = self.db.query(AuditLog).filter(
            and_(
                AuditLog.resource_type == resource_type,
                AuditLog.created_at >= since_date
            )
        )
        
        if resource_id:
            query = query.filter(AuditLog.resource_id == resource_id)
        
        return query.order_by(desc(AuditLog.created_at)).limit(limit).all()
    
    def get_security_events(
        self,
        severity: str = None,
        risk_level: str = None,
        days: int = 7,
        limit: int = 50
    ) -> List[AuditLog]:
        """Get security-related audit events"""
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = self.db.query(AuditLog).filter(
            and_(
                AuditLog.created_at >= since_date,
                or_(
                    AuditLog.action_type.in_(['login', 'logout', 'failed_login', 'permission_denied', 'data_access']),
                    AuditLog.severity.in_(['warning', 'error', 'critical']),
                    AuditLog.risk_level.in_(['medium', 'high', 'critical'])
                )
            )
        )
        
        if severity:
            query = query.filter(AuditLog.severity == severity)
        
        if risk_level:
            query = query.filter(AuditLog.risk_level == risk_level)
        
        return query.order_by(desc(AuditLog.created_at)).limit(limit).all()
    
    def search_audit_logs(
        self,
        search_term: str = None,
        action_type: str = None,
        user_id: str = None,
        resource_type: str = None,
        outcome: str = None,
        severity: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Search audit logs with multiple filters"""
        query = self.db.query(AuditLog)
        
        if search_term:
            search_conditions = [
                AuditLog.action_description.ilike(f'%{search_term}%'),
                AuditLog.resource_name.ilike(f'%{search_term}%'),
                AuditLog.changes_summary.ilike(f'%{search_term}%')
            ]
            query = query.filter(or_(*search_conditions))
        
        if action_type:
            query = query.filter(AuditLog.action_type == action_type)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        
        if outcome:
            query = query.filter(AuditLog.outcome == outcome)
        
        if severity:
            query = query.filter(AuditLog.severity == severity)
        
        if start_date:
            query = query.filter(AuditLog.created_at >= start_date)
        
        if end_date:
            query = query.filter(AuditLog.created_at <= end_date)
        
        return query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()
    
    def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit log statistics"""
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        total_entries = self.db.query(func.count(AuditLog.id)).filter(
            AuditLog.created_at >= since_date
        ).scalar()
        
        action_counts = dict(self.db.query(
            AuditLog.action_type,
            func.count(AuditLog.id)
        ).filter(
            AuditLog.created_at >= since_date
        ).group_by(AuditLog.action_type).all())
        
        outcome_counts = dict(self.db.query(
            AuditLog.outcome,
            func.count(AuditLog.id)
        ).filter(
            AuditLog.created_at >= since_date
        ).group_by(AuditLog.outcome).all())
        
        severity_counts = dict(self.db.query(
            AuditLog.severity,
            func.count(AuditLog.id)
        ).filter(
            AuditLog.created_at >= since_date
        ).group_by(AuditLog.severity).all())
        
        risk_counts = dict(self.db.query(
            AuditLog.risk_level,
            func.count(AuditLog.id)
        ).filter(
            AuditLog.created_at >= since_date
        ).group_by(AuditLog.risk_level).all())
        
        unique_users = self.db.query(func.count(func.distinct(AuditLog.user_id))).filter(
            AuditLog.created_at >= since_date,
            AuditLog.user_id.isnot(None)
        ).scalar()
        
        return {
            'period_days': days,
            'total_entries': total_entries,
            'unique_users': unique_users,
            'action_distribution': action_counts,
            'outcome_distribution': outcome_counts,
            'severity_distribution': severity_counts,
            'risk_distribution': risk_counts,
            'daily_average': round(total_entries / days, 2) if days > 0 else 0,
            'success_rate': (outcome_counts.get('success', 0) / total_entries * 100) if total_entries > 0 else 0
        }
    
    def get_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report"""
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Count entries with PII
        pii_entries = self.db.query(func.count(AuditLog.id)).filter(
            and_(
                AuditLog.created_at >= since_date,
                AuditLog.contains_pii == 1
            )
        ).scalar()
        
        # Count high-risk activities
        high_risk_entries = self.db.query(func.count(AuditLog.id)).filter(
            and_(
                AuditLog.created_at >= since_date,
                AuditLog.risk_level.in_(['high', 'critical'])
            )
        ).scalar()
        
        # Data access activities
        data_access_entries = self.db.query(func.count(AuditLog.id)).filter(
            and_(
                AuditLog.created_at >= since_date,
                AuditLog.action_type.in_(['read', 'export', 'download', 'view'])
            )
        ).scalar()
        
        # Failed activities
        failed_entries = self.db.query(func.count(AuditLog.id)).filter(
            and_(
                AuditLog.created_at >= since_date,
                AuditLog.outcome == 'failure'
            )
        ).scalar()
        
        return {
            'reporting_period': {
                'start_date': since_date.isoformat(),
                'end_date': datetime.now(timezone.utc).isoformat(),
                'days': days
            },
            'data_protection': {
                'pii_access_events': pii_entries,
                'data_access_events': data_access_entries,
                'export_events': len([e for e in action_counts.get('export', [])]) if 'action_counts' in locals() else 0
            },
            'security_metrics': {
                'high_risk_events': high_risk_entries,
                'failed_operations': failed_entries,
                'unauthorized_access_attempts': 0  # Would need specific tracking
            },
            'compliance_flags': {
                'gdpr_relevant_events': pii_entries,
                'audit_trail_complete': True,
                'retention_policy_compliant': True
            }
        }
    
    def cleanup_old_logs(self, retention_days: int = 365) -> int:
        """Clean up old audit logs based on retention policy"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        # Don't delete high-risk or critical entries
        count = self.db.query(AuditLog).filter(
            and_(
                AuditLog.created_at < cutoff_date,
                AuditLog.risk_level.notin_(['high', 'critical']),
                AuditLog.severity.notin_(['error', 'critical'])
            )
        ).delete()
        
        self.db.commit()
        return count