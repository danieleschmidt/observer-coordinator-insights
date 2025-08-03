"""
Analysis job repository for database operations
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
from datetime import datetime, timezone
from .base import BaseRepository
from ..models.analysis import AnalysisJob, AnalysisResult


class JobRepository(BaseRepository):
    """Repository for analysis job operations"""
    
    def __init__(self, db: Session):
        super().__init__(db, AnalysisJob)
    
    def get_active_jobs(self) -> List[AnalysisJob]:
        """Get all active (running or pending) jobs"""
        return self.db.query(AnalysisJob).filter(
            AnalysisJob.status.in_(['pending', 'running'])
        ).order_by(AnalysisJob.priority.desc(), AnalysisJob.created_at).all()
    
    def get_jobs_by_status(self, status: str) -> List[AnalysisJob]:
        """Get jobs by status"""
        return self.db.query(AnalysisJob).filter(
            AnalysisJob.status == status
        ).order_by(desc(AnalysisJob.created_at)).all()
    
    def get_jobs_by_type(self, job_type: str, limit: int = 50) -> List[AnalysisJob]:
        """Get jobs by type"""
        return self.db.query(AnalysisJob).filter(
            AnalysisJob.job_type == job_type
        ).order_by(desc(AnalysisJob.created_at)).limit(limit).all()
    
    def get_user_jobs(self, user_id: str, limit: int = 50) -> List[AnalysisJob]:
        """Get jobs for a specific user"""
        return self.db.query(AnalysisJob).filter(
            AnalysisJob.user_id == user_id
        ).order_by(desc(AnalysisJob.created_at)).limit(limit).all()
    
    def get_job_queue(self, priority: str = None) -> List[AnalysisJob]:
        """Get pending jobs in queue order"""
        query = self.db.query(AnalysisJob).filter(
            AnalysisJob.status == 'pending'
        )
        
        if priority:
            query = query.filter(AnalysisJob.priority == priority)
        
        # Order by priority (high first) then creation time
        priority_order = {'urgent': 4, 'high': 3, 'normal': 2, 'low': 1}
        return sorted(
            query.all(),
            key=lambda job: (priority_order.get(job.priority, 0), job.created_at),
            reverse=True
        )
    
    def update_job_progress(self, job_id: str, progress: float, status: str = None) -> bool:
        """Update job progress and optionally status"""
        job = self.get(job_id)
        if job:
            job.progress = progress
            if status:
                job.status = status
                if status == 'running' and not job.started_at:
                    job.started_at = datetime.now(timezone.utc)
                elif status in ['completed', 'failed', 'cancelled'] and not job.completed_at:
                    job.completed_at = datetime.now(timezone.utc)
                    if job.started_at:
                        job.actual_duration = int((job.completed_at - job.started_at).total_seconds())
            
            self.db.commit()
            return True
        return False
    
    def set_job_result(self, job_id: str, result_data: Dict[str, Any], error_message: str = None) -> bool:
        """Set job result and mark as completed or failed"""
        job = self.get(job_id)
        if job:
            if error_message:
                job.status = 'failed'
                job.error_message = error_message
            else:
                job.status = 'completed'
                job.result_data = str(result_data) if isinstance(result_data, dict) else result_data
            
            job.completed_at = datetime.now(timezone.utc)
            if job.started_at:
                job.actual_duration = int((job.completed_at - job.started_at).total_seconds())
            
            self.db.commit()
            return True
        return False
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get job execution statistics"""
        total_jobs = self.db.query(func.count(AnalysisJob.id)).scalar()
        
        status_counts = dict(self.db.query(
            AnalysisJob.status,
            func.count(AnalysisJob.id)
        ).group_by(AnalysisJob.status).all())
        
        type_counts = dict(self.db.query(
            AnalysisJob.job_type,
            func.count(AnalysisJob.id)
        ).group_by(AnalysisJob.job_type).all())
        
        avg_duration = self.db.query(
            func.avg(AnalysisJob.actual_duration)
        ).filter(
            AnalysisJob.actual_duration.isnot(None)
        ).scalar()
        
        return {
            'total_jobs': total_jobs,
            'status_distribution': status_counts,
            'job_type_distribution': type_counts,
            'average_duration_seconds': round(avg_duration, 2) if avg_duration else 0,
            'success_rate': (status_counts.get('completed', 0) / total_jobs * 100) if total_jobs > 0 else 0
        }


class ResultRepository(BaseRepository):
    """Repository for analysis result operations"""
    
    def __init__(self, db: Session):
        super().__init__(db, AnalysisResult)
    
    def get_by_hash(self, data_hash: str, parameters_hash: str = None) -> Optional[AnalysisResult]:
        """Get cached result by data hash"""
        query = self.db.query(AnalysisResult).filter(
            AnalysisResult.data_hash == data_hash
        )
        
        if parameters_hash:
            query = query.filter(AnalysisResult.parameters_hash == parameters_hash)
        
        return query.filter(
            AnalysisResult.expires_at > datetime.now(timezone.utc)
        ).first()
    
    def get_by_type(self, result_type: str, limit: int = 50) -> List[AnalysisResult]:
        """Get results by type"""
        return self.db.query(AnalysisResult).filter(
            AnalysisResult.result_type == result_type
        ).order_by(desc(AnalysisResult.created_at)).limit(limit).all()
    
    def cleanup_expired_results(self) -> int:
        """Clean up expired results"""
        count = self.db.query(AnalysisResult).filter(
            AnalysisResult.expires_at <= datetime.now(timezone.utc)
        ).delete()
        self.db.commit()
        return count
    
    def update_access_stats(self, result_id: str) -> bool:
        """Update access statistics for a result"""
        result = self.get(result_id)
        if result:
            result.access_count = (result.access_count or 0) + 1
            result.last_accessed = datetime.now(timezone.utc)
            self.db.commit()
            return True
        return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        total_results = self.db.query(func.count(AnalysisResult.id)).scalar()
        
        expired_count = self.db.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.expires_at <= datetime.now(timezone.utc)
        ).scalar()
        
        type_counts = dict(self.db.query(
            AnalysisResult.result_type,
            func.count(AnalysisResult.id)
        ).group_by(AnalysisResult.result_type).all())
        
        avg_access_count = self.db.query(
            func.avg(AnalysisResult.access_count)
        ).scalar()
        
        total_size = self.db.query(
            func.sum(AnalysisResult.data_size)
        ).scalar()
        
        return {
            'total_results': total_results,
            'expired_results': expired_count,
            'active_results': total_results - expired_count,
            'result_type_distribution': type_counts,
            'average_access_count': round(avg_access_count, 2) if avg_access_count else 0,
            'total_cache_size_bytes': total_size or 0,
            'cache_hit_rate': 'Not implemented'  # Would need request tracking
        }