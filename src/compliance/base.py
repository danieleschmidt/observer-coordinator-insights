"""
Base compliance framework and common enums/classes
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import uuid

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataCategory(Enum):
    """Categories of personal data"""
    BASIC_PERSONAL = "basic_personal"         # Name, email, basic info
    SENSITIVE_PERSONAL = "sensitive_personal" # Health, biometric, etc.
    BEHAVIORAL = "behavioral"                 # Usage patterns, preferences
    DERIVED = "derived"                       # Analytics results, insights
    EMPLOYMENT = "employment"                 # Work-related data
    FINANCIAL = "financial"                   # Payment, salary info


class ProcessingPurpose(Enum):
    """Purposes for data processing"""
    ANALYTICS = "analytics"
    CLUSTERING = "clustering"
    INSIGHTS = "insights"  
    REPORTING = "reporting"
    OPTIMIZATION = "optimization"
    RESEARCH = "research"
    COMPLIANCE = "compliance"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data_category: DataCategory = DataCategory.BASIC_PERSONAL
    processing_purpose: ProcessingPurpose = ProcessingPurpose.ANALYTICS
    lawful_basis: str = "legitimate_interests"
    data_subjects_count: int = 0
    retention_period: timedelta = timedelta(days=365)
    jurisdiction: str = "EU"
    processor_id: str = "system"
    controller_id: str = "organization"
    cross_border_transfer: bool = False
    transfer_countries: List[str] = field(default_factory=list)
    safeguards_applied: List[str] = field(default_factory=list)
    consent_obtained: bool = False
    consent_id: Optional[str] = None


@dataclass
class ComplianceViolation:
    """Record of compliance violation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: ComplianceLevel = ComplianceLevel.MEDIUM
    regulation: str = "GDPR"
    article: str = ""
    description: str = ""
    data_category: DataCategory = DataCategory.BASIC_PERSONAL
    affected_subjects: int = 0
    mitigation_required: bool = True
    mitigation_deadline: Optional[datetime] = None
    resolved: bool = False
    resolution_notes: str = ""


class ComplianceFramework(ABC):
    """Abstract base class for compliance frameworks"""
    
    def __init__(self, jurisdiction: str):
        self.jurisdiction = jurisdiction
        self.processing_records: List[DataProcessingRecord] = []
        self.violations: List[ComplianceViolation] = []
        self.enabled = True
        
    @abstractmethod
    def validate_processing(self, processing_record: DataProcessingRecord) -> List[ComplianceViolation]:
        """Validate data processing against compliance requirements"""
        pass
    
    @abstractmethod
    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """Get data retention requirements for category"""
        pass
    
    @abstractmethod
    def requires_consent(self, data_category: DataCategory, purpose: ProcessingPurpose) -> bool:
        """Check if explicit consent is required"""
        pass
    
    @abstractmethod
    def validate_cross_border_transfer(self, source_country: str, 
                                     target_country: str, 
                                     data_category: DataCategory) -> Dict[str, Any]:
        """Validate cross-border data transfer"""
        pass
    
    def log_processing_activity(self, processing_record: DataProcessingRecord):
        """Log data processing activity"""
        # Validate the processing
        violations = self.validate_processing(processing_record)
        
        # Record violations if any
        for violation in violations:
            self.violations.append(violation)
            logger.error(f"Compliance violation: {violation.description}")
        
        # Log the processing
        self.processing_records.append(processing_record)
        logger.info(f"Logged processing activity: {processing_record.id}")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status"""
        total_violations = len(self.violations)
        unresolved_violations = len([v for v in self.violations if not v.resolved])
        
        # Calculate risk level
        critical_violations = len([v for v in self.violations if v.level == ComplianceLevel.CRITICAL])
        high_violations = len([v for v in self.violations if v.level == ComplianceLevel.HIGH])
        
        if critical_violations > 0:
            risk_level = ComplianceLevel.CRITICAL
        elif high_violations > 0:
            risk_level = ComplianceLevel.HIGH
        elif unresolved_violations > 5:
            risk_level = ComplianceLevel.MEDIUM
        else:
            risk_level = ComplianceLevel.LOW
        
        return {
            'jurisdiction': self.jurisdiction,
            'enabled': self.enabled,
            'total_processing_records': len(self.processing_records),
            'total_violations': total_violations,
            'unresolved_violations': unresolved_violations,
            'risk_level': risk_level.value,
            'last_assessment': datetime.utcnow().isoformat(),
            'violations_by_level': {
                'critical': len([v for v in self.violations if v.level == ComplianceLevel.CRITICAL]),
                'high': len([v for v in self.violations if v.level == ComplianceLevel.HIGH]),
                'medium': len([v for v in self.violations if v.level == ComplianceLevel.MEDIUM]),
                'low': len([v for v in self.violations if v.level == ComplianceLevel.LOW])
            }
        }
    
    def resolve_violation(self, violation_id: str, resolution_notes: str):
        """Mark a violation as resolved"""
        for violation in self.violations:
            if violation.id == violation_id:
                violation.resolved = True
                violation.resolution_notes = resolution_notes
                logger.info(f"Resolved compliance violation: {violation_id}")
                return True
        
        logger.warning(f"Violation not found: {violation_id}")
        return False
    
    def get_data_subject_count(self, data_category: DataCategory = None) -> int:
        """Get count of data subjects in processing records"""
        if data_category:
            records = [r for r in self.processing_records if r.data_category == data_category]
        else:
            records = self.processing_records
        
        return sum(record.data_subjects_count for record in records)
    
    def cleanup_expired_data(self) -> int:
        """Remove processing records that have exceeded retention period"""
        current_time = datetime.utcnow()
        expired_count = 0
        
        unexpired_records = []
        for record in self.processing_records:
            if current_time - record.timestamp > record.retention_period:
                expired_count += 1
                logger.info(f"Removed expired processing record: {record.id}")
            else:
                unexpired_records.append(record)
        
        self.processing_records = unexpired_records
        return expired_count