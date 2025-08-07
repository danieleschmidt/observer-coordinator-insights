"""
CCPA (California Consumer Privacy Act) compliance implementation
For California, USA data protection requirements
"""

import logging
from datetime import timedelta
from typing import Dict, List, Any

from .base import ComplianceFramework, ComplianceViolation, ComplianceLevel
from .base import DataProcessingRecord, DataCategory, ProcessingPurpose

logger = logging.getLogger(__name__)


class CCPACompliance(ComplianceFramework):
    """CCPA compliance implementation for California jurisdiction"""
    
    def __init__(self):
        super().__init__("US-CA")
        self.regulation_name = "CCPA"
        self.revenue_threshold = 25000000  # $25M annual revenue threshold
        self.consumer_threshold = 50000    # 50K consumers threshold
        
    def validate_processing(self, processing_record: DataProcessingRecord) -> List[ComplianceViolation]:
        """Validate data processing against CCPA requirements"""
        violations = []
        
        # Check if sale of personal information requires disclosure
        if processing_record.processing_purpose == ProcessingPurpose.OPTIMIZATION:
            violations.append(ComplianceViolation(
                level=ComplianceLevel.MEDIUM,
                regulation="CCPA",
                article="Section 1798.135",
                description="May require 'Do Not Sell My Personal Information' disclosure",
                data_category=processing_record.data_category
            ))
        
        # Sensitive personal information handling
        if processing_record.data_category == DataCategory.SENSITIVE_PERSONAL:
            violations.append(ComplianceViolation(
                level=ComplianceLevel.HIGH,
                regulation="CCPA", 
                article="Section 1798.121",
                description="Sensitive personal information requires opt-out rights",
                data_category=processing_record.data_category
            ))
        
        return violations
    
    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """Get CCPA retention requirements"""
        # CCPA requires disclosure of retention periods but doesn't mandate specific periods
        return timedelta(days=1095)  # 3 years default
    
    def requires_consent(self, data_category: DataCategory, purpose: ProcessingPurpose) -> bool:
        """Check if consent is required under CCPA"""
        # CCPA focuses on disclosure and opt-out rather than opt-in consent
        return data_category == DataCategory.SENSITIVE_PERSONAL
    
    def validate_cross_border_transfer(self, source_country: str, target_country: str, 
                                     data_category: DataCategory) -> Dict[str, Any]:
        """Validate cross-border data transfer under CCPA"""
        return {
            'allowed': True,  # CCPA doesn't restrict international transfers
            'requires_safeguards': False,
            'disclosure_required': True,
            'consumer_rights_maintained': True
        }
    
    def get_consumer_rights(self) -> List[str]:
        """Get consumer rights under CCPA"""
        return [
            'right_to_know',           # What personal info is collected
            'right_to_delete',         # Delete personal information
            'right_to_opt_out',        # Opt out of sale
            'right_to_non_discrimination',  # No discrimination for exercising rights
            'right_to_limit'           # Limit use of sensitive personal info
        ]