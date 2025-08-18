"""PDPA (Personal Data Protection Act) compliance implementation
For Singapore data protection requirements
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List

from .base import (
    ComplianceFramework,
    ComplianceLevel,
    ComplianceViolation,
    DataCategory,
    DataProcessingRecord,
    ProcessingPurpose,
)


logger = logging.getLogger(__name__)


class PDPACompliance(ComplianceFramework):
    """PDPA compliance implementation for Singapore jurisdiction"""

    def __init__(self):
        super().__init__("SG")
        self.regulation_name = "PDPA"

    def validate_processing(self, processing_record: DataProcessingRecord) -> List[ComplianceViolation]:
        """Validate data processing against PDPA requirements"""
        violations = []

        # Consent requirement under Section 13
        if self.requires_consent(processing_record.data_category, processing_record.processing_purpose):
            if not processing_record.consent_obtained:
                violations.append(ComplianceViolation(
                    level=ComplianceLevel.HIGH,
                    regulation="PDPA",
                    article="Section 13",
                    description="Consent required for personal data collection",
                    data_category=processing_record.data_category
                ))

        # Purpose limitation under Section 15
        if processing_record.processing_purpose == ProcessingPurpose.RESEARCH:
            if not processing_record.consent_obtained:
                violations.append(ComplianceViolation(
                    level=ComplianceLevel.MEDIUM,
                    regulation="PDPA",
                    article="Section 15",
                    description="Research purposes may require additional consent",
                    data_category=processing_record.data_category
                ))

        return violations

    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """Get PDPA retention requirements"""
        # PDPA requires data to be destroyed when no longer needed
        retention_periods = {
            DataCategory.BASIC_PERSONAL: timedelta(days=1095),    # 3 years
            DataCategory.EMPLOYMENT: timedelta(days=2190),        # 6 years
            DataCategory.FINANCIAL: timedelta(days=2555)          # 7 years
        }
        return retention_periods.get(data_category, timedelta(days=1095))

    def requires_consent(self, data_category: DataCategory, purpose: ProcessingPurpose) -> bool:
        """Check if consent is required under PDPA"""
        # PDPA generally requires consent unless specific exceptions apply
        exceptions = [
            ProcessingPurpose.COMPLIANCE,  # Legal obligation
        ]
        return purpose not in exceptions

    def validate_cross_border_transfer(self, source_country: str, target_country: str,
                                     data_category: DataCategory) -> Dict[str, Any]:
        """Validate cross-border data transfer under PDPA"""
        # PDPA Section 26 - Transfer of personal data outside Singapore
        return {
            'allowed': True,
            'requires_safeguards': True,
            'adequate_protection_required': True,
            'recommended_safeguards': ['contractual_protection', 'recipient_compliance']
        }
