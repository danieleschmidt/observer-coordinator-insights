"""GDPR (General Data Protection Regulation) compliance implementation
For European Union data protection requirements
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


class GDPRCompliance(ComplianceFramework):
    """GDPR compliance implementation for EU jurisdiction"""

    def __init__(self):
        super().__init__("EU")
        self.regulation_name = "GDPR"
        self.adequate_countries = {
            'AD', 'AR', 'CA', 'FO', 'GG', 'IL', 'IM', 'JP', 'JE', 'NZ', 'CH', 'UY', 'GB'
        }

    def validate_processing(self, processing_record: DataProcessingRecord) -> List[ComplianceViolation]:
        """Validate data processing against GDPR requirements"""
        violations = []

        # Article 6 - Lawfulness of processing
        if not self._validate_lawful_basis(processing_record):
            violations.append(ComplianceViolation(
                level=ComplianceLevel.HIGH,
                regulation="GDPR",
                article="Article 6",
                description="No valid lawful basis for processing",
                data_category=processing_record.data_category
            ))

        # Article 9 - Special categories of personal data
        if processing_record.data_category == DataCategory.SENSITIVE_PERSONAL:
            if not self._validate_special_category_processing(processing_record):
                violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    regulation="GDPR",
                    article="Article 9",
                    description="Invalid processing of special category data",
                    data_category=processing_record.data_category
                ))

        # Article 5 - Principles relating to processing (data minimisation)
        if processing_record.data_subjects_count > 10000:
            violations.append(ComplianceViolation(
                level=ComplianceLevel.MEDIUM,
                regulation="GDPR",
                article="Article 5(1)(c)",
                description="Large-scale processing may violate data minimisation principle",
                data_category=processing_record.data_category,
                affected_subjects=processing_record.data_subjects_count
            ))

        # Article 44-49 - Cross-border transfers
        if processing_record.cross_border_transfer:
            transfer_violations = self._validate_transfer(processing_record)
            violations.extend(transfer_violations)

        # Article 7 - Consent requirements
        if self.requires_consent(processing_record.data_category, processing_record.processing_purpose):
            if not processing_record.consent_obtained:
                violations.append(ComplianceViolation(
                    level=ComplianceLevel.HIGH,
                    regulation="GDPR",
                    article="Article 7",
                    description="Consent required but not obtained",
                    data_category=processing_record.data_category
                ))

        return violations

    def _validate_lawful_basis(self, processing_record: DataProcessingRecord) -> bool:
        """Validate lawful basis under Article 6"""
        valid_bases = [
            'consent', 'contract', 'legal_obligation',
            'vital_interests', 'public_task', 'legitimate_interests'
        ]

        if processing_record.lawful_basis not in valid_bases:
            return False

        # Consent must be freely given, specific, informed, and unambiguous
        if processing_record.lawful_basis == 'consent':
            return processing_record.consent_obtained and processing_record.consent_id is not None

        # Legitimate interests requires balancing test
        if processing_record.lawful_basis == 'legitimate_interests':
            # For high-risk processing, legitimate interests may not be appropriate
            if processing_record.data_category == DataCategory.SENSITIVE_PERSONAL:
                return False
            if processing_record.data_subjects_count > 50000:  # Large scale
                return False

        return True

    def _validate_special_category_processing(self, processing_record: DataProcessingRecord) -> bool:
        """Validate special category data processing under Article 9"""
        # Special categories require explicit consent or other specific conditions
        valid_conditions = [
            'explicit_consent', 'employment_law', 'vital_interests',
            'legitimate_activities', 'made_public', 'legal_claims',
            'substantial_public_interest', 'healthcare', 'public_health',
            'research'
        ]

        # For our clustering system, research purposes might be applicable
        if processing_record.processing_purpose in [ProcessingPurpose.RESEARCH, ProcessingPurpose.ANALYTICS]:
            return processing_record.consent_obtained or 'research' in processing_record.safeguards_applied

        return processing_record.consent_obtained

    def _validate_transfer(self, processing_record: DataProcessingRecord) -> List[ComplianceViolation]:
        """Validate cross-border transfers under Chapter V"""
        violations = []

        for country in processing_record.transfer_countries:
            if country not in self.adequate_countries:
                # Need appropriate safeguards
                required_safeguards = ['standard_contractual_clauses', 'binding_corporate_rules', 'adequacy_decision']

                if not any(safeguard in processing_record.safeguards_applied for safeguard in required_safeguards):
                    violations.append(ComplianceViolation(
                        level=ComplianceLevel.CRITICAL,
                        regulation="GDPR",
                        article="Article 44",
                        description=f"Transfer to {country} without adequate safeguards",
                        data_category=processing_record.data_category
                    ))

        return violations

    def get_retention_requirements(self, data_category: DataCategory) -> timedelta:
        """Get GDPR retention requirements"""
        # GDPR doesn't specify exact retention periods but requires they be necessary
        retention_periods = {
            DataCategory.BASIC_PERSONAL: timedelta(days=1095),      # 3 years
            DataCategory.SENSITIVE_PERSONAL: timedelta(days=365),   # 1 year (shorter for sensitive)
            DataCategory.BEHAVIORAL: timedelta(days=730),           # 2 years
            DataCategory.DERIVED: timedelta(days=1095),             # 3 years
            DataCategory.EMPLOYMENT: timedelta(days=2190),          # 6 years (employment records)
            DataCategory.FINANCIAL: timedelta(days=2555)            # 7 years (financial records)
        }

        return retention_periods.get(data_category, timedelta(days=365))

    def requires_consent(self, data_category: DataCategory, purpose: ProcessingPurpose) -> bool:
        """Check if explicit consent is required"""
        # Special categories always need consent unless specific exemption applies
        if data_category == DataCategory.SENSITIVE_PERSONAL:
            return True

        # For research and analytics, legitimate interests might suffice
        if purpose in [ProcessingPurpose.RESEARCH, ProcessingPurpose.ANALYTICS]:
            return data_category == DataCategory.SENSITIVE_PERSONAL

        # Marketing purposes typically need consent
        if purpose == ProcessingPurpose.OPTIMIZATION:
            return True

        return False

    def validate_cross_border_transfer(self, source_country: str, target_country: str,
                                     data_category: DataCategory) -> Dict[str, Any]:
        """Validate cross-border data transfer"""
        result = {
            'allowed': False,
            'requires_safeguards': True,
            'adequacy_decision': target_country in self.adequate_countries,
            'recommended_safeguards': []
        }

        if target_country in self.adequate_countries:
            result['allowed'] = True
            result['requires_safeguards'] = False
        else:
            # Recommend appropriate safeguards
            result['recommended_safeguards'] = [
                'standard_contractual_clauses',
                'data_processing_agreement',
                'impact_assessment'
            ]

            if data_category == DataCategory.SENSITIVE_PERSONAL:
                result['recommended_safeguards'].extend([
                    'explicit_consent',
                    'additional_security_measures',
                    'regular_compliance_audits'
                ])

        return result

    def get_data_subject_rights(self) -> List[str]:
        """Get list of data subject rights under GDPR"""
        return [
            'right_to_information',        # Article 13-14
            'right_of_access',            # Article 15
            'right_to_rectification',     # Article 16
            'right_to_erasure',           # Article 17
            'right_to_restrict_processing', # Article 18
            'right_to_data_portability',  # Article 20
            'right_to_object',            # Article 21
            'right_not_to_be_subject_to_automated_decision_making'  # Article 22
        ]

    def validate_automated_decision_making(self, processing_record: DataProcessingRecord) -> List[ComplianceViolation]:
        """Validate automated decision-making under Article 22"""
        violations = []

        # Our clustering system involves automated processing
        if processing_record.processing_purpose in [ProcessingPurpose.CLUSTERING, ProcessingPurpose.ANALYTICS]:
            # Check if it produces legal or similarly significant effects
            if processing_record.data_category == DataCategory.EMPLOYMENT:
                violations.append(ComplianceViolation(
                    level=ComplianceLevel.HIGH,
                    regulation="GDPR",
                    article="Article 22",
                    description="Automated processing of employment data may require human oversight",
                    data_category=processing_record.data_category
                ))

        return violations

    def generate_privacy_notice(self, processing_purposes: List[ProcessingPurpose]) -> Dict[str, Any]:
        """Generate privacy notice content for GDPR compliance"""
        return {
            'controller_identity': 'Your Organization Name',
            'data_protection_officer': 'dpo@yourorg.com',
            'processing_purposes': [purpose.value for purpose in processing_purposes],
            'lawful_basis': 'legitimate_interests',
            'retention_period': '3 years or as required by law',
            'data_subject_rights': self.get_data_subject_rights(),
            'complaint_authority': 'Your national data protection authority',
            'automated_decision_making': True,
            'profiling': True,
            'third_country_transfers': False
        }
