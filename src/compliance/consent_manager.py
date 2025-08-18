"""Consent management system for global compliance
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ConsentStatus(Enum):
    """Consent status values"""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


@dataclass
class ConsentRecord:
    """Individual consent record"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_id: str = ""
    purpose: str = ""
    data_categories: List[str] = field(default_factory=list)
    status: ConsentStatus = ConsentStatus.PENDING
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    jurisdiction: str = "EU"
    lawful_basis: str = "consent"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsentManager:
    """Global consent management system"""

    def __init__(self):
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.consent_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize consent templates for different jurisdictions"""
        self.consent_templates = {
            'EU': {
                'title': 'Data Processing Consent',
                'description': 'We process your personal data for organizational analytics and team optimization.',
                'required_info': [
                    'controller_identity',
                    'processing_purposes',
                    'data_categories',
                    'retention_period',
                    'rights_information',
                    'withdrawal_method'
                ],
                'opt_in_required': True,
                'granular_consent': True
            },
            'US-CA': {
                'title': 'Privacy Rights Notice',
                'description': 'Information about your privacy rights under California law.',
                'required_info': [
                    'data_collection_notice',
                    'sale_opt_out',
                    'consumer_rights',
                    'contact_information'
                ],
                'opt_in_required': False,
                'granular_consent': True
            },
            'SG': {
                'title': 'Personal Data Collection Notice',
                'description': 'Notice of personal data collection and use.',
                'required_info': [
                    'collection_purpose',
                    'data_categories',
                    'retention_policy',
                    'transfer_information'
                ],
                'opt_in_required': True,
                'granular_consent': False
            }
        }

    def request_consent(self, data_subject_id: str, purpose: str,
                       data_categories: List[str], jurisdiction: str = "EU",
                       validity_period: timedelta = None) -> str:
        """Request consent from data subject"""
        consent_record = ConsentRecord(
            data_subject_id=data_subject_id,
            purpose=purpose,
            data_categories=data_categories,
            jurisdiction=jurisdiction,
            expires_at=datetime.utcnow() + (validity_period or timedelta(days=365))
        )

        self.consent_records[consent_record.id] = consent_record

        logger.info(f"Consent requested: {consent_record.id} for subject {data_subject_id}")
        return consent_record.id

    def grant_consent(self, consent_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Grant consent"""
        if consent_id not in self.consent_records:
            logger.error(f"Consent record not found: {consent_id}")
            return False

        consent = self.consent_records[consent_id]
        consent.status = ConsentStatus.GRANTED
        consent.granted_at = datetime.utcnow()
        consent.metadata.update(metadata or {})

        logger.info(f"Consent granted: {consent_id}")
        return True

    def withdraw_consent(self, consent_id: str, reason: str = "") -> bool:
        """Withdraw consent"""
        if consent_id not in self.consent_records:
            logger.error(f"Consent record not found: {consent_id}")
            return False

        consent = self.consent_records[consent_id]
        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.utcnow()
        consent.metadata['withdrawal_reason'] = reason

        logger.info(f"Consent withdrawn: {consent_id}")
        return True

    def check_consent(self, data_subject_id: str, purpose: str,
                     data_category: str = None) -> Dict[str, Any]:
        """Check if valid consent exists"""
        current_time = datetime.utcnow()

        for consent in self.consent_records.values():
            if (consent.data_subject_id == data_subject_id and
                consent.purpose == purpose and
                consent.status == ConsentStatus.GRANTED):

                # Check expiration
                if consent.expires_at and current_time > consent.expires_at:
                    consent.status = ConsentStatus.EXPIRED
                    continue

                # Check data category if specified
                if data_category and data_category not in consent.data_categories:
                    continue

                return {
                    'valid': True,
                    'consent_id': consent.id,
                    'granted_at': consent.granted_at,
                    'expires_at': consent.expires_at
                }

        return {'valid': False, 'reason': 'No valid consent found'}

    def get_consent_status(self, data_subject_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a data subject"""
        records = []

        for consent in self.consent_records.values():
            if consent.data_subject_id == data_subject_id:
                records.append({
                    'id': consent.id,
                    'purpose': consent.purpose,
                    'data_categories': consent.data_categories,
                    'status': consent.status.value,
                    'granted_at': consent.granted_at.isoformat() if consent.granted_at else None,
                    'withdrawn_at': consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    'expires_at': consent.expires_at.isoformat() if consent.expires_at else None,
                    'jurisdiction': consent.jurisdiction
                })

        return records

    def cleanup_expired_consents(self) -> int:
        """Remove expired consent records"""
        current_time = datetime.utcnow()
        expired_count = 0

        for consent_id, consent in list(self.consent_records.items()):
            if (consent.expires_at and current_time > consent.expires_at and
                consent.status != ConsentStatus.WITHDRAWN):
                consent.status = ConsentStatus.EXPIRED
                expired_count += 1

        logger.info(f"Marked {expired_count} consent records as expired")
        return expired_count

    def generate_consent_form(self, jurisdiction: str, purposes: List[str],
                            data_categories: List[str]) -> Dict[str, Any]:
        """Generate consent form based on jurisdiction requirements"""
        template = self.consent_templates.get(jurisdiction, self.consent_templates['EU'])

        form = {
            'jurisdiction': jurisdiction,
            'title': template['title'],
            'description': template['description'],
            'purposes': purposes,
            'data_categories': data_categories,
            'opt_in_required': template['opt_in_required'],
            'granular_consent': template['granular_consent'],
            'required_info': template['required_info'],
            'generated_at': datetime.utcnow().isoformat()
        }

        return form
