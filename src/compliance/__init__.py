"""Global Compliance Framework for Neuromorphic Clustering System
Supports GDPR, CCPA, PDPA and other international data protection regulations

Key Features:
- Cross-border data transfer restrictions
- Consent management and user rights
- Jurisdiction-specific audit logging
- Regional compliance validation
- Data residency controls
"""

from .audit_logger import ComplianceAuditLogger
from .base import ComplianceFramework, ComplianceLevel, DataCategory
from .ccpa import CCPACompliance
from .consent_manager import ConsentManager
from .cross_border import CrossBorderDataTransferManager
from .data_subject_rights import DataSubjectRightsManager
from .gdpr import GDPRCompliance
from .pdpa import PDPACompliance


__all__ = [
    'CCPACompliance',
    'ComplianceAuditLogger',
    'ComplianceFramework',
    'ComplianceLevel',
    'ConsentManager',
    'CrossBorderDataTransferManager',
    'DataCategory',
    'DataSubjectRightsManager',
    'GDPRCompliance',
    'PDPACompliance'
]

# Supported jurisdictions
SUPPORTED_JURISDICTIONS = {
    'EU': 'European Union (GDPR)',
    'US-CA': 'California, USA (CCPA)',
    'SG': 'Singapore (PDPA)',
    'UK': 'United Kingdom (UK GDPR)',
    'CA': 'Canada (PIPEDA)',
    'AU': 'Australia (Privacy Act)'
}

# Data processing lawful bases
LAWFUL_BASES = {
    'consent': 'Consent of the data subject',
    'contract': 'Performance of a contract',
    'legal_obligation': 'Compliance with legal obligation',
    'vital_interests': 'Protection of vital interests',
    'public_task': 'Performance of public task',
    'legitimate_interests': 'Legitimate interests'
}
