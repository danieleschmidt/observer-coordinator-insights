#!/usr/bin/env python3
"""
Advanced Security Framework - Generation 2 Robustness
Comprehensive security, encryption, and compliance implementation
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import base64
import uuid

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


class AccessType(Enum):
    """Types of access operations"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ComplianceStandard(Enum):
    """Compliance standards supported"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.now)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    additional_claims: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessAttempt:
    """Record of access attempt for auditing"""
    id: str
    user_id: str
    resource: str
    access_type: AccessType
    granted: bool
    security_context: SecurityContext
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'resource': self.resource,
            'access_type': self.access_type.value,
            'granted': self.granted,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'ip_address': self.security_context.ip_address,
            'session_id': self.security_context.session_id
        }


class EncryptionManager:
    """Advanced encryption and key management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or self._generate_master_key()
        self._encryption_keys: Dict[str, bytes] = {}
        self._key_rotation_schedule: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)
    
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key"""
        return Fernet.generate_key()
    
    def create_encryption_key(self, identifier: str, rotate_after_days: int = 30) -> str:
        """Create new encryption key with rotation schedule"""
        key = Fernet.generate_key()
        self._encryption_keys[identifier] = key
        self._key_rotation_schedule[identifier] = datetime.now() + timedelta(days=rotate_after_days)
        
        self.logger.info(f"Created encryption key: {identifier}")
        return identifier
    
    def encrypt_data(self, data: Union[str, bytes], key_identifier: str = "default") -> str:
        """Encrypt data with specified key"""
        if key_identifier not in self._encryption_keys:
            self.create_encryption_key(key_identifier)
        
        # Convert to bytes if string
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Encrypt with Fernet
        fernet = Fernet(self._encryption_keys[key_identifier])
        encrypted_data = fernet.encrypt(data)
        
        # Return base64 encoded for storage
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str, key_identifier: str = "default") -> str:
        """Decrypt data with specified key"""
        if key_identifier not in self._encryption_keys:
            raise ValueError(f"Encryption key not found: {key_identifier}")
        
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Decrypt with Fernet
            fernet = Fernet(self._encryption_keys[key_identifier])
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data.decode('utf-8')
        
        except Exception as e:
            self.logger.error(f"Decryption failed for key {key_identifier}: {e}")
            raise
    
    def rotate_keys(self) -> List[str]:
        """Rotate encryption keys that are due for rotation"""
        rotated_keys = []
        now = datetime.now()
        
        for key_id, rotation_date in self._key_rotation_schedule.items():
            if now >= rotation_date:
                # Generate new key
                old_key = self._encryption_keys[key_id]
                new_key = Fernet.generate_key()
                
                # Store with versioning
                self._encryption_keys[f"{key_id}_old"] = old_key
                self._encryption_keys[key_id] = new_key
                
                # Update rotation schedule
                self._key_rotation_schedule[key_id] = now + timedelta(days=30)
                
                rotated_keys.append(key_id)
                self.logger.info(f"Rotated encryption key: {key_id}")
        
        return rotated_keys


class AccessControlManager:
    """Role-based access control (RBAC) system"""
    
    def __init__(self):
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.user_roles: Dict[str, List[str]] = {}
        self.resource_permissions: Dict[str, Dict[AccessType, SecurityLevel]] = {}
        self.access_log: List[AccessAttempt] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize standard security roles"""
        self.roles = {
            "admin": {
                "permissions": [access_type for access_type in AccessType],
                "security_level": SecurityLevel.TOP_SECRET,
                "description": "Full system administrator"
            },
            "analyst": {
                "permissions": [AccessType.READ, AccessType.WRITE],
                "security_level": SecurityLevel.CONFIDENTIAL,
                "description": "Data analyst with read/write access"
            },
            "viewer": {
                "permissions": [AccessType.READ],
                "security_level": SecurityLevel.INTERNAL,
                "description": "Read-only access"
            },
            "guest": {
                "permissions": [AccessType.READ],
                "security_level": SecurityLevel.PUBLIC,
                "description": "Limited guest access"
            }
        }
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
        if role not in self.roles:
            self.logger.warning(f"Unknown role assignment attempted: {role}")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            self.logger.info(f"Assigned role '{role}' to user {user_id}")
        
        return True
    
    def check_access(self, 
                    user_id: str,
                    resource: str,
                    access_type: AccessType,
                    security_context: SecurityContext) -> bool:
        """Check if user has access to resource"""
        
        attempt_id = str(uuid.uuid4())
        
        # Check if user has any roles
        user_roles = self.user_roles.get(user_id, [])
        if not user_roles:
            self._log_access_attempt(attempt_id, user_id, resource, access_type, 
                                   False, security_context, "No roles assigned")
            return False
        
        # Check resource-specific permissions
        resource_perms = self.resource_permissions.get(resource, {})
        required_level = resource_perms.get(access_type, SecurityLevel.INTERNAL)
        
        # Check if any user role has sufficient permissions
        for role in user_roles:
            role_info = self.roles.get(role, {})
            role_permissions = role_info.get("permissions", [])
            role_level = role_info.get("security_level", SecurityLevel.PUBLIC)
            
            # Check permission type and security level
            if access_type in role_permissions and role_level.value >= required_level.value:
                self._log_access_attempt(attempt_id, user_id, resource, access_type,
                                       True, security_context, f"Granted via role: {role}")
                return True
        
        self._log_access_attempt(attempt_id, user_id, resource, access_type,
                               False, security_context, "Insufficient permissions")
        return False
    
    def _log_access_attempt(self,
                           attempt_id: str,
                           user_id: str,
                           resource: str,
                           access_type: AccessType,
                           granted: bool,
                           security_context: SecurityContext,
                           reason: str):
        """Log access attempt for auditing"""
        attempt = AccessAttempt(
            id=attempt_id,
            user_id=user_id,
            resource=resource,
            access_type=access_type,
            granted=granted,
            security_context=security_context,
            reason=reason
        )
        
        self.access_log.append(attempt)
        
        log_level = logging.INFO if granted else logging.WARNING
        self.logger.log(log_level, f"Access {granted and 'granted' or 'denied'}: "
                                  f"{user_id} -> {resource} ({access_type.value}) - {reason}")
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics"""
        if not self.access_log:
            return {"message": "No access attempts logged"}
        
        total_attempts = len(self.access_log)
        granted_attempts = sum(1 for attempt in self.access_log if attempt.granted)
        denied_attempts = total_attempts - granted_attempts
        
        # Recent activity (last hour)
        now = datetime.now()
        recent_attempts = [
            attempt for attempt in self.access_log
            if now - attempt.timestamp <= timedelta(hours=1)
        ]
        
        return {
            "total_attempts": total_attempts,
            "granted": granted_attempts,
            "denied": denied_attempts,
            "success_rate": granted_attempts / max(1, total_attempts),
            "recent_activity": len(recent_attempts),
            "unique_users": len(set(attempt.user_id for attempt in self.access_log)),
            "most_accessed_resources": self._get_resource_access_counts()
        }
    
    def _get_resource_access_counts(self) -> Dict[str, int]:
        """Get resource access frequency"""
        counts = {}
        for attempt in self.access_log:
            counts[attempt.resource] = counts.get(attempt.resource, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])


class ComplianceManager:
    """Manage compliance with various standards"""
    
    def __init__(self):
        self.enabled_standards: List[ComplianceStandard] = []
        self.data_retention_policies: Dict[str, int] = {}  # days
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def enable_compliance_standard(self, standard: ComplianceStandard):
        """Enable compliance standard with default policies"""
        if standard not in self.enabled_standards:
            self.enabled_standards.append(standard)
            self._apply_standard_policies(standard)
            self.logger.info(f"Enabled compliance standard: {standard.value}")
    
    def _apply_standard_policies(self, standard: ComplianceStandard):
        """Apply default policies for compliance standard"""
        if standard == ComplianceStandard.GDPR:
            self.data_retention_policies.update({
                "personal_data": 365,  # 1 year
                "analytics_data": 730,  # 2 years
                "audit_logs": 2555     # 7 years
            })
        elif standard == ComplianceStandard.CCPA:
            self.data_retention_policies.update({
                "consumer_data": 365,
                "opt_out_requests": 730
            })
        elif standard == ComplianceStandard.HIPAA:
            self.data_retention_policies.update({
                "health_records": 2555,  # 7 years minimum
                "audit_logs": 2555
            })
    
    def record_consent(self, user_id: str, purpose: str, granted: bool, 
                      ip_address: str = "", user_agent: str = ""):
        """Record user consent for GDPR compliance"""
        consent_id = str(uuid.uuid4())
        
        self.consent_records[consent_id] = {
            "user_id": user_id,
            "purpose": purpose,
            "granted": granted,
            "timestamp": datetime.now().isoformat(),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "consent_id": consent_id
        }
        
        # Add to audit trail
        self.audit_trail.append({
            "action": "consent_recorded",
            "details": {
                "user_id": user_id,
                "purpose": purpose,
                "granted": granted,
                "consent_id": consent_id
            },
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info(f"Recorded consent: {user_id} - {purpose} - {granted}")
    
    def check_data_retention(self, data_type: str, created_date: datetime) -> bool:
        """Check if data should be retained based on policies"""
        retention_days = self.data_retention_policies.get(data_type, 365)  # Default 1 year
        retention_limit = datetime.now() - timedelta(days=retention_days)
        
        return created_date > retention_limit
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance status report"""
        return {
            "enabled_standards": [std.value for std in self.enabled_standards],
            "data_retention_policies": self.data_retention_policies,
            "consent_statistics": {
                "total_records": len(self.consent_records),
                "granted_consents": sum(1 for record in self.consent_records.values() 
                                      if record["granted"]),
                "recent_consents": len([
                    record for record in self.consent_records.values()
                    if datetime.fromisoformat(record["timestamp"]) > 
                       datetime.now() - timedelta(days=30)
                ])
            },
            "audit_trail_entries": len(self.audit_trail),
            "generated_at": datetime.now().isoformat()
        }


class SecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.access_control = AccessControlManager()
        self.compliance_manager = ComplianceManager()
        self.logger = logging.getLogger(__name__)
        
        # Security configuration
        self.config = {
            "session_timeout_minutes": 30,
            "max_failed_login_attempts": 5,
            "password_min_length": 12,
            "require_mfa": True,
            "log_all_access": True,
            "encrypt_sensitive_data": True
        }
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with GDPR compliance by default
        self.compliance_manager.enable_compliance_standard(ComplianceStandard.GDPR)
    
    def initialize_user_security(self, user_id: str, role: str = "viewer") -> bool:
        """Initialize security for new user"""
        try:
            # Assign default role
            success = self.access_control.assign_role(user_id, role)
            
            if success:
                # Initialize rate limiting
                self.rate_limits[user_id] = {
                    "requests": 0,
                    "window_start": datetime.now(),
                    "blocked_until": None
                }
                
                self.logger.info(f"Initialized security for user: {user_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security for {user_id}: {e}")
            return False
    
    def check_rate_limit(self, user_id: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if user is within rate limits"""
        if user_id not in self.rate_limits:
            self.initialize_user_security(user_id)
        
        user_limits = self.rate_limits[user_id]
        now = datetime.now()
        
        # Check if currently blocked
        if user_limits.get("blocked_until") and now < user_limits["blocked_until"]:
            return False
        
        # Reset window if needed
        window_duration = timedelta(minutes=window_minutes)
        if now - user_limits["window_start"] > window_duration:
            user_limits["requests"] = 0
            user_limits["window_start"] = now
            user_limits["blocked_until"] = None
        
        # Check limit
        if user_limits["requests"] >= max_requests:
            # Block for 15 minutes
            user_limits["blocked_until"] = now + timedelta(minutes=15)
            self.logger.warning(f"Rate limit exceeded for user: {user_id}")
            return False
        
        # Increment counter
        user_limits["requests"] += 1
        return True
    
    def secure_data_operation(self, 
                            data: Any,
                            user_id: str,
                            operation: str,
                            security_context: SecurityContext) -> Dict[str, Any]:
        """Perform secure data operation with full security checks"""
        
        # Rate limiting
        if not self.check_rate_limit(user_id):
            return {"success": False, "error": "Rate limit exceeded"}
        
        # Access control
        access_granted = self.access_control.check_access(
            user_id, operation, AccessType.WRITE, security_context
        )
        
        if not access_granted:
            return {"success": False, "error": "Access denied"}
        
        try:
            # Encrypt sensitive data
            if self.config["encrypt_sensitive_data"]:
                if isinstance(data, (str, dict)):
                    data_str = json.dumps(data) if isinstance(data, dict) else data
                    encrypted_data = self.encryption_manager.encrypt_data(data_str)
                    
                    return {
                        "success": True,
                        "data": encrypted_data,
                        "encrypted": True,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "success": True,
                "data": data,
                "encrypted": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Secure operation failed: {e}")
            return {"success": False, "error": "Operation failed"}
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        return {
            "access_control": self.access_control.get_access_statistics(),
            "compliance": self.compliance_manager.get_compliance_report(),
            "rate_limiting": {
                "tracked_users": len(self.rate_limits),
                "currently_blocked": sum(
                    1 for limits in self.rate_limits.values()
                    if limits.get("blocked_until") and 
                       datetime.now() < limits["blocked_until"]
                )
            },
            "encryption": {
                "active_keys": len(self.encryption_manager._encryption_keys),
                "keys_due_rotation": len([
                    key_id for key_id, rotation_date in 
                    self.encryption_manager._key_rotation_schedule.items()
                    if datetime.now() >= rotation_date
                ])
            },
            "configuration": self.config.copy(),
            "generated_at": datetime.now().isoformat()
        }


# Global security framework instance
security_framework = SecurityFramework()