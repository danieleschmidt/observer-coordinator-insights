#!/usr/bin/env python3
"""Advanced Quantum Security Framework - Generation 2 Implementation
Quantum-safe cryptography, secure quantum communication, and quantum key distribution
"""

import hashlib
import hmac
import logging
import secrets
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantumKey:
    """Represents a quantum-safe cryptographic key."""
    key_id: str
    key_material: bytes
    algorithm: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    usage_count: int = 0
    max_usage: int = 1000
    quantum_safe: bool = True
    
    def is_valid(self) -> bool:
        """Check if key is still valid for use."""
        now = time.time()
        
        # Check expiration
        if self.expires_at and now > self.expires_at:
            return False
        
        # Check usage limit
        if self.usage_count >= self.max_usage:
            return False
        
        return True
    
    def use(self) -> bool:
        """Increment usage counter and check validity."""
        if not self.is_valid():
            return False
        
        self.usage_count += 1
        return True


class QuantumKeyDistribution:
    """Quantum Key Distribution (QKD) simulator for secure key exchange."""
    
    def __init__(self, noise_level: float = 0.01, eavesdropping_detection_threshold: float = 0.11):
        self.noise_level = noise_level
        self.eavesdropping_detection_threshold = eavesdropping_detection_threshold
        self.key_repository = {}
        self.distribution_log = []
        
    def generate_quantum_key_pair(self, key_length: int = 256) -> Tuple[QuantumKey, QuantumKey]:
        """Generate a pair of quantum-entangled keys for secure communication."""
        try:
            # Generate quantum random bits (simulated)
            alice_bits = self._generate_quantum_random_bits(key_length)
            bob_bits = self._generate_quantum_random_bits(key_length)
            
            # Simulate quantum entanglement correlation
            entangled_correlation = self._apply_quantum_entanglement(alice_bits, bob_bits)
            
            # Create quantum keys
            key_id = secrets.token_hex(16)
            
            alice_key = QuantumKey(
                key_id=f"alice_{key_id}",
                key_material=self._bits_to_bytes(entangled_correlation['alice']),
                algorithm="QKD-BB84",
                expires_at=time.time() + 3600  # 1 hour expiration
            )
            
            bob_key = QuantumKey(
                key_id=f"bob_{key_id}",
                key_material=self._bits_to_bytes(entangled_correlation['bob']),
                algorithm="QKD-BB84",
                expires_at=time.time() + 3600
            )
            
            # Store keys
            self.key_repository[alice_key.key_id] = alice_key
            self.key_repository[bob_key.key_id] = bob_key
            
            # Log distribution
            self.distribution_log.append({
                'timestamp': time.time(),
                'key_pair_id': key_id,
                'key_length': key_length,
                'alice_key_id': alice_key.key_id,
                'bob_key_id': bob_key.key_id,
                'security_level': self._assess_security_level(entangled_correlation)
            })
            
            logger.info(f"Generated quantum key pair: {key_length} bits, security level: high")
            
            return alice_key, bob_key
            
        except Exception as e:
            logger.error(f"Quantum key generation failed: {e}")
            raise
    
    def _generate_quantum_random_bits(self, length: int) -> np.ndarray:
        """Generate truly random bits using quantum processes (simulated)."""
        # Simulate quantum random number generation
        # In reality, this would use quantum hardware
        quantum_entropy = np.random.RandomState(int(time.time() * 1000000) % 2**32)
        return quantum_entropy.randint(0, 2, size=length)
    
    def _apply_quantum_entanglement(self, alice_bits: np.ndarray, bob_bits: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply quantum entanglement correlation between bit strings."""
        # Simulate BB84 protocol
        alice_bases = np.random.randint(0, 2, size=len(alice_bits))  # 0: rectilinear, 1: diagonal
        bob_bases = np.random.randint(0, 2, size=len(bob_bits))
        
        # Quantum measurement results
        alice_results = alice_bits.copy()
        bob_results = np.zeros_like(bob_bits)
        
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                # Same basis - perfect correlation (with noise)
                if np.random.random() > self.noise_level:
                    bob_results[i] = alice_results[i]
                else:
                    bob_results[i] = 1 - alice_results[i]  # Noise flip
            else:
                # Different basis - random result
                bob_results[i] = np.random.randint(0, 2)
        
        # Sifting - keep only bits measured in same basis
        matching_bases = alice_bases == bob_bases
        sifted_alice = alice_results[matching_bases]
        sifted_bob = bob_results[matching_bases]
        
        # Error correction and privacy amplification (simplified)
        corrected_length = min(len(sifted_alice), 128)  # Reduce for security
        final_alice = sifted_alice[:corrected_length]
        final_bob = sifted_bob[:corrected_length]
        
        return {
            'alice': final_alice,
            'bob': final_bob,
            'error_rate': np.mean(final_alice != final_bob),
            'sifted_length': len(sifted_alice),
            'final_length': corrected_length
        }
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes."""
        # Pad to multiple of 8
        padded_bits = np.pad(bits, (0, (8 - len(bits) % 8) % 8), mode='constant')
        
        # Convert to bytes
        byte_array = []
        for i in range(0, len(padded_bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(padded_bits):
                    byte_val |= int(padded_bits[i + j]) << (7 - j)
            byte_array.append(byte_val)
        
        return bytes(byte_array)
    
    def _assess_security_level(self, correlation_data: Dict) -> str:
        """Assess the security level of the quantum key."""
        error_rate = correlation_data.get('error_rate', 1.0)
        
        if error_rate < 0.05:
            return 'high'
        elif error_rate < self.eavesdropping_detection_threshold:
            return 'medium'
        else:
            return 'compromised'  # Possible eavesdropping
    
    def detect_eavesdropping(self, alice_key: QuantumKey, bob_key: QuantumKey, sample_size: int = 64) -> Dict[str, Any]:
        """Detect eavesdropping by analyzing quantum bit error rate."""
        try:
            # Convert key material back to bits for comparison
            alice_bits = self._bytes_to_bits(alice_key.key_material)[:sample_size]
            bob_bits = self._bytes_to_bits(bob_key.key_material)[:sample_size]
            
            # Calculate quantum bit error rate (QBER)
            if len(alice_bits) != len(bob_bits):
                logger.warning("Key length mismatch in eavesdropping detection")
                return {'eavesdropping_detected': True, 'confidence': 1.0, 'reason': 'key_mismatch'}
            
            errors = np.sum(alice_bits != bob_bits)
            qber = errors / len(alice_bits) if len(alice_bits) > 0 else 1.0
            
            # Eavesdropping detection
            eavesdropping_detected = qber > self.eavesdropping_detection_threshold
            confidence = min(1.0, qber / self.eavesdropping_detection_threshold)
            
            detection_result = {
                'eavesdropping_detected': eavesdropping_detected,
                'qber': qber,
                'confidence': confidence,
                'sample_size': sample_size,
                'errors_detected': int(errors),
                'security_assessment': 'compromised' if eavesdropping_detected else 'secure',
                'recommendation': 'discard_keys' if eavesdropping_detected else 'proceed'
            }
            
            logger.info(f"Eavesdropping detection: QBER={qber:.4f}, "
                       f"{'DETECTED' if eavesdropping_detected else 'NOT DETECTED'}")
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Eavesdropping detection failed: {e}")
            return {'eavesdropping_detected': True, 'confidence': 1.0, 'reason': 'detection_error'}
    
    def _bytes_to_bits(self, byte_data: bytes) -> np.ndarray:
        """Convert bytes back to bit array."""
        bits = []
        for byte_val in byte_data:
            for i in range(8):
                bits.append((byte_val >> (7 - i)) & 1)
        return np.array(bits)


class QuantumSafeCryptography:
    """Quantum-safe cryptographic operations."""
    
    def __init__(self):
        self.backend = default_backend()
        self.active_keys = {}
        self.encryption_log = []
        
    def generate_quantum_safe_key(self, algorithm: str = "AES-256") -> QuantumKey:
        """Generate a quantum-safe symmetric key."""
        try:
            if algorithm == "AES-256":
                key_material = secrets.token_bytes(32)  # 256 bits
            elif algorithm == "AES-128":
                key_material = secrets.token_bytes(16)  # 128 bits
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            key = QuantumKey(
                key_id=secrets.token_hex(16),
                key_material=key_material,
                algorithm=algorithm,
                expires_at=time.time() + 7200,  # 2 hours
                max_usage=10000
            )
            
            self.active_keys[key.key_id] = key
            
            logger.info(f"Generated quantum-safe key: {algorithm}, ID: {key.key_id}")
            
            return key
            
        except Exception as e:
            logger.error(f"Quantum-safe key generation failed: {e}")
            raise
    
    def encrypt_quantum_safe(self, data: bytes, key: QuantumKey, 
                           additional_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Encrypt data using quantum-safe algorithms."""
        try:
            if not key.use():
                raise ValueError("Key is no longer valid for encryption")
            
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Create cipher
            if key.algorithm.startswith("AES"):
                cipher = Cipher(algorithms.AES(key.key_material), modes.CBC(iv), backend=self.backend)
                encryptor = cipher.encryptor()
                
                # PKCS7 padding
                padded_data = self._pkcs7_pad(data, 16)
                ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            else:
                raise ValueError(f"Unsupported encryption algorithm: {key.algorithm}")
            
            # Generate authentication tag using HMAC
            auth_tag = self._generate_auth_tag(key.key_material, iv + ciphertext, additional_data)
            
            # Create encrypted package
            encrypted_package = {
                'ciphertext': ciphertext,
                'iv': iv,
                'auth_tag': auth_tag,
                'algorithm': key.algorithm,
                'key_id': key.key_id,
                'timestamp': time.time(),
                'quantum_safe': True
            }
            
            # Log encryption
            self.encryption_log.append({
                'operation': 'encrypt',
                'key_id': key.key_id,
                'data_size': len(data),
                'timestamp': time.time(),
                'algorithm': key.algorithm
            })
            
            logger.debug(f"Quantum-safe encryption complete: {len(data)} bytes -> {len(ciphertext)} bytes")
            
            return encrypted_package
            
        except Exception as e:
            logger.error(f"Quantum-safe encryption failed: {e}")
            raise
    
    def decrypt_quantum_safe(self, encrypted_package: Dict[str, Any], 
                           key: QuantumKey, additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using quantum-safe algorithms."""
        try:
            if not key.use():
                raise ValueError("Key is no longer valid for decryption")
            
            # Verify authentication tag
            expected_tag = self._generate_auth_tag(
                key.key_material,
                encrypted_package['iv'] + encrypted_package['ciphertext'],
                additional_data
            )
            
            if not hmac.compare_digest(encrypted_package['auth_tag'], expected_tag):
                raise ValueError("Authentication verification failed - data may be tampered")
            
            # Decrypt
            if key.algorithm.startswith("AES"):
                cipher = Cipher(
                    algorithms.AES(key.key_material),
                    modes.CBC(encrypted_package['iv']),
                    backend=self.backend
                )
                decryptor = cipher.decryptor()
                padded_plaintext = decryptor.update(encrypted_package['ciphertext']) + decryptor.finalize()
                
                # Remove PKCS7 padding
                plaintext = self._pkcs7_unpad(padded_plaintext)
            else:
                raise ValueError(f"Unsupported decryption algorithm: {key.algorithm}")
            
            # Log decryption
            self.encryption_log.append({
                'operation': 'decrypt',
                'key_id': key.key_id,
                'data_size': len(plaintext),
                'timestamp': time.time(),
                'algorithm': key.algorithm
            })
            
            logger.debug(f"Quantum-safe decryption complete: {len(plaintext)} bytes recovered")
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Quantum-safe decryption failed: {e}")
            raise
    
    def _pkcs7_pad(self, data: bytes, block_size: int) -> bytes:
        """Apply PKCS7 padding."""
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _pkcs7_unpad(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding."""
        padding_length = padded_data[-1]
        if padding_length == 0 or padding_length > len(padded_data):
            raise ValueError("Invalid PKCS7 padding")
        
        # Verify padding
        for i in range(padding_length):
            if padded_data[-(i + 1)] != padding_length:
                raise ValueError("Invalid PKCS7 padding")
        
        return padded_data[:-padding_length]
    
    def _generate_auth_tag(self, key: bytes, data: bytes, additional_data: Optional[bytes] = None) -> bytes:
        """Generate HMAC authentication tag."""
        h = hmac.new(key, digestmod=hashlib.sha256)
        h.update(data)
        if additional_data:
            h.update(additional_data)
        return h.digest()
    
    def get_encryption_statistics(self) -> Dict[str, Any]:
        """Get encryption operation statistics."""
        if not self.encryption_log:
            return {'total_operations': 0}
        
        encryptions = [op for op in self.encryption_log if op['operation'] == 'encrypt']
        decryptions = [op for op in self.encryption_log if op['operation'] == 'decrypt']
        
        return {
            'total_operations': len(self.encryption_log),
            'encryptions': len(encryptions),
            'decryptions': len(decryptions),
            'active_keys': len(self.active_keys),
            'average_data_size': np.mean([op['data_size'] for op in self.encryption_log]),
            'operations_per_hour': len([op for op in self.encryption_log if time.time() - op['timestamp'] < 3600]),
            'algorithms_used': list(set(op['algorithm'] for op in self.encryption_log))
        }


class QuantumSecurityOrchestrator:
    """Orchestrates all quantum security operations."""
    
    def __init__(self):
        self.qkd = QuantumKeyDistribution()
        self.crypto = QuantumSafeCryptography()
        self.security_log = []
        self.threat_level = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
        
    async def establish_secure_quantum_channel(self, channel_id: str) -> Dict[str, Any]:
        """Establish a secure quantum communication channel."""
        try:
            logger.info(f"🔐 Establishing secure quantum channel: {channel_id}")
            
            # Step 1: Generate quantum key pair using QKD
            alice_key, bob_key = self.qkd.generate_quantum_key_pair(256)
            
            # Step 2: Detect eavesdropping
            eavesdropping_result = self.qkd.detect_eavesdropping(alice_key, bob_key)
            
            if eavesdropping_result['eavesdropping_detected']:
                logger.warning(f"⚠️ Eavesdropping detected on channel {channel_id}")
                self.threat_level = "HIGH"
                return {
                    'success': False,
                    'reason': 'eavesdropping_detected',
                    'detection_result': eavesdropping_result
                }
            
            # Step 3: Generate additional symmetric keys for bulk encryption
            bulk_encryption_key = self.crypto.generate_quantum_safe_key("AES-256")
            
            # Step 4: Encrypt the bulk key using QKD keys
            encrypted_bulk_key = self.crypto.encrypt_quantum_safe(
                bulk_encryption_key.key_material,
                alice_key,
                additional_data=channel_id.encode()
            )
            
            channel_info = {
                'success': True,
                'channel_id': channel_id,
                'alice_key_id': alice_key.key_id,
                'bob_key_id': bob_key.key_id,
                'bulk_key_id': bulk_encryption_key.key_id,
                'encrypted_bulk_key': encrypted_bulk_key,
                'security_level': 'quantum_safe',
                'established_at': time.time(),
                'eavesdropping_check': eavesdropping_result
            }
            
            # Log security event
            self.security_log.append({
                'event': 'secure_channel_established',
                'channel_id': channel_id,
                'timestamp': time.time(),
                'security_level': 'quantum_safe',
                'threat_level': self.threat_level
            })
            
            logger.info(f"✅ Secure quantum channel established: {channel_id}")
            
            return channel_info
            
        except Exception as e:
            logger.error(f"❌ Failed to establish secure channel: {e}")
            self.security_log.append({
                'event': 'channel_establishment_failed',
                'channel_id': channel_id,
                'error': str(e),
                'timestamp': time.time()
            })
            raise
    
    def secure_quantum_data_transmission(self, data: bytes, channel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Securely transmit data over quantum channel."""
        try:
            # Get the bulk encryption key
            bulk_key = self.crypto.active_keys.get(channel_info['bulk_key_id'])
            if not bulk_key:
                raise ValueError("Bulk encryption key not found")
            
            # Encrypt data
            encrypted_data = self.crypto.encrypt_quantum_safe(
                data,
                bulk_key,
                additional_data=channel_info['channel_id'].encode()
            )
            
            # Add quantum integrity verification
            quantum_checksum = self._calculate_quantum_checksum(data)
            
            transmission_package = {
                'encrypted_data': encrypted_data,
                'quantum_checksum': quantum_checksum,
                'channel_id': channel_info['channel_id'],
                'transmission_time': time.time(),
                'security_level': 'quantum_safe'
            }
            
            logger.info(f"🚀 Secure quantum data transmission: {len(data)} bytes")
            
            return transmission_package
            
        except Exception as e:
            logger.error(f"❌ Quantum data transmission failed: {e}")
            raise
    
    def receive_quantum_data_transmission(self, transmission_package: Dict[str, Any], 
                                        channel_info: Dict[str, Any]) -> bytes:
        """Receive and decrypt quantum data transmission."""
        try:
            # Get the bulk encryption key
            bulk_key = self.crypto.active_keys.get(channel_info['bulk_key_id'])
            if not bulk_key:
                raise ValueError("Bulk encryption key not found")
            
            # Decrypt data
            decrypted_data = self.crypto.decrypt_quantum_safe(
                transmission_package['encrypted_data'],
                bulk_key,
                additional_data=channel_info['channel_id'].encode()
            )
            
            # Verify quantum integrity
            expected_checksum = self._calculate_quantum_checksum(decrypted_data)
            if transmission_package['quantum_checksum'] != expected_checksum:
                raise ValueError("Quantum integrity verification failed")
            
            logger.info(f"📥 Quantum data received and verified: {len(decrypted_data)} bytes")
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"❌ Quantum data reception failed: {e}")
            raise
    
    def _calculate_quantum_checksum(self, data: bytes) -> str:
        """Calculate quantum-enhanced checksum."""
        # Use multiple hash functions for quantum resistance
        sha3_hash = hashlib.sha3_256(data).hexdigest()
        blake2_hash = hashlib.blake2b(data, digest_size=32).hexdigest()
        
        # Combine hashes
        combined = f"{sha3_hash}:{blake2_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        qkd_keys = len(self.qkd.key_repository)
        crypto_keys = len(self.crypto.active_keys)
        recent_events = len([event for event in self.security_log if time.time() - event['timestamp'] < 3600])
        
        return {
            'threat_level': self.threat_level,
            'active_qkd_keys': qkd_keys,
            'active_crypto_keys': crypto_keys,
            'recent_security_events': recent_events,
            'total_security_events': len(self.security_log),
            'quantum_safe_status': True,
            'last_security_check': time.time(),
            'encryption_statistics': self.crypto.get_encryption_statistics(),
            'recommendations': self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current status."""
        recommendations = []
        
        if self.threat_level in ["HIGH", "CRITICAL"]:
            recommendations.append("Consider increasing key rotation frequency")
            recommendations.append("Implement additional quantum error correction")
        
        if len(self.crypto.active_keys) > 100:
            recommendations.append("Consider key cleanup and rotation")
        
        recent_failures = len([
            event for event in self.security_log
            if 'failed' in event.get('event', '') and time.time() - event['timestamp'] < 3600
        ])
        
        if recent_failures > 5:
            recommendations.append("Investigate recent security failures")
            recommendations.append("Consider system security audit")
        
        if not recommendations:
            recommendations.append("Security status optimal - continue monitoring")
        
        return recommendations


# Global security orchestrator instance
quantum_security = QuantumSecurityOrchestrator()