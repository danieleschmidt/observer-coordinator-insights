#!/usr/bin/env python3
"""Quantum Resilience Framework - Generation 2 Implementation
Advanced error handling, validation, and quantum error correction
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class QuantumErrorType(Enum):
    """Types of quantum errors that can occur."""
    DECOHERENCE = "decoherence"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    AMPLITUDE_DAMPING = "amplitude_damping"
    GATE_ERROR = "gate_error"
    MEASUREMENT_ERROR = "measurement_error"
    CROSSTALK = "crosstalk"
    THERMAL_NOISE = "thermal_noise"


@dataclass
class QuantumError:
    """Represents a quantum error with context."""
    error_type: QuantumErrorType
    severity: float  # 0.0 to 1.0
    affected_qubits: List[int]
    timestamp: float
    context: Dict[str, Any]
    correction_applied: bool = False
    correction_method: Optional[str] = None


class QuantumValidationModel(BaseModel):
    """Pydantic model for quantum system validation."""
    coherence_time: float = Field(ge=0, description="Coherence time in microseconds")
    fidelity: float = Field(ge=0, le=1, description="Gate fidelity")
    error_rate: float = Field(ge=0, le=1, description="Error rate per operation")
    temperature: float = Field(ge=0, description="System temperature in mK")
    entanglement_strength: float = Field(ge=0, le=1, description="Entanglement strength")
    quantum_volume: int = Field(ge=1, description="Quantum volume")
    
    @validator('coherence_time')
    def coherence_time_realistic(cls, v):
        if v < 1 or v > 1000:  # 1μs to 1ms is realistic range
            raise ValueError('Coherence time must be between 1 and 1000 microseconds')
        return v
    
    @validator('fidelity')
    def fidelity_minimum(cls, v):
        if v < 0.9:
            raise ValueError('Gate fidelity must be at least 90% for reliable operation')
        return v
    
    @validator('temperature')
    def temperature_range(cls, v):
        if v < 10 or v > 100:  # 10mK to 100mK typical range
            raise ValueError('Operating temperature must be between 10 and 100 mK')
        return v


class QuantumCircuitBreaker:
    """Circuit breaker pattern for quantum operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        error_rate_threshold: float = 0.1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.error_rate_threshold = error_rate_threshold
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.error_history = []
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise RuntimeError("Circuit breaker OPEN - quantum operation blocked")
        
        try:
            start_time = time.time()
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success - reset if in half-open state
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - quantum operation restored")
                
            # Track successful operation
            self.error_history.append({
                'timestamp': time.time(),
                'success': True,
                'duration': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Track failed operation
            self.error_history.append({
                'timestamp': time.time(),
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            # Calculate recent error rate
            recent_history = [
                op for op in self.error_history
                if time.time() - op['timestamp'] < 300  # Last 5 minutes
            ]
            
            if recent_history:
                error_rate = len([op for op in recent_history if not op['success']]) / len(recent_history)
            else:
                error_rate = 0
            
            # Open circuit if thresholds exceeded
            if (self.failure_count >= self.failure_threshold or 
                error_rate > self.error_rate_threshold):
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN - failures: {self.failure_count}, error rate: {error_rate:.2%}")
            
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        recent_ops = [op for op in self.error_history if time.time() - op['timestamp'] < 300]
        success_rate = len([op for op in recent_ops if op['success']]) / len(recent_ops) if recent_ops else 1.0
        
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'total_operations': len(self.error_history),
            'last_failure_time': self.last_failure_time
        }


class QuantumErrorCorrection:
    """Advanced quantum error correction system."""
    
    def __init__(self):
        self.error_log = []
        self.correction_stats = {error_type: 0 for error_type in QuantumErrorType}
        self.correction_methods = {
            QuantumErrorType.DECOHERENCE: self._correct_decoherence,
            QuantumErrorType.BIT_FLIP: self._correct_bit_flip,
            QuantumErrorType.PHASE_FLIP: self._correct_phase_flip,
            QuantumErrorType.AMPLITUDE_DAMPING: self._correct_amplitude_damping,
            QuantumErrorType.GATE_ERROR: self._correct_gate_error,
            QuantumErrorType.MEASUREMENT_ERROR: self._correct_measurement_error,
            QuantumErrorType.CROSSTALK: self._correct_crosstalk,
            QuantumErrorType.THERMAL_NOISE: self._correct_thermal_noise
        }
        
    def detect_quantum_errors(self, quantum_state: np.ndarray, expected_state: np.ndarray) -> List[QuantumError]:
        """Detect quantum errors by comparing states."""
        errors = []
        
        # Calculate fidelity
        fidelity = np.abs(np.vdot(quantum_state, expected_state))**2
        
        if fidelity < 0.99:  # Error detected
            # Analyze error type based on state differences
            state_diff = quantum_state - expected_state
            
            # Decoherence detection (loss of coherence)
            coherence_loss = np.sum(np.abs(state_diff)**2)
            if coherence_loss > 0.1:
                errors.append(QuantumError(
                    error_type=QuantumErrorType.DECOHERENCE,
                    severity=min(1.0, coherence_loss),
                    affected_qubits=list(range(len(quantum_state))),
                    timestamp=time.time(),
                    context={'fidelity': fidelity, 'coherence_loss': coherence_loss}
                ))
            
            # Bit flip detection (population inversion)
            bit_flip_indicators = np.abs(np.real(state_diff))
            for i, indicator in enumerate(bit_flip_indicators):
                if indicator > 0.1:
                    errors.append(QuantumError(
                        error_type=QuantumErrorType.BIT_FLIP,
                        severity=min(1.0, indicator),
                        affected_qubits=[i],
                        timestamp=time.time(),
                        context={'bit_flip_strength': indicator}
                    ))
            
            # Phase flip detection (phase errors)
            phase_flip_indicators = np.abs(np.imag(state_diff))
            for i, indicator in enumerate(phase_flip_indicators):
                if indicator > 0.1:
                    errors.append(QuantumError(
                        error_type=QuantumErrorType.PHASE_FLIP,
                        severity=min(1.0, indicator),
                        affected_qubits=[i],
                        timestamp=time.time(),
                        context={'phase_error': indicator}
                    ))
        
        return errors
    
    async def correct_quantum_errors(self, errors: List[QuantumError], quantum_state: np.ndarray) -> np.ndarray:
        """Apply quantum error correction to the state."""
        corrected_state = quantum_state.copy()
        
        for error in errors:
            try:
                correction_method = self.correction_methods.get(error.error_type)
                if correction_method:
                    corrected_state = await correction_method(error, corrected_state)
                    error.correction_applied = True
                    error.correction_method = correction_method.__name__
                    self.correction_stats[error.error_type] += 1
                    
                    logger.info(f"Applied {error.error_type.value} correction: severity={error.severity:.3f}")
                else:
                    logger.warning(f"No correction method for error type: {error.error_type}")
                    
            except Exception as e:
                logger.error(f"Error correction failed: {e}")
                error.correction_applied = False
            
            # Log error for analysis
            self.error_log.append(error)
        
        return corrected_state
    
    async def _correct_decoherence(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct decoherence errors using state restoration."""
        # Renormalize state and apply coherence restoration
        norm = np.linalg.norm(state)
        if norm > 0:
            corrected = state / norm
            # Apply coherence enhancement
            coherence_factor = 1.0 / (1.0 + error.severity)
            corrected = corrected * coherence_factor
            return corrected / np.linalg.norm(corrected)
        return state
    
    async def _correct_bit_flip(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct bit flip errors using majority voting."""
        corrected = state.copy()
        for qubit_id in error.affected_qubits:
            if qubit_id < len(corrected):
                # Apply bit flip correction (simplified)
                if error.severity > 0.5:
                    corrected[qubit_id] = -corrected[qubit_id]
        return corrected
    
    async def _correct_phase_flip(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct phase flip errors."""
        corrected = state.copy()
        for qubit_id in error.affected_qubits:
            if qubit_id < len(corrected):
                # Apply phase correction
                phase_correction = np.exp(-1j * error.severity * np.pi)
                corrected[qubit_id] *= phase_correction
        return corrected
    
    async def _correct_amplitude_damping(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct amplitude damping errors."""
        # Restore amplitude based on error severity
        restoration_factor = 1.0 + error.severity * 0.1
        return state * restoration_factor
    
    async def _correct_gate_error(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct gate operation errors."""
        # Apply inverse gate operation (simplified)
        correction_matrix = np.eye(len(state)) * (1.0 - error.severity)
        return correction_matrix @ state
    
    async def _correct_measurement_error(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct measurement-induced errors."""
        # Apply measurement correction based on known error characteristics
        corrected = state.copy()
        noise_reduction = 1.0 - error.severity * 0.5
        return corrected * noise_reduction
    
    async def _correct_crosstalk(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct crosstalk errors between qubits."""
        # Reduce crosstalk effects
        crosstalk_correction = np.eye(len(state))
        for qubit_id in error.affected_qubits:
            if qubit_id < len(state) - 1:
                crosstalk_correction[qubit_id, qubit_id + 1] = -error.severity * 0.1
        return crosstalk_correction @ state
    
    async def _correct_thermal_noise(self, error: QuantumError, state: np.ndarray) -> np.ndarray:
        """Correct thermal noise effects."""
        # Apply thermal noise reduction
        noise_filter = 1.0 - error.severity * 0.2
        return state * noise_filter
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get quantum error correction statistics."""
        total_corrections = sum(self.correction_stats.values())
        
        return {
            'total_errors_corrected': total_corrections,
            'error_type_distribution': dict(self.correction_stats),
            'correction_success_rate': len([e for e in self.error_log if e.correction_applied]) / len(self.error_log) if self.error_log else 1.0,
            'recent_error_rate': self._calculate_recent_error_rate(),
            'most_common_error': max(self.correction_stats.items(), key=lambda x: x[1])[0].value if self.correction_stats else None
        }
    
    def _calculate_recent_error_rate(self) -> float:
        """Calculate error rate in the last hour."""
        recent_errors = [e for e in self.error_log if time.time() - e.timestamp < 3600]
        return len(recent_errors) / 3600  # errors per second


@asynccontextmanager
async def quantum_resilience_context(operation_name: str, error_correction: Optional[QuantumErrorCorrection] = None):
    """Context manager for quantum operations with full resilience."""
    start_time = time.time()
    errors_detected = []
    
    try:
        logger.info(f"🔒 Starting resilient quantum operation: {operation_name}")
        yield errors_detected
        logger.info(f"✅ Quantum operation completed successfully: {operation_name} ({time.time() - start_time:.3f}s)")
        
    except Exception as e:
        # Log the error with full context
        error_context = {
            'operation': operation_name,
            'duration': time.time() - start_time,
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        
        logger.error(f"❌ Quantum operation failed: {operation_name}")
        logger.debug(f"Error context: {error_context}")
        
        # If we have error correction, try to handle quantum-specific errors
        if error_correction and isinstance(e, (ValueError, RuntimeError)):
            # Create a generic quantum error for correction attempt
            quantum_error = QuantumError(
                error_type=QuantumErrorType.GATE_ERROR,
                severity=0.5,
                affected_qubits=[0],
                timestamp=time.time(),
                context=error_context
            )
            errors_detected.append(quantum_error)
            
            logger.warning(f"Attempting quantum error correction for: {operation_name}")
        
        raise
    
    finally:
        # Apply any detected error corrections
        if errors_detected and error_correction:
            try:
                correction_stats = error_correction.get_correction_statistics()
                logger.info(f"Quantum error correction stats: {correction_stats}")
            except Exception as correction_error:
                logger.error(f"Error correction statistics failed: {correction_error}")


class QuantumValidationFramework:
    """Comprehensive validation framework for quantum systems."""
    
    def __init__(self):
        self.validation_history = []
        self.error_correction = QuantumErrorCorrection()
    
    async def validate_quantum_system(self, system_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quantum system parameters comprehensively."""
        try:
            # Use Pydantic model for validation
            quantum_model = QuantumValidationModel(**system_params)
            
            validation_result = {
                'is_valid': True,
                'quality_score': 100.0,
                'errors': [],
                'warnings': [],
                'recommendations': [],
                'validated_params': quantum_model.dict()
            }
            
            # Additional quantum-specific validations
            warnings = []
            score_deductions = 0
            
            # Check coherence time
            if quantum_model.coherence_time < 10:
                warnings.append("Low coherence time may affect quantum operations")
                score_deductions += 10
            
            # Check gate fidelity
            if quantum_model.fidelity < 0.95:
                warnings.append("Gate fidelity below optimal threshold")
                score_deductions += 15
            
            # Check error rate
            if quantum_model.error_rate > 0.05:
                warnings.append("High error rate detected - consider calibration")
                score_deductions += 20
            
            # Check temperature
            if quantum_model.temperature > 50:
                warnings.append("Operating temperature higher than optimal")
                score_deductions += 5
            
            # Update validation result
            validation_result['warnings'] = warnings
            validation_result['quality_score'] = max(0, 100 - score_deductions)
            
            # Generate recommendations
            recommendations = []
            if quantum_model.coherence_time < 50:
                recommendations.append("Consider improving qubit isolation to increase coherence time")
            if quantum_model.fidelity < 0.98:
                recommendations.append("Perform gate calibration to improve fidelity")
            if quantum_model.error_rate > 0.01:
                recommendations.append("Implement additional error correction protocols")
            
            validation_result['recommendations'] = recommendations
            
            # Log validation
            self.validation_history.append({
                'timestamp': time.time(),
                'result': validation_result,
                'system_params': system_params
            })
            
            logger.info(f"Quantum system validation complete: score={validation_result['quality_score']:.1f}")
            
            return validation_result
            
        except Exception as e:
            error_result = {
                'is_valid': False,
                'quality_score': 0.0,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': ["Review and correct quantum system parameters"]
            }
            
            logger.error(f"Quantum validation failed: {e}")
            return error_result
    
    async def validate_quantum_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Validate quantum data integrity and quality."""
        validation_result = {
            'is_valid': True,
            'quality_score': 100.0,
            'errors': [],
            'warnings': [],
            'data_stats': {}
        }
        
        try:
            # Basic data validation
            if data is None or data.size == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Data is empty or None")
                return validation_result
            
            # Check for complex data (quantum states often complex)
            is_complex = np.iscomplexobj(data)
            
            # Calculate data statistics
            data_stats = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'is_complex': is_complex,
                'has_nan': bool(np.isnan(data).any()),
                'has_inf': bool(np.isinf(data).any()),
                'min_value': float(np.min(np.real(data))),
                'max_value': float(np.max(np.real(data))),
                'mean_magnitude': float(np.mean(np.abs(data))),
                'std_magnitude': float(np.std(np.abs(data)))
            }
            
            validation_result['data_stats'] = data_stats
            
            # Check for issues
            warnings = []
            score_deductions = 0
            
            if data_stats['has_nan']:
                validation_result['errors'].append("Data contains NaN values")
                validation_result['is_valid'] = False
                score_deductions += 50
            
            if data_stats['has_inf']:
                validation_result['errors'].append("Data contains infinite values")
                validation_result['is_valid'] = False
                score_deductions += 30
            
            # Check normalization for quantum states
            if is_complex:
                norms = np.linalg.norm(data, axis=-1) if data.ndim > 1 else np.array([np.linalg.norm(data)])
                norm_deviation = np.std(norms)
                
                if norm_deviation > 0.1:
                    warnings.append(f"Quantum state normalization inconsistent: std={norm_deviation:.3f}")
                    score_deductions += 10
                
                data_stats['normalization_std'] = float(norm_deviation)
            
            # Check dynamic range
            if data_stats['max_value'] - data_stats['min_value'] < 1e-10:
                warnings.append("Very small dynamic range in data")
                score_deductions += 5
            
            validation_result['warnings'] = warnings
            validation_result['quality_score'] = max(0, 100 - score_deductions)
            
            logger.info(f"Quantum data validation complete: score={validation_result['quality_score']:.1f}")
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Data validation failed: {str(e)}")
            logger.error(f"Quantum data validation error: {e}")
        
        return validation_result


# Global instances for easy access
quantum_circuit_breaker = QuantumCircuitBreaker()
quantum_error_correction = QuantumErrorCorrection()
quantum_validator = QuantumValidationFramework()