"""Configuration Validation for Generation 2 Robustness Features
Validates and enforces configuration constraints for enhanced system reliability
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


class ConfigValidationLevel(Enum):
    """Validation levels for configuration checks"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


@dataclass
class ValidationRule:
    """Configuration validation rule"""
    field_path: str
    rule_type: str  # 'range', 'enum', 'type', 'required', 'regex'
    constraint: Any
    message: str
    level: ConfigValidationLevel = ConfigValidationLevel.STRICT


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationValidator:
    """Validates Generation 2 configuration parameters"""

    def __init__(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.STRICT):
        self.validation_level = validation_level
        self.rules = self._define_validation_rules()

    def _define_validation_rules(self) -> List[ValidationRule]:
        """Define validation rules for all Generation 2 components"""
        return [
            # Circuit Breaker Configuration
            ValidationRule(
                "circuit_breaker.failure_threshold",
                "range",
                {"min": 1, "max": 20},
                "Circuit breaker failure threshold must be between 1 and 20",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "circuit_breaker.recovery_timeout",
                "range",
                {"min": 5, "max": 3600},
                "Circuit breaker recovery timeout must be between 5 and 3600 seconds",
                ConfigValidationLevel.STRICT
            ),

            # Differential Privacy Configuration
            ValidationRule(
                "privacy.epsilon",
                "range",
                {"min": 0.01, "max": 10.0},
                "Privacy epsilon must be between 0.01 and 10.0",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "privacy.delta",
                "range",
                {"min": 1e-10, "max": 0.1},
                "Privacy delta must be between 1e-10 and 0.1",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "privacy.mechanism",
                "enum",
                ["laplace", "gaussian", "exponential"],
                "Privacy mechanism must be one of: laplace, gaussian, exponential",
                ConfigValidationLevel.STRICT
            ),

            # Neuromorphic Clustering Configuration
            ValidationRule(
                "neuromorphic.method",
                "enum",
                ["echo_state_network", "spiking_neural_network", "liquid_state_machine", "hybrid_reservoir"],
                "Neuromorphic method must be a valid clustering method",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "neuromorphic.n_clusters",
                "range",
                {"min": 2, "max": 50},
                "Number of clusters must be between 2 and 50",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "neuromorphic.enable_fallback",
                "type",
                bool,
                "Enable fallback must be a boolean",
                ConfigValidationLevel.MODERATE
            ),

            # Resource Monitoring Configuration
            ValidationRule(
                "monitoring.memory_warning_threshold",
                "range",
                {"min": 50.0, "max": 95.0},
                "Memory warning threshold must be between 50% and 95%",
                ConfigValidationLevel.MODERATE
            ),
            ValidationRule(
                "monitoring.memory_critical_threshold",
                "range",
                {"min": 80.0, "max": 99.0},
                "Memory critical threshold must be between 80% and 99%",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "monitoring.cpu_warning_threshold",
                "range",
                {"min": 50.0, "max": 95.0},
                "CPU warning threshold must be between 50% and 95%",
                ConfigValidationLevel.MODERATE
            ),

            # Quality Gates Configuration
            ValidationRule(
                "quality_gates.silhouette_threshold",
                "range",
                {"min": -1.0, "max": 1.0},
                "Silhouette threshold must be between -1.0 and 1.0",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "quality_gates.min_cluster_size",
                "range",
                {"min": 2, "max": 1000},
                "Minimum cluster size must be between 2 and 1000",
                ConfigValidationLevel.STRICT
            ),

            # Retry Configuration
            ValidationRule(
                "retry.max_retries",
                "range",
                {"min": 0, "max": 10},
                "Maximum retries must be between 0 and 10",
                ConfigValidationLevel.MODERATE
            ),
            ValidationRule(
                "retry.base_delay",
                "range",
                {"min": 0.1, "max": 60.0},
                "Base delay must be between 0.1 and 60 seconds",
                ConfigValidationLevel.MODERATE
            ),

            # Timeout Configuration
            ValidationRule(
                "timeouts.clustering_timeout",
                "range",
                {"min": 30, "max": 3600},
                "Clustering timeout must be between 30 and 3600 seconds",
                ConfigValidationLevel.STRICT
            ),
            ValidationRule(
                "timeouts.feature_extraction_timeout",
                "range",
                {"min": 10, "max": 1800},
                "Feature extraction timeout must be between 10 and 1800 seconds",
                ConfigValidationLevel.MODERATE
            ),
        ]

    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate complete configuration"""
        result = ValidationResult(is_valid=True, validated_config=config.copy())

        for rule in self.rules:
            if rule.level.value not in self._get_applicable_levels():
                continue

            validation_error = self._validate_rule(config, rule)
            if validation_error:
                if rule.level == ConfigValidationLevel.STRICT:
                    result.errors.append(validation_error)
                    result.is_valid = False
                else:
                    result.warnings.append(validation_error)

        # Apply defaults for missing values
        result.validated_config = self._apply_defaults(result.validated_config)

        return result

    def _get_applicable_levels(self) -> List[str]:
        """Get applicable validation levels based on current setting"""
        if self.validation_level == ConfigValidationLevel.STRICT:
            return ["strict", "moderate", "permissive"]
        elif self.validation_level == ConfigValidationLevel.MODERATE:
            return ["moderate", "permissive"]
        else:
            return ["permissive"]

    def _validate_rule(self, config: Dict[str, Any], rule: ValidationRule) -> Optional[str]:
        """Validate a single rule against configuration"""
        try:
            value = self._get_nested_value(config, rule.field_path)

            if rule.rule_type == "required" and value is None:
                return f"Required field '{rule.field_path}' is missing"

            if value is None:
                return None  # Optional field not provided

            if rule.rule_type == "range":
                min_val = rule.constraint.get("min")
                max_val = rule.constraint.get("max")
                if min_val is not None and value < min_val:
                    return f"{rule.message}: got {value}, minimum is {min_val}"
                if max_val is not None and value > max_val:
                    return f"{rule.message}: got {value}, maximum is {max_val}"

            elif rule.rule_type == "enum":
                if value not in rule.constraint:
                    return f"{rule.message}: got '{value}', valid options are {rule.constraint}"

            elif rule.rule_type == "type":
                if not isinstance(value, rule.constraint):
                    return f"{rule.message}: got {type(value).__name__}, expected {rule.constraint.__name__}"

            return None

        except Exception as e:
            return f"Error validating '{rule.field_path}': {e!s}"

    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from configuration using dot notation"""
        keys = field_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing configuration"""
        defaults = {
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60
            },
            "privacy": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "mechanism": "laplace",
                "enabled": True
            },
            "neuromorphic": {
                "method": "hybrid_reservoir",
                "n_clusters": 4,
                "enable_fallback": True,
                "random_state": 42
            },
            "monitoring": {
                "memory_warning_threshold": 80.0,
                "memory_critical_threshold": 95.0,
                "cpu_warning_threshold": 70.0,
                "cpu_critical_threshold": 90.0
            },
            "quality_gates": {
                "silhouette_threshold": 0.3,
                "calinski_harabasz_threshold": 50.0,
                "min_cluster_size": 5,
                "enabled": True
            },
            "retry": {
                "max_retries": 3,
                "base_delay": 1.0,
                "max_delay": 60.0
            },
            "timeouts": {
                "clustering_timeout": 600,
                "feature_extraction_timeout": 300
            }
        }

        return self._merge_with_defaults(config, defaults)

    def _merge_with_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration with defaults"""
        result = defaults.copy()

        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_with_defaults(value, result[key])
            else:
                result[key] = value

        return result

    def validate_from_file(self, file_path: str) -> ValidationResult:
        """Validate configuration from file"""
        try:
            with open(file_path) as f:
                if file_path.endswith('.json'):
                    config = json.load(f)
                elif file_path.endswith(('.yml', '.yaml')):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")

            return self.validate_configuration(config)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to load configuration file: {e!s}"]
            )

    def generate_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for configuration validation"""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Generation 2 Robustness Configuration",
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }

        # Build schema from validation rules
        for rule in self.rules:
            self._add_rule_to_schema(schema, rule)

        return schema

    def _add_rule_to_schema(self, schema: Dict[str, Any], rule: ValidationRule):
        """Add validation rule to JSON schema"""
        keys = rule.field_path.split('.')
        current = schema["properties"]

        # Navigate/create nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {"type": "object", "properties": {}}
            current = current[key]["properties"]

        # Add rule for final key
        final_key = keys[-1]
        if final_key not in current:
            current[final_key] = {}

        property_def = current[final_key]

        if rule.rule_type == "range":
            property_def["type"] = "number"
            if "min" in rule.constraint:
                property_def["minimum"] = rule.constraint["min"]
            if "max" in rule.constraint:
                property_def["maximum"] = rule.constraint["max"]

        elif rule.rule_type == "enum":
            property_def["enum"] = rule.constraint

        elif rule.rule_type == "type":
            if rule.constraint == bool:
                property_def["type"] = "boolean"
            elif rule.constraint == int:
                property_def["type"] = "integer"
            elif rule.constraint == float:
                property_def["type"] = "number"
            elif rule.constraint == str:
                property_def["type"] = "string"


# Global configuration validator
config_validator = ConfigurationValidator()


def validate_config(config: Dict[str, Any], level: ConfigValidationLevel = ConfigValidationLevel.STRICT) -> ValidationResult:
    """Convenience function for configuration validation"""
    validator = ConfigurationValidator(level)
    return validator.validate_configuration(config)


def load_and_validate_config(file_path: str, level: ConfigValidationLevel = ConfigValidationLevel.STRICT) -> ValidationResult:
    """Load and validate configuration from file"""
    validator = ConfigurationValidator(level)
    return validator.validate_from_file(file_path)


if __name__ == "__main__":
    # Example usage and testing
    test_config = {
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout": 120
        },
        "privacy": {
            "epsilon": 1.0,
            "mechanism": "laplace"
        },
        "neuromorphic": {
            "method": "hybrid_reservoir",
            "n_clusters": 4
        }
    }

    result = validate_config(test_config)
    print(f"Configuration valid: {result.is_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    # Generate schema
    schema = config_validator.generate_schema()
    print(f"Generated schema with {len(schema['properties'])} top-level properties")
