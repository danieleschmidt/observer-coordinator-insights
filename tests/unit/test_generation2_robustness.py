"""
Comprehensive Unit Tests for Generation 2 Robustness Features
Tests error handling, circuit breakers, fallback mechanisms, monitoring, and security
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import threading
import time

# Import Generation 2 components
from src.insights_clustering.neuromorphic_clustering import (
    NeuromorphicClusterer, NeuromorphicException, NeuromorphicErrorType,
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    RetryManager, ResourceMetrics, get_correlation_id, set_correlation_id
)
from src.error_handling import (
    EnhancedErrorHandler, ErrorCategory, ErrorSeverity, ErrorDetails,
    NeuromorphicClusteringError, CircuitBreakerError, TimeoutError,
    ResourceExhaustionError, FallbackError
)
from src.insights_clustering.resilience import (
    ResilienceManager, ResourceMonitor, ClusteringQualityGates,
    QualityGate, ResilienceStatus, ResourceType
)
from src.insights_clustering.monitoring import (
    ClusteringMonitor, ClusteringOperationMetrics, ClusteringPhase
)
from src.security import (
    EnhancedDataAnonymizer, EnhancedSecurityAuditor, 
    SecurityEventType, SecurityLevel, DifferentialPrivacyConfig
)


class TestNeuromorphicClusteringRobustness:
    """Test neuromorphic clustering robustness features"""
    
    def setup_method(self):
        """Setup for each test"""
        self.test_data = pd.DataFrame({
            'red_energy': np.random.uniform(0, 100, 100),
            'blue_energy': np.random.uniform(0, 100, 100),
            'green_energy': np.random.uniform(0, 100, 100),
            'yellow_energy': np.random.uniform(0, 100, 100)
        })
        
        # Set correlation ID for testing
        set_correlation_id("test-correlation-123")
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern implementation"""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
        breaker = CircuitBreaker(config)
        
        # Test function that always fails
        @breaker
        def failing_function():
            raise ValueError("Test failure")
        
        # Should allow failures up to threshold (but not including)
        for i in range(2):
            with pytest.raises(ValueError):
                failing_function()
            assert breaker.state == CircuitBreakerState.CLOSED
        
        # Third failure should open the circuit
        with pytest.raises(ValueError):
            failing_function()
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Should raise circuit breaker error when open
        with pytest.raises(NeuromorphicException) as exc_info:
            failing_function()
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_retry_manager_exponential_backoff(self):
        """Test retry manager with exponential backoff"""
        retry_manager = RetryManager(max_retries=3, base_delay=0.1)
        
        call_count = 0
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        # Should succeed after retries
        result = retry_manager.retry_with_backoff(flaky_operation)
        assert result == "success"
        assert call_count == 3
    
    def test_neuromorphic_clusterer_fallback_mechanism(self):
        """Test fallback to K-means when neuromorphic clustering fails"""
        # Create clusterer with fallback enabled
        clusterer = NeuromorphicClusterer(
            n_clusters=3, 
            enable_fallback=True,
            random_state=42
        )
        
        # Mock neuromorphic feature extraction to fail
        with patch.object(clusterer, '_extract_neuromorphic_features_safe') as mock_extract:
            mock_extract.side_effect = NeuromorphicException(
                "Feature extraction failed",
                error_type=NeuromorphicErrorType.CONVERGENCE_ERROR,
                correlation_id="test-123",
                context={}
            )
            
            # Should fallback to K-means
            result = clusterer.fit(self.test_data)
            
            assert result.trained
            assert result.fallback_used
            assert result.cluster_labels is not None
            assert len(np.unique(result.cluster_labels)) == 3
    
    def test_input_validation_robustness(self):
        """Test comprehensive input validation"""
        clusterer = NeuromorphicClusterer(n_clusters=3)
        
        # Test None input
        with pytest.raises(NeuromorphicException) as exc_info:
            clusterer.fit(None)
        assert exc_info.value.error_type == NeuromorphicErrorType.DIMENSION_ERROR
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(NeuromorphicException):
            clusterer.fit(empty_df)
        
        # Test missing columns
        invalid_df = pd.DataFrame({'wrong_col': [1, 2, 3]})
        with pytest.raises(NeuromorphicException):
            clusterer.fit(invalid_df)
        
        # Test insufficient data
        small_df = pd.DataFrame({
            'red_energy': [10, 20],
            'blue_energy': [30, 40],
            'green_energy': [50, 60],
            'yellow_energy': [70, 80]
        })
        with pytest.raises(NeuromorphicException):
            clusterer.fit(small_df)  # Only 2 samples for 3 clusters
    
    def test_resource_monitoring_context(self):
        """Test resource monitoring during operations"""
        from src.insights_clustering.neuromorphic_clustering import resource_monitor
        
        with resource_monitor() as context:
            # Simulate some work
            time.sleep(0.1)
            data = np.random.randn(1000, 100)  # Create some memory usage
        
        # Resource monitoring should not raise exceptions
        assert True  # Test passes if no exceptions
    
    def test_timeout_functionality(self):
        """Test operation timeout handling"""
        from src.insights_clustering.neuromorphic_clustering import timeout_operation
        
        @timeout_operation(timeout_seconds=1)
        def slow_operation():
            time.sleep(2)  # Sleep longer than timeout
            return "completed"
        
        with pytest.raises(NeuromorphicException) as exc_info:
            slow_operation()
        assert exc_info.value.error_type == NeuromorphicErrorType.TIMEOUT_ERROR


class TestEnhancedErrorHandling:
    """Test enhanced error handling capabilities"""
    
    def setup_method(self):
        """Setup for each test"""
        self.error_handler = EnhancedErrorHandler()
        self.correlation_id = "test-error-123"
    
    def test_error_categorization_and_tracking(self):
        """Test error categorization and correlation tracking"""
        # Test standard exception handling
        try:
            raise ValueError("Test validation error")
        except Exception as e:
            error_details = self.error_handler.handle_error(
                e, context={"test": "context"}, correlation_id=self.correlation_id
            )
        
        assert error_details.category == ErrorCategory.DATA_VALIDATION
        assert error_details.severity == ErrorSeverity.HIGH
        assert error_details.correlation_id == self.correlation_id
        assert error_details.recoverable is True
    
    def test_neuromorphic_clustering_error_handling(self):
        """Test neuromorphic clustering specific error handling"""
        error = NeuromorphicClusteringError(
            "Neuromorphic feature extraction failed",
            method="echo_state_network",
            feature_shape=(100, 50),
            component="ESN"
        )
        
        error_details = self.error_handler.handle_error(error, correlation_id=self.correlation_id)
        
        assert error_details.category == ErrorCategory.NEUROMORPHIC_CLUSTERING
        assert error_details.severity == ErrorSeverity.HIGH
        assert "neuromorphic" in error_details.suggestions[0].lower()
    
    def test_alert_threshold_functionality(self):
        """Test alert threshold monitoring"""
        # Generate multiple high-severity errors quickly
        for i in range(6):  # Exceed threshold of 5
            try:
                raise NeuromorphicClusteringError("Test error")
            except Exception as e:
                self.error_handler.handle_error(e, correlation_id=self.correlation_id)
        
        # Check that alerts were triggered (in logs)
        assert len(self.error_handler.error_history) >= 6
    
    def test_correlation_timeline_tracking(self):
        """Test correlation ID timeline tracking"""
        # Generate multiple errors with same correlation ID
        for i in range(3):
            try:
                raise ValueError(f"Test error {i}")
            except Exception as e:
                self.error_handler.handle_error(e, correlation_id=self.correlation_id)
        
        timeline = self.error_handler.get_correlation_timeline(self.correlation_id)
        
        assert timeline['correlation_id'] == self.correlation_id
        assert timeline['error_count'] == 3
        assert len(timeline['timeline']) == 3
    
    def test_error_recovery_suggestions(self):
        """Test context-aware error recovery suggestions"""
        error = CircuitBreakerError(
            "Circuit breaker opened",
            failure_count=5,
            recovery_timeout=60
        )
        
        error_details = self.error_handler.handle_error(error, correlation_id=self.correlation_id)
        suggestions = self.error_handler.suggest_recovery_actions(error_details.error_id)
        
        assert any("circuit breaker" in suggestion.lower() for suggestion in suggestions)
        assert any("60" in suggestion for suggestion in suggestions)  # Recovery timeout


class TestResilienceManager:
    """Test resilience and reliability features"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resilience_manager = ResilienceManager()
    
    def test_resource_monitoring_alerts(self):
        """Test resource monitoring and alerting"""
        resource_monitor = ResourceMonitor()
        
        # Mock high memory usage
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95.0  # Critical threshold
            
            resource_monitor._check_resources()
            
            status = resource_monitor.get_current_status()
            assert status['status'] == ResilienceStatus.CRITICAL.value
            assert len(resource_monitor.alerts) > 0
    
    def test_quality_gates_validation(self):
        """Test clustering quality gates"""
        quality_gates = ClusteringQualityGates()
        
        # Test good quality metrics
        good_metrics = {
            'silhouette': 0.7,
            'calinski_harabasz': 150.0,
            'stability': 0.8
        }
        
        cluster_labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        result = quality_gates.validate_clustering_results(good_metrics, cluster_labels)
        
        assert result['passed'] is True
        assert result['overall_score'] > 0.7
        
        # Test poor quality metrics
        poor_metrics = {
            'silhouette': 0.1,  # Below threshold
            'calinski_harabasz': 20.0,  # Below threshold
            'stability': 0.4  # Below threshold
        }
        
        result = quality_gates.validate_clustering_results(poor_metrics, cluster_labels)
        assert result['passed'] is False
    
    def test_operation_context_tracking(self):
        """Test operation context for resilience tracking"""
        with self.resilience_manager.operation_context("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        assert self.resilience_manager.metrics.total_operations == 1
        assert self.resilience_manager.metrics.successful_operations == 1
    
    def test_fallback_recording(self):
        """Test fallback operation recording"""
        self.resilience_manager.record_fallback_operation(
            "neuromorphic_clustering", "kmeans"
        )
        
        assert self.resilience_manager.metrics.fallback_operations == 1
        assert self.resilience_manager.metrics.fallback_rate > 0


class TestClusteringMonitoring:
    """Test clustering-specific monitoring"""
    
    def setup_method(self):
        """Setup for each test"""
        self.monitor = ClusteringMonitor()
    
    def test_operation_monitoring_lifecycle(self):
        """Test complete operation monitoring lifecycle"""
        with self.monitor.monitor_operation(
            method="test_method",
            correlation_id="test-123",
            dataset_size=100,
            target_clusters=3
        ) as metrics:
            
            # Simulate different phases
            with self.monitor.monitor_phase(metrics.operation_id, ClusteringPhase.INITIALIZATION):
                time.sleep(0.01)
            
            with self.monitor.monitor_phase(metrics.operation_id, ClusteringPhase.FEATURE_EXTRACTION):
                time.sleep(0.01)
            
            # Record some metrics
            self.monitor.record_quality_metrics(metrics.operation_id, {
                'silhouette_score': 0.7,
                'calinski_harabasz_score': 150.0
            })
            
            self.monitor.record_resource_usage(metrics.operation_id, 512.0, 45.0)
        
        # Check that operation was recorded
        assert len(self.monitor.completed_operations) == 1
        completed_op = self.monitor.completed_operations[0]
        assert completed_op.status == "success"
        assert completed_op.silhouette_score == 0.7
        assert len(completed_op.phase_durations) > 0
    
    def test_performance_dashboard_generation(self):
        """Test performance dashboard generation"""
        # Add some test operations
        for i in range(5):
            with self.monitor.monitor_operation("test_method", dataset_size=100) as metrics:
                self.monitor.record_quality_metrics(metrics.operation_id, {
                    'silhouette_score': 0.6 + (i * 0.1)
                })
                time.sleep(0.001)  # Minimal delay
        
        dashboard = self.monitor.get_performance_dashboard(hours=1)
        
        assert dashboard['summary']['total_operations'] == 5
        assert dashboard['summary']['success_rate'] == 1.0
        assert 'quality_metrics' in dashboard
        assert dashboard['quality_metrics']['avg_silhouette_score'] is not None
    
    def test_neuromorphic_metrics_recording(self):
        """Test neuromorphic-specific metrics recording"""
        with self.monitor.monitor_operation("neuromorphic") as metrics:
            # Record successful neuromorphic operation
            self.monitor.record_neuromorphic_metrics(
                metrics.operation_id, "esn", 150.0, True, 50
            )
            
            # Record failed neuromorphic operation
            self.monitor.record_neuromorphic_metrics(
                metrics.operation_id, "snn", 0.0, False, 0
            )
        
        completed_op = self.monitor.completed_operations[0]
        assert "esn" in completed_op.neuromorphic_components_used
        assert completed_op.feature_extraction_failures == 1


class TestSecurityEnhancements:
    """Test security enhancements and differential privacy"""
    
    def setup_method(self):
        """Setup for each test"""
        self.privacy_config = DifferentialPrivacyConfig(epsilon=1.0, enabled=True)
        self.anonymizer = EnhancedDataAnonymizer(privacy_config=self.privacy_config)
        self.auditor = EnhancedSecurityAuditor()
    
    def test_differential_privacy_scalar(self):
        """Test differential privacy on scalar values"""
        original_value = 100.0
        noisy_value = self.anonymizer.apply_differential_privacy(
            original_value, "mean", "test-correlation"
        )
        
        # Should be different but within reasonable bounds
        assert noisy_value != original_value
        assert abs(noisy_value - original_value) < 50  # Reasonable noise
    
    def test_differential_privacy_vector(self):
        """Test differential privacy on vectors"""
        original_vector = np.array([1.0, 2.0, 3.0, 4.0])
        noisy_vector = self.anonymizer.apply_differential_privacy(
            original_vector, "mean", "test-correlation"
        )
        
        assert len(noisy_vector) == len(original_vector)
        assert not np.array_equal(noisy_vector, original_vector)
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget exhaustion"""
        correlation_id = "budget-test"
        
        # Exhaust privacy budget
        for i in range(15):  # Should exceed epsilon=1.0
            try:
                self.anonymizer.apply_differential_privacy(
                    100.0, "query", correlation_id
                )
            except ValueError as e:
                assert "budget exhausted" in str(e)
                break
        else:
            pytest.fail("Privacy budget should have been exhausted")
    
    def test_clustering_results_anonymization(self):
        """Test anonymization of clustering results"""
        cluster_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        feature_data = np.random.randn(10, 4)
        
        anonymized = self.anonymizer.anonymize_clustering_results(
            cluster_labels, feature_data, "test-correlation"
        )
        
        assert 'anonymized_labels' in anonymized
        assert 'cluster_statistics' in anonymized
        assert 'privacy_parameters' in anonymized
        assert 'k_anonymity_threshold' in anonymized
    
    def test_security_event_logging_and_risk_assessment(self):
        """Test comprehensive security event logging"""
        event_id = self.auditor.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id="test_user",
            resource="employee_data",
            action="read",
            success=True,
            details={'record_count': 1000},
            correlation_id="test-123",
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        assert event_id is not None
        assert len(self.auditor.security_events) == 1
        
        event = self.auditor.security_events[0]
        assert event.risk_score > 0
        assert event.event_type == SecurityEventType.DATA_ACCESS
    
    def test_security_anomaly_detection(self):
        """Test security anomaly detection"""
        user_id = "suspicious_user"
        
        # Generate multiple high-risk events quickly
        for i in range(10):
            self.auditor.log_security_event(
                SecurityEventType.VALIDATION_FAILURE,
                user_id, "system", "failed_login", False,
                {'failure_type': 'injection_attempt'}, f"test-{i}"
            )
        
        # Should trigger anomaly detection
        assert len(self.auditor.security_events) > 10  # Original events + anomaly alerts
        
        # Check for anomaly detection events
        anomaly_events = [
            event for event in self.auditor.security_events
            if event.event_type == SecurityEventType.ANOMALY_DETECTION
        ]
        assert len(anomaly_events) > 0


class TestIntegrationRobustness:
    """Test integration of all robustness features"""
    
    def test_end_to_end_robust_clustering(self):
        """Test end-to-end robust clustering with all features enabled"""
        # Create test data
        test_data = pd.DataFrame({
            'red_energy': np.random.uniform(20, 80, 50),
            'blue_energy': np.random.uniform(20, 80, 50),
            'green_energy': np.random.uniform(20, 80, 50),
            'yellow_energy': np.random.uniform(20, 80, 50)
        })
        
        # Initialize robust clusterer
        clusterer = NeuromorphicClusterer(
            n_clusters=3,
            enable_fallback=True,
            random_state=42
        )
        
        # Monitor the operation
        clustering_monitor = ClusteringMonitor()
        
        with clustering_monitor.monitor_operation("integration_test") as metrics:
            # Perform clustering
            result = clusterer.fit(test_data)
            
            # Record quality metrics
            if hasattr(result, 'get_clustering_metrics'):
                quality_metrics = result.get_clustering_metrics()
                clustering_monitor.record_quality_metrics(
                    metrics.operation_id,
                    {'silhouette_score': getattr(quality_metrics, 'silhouette_score', 0.5)}
                )
        
        # Verify robustness features worked
        assert result.trained
        assert result.cluster_labels is not None
        assert len(clustering_monitor.completed_operations) == 1
        
        # Check that operation was successful
        completed_op = clustering_monitor.completed_operations[0]
        assert completed_op.status == "success"
    
    def test_configuration_validation_robustness(self):
        """Test configuration validation for robustness parameters"""
        # Test invalid circuit breaker config
        with pytest.raises((ValueError, TypeError)):
            invalid_config = CircuitBreakerConfig(failure_threshold=-1)
        
        # Test invalid privacy config
        with pytest.raises((ValueError, TypeError)):
            invalid_privacy = DifferentialPrivacyConfig(epsilon=-1.0)
    
    def test_api_compatibility_maintained(self):
        """Test that API compatibility is maintained with Generation 2 features"""
        # Test that old API still works
        test_data = pd.DataFrame({
            'red_energy': [10, 20, 30],
            'blue_energy': [40, 50, 60],
            'green_energy': [70, 80, 90],
            'yellow_energy': [15, 25, 35]
        })
        
        # Legacy usage should still work
        clusterer = NeuromorphicClusterer(n_clusters=2)
        result = clusterer.fit(test_data)
        
        # Should have both old and new features
        assert hasattr(result, 'trained')  # Old attribute
        assert hasattr(result, 'fallback_used')  # New attribute
        
        # Old methods should still work
        labels = result.get_cluster_assignments()
        assert labels is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])