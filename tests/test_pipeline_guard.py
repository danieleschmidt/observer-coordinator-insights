"""
Comprehensive tests for Self-Healing Pipeline Guard system
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

# Import pipeline guard components
from src.pipeline_guard.pipeline_guard import (
    SelfHealingPipelineGuard, 
    PipelineComponent, 
    PipelineState
)
from src.pipeline_guard.monitoring import PipelineMonitor, HealthChecker
from src.pipeline_guard.recovery import (
    RecoveryEngine, 
    FailureAnalyzer, 
    RecoveryAction, 
    RecoveryStrategy,
    FailureEvent
)
from src.pipeline_guard.predictor import (
    FailurePredictor, 
    NeuromorphicPredictor,
    PredictionResult
)
from src.pipeline_guard.integration import PipelineGuardIntegrator


class TestPipelineComponent:
    """Test PipelineComponent functionality"""
    
    def test_component_creation(self):
        """Test basic component creation"""
        def mock_health_check():
            return True
        
        component = PipelineComponent(
            name="test_component",
            component_type="test",
            health_check=mock_health_check,
            critical=True,
            max_failures=3
        )
        
        assert component.name == "test_component"
        assert component.component_type == "test"
        assert component.critical is True
        assert component.max_failures == 3
        assert component.failure_count == 0
        assert component.state == PipelineState.HEALTHY
        assert component.health_check() is True
    
    def test_component_with_recovery_actions(self):
        """Test component with recovery actions"""
        def mock_health_check():
            return True
        
        def mock_recovery():
            return True
        
        recovery_action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RESTART,
            action=mock_recovery
        )
        
        component = PipelineComponent(
            name="test_component",
            component_type="test",
            health_check=mock_health_check,
            recovery_actions=[recovery_action],
            dependencies=["dep1", "dep2"]
        )
        
        assert len(component.recovery_actions) == 1
        assert component.recovery_actions[0].name == "test_recovery"
        assert component.dependencies == ["dep1", "dep2"]


class TestSelfHealingPipelineGuard:
    """Test core SelfHealingPipelineGuard functionality"""
    
    @pytest.fixture
    def pipeline_guard(self):
        """Create a pipeline guard instance for testing"""
        return SelfHealingPipelineGuard(monitoring_interval=1, recovery_timeout=30)
    
    @pytest.fixture
    def mock_component(self):
        """Create a mock component for testing"""
        def mock_health_check():
            return True
        
        def mock_recovery():
            return True
        
        return PipelineComponent(
            name="test_component",
            component_type="test",
            health_check=mock_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="restart",
                    strategy=RecoveryStrategy.RESTART,
                    action=mock_recovery
                )
            ]
        )
    
    def test_initialization(self, pipeline_guard):
        """Test pipeline guard initialization"""
        assert pipeline_guard.monitoring_interval == 1
        assert pipeline_guard.recovery_timeout == 30
        assert not pipeline_guard.is_running
        assert len(pipeline_guard.components) == 0
    
    def test_component_registration(self, pipeline_guard, mock_component):
        """Test component registration"""
        pipeline_guard.register_component(mock_component)
        
        assert "test_component" in pipeline_guard.components
        assert pipeline_guard.components["test_component"] == mock_component
        assert "test_component" in pipeline_guard.component_graph
    
    def test_monitoring_start_stop(self, pipeline_guard):
        """Test monitoring start and stop"""
        pipeline_guard.start_monitoring()
        
        assert pipeline_guard.is_running is True
        assert pipeline_guard.monitor_thread is not None
        
        time.sleep(0.1)  # Let monitoring start
        
        pipeline_guard.stop_monitoring()
        
        assert pipeline_guard.is_running is False
    
    def test_component_failure_handling(self, pipeline_guard):
        """Test component failure detection and handling"""
        # Create a component that fails health check
        failure_count = 0
        
        def failing_health_check():
            nonlocal failure_count
            failure_count += 1
            return failure_count <= 2  # Fail after 2 checks
        
        def mock_recovery():
            nonlocal failure_count
            failure_count = 0  # Reset to healthy
            return True
        
        component = PipelineComponent(
            name="failing_component",
            component_type="test",
            health_check=failing_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="reset",
                    strategy=RecoveryStrategy.RESTART,
                    action=mock_recovery
                )
            ],
            max_failures=2
        )
        
        pipeline_guard.register_component(component)
        
        # Simulate failure detection
        pipeline_guard._check_all_components()
        assert component.state == PipelineState.HEALTHY
        
        pipeline_guard._check_all_components()
        assert component.state == PipelineState.HEALTHY
        
        pipeline_guard._check_all_components()
        assert component.state == PipelineState.DEGRADED
    
    def test_system_status(self, pipeline_guard, mock_component):
        """Test system status reporting"""
        pipeline_guard.register_component(mock_component)
        status = pipeline_guard.get_system_status()
        
        assert 'system' in status
        assert 'components' in status
        assert 'statistics' in status
        assert 'monitoring' in status
        
        assert status['system']['total_components'] == 1
        assert status['system']['healthy_components'] == 1
        assert 'test_component' in status['components']
    
    def test_context_manager(self, mock_component):
        """Test context manager functionality"""
        with SelfHealingPipelineGuard(monitoring_interval=1) as guard:
            guard.register_component(mock_component)
            assert guard.is_running
        
        # Should be stopped after context exit
        assert not guard.is_running
    
    def test_force_recovery(self, pipeline_guard, mock_component):
        """Test manual recovery triggering"""
        pipeline_guard.register_component(mock_component)
        
        result = pipeline_guard.force_recovery("test_component")
        assert result is True
        
        # Test non-existent component
        result = pipeline_guard.force_recovery("non_existent")
        assert result is False


class TestPipelineMonitor:
    """Test PipelineMonitor functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        return PipelineMonitor(metric_retention_hours=1)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert len(monitor.health_checks) == 0
        assert len(monitor.health_metrics) == 0
        assert not monitor.is_collecting
    
    def test_component_addition(self, monitor):
        """Test adding components to monitor"""
        def mock_health_check():
            return True
        
        monitor.add_component("test_component", mock_health_check)
        
        assert "test_component" in monitor.health_checks
        assert "test_component" in monitor.component_performance
    
    def test_health_check_recording(self, monitor):
        """Test health check result recording"""
        def mock_health_check():
            return True
        
        monitor.add_component("test_component", mock_health_check)
        monitor.record_health_check("test_component", True, 0.1)
        
        assert len(monitor.health_metrics["test_component"]) == 1
        
        perf = monitor.component_performance["test_component"]
        assert perf['success_count'] == 1
        assert perf['failure_count'] == 0
        assert perf['avg_response_time'] == 0.1
    
    def test_health_trend_analysis(self, monitor):
        """Test health trend analysis"""
        def mock_health_check():
            return True
        
        monitor.add_component("test_component", mock_health_check)
        
        # Record multiple health checks
        for i in range(5):
            monitor.record_health_check("test_component", i < 3, 0.1)
        
        trend = monitor.get_component_health_trend("test_component", hours=1)
        
        assert trend['component'] == "test_component"
        assert trend['total_checks'] == 5
        assert trend['success_rate'] == 0.6  # 3/5 successful
    
    def test_system_performance_summary(self, monitor):
        """Test system performance summary"""
        summary = monitor.get_system_performance_summary()
        
        # Should return error when no metrics available
        assert 'error' in summary


class TestHealthChecker:
    """Test HealthChecker functionality"""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance"""
        return HealthChecker()
    
    @pytest.fixture
    def mock_component(self):
        """Create a mock component"""
        def mock_health_check():
            return True
        
        return PipelineComponent(
            name="test_component",
            component_type="test",
            health_check=mock_health_check
        )
    
    def test_basic_health_check(self, health_checker, mock_component):
        """Test basic health checking"""
        result = health_checker.check_component_health(mock_component)
        assert result is True
    
    def test_failing_health_check(self, health_checker):
        """Test health check with failure"""
        def failing_health_check():
            raise Exception("Health check failed")
        
        component = PipelineComponent(
            name="failing_component",
            component_type="test",
            health_check=failing_health_check
        )
        
        result = health_checker.check_component_health(component)
        assert result is False
    
    def test_retry_health_check(self, health_checker):
        """Test retry health check strategy"""
        call_count = 0
        
        def intermittent_health_check():
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # Succeed on second try
        
        component = PipelineComponent(
            name="intermittent_component",
            component_type="test",
            health_check=intermittent_health_check
        )
        
        result = health_checker.check_component_health(component, strategy='retry')
        assert result is True
        assert call_count >= 2
    
    def test_batch_health_check(self, health_checker):
        """Test batch health checking"""
        components = []
        for i in range(3):
            def health_check():
                return i % 2 == 0  # Alternate success/failure
            
            component = PipelineComponent(
                name=f"component_{i}",
                component_type="test",
                health_check=health_check
            )
            components.append(component)
        
        results = health_checker.batch_health_check(components)
        
        assert len(results) == 3
        assert all(name in results for name in [f"component_{i}" for i in range(3)])
    
    def test_health_summary(self, health_checker):
        """Test health summary generation"""
        components = []
        for i in range(4):
            def health_check():
                return i < 3  # 3 out of 4 healthy
            
            component = PipelineComponent(
                name=f"component_{i}",
                component_type="test",
                health_check=health_check
            )
            components.append(component)
        
        summary = health_checker.get_health_summary(components)
        
        assert summary['total_components'] == 4
        assert summary['healthy_components'] == 3
        assert summary['health_percentage'] == 75.0


class TestRecoveryEngine:
    """Test RecoveryEngine functionality"""
    
    @pytest.fixture
    def recovery_engine(self):
        """Create a recovery engine instance"""
        return RecoveryEngine()
    
    def test_recovery_action_registration(self, recovery_engine):
        """Test recovery action registration"""
        def mock_recovery():
            return True
        
        action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RESTART,
            action=mock_recovery
        )
        
        recovery_engine.register_recovery_action("test_component", action)
        
        assert "test_component" in recovery_engine.recovery_actions
        assert len(recovery_engine.recovery_actions["test_component"]) == 1
    
    def test_successful_recovery(self, recovery_engine):
        """Test successful recovery execution"""
        def mock_recovery():
            return True
        
        action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RESTART,
            action=mock_recovery
        )
        
        recovery_engine.register_recovery_action("test_component", action)
        
        result = recovery_engine.execute_recovery("test_component")
        
        assert result is True
        assert recovery_engine.recovery_stats['successful_recoveries'] == 1
    
    def test_failed_recovery(self, recovery_engine):
        """Test failed recovery execution"""
        def mock_recovery():
            return False
        
        action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RESTART,
            action=mock_recovery
        )
        
        recovery_engine.register_recovery_action("test_component", action)
        
        result = recovery_engine.execute_recovery("test_component")
        
        assert result is False
        assert recovery_engine.recovery_stats['failed_recoveries'] == 1
    
    def test_recovery_with_rollback(self, recovery_engine):
        """Test recovery with rollback action"""
        rollback_called = False
        
        def mock_recovery():
            return False
        
        def mock_rollback():
            nonlocal rollback_called
            rollback_called = True
            return True
        
        action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RESTART,
            action=mock_recovery,
            rollback_action=mock_rollback
        )
        
        recovery_engine.register_recovery_action("test_component", action)
        
        result = recovery_engine.execute_recovery("test_component")
        
        assert result is False
        assert rollback_called is True
    
    def test_recovery_statistics(self, recovery_engine):
        """Test recovery statistics tracking"""
        def mock_recovery():
            return True
        
        action = RecoveryAction(
            name="test_recovery",
            strategy=RecoveryStrategy.RESTART,
            action=mock_recovery
        )
        
        recovery_engine.register_recovery_action("test_component", action)
        recovery_engine.execute_recovery("test_component")
        
        stats = recovery_engine.get_recovery_statistics()
        
        assert stats['total_attempts'] == 1
        assert stats['successful_recoveries'] == 1
        assert stats['overall_success_rate'] == 1.0
        assert 'restart' in stats['strategy_statistics']


class TestFailureAnalyzer:
    """Test FailureAnalyzer functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a failure analyzer instance"""
        return FailureAnalyzer()
    
    def test_failure_recording(self, analyzer):
        """Test failure event recording"""
        error = Exception("Test error")
        
        failure_event = analyzer.record_failure("test_component", error)
        
        assert failure_event.component_name == "test_component"
        assert failure_event.failure_type == "Exception"
        assert failure_event.error_message == "Test error"
        assert len(analyzer.failure_history) == 1
    
    def test_failure_analysis(self, analyzer):
        """Test failure analysis"""
        from src.pipeline_guard.pipeline_guard import PipelineComponent
        
        def mock_health_check():
            return True
        
        component = PipelineComponent(
            name="test_component",
            component_type="test",
            health_check=mock_health_check
        )
        
        # Record some failures
        for i in range(3):
            error = Exception(f"Test error {i}")
            analyzer.record_failure("test_component", error)
        
        analysis = analyzer.analyze_failure(component)
        
        assert analysis['component'] == "test_component"
        assert analysis['total_failures'] == 3
        assert analysis['most_common_failure_type'] == "Exception"
        assert len(analysis['recommendations']) > 0
    
    def test_failure_trends(self, analyzer):
        """Test failure trend analysis"""
        # Record failures
        for i in range(5):
            error = Exception(f"Test error {i}")
            analyzer.record_failure(f"component_{i % 2}", error)
        
        trends = analyzer.get_failure_trends(hours=1)
        
        assert trends['total_failures'] == 5
        assert trends['unique_components'] == 2
        assert 'component_0' in trends['component_trends']
        assert 'component_1' in trends['component_trends']


class TestFailurePredictor:
    """Test FailurePredictor functionality"""
    
    @pytest.fixture
    def predictor(self):
        """Create a failure predictor instance"""
        return FailurePredictor()
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initialization"""
        assert predictor.neuromorphic_predictor is not None
        assert predictor.statistical_predictor is not None
        assert len(predictor.component_history) == 0
    
    def test_component_metrics_update(self, predictor):
        """Test updating component metrics"""
        metrics = {
            'failure_count': 1,
            'state': 'degraded',
            'critical': False
        }
        
        predictor.update_component_metrics("test_component", metrics, failed=True)
        
        assert len(predictor.component_history["test_component"]) == 1
        assert len(predictor.failure_history["test_component"]) == 1
    
    def test_single_component_prediction(self, predictor):
        """Test predicting failure for single component"""
        metrics = {
            'failure_count': 2,
            'state': 'degraded',
            'critical': True,
            'last_failure': time.time() - 3600
        }
        
        prediction = predictor.predict_single_component("test_component", metrics)
        
        assert 0.0 <= prediction <= 1.0
    
    def test_detailed_prediction(self, predictor):
        """Test detailed prediction with explanations"""
        metrics = {
            'failure_count': 3,
            'state': 'failing',
            'critical': True,
            'last_failure': time.time() - 1800
        }
        
        result = predictor.get_detailed_prediction("test_component", metrics)
        
        assert isinstance(result, PredictionResult)
        assert result.component_name == "test_component"
        assert 0.0 <= result.failure_probability <= 1.0
        assert len(result.contributing_factors) > 0
        assert len(result.recommended_actions) > 0
    
    def test_batch_prediction(self, predictor):
        """Test predicting failures for multiple components"""
        all_metrics = {
            "component_1": {'failure_count': 1, 'state': 'healthy'},
            "component_2": {'failure_count': 3, 'state': 'failing'},
        }
        
        predictions = predictor.predict_component_failures(all_metrics)
        
        assert len(predictions) == 2
        assert "component_1" in predictions
        assert "component_2" in predictions
        assert all(0.0 <= p <= 1.0 for p in predictions.values())


class TestNeuromorphicPredictor:
    """Test NeuromorphicPredictor functionality"""
    
    @pytest.fixture
    def neuromorphic_predictor(self):
        """Create a neuromorphic predictor instance"""
        return NeuromorphicPredictor(reservoir_size=50)
    
    def test_network_initialization(self, neuromorphic_predictor):
        """Test neuromorphic network initialization"""
        neuromorphic_predictor.initialize_network(input_size=8, output_size=1)
        
        assert neuromorphic_predictor.input_weights is not None
        assert neuromorphic_predictor.reservoir_weights is not None
        assert neuromorphic_predictor.output_weights is not None
        assert neuromorphic_predictor.reservoir_state is not None
    
    def test_feature_extraction(self, neuromorphic_predictor):
        """Test feature extraction from component metrics"""
        metrics = {
            'failure_count': 2,
            'state': 'degraded',
            'critical': True,
            'last_failure': time.time() - 3600
        }
        
        features = neuromorphic_predictor._extract_features(metrics)
        
        assert len(features) == 8  # Expected number of features
        assert all(isinstance(f, (int, float, np.float64)) for f in features)
    
    def test_training(self, neuromorphic_predictor):
        """Test neuromorphic predictor training"""
        # Create training data
        training_data = []
        labels = []
        
        for i in range(10):
            metrics = {
                'failure_count': i % 3,
                'state': 'healthy' if i % 2 == 0 else 'degraded',
                'critical': i % 4 == 0,
                'last_failure': time.time() - (i * 3600)
            }
            training_data.append(metrics)
            labels.append(i % 3 > 1)  # True for high failure counts
        
        neuromorphic_predictor.train(training_data, labels)
        
        assert neuromorphic_predictor.is_trained
    
    def test_prediction_after_training(self, neuromorphic_predictor):
        """Test prediction after training"""
        # First train the predictor
        training_data = []
        labels = []
        
        for i in range(20):
            metrics = {
                'failure_count': i % 3,
                'state': 'healthy' if i % 2 == 0 else 'degraded',
                'critical': i % 4 == 0,
                'last_failure': time.time() - (i * 3600)
            }
            training_data.append(metrics)
            labels.append(i % 3 > 1)
        
        neuromorphic_predictor.train(training_data, labels)
        
        # Now test prediction
        test_metrics = {
            'failure_count': 2,
            'state': 'failing',
            'critical': True,
            'last_failure': time.time() - 1800
        }
        
        prediction = neuromorphic_predictor.predict(test_metrics)
        
        assert 0.0 <= prediction <= 1.0
    
    def test_model_persistence(self, neuromorphic_predictor):
        """Test saving and loading models"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Initialize and train a simple model
            neuromorphic_predictor.initialize_network(8, 1)
            neuromorphic_predictor.is_trained = True
            
            # Save model
            neuromorphic_predictor.save_model(tmp_path)
            assert tmp_path.exists()
            
            # Create new predictor and load model
            new_predictor = NeuromorphicPredictor()
            new_predictor.load_model(tmp_path)
            
            assert new_predictor.is_trained
            assert new_predictor.reservoir_size == neuromorphic_predictor.reservoir_size
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestPipelineGuardIntegration:
    """Test PipelineGuardIntegrator functionality"""
    
    @pytest.fixture
    def integrator(self):
        """Create an integrator instance"""
        config = {
            'monitoring_interval': 5,
            'recovery_timeout': 60,
            'max_concurrent_recoveries': 2
        }
        return PipelineGuardIntegrator(config)
    
    def test_integrator_initialization(self, integrator):
        """Test integrator initialization"""
        assert integrator.pipeline_guard is not None
        assert not integrator.is_integrated
        assert len(integrator.integration_components) == 0
    
    def test_component_registration(self, integrator):
        """Test component registration through integrator"""
        # This will test the registration without full Observer Coordinator integration
        assert len(integrator.integration_components) == 0
        
        # The integration methods should handle missing dependencies gracefully
        result = integrator.integrate_with_observer_coordinator()
        
        # Should return False if integration dependencies not available
        # but not crash
        assert isinstance(result, bool)
    
    def test_configuration_export(self, integrator):
        """Test configuration export"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            integrator.export_configuration(tmp_path)
            assert tmp_path.exists()
            
            with open(tmp_path) as f:
                config = json.load(f)
            
            assert 'pipeline_guard_config' in config
            assert 'components' in config
            assert 'integration_status' in config
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_event_handler_registration(self, integrator):
        """Test event handler registration"""
        def mock_handler(component_name, details):
            pass
        
        integrator.add_event_handler('component_failure', mock_handler)
        
        assert len(integrator.event_handlers['component_failure']) == 1
    
    def test_integration_status(self, integrator):
        """Test integration status reporting"""
        status = integrator.get_integration_status()
        
        assert 'integration' in status
        assert 'pipeline_guard' in status
        assert 'timestamp' in status
        
        integration_info = status['integration']
        assert 'is_integrated' in integration_info
        assert 'integration_available' in integration_info
        assert 'registered_components' in integration_info
    
    def test_context_manager(self):
        """Test integrator context manager"""
        config = {'monitoring_interval': 1}
        
        with PipelineGuardIntegrator(config) as integrator:
            assert integrator.pipeline_guard.is_running
        
        # Should be stopped after context exit
        assert not integrator.pipeline_guard.is_running


@pytest.mark.integration
class TestPipelineGuardEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_pipeline_guard_workflow(self):
        """Test complete pipeline guard workflow"""
        # Create pipeline guard
        guard = SelfHealingPipelineGuard(monitoring_interval=1)
        
        # Track recovery attempts
        recovery_called = False
        
        def mock_health_check():
            return not recovery_called  # Fail before recovery
        
        def mock_recovery():
            nonlocal recovery_called
            recovery_called = True
            return True
        
        # Create component with recovery
        component = PipelineComponent(
            name="test_workflow_component",
            component_type="test",
            health_check=mock_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="test_recovery",
                    strategy=RecoveryStrategy.RESTART,
                    action=mock_recovery
                )
            ],
            max_failures=1
        )
        
        # Register component
        guard.register_component(component)
        
        # Start monitoring
        guard.start_monitoring()
        
        try:
            # Wait for failure detection and recovery
            time.sleep(3)
            
            # Check that recovery was triggered
            assert recovery_called
            
            # Check final component state
            assert component.state == PipelineState.HEALTHY
            
        finally:
            guard.stop_monitoring()
    
    def test_integrator_with_real_components(self):
        """Test integrator with realistic component simulation"""
        integrator = PipelineGuardIntegrator({
            'monitoring_interval': 1,
            'recovery_timeout': 30
        })
        
        # Simulate some integration (without full Observer Coordinator)
        integrator.integrate_with_observer_coordinator()
        
        # Start monitoring
        integrator.start_integrated_monitoring()
        
        try:
            # Let it run briefly
            time.sleep(2)
            
            # Check status
            status = integrator.get_integration_status()
            assert status['pipeline_guard']['system']['is_running']
            
        finally:
            integrator.stop_integrated_monitoring()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])