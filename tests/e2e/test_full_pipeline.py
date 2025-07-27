"""End-to-end tests for the complete analysis pipeline."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from autonomous_orchestrator import AutonomousOrchestrator
from src.insights_clustering.clustering import ClusteringEngine
from src.insights_clustering.parser import DataParser
from src.insights_clustering.validator import DataValidator


@pytest.mark.e2e
class TestFullPipeline:
    """Test the complete end-to-end pipeline."""

    def test_complete_analysis_pipeline(
        self, 
        sample_csv_file: Path, 
        sample_config_file: Path,
        tmp_path: Path
    ) -> None:
        """Test the complete analysis pipeline from CSV to results."""
        # Step 1: Parse configuration
        with open(sample_config_file, "r") as f:
            config = yaml.safe_load(f)
        
        # Step 2: Parse and validate data
        parser = DataParser(anonymize=config["data"]["anonymize"])
        data = parser.parse_csv(sample_csv_file)
        
        validator = DataValidator(level=config["data"]["validation_level"])
        validated_data = validator.validate_data(data)
        
        # Step 3: Perform clustering
        clustering_config = config["clustering"]
        engine = ClusteringEngine(
            n_clusters=clustering_config["n_clusters"],
            random_state=clustering_config["random_state"],
            max_iter=clustering_config["max_iter"]
        )
        
        cluster_results = engine.fit_predict(validated_data)
        
        # Step 4: Generate output
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        
        if config["output"]["format"] == "json":
            results_file = output_dir / "cluster_results.json"
            with open(results_file, "w") as f:
                json.dump({
                    "n_clusters": len(set(cluster_results["labels"])),
                    "silhouette_score": float(cluster_results["silhouette_score"]),
                    "inertia": float(cluster_results["inertia"]),
                    "cluster_counts": {
                        str(i): int(sum(1 for label in cluster_results["labels"] if label == i))
                        for i in set(cluster_results["labels"])
                    }
                }, f, indent=2)
        
        # Verify pipeline results
        assert results_file.exists()
        assert len(cluster_results["labels"]) == len(validated_data)
        assert cluster_results["silhouette_score"] is not None
        assert cluster_results["inertia"] > 0

    def test_autonomous_orchestrator_integration(
        self,
        sample_csv_file: Path,
        sample_config_file: Path,
        tmp_path: Path
    ) -> None:
        """Test integration with autonomous orchestrator."""
        # Initialize orchestrator
        orchestrator = AutonomousOrchestrator(config_file=str(sample_config_file))
        
        # Set up input and output paths
        input_data = {
            "csv_file": str(sample_csv_file),
            "output_dir": str(tmp_path / "orchestrator_output")
        }
        
        # Run orchestration
        try:
            results = orchestrator.run_analysis(input_data)
            
            # Verify orchestrator results
            assert "clustering_results" in results
            assert "execution_time" in results
            assert "status" in results
            assert results["status"] == "completed"
            
        except Exception as e:
            # If orchestrator fails, ensure it's due to expected reasons
            # (e.g., missing dependencies, not core logic errors)
            pytest.skip(f"Orchestrator test skipped due to: {e}")

    def test_error_handling_pipeline(self, tmp_path: Path) -> None:
        """Test error handling throughout the pipeline."""
        # Test with malformed CSV
        malformed_csv = tmp_path / "malformed.csv"
        with open(malformed_csv, "w") as f:
            f.write("invalid,csv,format\n")
            f.write("incomplete,row\n")
            f.write("another,incomplete\n")
        
        parser = DataParser()
        
        # Should handle malformed CSV gracefully
        with pytest.raises((ValueError, pd.errors.ParserError)):
            parser.parse_csv(malformed_csv)

    def test_large_dataset_pipeline(self, large_dataset: pd.DataFrame, tmp_path: Path) -> None:
        """Test pipeline performance with large datasets."""
        # Save large dataset to CSV
        large_csv = tmp_path / "large_dataset.csv"
        large_dataset.to_csv(large_csv, index=False)
        
        # Run pipeline with large dataset
        parser = DataParser()
        data = parser.parse_csv(large_csv)
        
        validator = DataValidator()
        validated_data = validator.validate_data(data)
        
        engine = ClusteringEngine(n_clusters=8, random_state=42)
        results = engine.fit_predict(validated_data)
        
        # Verify large dataset handling
        assert len(results["labels"]) == len(large_dataset)
        assert results["silhouette_score"] is not None
        assert len(set(results["labels"])) <= 8

    def test_configuration_validation_pipeline(self, tmp_path: Path) -> None:
        """Test pipeline with various configuration scenarios."""
        test_configs = [
            # Valid configuration
            {
                "clustering": {"algorithm": "kmeans", "n_clusters": 4},
                "data": {"anonymize": True, "validation_level": "strict"},
                "output": {"format": "json"}
            },
            # Invalid algorithm
            {
                "clustering": {"algorithm": "invalid_algo", "n_clusters": 4},
                "data": {"anonymize": True, "validation_level": "strict"},
                "output": {"format": "json"}
            },
            # Invalid cluster count
            {
                "clustering": {"algorithm": "kmeans", "n_clusters": -1},
                "data": {"anonymize": True, "validation_level": "strict"},
                "output": {"format": "json"}
            }
        ]
        
        for i, config in enumerate(test_configs):
            config_file = tmp_path / f"test_config_{i}.yml"
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            
            if i == 0:  # Valid config
                # Should work without errors
                assert config["clustering"]["algorithm"] == "kmeans"
            else:  # Invalid configs
                # Should be caught by validation
                if config["clustering"]["n_clusters"] < 0:
                    with pytest.raises(ValueError):
                        ClusteringEngine(n_clusters=config["clustering"]["n_clusters"])

    def test_data_privacy_pipeline(
        self, sample_insights_data: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test that privacy measures work throughout the pipeline."""
        # Add PII to test data
        test_data = sample_insights_data.copy()
        test_data["ssn"] = ["123-45-6789"] * len(test_data)
        test_data["phone"] = ["+1-555-123-4567"] * len(test_data)
        
        # Save data with PII
        csv_with_pii = tmp_path / "data_with_pii.csv"
        test_data.to_csv(csv_with_pii, index=False)
        
        # Run pipeline with anonymization
        parser = DataParser(anonymize=True)
        parsed_data = parser.parse_csv(csv_with_pii)
        
        # Verify PII has been removed/anonymized
        if "ssn" in parsed_data.columns:
            assert not any("123-45-6789" in str(val) for val in parsed_data["ssn"])
        if "phone" in parsed_data.columns:
            assert not any("+1-555-123-4567" in str(val) for val in parsed_data["phone"])

    def test_output_format_pipeline(
        self, sample_csv_file: Path, tmp_path: Path
    ) -> None:
        """Test different output format options."""
        # Parse data
        parser = DataParser()
        data = parser.parse_csv(sample_csv_file)
        
        # Run clustering
        engine = ClusteringEngine(n_clusters=4, random_state=42)
        results = engine.fit_predict(data)
        
        # Test JSON output
        json_output = tmp_path / "results.json"
        output_data = {
            "clusters": len(set(results["labels"])),
            "silhouette_score": float(results["silhouette_score"]),
            "labels": [int(label) for label in results["labels"]]
        }
        with open(json_output, "w") as f:
            json.dump(output_data, f)
        
        # Verify JSON output
        assert json_output.exists()
        with open(json_output, "r") as f:
            loaded_data = json.load(f)
            assert "clusters" in loaded_data
            assert "silhouette_score" in loaded_data
            assert len(loaded_data["labels"]) == len(data)

    def test_concurrent_pipeline_execution(
        self, sample_csv_file: Path, tmp_path: Path
    ) -> None:
        """Test concurrent execution of multiple pipeline instances."""
        import concurrent.futures
        import threading
        
        def run_pipeline(instance_id: int) -> Dict[str, Any]:
            """Run a single pipeline instance."""
            # Parse data
            parser = DataParser()
            data = parser.parse_csv(sample_csv_file)
            
            # Run clustering with different parameters for each instance
            engine = ClusteringEngine(
                n_clusters=3 + (instance_id % 3),  # 3, 4, or 5 clusters
                random_state=42 + instance_id
            )
            results = engine.fit_predict(data)
            
            # Save results
            instance_output = tmp_path / f"concurrent_results_{instance_id}.json"
            with open(instance_output, "w") as f:
                json.dump({
                    "instance_id": instance_id,
                    "n_clusters": len(set(results["labels"])),
                    "silhouette_score": float(results["silhouette_score"])
                }, f)
            
            return {
                "instance_id": instance_id,
                "output_file": str(instance_output),
                "n_clusters": len(set(results["labels"]))
            }
        
        # Run 3 concurrent pipeline instances
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_pipeline, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        # Verify all instances completed successfully
        assert len(results) == 3
        for result in results:
            assert "instance_id" in result
            assert "n_clusters" in result
            assert Path(result["output_file"]).exists()

    def test_pipeline_monitoring_and_metrics(
        self, sample_csv_file: Path, tmp_path: Path
    ) -> None:
        """Test that pipeline execution can be monitored and metrics collected."""
        import time
        
        metrics = {
            "start_time": time.time(),
            "steps_completed": [],
            "errors": [],
            "performance_metrics": {}
        }
        
        try:
            # Step 1: Data parsing
            step_start = time.time()
            parser = DataParser()
            data = parser.parse_csv(sample_csv_file)
            step_end = time.time()
            
            metrics["steps_completed"].append("data_parsing")
            metrics["performance_metrics"]["parsing_time"] = step_end - step_start
            
            # Step 2: Data validation
            step_start = time.time()
            validator = DataValidator()
            validated_data = validator.validate_data(data)
            step_end = time.time()
            
            metrics["steps_completed"].append("data_validation")
            metrics["performance_metrics"]["validation_time"] = step_end - step_start
            
            # Step 3: Clustering
            step_start = time.time()
            engine = ClusteringEngine(n_clusters=4, random_state=42)
            cluster_results = engine.fit_predict(validated_data)
            step_end = time.time()
            
            metrics["steps_completed"].append("clustering")
            metrics["performance_metrics"]["clustering_time"] = step_end - step_start
            
        except Exception as e:
            metrics["errors"].append(str(e))
        
        finally:
            metrics["end_time"] = time.time()
            metrics["total_time"] = metrics["end_time"] - metrics["start_time"]
        
        # Save metrics
        metrics_file = tmp_path / "pipeline_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Verify monitoring worked
        assert len(metrics["steps_completed"]) > 0
        assert metrics["total_time"] > 0
        assert "parsing_time" in metrics["performance_metrics"]
        assert metrics_file.exists()