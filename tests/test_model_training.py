"""
Tests for Model Training Pipeline

Tests cover:
- Synthetic data generation
- Ensemble model training
- Model registry operations
- A/B testing
"""

from datetime import datetime
import json

import pytest
import numpy as np

from ai.feature_extractor import FeatureVector
from ai.synthetic_data_generator import SyntheticDataGenerator, USER_PROFILES
from ai.model_registry import ModelRegistry, ModelMetadata
from ai.model_trainer import EnsembleModelTrainer, AnomalyModelTrainer


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = SyntheticDataGenerator(random_seed=42)

    def test_generate_normal_session(self):
        """Should generate normal behavior session."""
        features_list = self.generator.generate_normal_session(
            user_id="test_user",
            profile_name="regular_developer",
            session_duration_hours=1.0,
        )
        
        assert len(features_list) > 0
        
        # Check feature structure
        for features in features_list:
            assert "time_since_last" in features
            assert "unusual_hour" in features
            assert "file_sensitivity_score" in features
            
            # Normal session should have reasonable values
            assert features["file_sensitivity_score"] <= 1.0
            assert features["request_rate_per_min"] <= 60

    def test_generate_anomalous_session_rapid_fire(self):
        """Should generate rapid fire anomaly."""
        features_list = self.generator.generate_anomalous_session(
            user_id="test_user",
            anomaly_type="rapid_fire",
        )
        
        assert len(features_list) > 0
        
        # Rapid fire should have low time_since_last and high rate
        avg_time = np.mean([f["time_since_last"] for f in features_list])
        avg_rate = np.mean([f["request_rate_per_min"] for f in features_list])
        
        assert avg_time < 5  # Very rapid
        assert avg_rate > 10  # High rate

    def test_generate_anomalous_session_sensitive_access(self):
        """Should generate sensitive file access anomaly."""
        features_list = self.generator.generate_anomalous_session(
            user_id="test_user",
            anomaly_type="sensitive_access",
        )
        
        # Should have high sensitivity scores
        avg_sensitivity = np.mean([f["file_sensitivity_score"] for f in features_list])
        assert avg_sensitivity > 0.5

    def test_generate_anomalous_session_night_access(self):
        """Should generate night access anomaly."""
        features_list = self.generator.generate_anomalous_session(
            user_id="test_user",
            anomaly_type="night_access",
        )
        
        # Should have unusual_hour flag
        assert all(f["unusual_hour"] == 1.0 for f in features_list)

    def test_generate_training_dataset(self):
        """Should generate complete training dataset with labels."""
        features_list, labels = self.generator.generate_training_dataset(
            n_normal_sessions=10,
            n_anomaly_sessions=2,
        )
        
        assert len(features_list) == len(labels)
        assert sum(l == 0 for l in labels) > sum(l == 1 for l in labels)  # More normal
        assert sum(l == 1 for l in labels) > 0  # Some anomalies

    def test_to_numpy(self):
        """Should convert features to numpy array."""
        features_list, _ = self.generator.generate_training_dataset(
            n_normal_sessions=5,
            n_anomaly_sessions=1,
        )
        
        X = self.generator.to_numpy(features_list)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(features_list)
        assert X.shape[1] == 12  # 12 features

    def test_all_user_profiles(self):
        """Should work with all user profiles."""
        for profile_name in USER_PROFILES:
            features_list = self.generator.generate_normal_session(
                user_id=f"test_{profile_name}",
                profile_name=profile_name,
                session_duration_hours=0.5,
            )
            
            assert len(features_list) > 0


class TestModelRegistry:
    """Test model registry operations."""

    def test_register_and_get_model(self, tmp_path):
        """Should register and retrieve model."""
        registry = ModelRegistry(registry_path=str(tmp_path / "models"))
        
        # Create dummy model
        model = {"test": "model"}
        
        model_id = registry.register(
            model=model,
            model_type="test",
            version="1.0.0",
            features=["f1", "f2"],
            training_samples=100,
            activate=True,
        )
        
        assert model_id is not None
        
        # Get active model
        result = registry.get_active_model()
        assert result is not None
        
        loaded_model, metadata = result
        assert loaded_model == model
        assert metadata.version == "1.0.0"
        assert metadata.is_active

    def test_model_versioning(self, tmp_path):
        """Should track multiple versions."""
        registry = ModelRegistry(registry_path=str(tmp_path / "models"))
        
        # Register v1
        registry.register(
            model={"v": 1},
            model_type="test",
            version="1.0.0",
            features=[],
            training_samples=100,
            activate=True,
        )
        
        # Register v2
        id_v2 = registry.register(
            model={"v": 2},
            model_type="test",
            version="2.0.0",
            features=[],
            training_samples=200,
            activate=True,
        )
        
        # V2 should be active
        result = registry.get_active_model()
        loaded, metadata = result
        assert loaded["v"] == 2
        assert metadata.version == "2.0.0"
        
        # List should show both
        models = registry.list_models()
        assert len(models) == 2

    def test_rollback(self, tmp_path):
        """Should rollback to previous version."""
        registry = ModelRegistry(registry_path=str(tmp_path / "models"))
        
        # Register v1
        id_v1 = registry.register(
            model={"v": 1},
            model_type="test",
            version="1.0.0",
            features=[],
            training_samples=100,
            activate=True,
        )
        
        # Register v2
        registry.register(
            model={"v": 2},
            model_type="test",
            version="2.0.0",
            features=[],
            training_samples=200,
            activate=True,
        )
        
        # Rollback
        success = registry.rollback()
        assert success
        
        # V1 should be active again
        result = registry.get_active_model()
        loaded, metadata = result
        assert loaded["v"] == 1

    def test_ab_test_setup(self, tmp_path):
        """Should setup A/B testing."""
        registry = ModelRegistry(registry_path=str(tmp_path / "models"))
        
        id_a = registry.register(
            model={"variant": "A"},
            model_type="test",
            version="1.0.0",
            features=[],
            training_samples=100,
        )
        
        id_b = registry.register(
            model={"variant": "B"},
            model_type="test",
            version="2.0.0",
            features=[],
            training_samples=100,
        )
        
        # Setup A/B test
        success = registry.setup_ab_test(id_a, id_b, weight_a=0.7)
        assert success
        
        # Both should be active
        active_models = registry.list_models(active_only=True)
        assert len(active_models) == 2
        
        # Check weights
        meta_a = [m for m in active_models if m.model_id == id_a][0]
        meta_b = [m for m in active_models if m.model_id == id_b][0]
        
        assert meta_a.traffic_weight == 0.7
        assert meta_b.traffic_weight == 0.3

    def test_cleanup_old_models(self, tmp_path):
        """Should cleanup old inactive models."""
        import time
        
        registry = ModelRegistry(registry_path=str(tmp_path / "models"))
        
        # Register multiple models
        for i in range(5):
            registry.register(
                model={"v": i},
                model_type="test",
                version=f"{i}.0.0",
                features=[],
                training_samples=100,
                activate=(i == 4),  # Only last one active
            )
            time.sleep(0.1)  # Ensure different timestamps
        
        # Cleanup, keeping 2
        deleted = registry.cleanup_old_models(keep_count=2)
        
        assert deleted == 2  # Should delete 2
        
        models = registry.list_models()
        assert len(models) == 3  # 1 active + 2 kept


class TestEnsembleModelTrainer:
    """Test ensemble model training."""

    def test_train_from_synthetic(self, tmp_path):
        """Should train from synthetic data."""
        trainer = EnsembleModelTrainer(contamination=0.1)
        
        registry_path = str(tmp_path / "models")
        model_path = str(tmp_path / "ensemble.pkl")
        
        result = trainer.train_from_synthetic(
            output_path=model_path,
            n_normal_sessions=20,
            n_anomaly_sessions=2,
            register=False,
        )
        
        assert result is not None
        assert result["success"]
        assert result["samples"] > 0

    def test_train_from_logs(self, tmp_path):
        """Should train from audit logs."""
        trainer = EnsembleModelTrainer()
        
        # Generate fake audit logs
        logs = []
        for i in range(150):
            logs.append({
                "user_id": f"user_{i % 5}",
                "tool_name": "read_file",
                "timestamp": (datetime.utcnow()).isoformat(),
                "ip_address": "192.168.1.1",
                "tool_input": json.dumps({"path": f"src/file_{i}.py"}),
            })
        
        model_path = str(tmp_path / "ensemble.pkl")
        
        result = trainer.train_from_logs(
            logs=logs,
            output_path=model_path,
            register=False,
        )
        
        assert result is not None
        assert result["success"]

    def test_train_registers_model(self, tmp_path):
        """Training should register model when requested."""
        registry = ModelRegistry(registry_path=str(tmp_path / "models"))
        trainer = EnsembleModelTrainer(registry=registry)
        
        model_path = str(tmp_path / "ensemble.pkl")
        
        result = trainer.train_from_synthetic(
            output_path=model_path,
            n_normal_sessions=20,
            n_anomaly_sessions=2,
            register=True,
            version="1.0.0-test",
        )
        
        assert result["model_id"] is not None
        
        # Should be in registry
        active = registry.get_active_model()
        assert active is not None


class TestLegacyAnomalyModelTrainer:
    """Test backward compatibility with legacy API."""

    def test_train_from_synthetic_legacy(self, tmp_path):
        """Legacy API should still work."""
        trainer = AnomalyModelTrainer(contamination=0.1)
        
        model_path = str(tmp_path / "legacy_model.pkl")
        
        success = trainer.train_from_synthetic(
            output_path=model_path,
            n_samples=500,
        )
        
        assert success

    def test_insufficient_data_handling(self, tmp_path):
        """Should handle insufficient data gracefully."""
        trainer = AnomalyModelTrainer()
        
        logs = [{"user_id": "u1", "tool_name": "read_file"}]  # Too few
        model_path = str(tmp_path / "model.pkl")
        
        success = trainer.train_from_logs(logs, model_path)
        
        assert not success  # Should fail gracefully
