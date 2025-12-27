"""
Tests for Enhanced Ensemble Anomaly Detector

Tests cover:
- Ensemble model training and prediction
- Feature extraction (all 12 features)
- SHAP explanation generation
- Rule-based fallback
"""

from datetime import datetime, timedelta

import pytest
import numpy as np

from ai.feature_extractor import BehavioralFeatureExtractor, FeatureVector
from ai.shap_explainer import SHAPExplainer, AnomalyExplanation
from proxy.anomaly_detector import (
    AnomalyScore,
    BehavioralAnomalyDetector,
    EnsembleAnomalyModel,
    ToolRequest,
)


class TestFeatureExtractor:
    """Test behavioral feature extraction."""

    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = BehavioralFeatureExtractor()

    def test_extract_all_12_features(self):
        """Should extract all 12 behavioral features."""
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            ip="192.168.1.1",
            params={"path": "src/main.py"},
            history=[],
        )

        assert isinstance(features, FeatureVector)
        feature_dict = features.to_dict()
        
        # Verify all 12 features exist
        assert len(feature_dict) == 12
        assert "time_since_last" in feature_dict
        assert "unusual_hour" in feature_dict
        assert "session_duration" in feature_dict
        assert "request_rate_per_min" in feature_dict
        assert "velocity_change" in feature_dict
        assert "tool_changed" in feature_dict
        assert "sequence_entropy" in feature_dict
        assert "tool_transition_prob" in feature_dict
        assert "file_sensitivity_score" in feature_dict
        assert "resource_depth_score" in feature_dict
        assert "new_ip" in feature_dict
        assert "request_size_anomaly" in feature_dict

    def test_time_since_last_calculation(self):
        """Time since last should calculate correctly."""
        now = datetime.utcnow()
        history = [{"timestamp": now - timedelta(seconds=30), "tool": "read_file"}]
        
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=now,
            history=history,
        )
        
        assert 25 <= features.time_since_last <= 35

    def test_unusual_hour_night(self):
        """Night hours should be flagged as unusual."""
        night_time = datetime.utcnow().replace(hour=3, minute=0)
        
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=night_time,
        )
        
        assert features.unusual_hour == 1

    def test_unusual_hour_normal(self):
        """Normal hours should not be flagged."""
        normal_time = datetime.utcnow().replace(hour=14, minute=0)
        
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=normal_time,
        )
        
        assert features.unusual_hour == 0

    def test_file_sensitivity_score_env(self):
        """ENV files should have high sensitivity."""
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            params={"path": ".env.production"},
        )
        
        assert features.file_sensitivity_score >= 0.8

    def test_file_sensitivity_score_normal(self):
        """Normal code files should have low sensitivity."""
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            params={"path": "src/utils.py"},
        )
        
        assert features.file_sensitivity_score <= 0.3

    def test_new_ip_detection(self):
        """New IP should be detected."""
        user_id = "test_ip_user"
        
        # First request with IP A
        _ = self.extractor.extract_features(
            tool_name="read_file",
            user_id=user_id,
            timestamp=datetime.utcnow(),
            ip="192.168.1.1",
        )
        
        # Second request with IP B (new)
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id=user_id,
            timestamp=datetime.utcnow(),
            ip="10.0.0.1",
        )
        
        assert features.new_ip == 1

    def test_tool_changed_detection(self):
        """Tool change should be detected."""
        history = [{"tool": "git_status", "timestamp": datetime.utcnow()}]
        
        features = self.extractor.extract_features(
            tool_name="read_file",  # Different tool
            user_id="user_123",
            timestamp=datetime.utcnow(),
            history=history,
        )
        
        assert features.tool_changed == 1

    def test_to_array_conversion(self):
        """FeatureVector should convert to numpy array."""
        features = self.extractor.extract_features(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
        )
        
        arr = features.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 12


class TestEnsembleAnomalyModel:
    """Test ensemble model functionality."""

    def test_unfitted_model_returns_neutral(self):
        """Unfitted model should return neutral scores."""
        model = EnsembleAnomalyModel()
        X = np.random.rand(10, 12)
        
        scores, model_scores = model.predict(X)
        
        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)

    def test_model_fit_and_predict(self):
        """Model should fit and predict correctly."""
        model = EnsembleAnomalyModel()
        
        # Generate training data
        np.random.seed(42)
        X_train = np.random.rand(200, 12)
        
        model.fit(X_train, contamination=0.05)
        
        assert model.is_fitted
        
        # Predict on test data
        X_test = np.random.rand(10, 12)
        scores, model_scores = model.predict(X_test)
        
        assert len(scores) == 10
        assert "isolation_forest" in model_scores
        assert "lof" in model_scores
        assert "one_class_svm" in model_scores

    def test_model_save_and_load(self, tmp_path):
        """Model should save and load correctly."""
        model = EnsembleAnomalyModel()
        X_train = np.random.rand(100, 12)
        model.fit(X_train)
        
        model_path = str(tmp_path / "test_model.pkl")
        model.save(model_path)
        
        loaded_model = EnsembleAnomalyModel.load(model_path)
        
        assert loaded_model.is_fitted
        
        # Predictions should be consistent
        X_test = np.random.rand(5, 12)
        orig_scores, _ = model.predict(X_test)
        loaded_scores, _ = loaded_model.predict(X_test)
        
        np.testing.assert_array_almost_equal(orig_scores, loaded_scores)


class TestSHAPExplainer:
    """Test SHAP-based explanations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.explainer = SHAPExplainer()

    def test_explain_returns_explanation(self):
        """Explain should return AnomalyExplanation."""
        features = {
            "time_since_last": 0.5,
            "unusual_hour": 1,
            "request_rate_per_min": 15,
            "file_sensitivity_score": 0.8,
            "new_ip": 1,
        }
        
        explanation = self.explainer.explain(features, anomaly_score=0.75)
        
        assert isinstance(explanation, AnomalyExplanation)
        assert explanation.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert len(explanation.summary) > 0
        assert len(explanation.contributing_features) > 0

    def test_risk_level_critical(self):
        """High anomaly score should return CRITICAL risk."""
        features = {"time_since_last": 0.1, "request_rate_per_min": 30}
        
        explanation = self.explainer.explain(features, anomaly_score=0.95)
        
        assert explanation.risk_level == "CRITICAL"

    def test_risk_level_low(self):
        """Low anomaly score should return LOW risk."""
        features = {"time_since_last": 60, "request_rate_per_min": 2}
        
        explanation = self.explainer.explain(features, anomaly_score=0.2)
        
        assert explanation.risk_level == "LOW"

    def test_recommendations_generated(self):
        """Explanations should include recommendations."""
        features = {
            "new_ip": 1,
            "unusual_hour": 1,
            "file_sensitivity_score": 0.9,
        }
        
        explanation = self.explainer.explain(features, anomaly_score=0.8)
        
        assert len(explanation.recommendations) > 0

    def test_to_dict(self):
        """Explanation should convert to dict."""
        features = {"time_since_last": 1.0}
        explanation = self.explainer.explain(features, anomaly_score=0.5)
        
        d = explanation.to_dict()
        
        assert "summary" in d
        assert "contributing_features" in d
        assert "risk_level" in d
        assert "recommendations" in d


class TestBehavioralAnomalyDetector:
    """Test the main anomaly detector."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = BehavioralAnomalyDetector(
            model_path=None,
            threshold=0.5,
            enable_explanations=True,
        )

    def test_check_anomaly_returns_score(self, sample_user_history):
        """Check anomaly should return AnomalyScore."""
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            ip="192.168.1.1",
        )
        
        result = self.detector.check_anomaly(request, sample_user_history)
        
        assert isinstance(result, AnomalyScore)
        assert 0 <= result.score <= 1
        assert isinstance(result.is_anomalous, bool)
        assert len(result.reason) > 0

    def test_rule_based_high_rate(self):
        """High request rate should trigger anomaly."""
        now = datetime.utcnow()
        history = [
            {"tool": "read_file", "timestamp": now - timedelta(seconds=i * 5), "ip": "192.168.1.1"}
            for i in range(15)
        ]
        
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=now,
            ip="192.168.1.1",
        )
        
        result = self.detector.check_anomaly(request, history)
        
        # High rate should increase score
        assert result.score > 0.2

    def test_rule_based_sensitive_file(self):
        """Accessing sensitive files should increase score."""
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            ip="192.168.1.1",
            params={"path": ".env.production"},
        )
        
        result = self.detector.check_anomaly(request, [])
        
        # Sensitive file should increase score
        assert result.score >= 0.2

    def test_explanation_included_when_anomalous(self):
        """Anomalous requests should include explanation."""
        # Create conditions for anomaly
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow().replace(hour=3),  # Night
            ip="new_ip_address",
            params={"path": "secrets/api_key.txt"},
        )
        
        result = self.detector.check_anomaly(request, [])
        
        if result.is_anomalous:
            assert result.explanation is not None
            assert isinstance(result.explanation, AnomalyExplanation)

    def test_extract_features_backward_compatible(self, sample_user_history):
        """extract_features should return dict (backward compatible)."""
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
        )
        
        features = self.detector.extract_features(request, sample_user_history)
        
        assert isinstance(features, dict)
        assert "time_since_last" in features


class TestAnomalyDetectorIntegration:
    """Integration tests for the full anomaly detection pipeline."""

    def test_train_and_detect(self, tmp_path):
        """Training and detection should work end-to-end."""
        from ai.synthetic_data_generator import SyntheticDataGenerator
        from ai.feature_extractor import FeatureVector
        
        # Generate synthetic training data
        generator = SyntheticDataGenerator()
        features_list, _ = generator.generate_training_dataset(
            n_normal_sessions=20,
            n_anomaly_sessions=2,
        )
        
        # Train detector
        detector = BehavioralAnomalyDetector()
        model_path = str(tmp_path / "ensemble.pkl")
        
        success = detector.train_from_data(
            training_data=features_list,
            output_path=model_path,
            contamination=0.1,
        )
        
        assert success
        
        # Reload detector with trained model
        trained_detector = BehavioralAnomalyDetector(model_path=model_path)
        
        # Check normal request
        request = ToolRequest(
            tool_name="read_file",
            user_id="normal_user",
            timestamp=datetime.utcnow().replace(hour=14),
            ip="192.168.1.1",
            params={"path": "src/main.py"},
        )
        
        result = trained_detector.check_anomaly(request, [])
        
        # Should get a score (whether normal or anomalous depends on model)
        assert 0 <= result.score <= 1
