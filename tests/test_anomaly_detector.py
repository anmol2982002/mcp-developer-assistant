"""
Tests for Anomaly Detector
"""

from datetime import datetime, timedelta

import pytest

from proxy.anomaly_detector import AnomalyScore, BehavioralAnomalyDetector, ToolRequest


class TestAnomalyDetector:
    """Test anomaly detection functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = BehavioralAnomalyDetector(model_path=None)

    def test_extract_features_normal(self, sample_user_history):
        """Normal request should have reasonable features."""
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            ip="192.168.1.1",
        )

        features = self.detector.extract_features(request, sample_user_history)

        assert features["unusual_hour"] == 0 or features["unusual_hour"] == 1
        assert features["new_ip"] == 0  # Same IP as history
        assert features["request_rate_per_min"] >= 0

    def test_extract_features_new_ip(self, sample_user_history):
        """New IP should be flagged."""
        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=datetime.utcnow(),
            ip="10.0.0.1",  # Different IP
        )

        features = self.detector.extract_features(request, sample_user_history)

        assert features["new_ip"] == 1

    def test_rule_based_high_rate(self):
        """High request rate should trigger anomaly."""
        # Create history with many recent requests
        history = []
        now = datetime.utcnow()
        for i in range(10):
            history.append({
                "tool": "read_file",
                "timestamp": now - timedelta(seconds=i * 5),
                "ip": "192.168.1.1",
            })

        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=now,
            ip="192.168.1.1",
        )

        result = self.detector.check_anomaly(request, history)

        # Should detect high rate
        assert result.score > 0.2 or "rate" in result.reason.lower()

    def test_rule_based_night_access(self):
        """Night-time access should be flagged."""
        # Create request at 3 AM
        night_time = datetime.utcnow().replace(hour=3, minute=30)

        request = ToolRequest(
            tool_name="read_file",
            user_id="user_123",
            timestamp=night_time,
            ip="192.168.1.1",
        )

        features = self.detector.extract_features(request, [])

        assert features["unusual_hour"] == 1

    def test_explain_anomaly(self):
        """Anomaly explanations should be human-readable."""
        features = {
            "time_since_last": 0.5,  # Very rapid
            "tool_changed": 0,
            "request_rate_per_min": 9,  # High rate
            "unusual_hour": 1,  # Night
            "new_ip": 1,  # New IP
        }

        explanation = self.detector._explain_anomaly(features)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
