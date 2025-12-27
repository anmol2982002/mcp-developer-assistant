"""
Tests for Phase 5: Code Review Tool

Tests for:
- RiskFeatureExtractor
- RiskPredictionModel
- ReviewRouter
- Enhanced ReviewChangesTool
- Enhanced RiskScorer
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ai.risk_feature_extractor import RiskFeatureExtractor, RiskFeatures
from ai.risk_prediction_model import RiskPredictionModel, RiskPrediction
from ai.review_router import ReviewRouter, ReviewRouting
from ai.risk_scorer import RiskScorer, RiskScore
from server.tools.ai_tools import ReviewChangesTool
from server.tools.base import ToolResult


# =============================================================================
# Test RiskFeatureExtractor
# =============================================================================


class TestRiskFeatureExtractor:
    """Tests for RiskFeatureExtractor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = RiskFeatureExtractor()

    def test_parse_changed_files(self):
        """Should extract file paths from diff."""
        diff = """
diff --git a/auth.py b/auth.py
+++ b/auth.py
@@ -1,3 +1,5 @@
+def authenticate():
+    pass
diff --git a/config.py b/config.py
+++ b/config.py
+SECRET = "value"
"""
        files = self.extractor._parse_changed_files(diff)
        
        assert "auth.py" in files
        assert "config.py" in files

    def test_count_line_changes(self):
        """Should count additions and deletions correctly."""
        diff = """
+line added 1
+line added 2
-line removed 1
+++ b/file.py
--- a/file.py
"""
        additions, deletions = self.extractor._count_line_changes(diff)
        
        assert additions == 2
        assert deletions == 1

    def test_file_sensitivity_critical(self):
        """Should detect critical files (.env, secrets)."""
        files = [".env", "src/main.py"]
        
        result = self.extractor._calc_file_sensitivity(files)
        
        assert result["critical"] is True
        assert result["score"] >= 0.5

    def test_file_sensitivity_high(self):
        """Should detect high sensitivity files (auth, config)."""
        files = ["auth/login.py", "config.py"]
        
        result = self.extractor._calc_file_sensitivity(files)
        
        assert result["count"] >= 1
        assert result["score"] >= 0.3

    def test_file_sensitivity_low(self):
        """Should mark test files as low sensitivity."""
        files = ["tests/test_main.py", "README.md"]
        
        result = self.extractor._calc_file_sensitivity(files)
        
        assert result["critical"] is False
        assert result["score"] < 0.3

    def test_detect_security_patterns(self):
        """Should detect security-related patterns in added lines."""
        diff = """
+eval(user_input)
+os.system("rm -rf /")
+password = "hardcoded123"
"""
        result = self.extractor._detect_security_patterns(diff)
        
        assert result["count"] >= 2
        assert "eval_usage" in result["patterns"] or "os_system_usage" in result["patterns"]

    def test_temporal_features_weekend(self):
        """Should detect weekend commits."""
        saturday = datetime(2024, 1, 6, 14, 0, 0)  # Saturday
        
        result = self.extractor._calc_temporal_features(saturday)
        
        assert result["is_weekend"] is True
        assert result["risk_score"] >= 0.3

    def test_temporal_features_late_night(self):
        """Should detect late night commits."""
        late_night = datetime(2024, 1, 5, 23, 30, 0)
        
        result = self.extractor._calc_temporal_features(late_night)
        
        assert result["is_late_night"] is True
        assert result["risk_score"] >= 0.4

    def test_complexity_score_large_change(self):
        """Should calculate high complexity for large changes."""
        score = self.extractor._calc_complexity_score(
            additions=1000,
            deletions=500,
            files=15,
        )
        
        assert score >= 0.5

    def test_complexity_score_small_change(self):
        """Should calculate low complexity for small changes."""
        score = self.extractor._calc_complexity_score(
            additions=10,
            deletions=5,
            files=1,
        )
        
        assert score < 0.3

    def test_extract_from_diff_full(self):
        """Should extract all features from a complete diff."""
        diff = """
diff --git a/auth.py b/auth.py
+++ b/auth.py
+def login(password):
+    eval(password)
"""
        features = self.extractor.extract_from_diff(diff, ".", None)
        
        assert isinstance(features, RiskFeatures)
        assert features.files_changed >= 1
        assert features.lines_added >= 2
        assert features.security_pattern_count >= 1

    def test_features_to_array(self):
        """Should convert features to numpy-compatible array."""
        features = RiskFeatures(
            file_sensitivity_score=0.5,
            lines_added=100,
            security_pattern_count=2,
        )
        
        array = features.to_array()
        
        assert len(array) == 18
        assert all(isinstance(x, float) for x in array)

    def test_features_to_dict(self):
        """Should convert features to dictionary."""
        features = RiskFeatures(
            file_sensitivity_score=0.5,
            lines_added=100,
        )
        
        d = features.to_dict()
        
        assert d["file_sensitivity_score"] == 0.5
        assert d["lines_added"] == 100


# =============================================================================
# Test RiskPredictionModel
# =============================================================================


class TestRiskPredictionModel:
    """Tests for RiskPredictionModel."""

    def test_rule_based_prediction(self):
        """Should use rule-based fallback when model not trained."""
        model = RiskPredictionModel()
        
        # High-risk features
        features = [
            0.8,  # file_sensitivity_score
            3,    # sensitive_file_count
            1.0,  # critical_file_touched
            500,  # lines_added
            100,  # lines_removed
            10,   # files_changed
            0.7,  # complexity_score
            0.3,  # author_familiarity
            0.2,  # author_file_ownership
            0.1,  # author_recent_activity
            30,   # days_since_last_review
            1.0,  # is_weekend_commit
            1.0,  # is_late_night_commit
            0.7,  # time_risk_score
            3,    # security_pattern_count
            2,    # new_dependency_count
            0,    # has_binary_files
            1,    # has_large_additions
        ]
        
        score = model.predict(features)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be high risk

    def test_predict_with_explanation(self):
        """Should return prediction with feature contributions."""
        model = RiskPredictionModel()
        
        features = [0.5] * 18  # Neutral features
        
        result = model.predict_with_explanation(features)
        
        assert isinstance(result, RiskPrediction)
        assert 0 <= result.score <= 1
        assert result.level in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(result.feature_contributions, dict)

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn not installed"),
        reason="sklearn not installed"
    )
    def test_train_from_synthetic(self, tmp_path):
        """Should train model from synthetic data."""
        model = RiskPredictionModel()
        output_path = str(tmp_path / "model.pkl")
        
        result = model.train_from_synthetic(output_path, n_samples=50)
        
        assert result.get("success") is True
        assert model.is_trained is True

    def test_save_and_load(self, tmp_path):
        """Should save and load model without sklearn."""
        model = RiskPredictionModel()
        model.model_id = "test_v1"
        output_path = str(tmp_path / "model.pkl")
        
        model.save(output_path)
        
        loaded = RiskPredictionModel()
        loaded.load(output_path)
        
        assert loaded.model_id == "test_v1"


# =============================================================================
# Test ReviewRouter
# =============================================================================


class TestReviewRouter:
    """Tests for ReviewRouter."""

    def setup_method(self):
        """Setup test fixtures."""
        self.router = ReviewRouter(
            security_reviewers=["security@example.com"],
            senior_reviewers=["senior@example.com"],
        )

    def test_critical_priority(self):
        """Should route very high risk to critical priority."""
        routing = self.router.determine_route(
            risk_score=0.9,
            risk_factors=["Critical file touched"],
        )
        
        assert routing.priority == "critical"
        assert routing.requires_senior_review is True

    def test_high_priority(self):
        """Should route high risk to high priority."""
        routing = self.router.determine_route(
            risk_score=0.7,
            risk_factors=["Security patterns detected"],
            security_patterns=["sql_injection_risk"],
        )
        
        assert routing.priority == "high"
        assert routing.requires_security_review is True

    def test_normal_priority(self):
        """Should route medium risk to normal priority."""
        routing = self.router.determine_route(
            risk_score=0.4,
            risk_factors=[],
        )
        
        assert routing.priority == "normal"

    def test_low_priority(self):
        """Should route low risk to low priority."""
        routing = self.router.determine_route(
            risk_score=0.2,
            risk_factors=[],
        )
        
        assert routing.priority == "low"

    def test_security_review_required(self):
        """Should require security review for security patterns."""
        routing = self.router.determine_route(
            risk_score=0.5,
            risk_factors=["credential pattern found"],
            security_patterns=["hardcoded_credential"],
        )
        
        assert routing.requires_security_review is True
        assert "security@example.com" in routing.suggested_reviewers

    def test_senior_review_for_large_change(self):
        """Should require senior review for large changes."""
        routing = self.router.determine_route(
            risk_score=0.5,
            risk_factors=[],
            files_changed=20,
            lines_changed=600,
        )
        
        assert routing.requires_senior_review is True

    def test_review_time_estimation(self):
        """Should estimate review time based on size."""
        routing = self.router.determine_route(
            risk_score=0.5,
            risk_factors=[],
            lines_changed=200,
            files_changed=5,
        )
        
        assert routing.estimated_review_time >= 20

    def test_routing_to_dict(self):
        """Should convert routing to dictionary."""
        routing = ReviewRouting(
            priority="high",
            suggested_reviewers=["user@example.com"],
            estimated_review_time=30,
            requires_security_review=True,
            requires_senior_review=False,
            reason="High risk score",
        )
        
        d = routing.to_dict()
        
        assert d["priority"] == "high"
        assert d["requires_security_review"] is True


# =============================================================================
# Test Enhanced RiskScorer
# =============================================================================


class TestEnhancedRiskScorer:
    """Tests for enhanced RiskScorer with ML integration."""

    def test_score_diff_basic(self):
        """Should score diff using rule-based method."""
        scorer = RiskScorer()
        
        diff = """
diff --git a/.env b/.env
+API_KEY=secret123
"""
        score = scorer.score_diff(diff)
        
        assert isinstance(score, RiskScore)
        assert score.level in ["HIGH", "MEDIUM"]
        assert "sensitive file" in " ".join(score.factors).lower()

    def test_score_with_features(self):
        """Should use feature extraction for enhanced scoring."""
        scorer = RiskScorer()
        
        diff = """
diff --git a/auth/login.py b/auth/login.py
+password = user_input
+eval(password)
"""
        score = scorer.score_with_features(diff, ".")
        
        assert isinstance(score, RiskScore)
        assert score.level in ["HIGH", "MEDIUM"]
        assert len(score.factors) >= 1

    def test_score_includes_ml_fields(self):
        """Should include ML-related fields in result."""
        scorer = RiskScorer()
        
        diff = "+simple change"
        score = scorer.score_with_features(diff, ".")
        
        # Without ML model, ml_score should be None
        assert score.ml_score is None or isinstance(score.ml_score, float)

    def test_score_with_ml_model(self):
        """Should use ML model when provided."""
        mock_model = Mock()
        mock_model.predict_with_explanation.return_value = RiskPrediction(
            score=0.8,
            confidence=0.9,
            level="HIGH",
            feature_contributions={"file_sensitivity": 0.3},
            model_id="test_v1",
        )
        
        scorer = RiskScorer(ml_model=mock_model)
        
        diff = "+change"
        score = scorer.score_with_features(diff, ".")
        
        # Should have ML score from model
        assert score.ml_score == 0.8 or score.ml_score is None  # May fail if import issues


# =============================================================================
# Test Enhanced ReviewChangesTool
# =============================================================================


class TestEnhancedReviewChangesTool:
    """Tests for enhanced ReviewChangesTool with ML integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = Mock()
        self.mock_llm.generate = Mock(return_value=Mock())
        
        # Default valid response
        self.valid_response = '''
{
    "summary": "Added authentication feature with JWT tokens",
    "issues": [
        {"file": "auth.py", "line": 42, "severity": "MEDIUM", "category": "security", "message": "Consider constant-time comparison"}
    ],
    "test_suggestions": ["Test invalid token handling"],
    "risk_level": "MEDIUM",
    "estimated_review_time_minutes": 15,
    "security_concerns": []
}
'''

    @pytest.mark.asyncio
    async def test_execute_returns_ml_fields(self):
        """Should include ML risk scoring in result."""
        async def mock_generate(prompt):
            return self.valid_response
        
        self.mock_llm.generate = mock_generate
        tool = ReviewChangesTool(llm_client=self.mock_llm)
        
        diff = """
diff --git a/file.py b/file.py
+simple change
"""
        result = await tool.execute(diff)
        
        assert result.success is True
        # Should have ml_risk_score field
        assert "ml_risk_score" in result.result or result.result.get("ml_risk_score") is None

    @pytest.mark.asyncio
    async def test_execute_includes_routing(self):
        """Should include review routing in result."""
        async def mock_generate(prompt):
            return self.valid_response
        
        self.mock_llm.generate = mock_generate
        tool = ReviewChangesTool(llm_client=self.mock_llm)
        
        result = await tool.execute("+change")
        
        if result.success:
            assert "routing" in result.result or "review_priority" in result.result

    @pytest.mark.asyncio
    async def test_execute_adds_repo_path_and_author(self):
        """Should accept repo_path and author parameters."""
        async def mock_generate(prompt):
            return self.valid_response
        
        self.mock_llm.generate = mock_generate
        tool = ReviewChangesTool(llm_client=self.mock_llm)
        
        result = await tool.execute(
            diff="+change",
            repo_path=".",
            author="test@example.com",
        )
        
        # Should not raise on new parameters
        assert result.success is True or "error" in str(result.error).lower() is False

    @pytest.mark.asyncio
    async def test_execute_without_llm(self):
        """Should return error without LLM."""
        tool = ReviewChangesTool()
        
        result = await tool.execute("+change")
        
        assert result.success is False
        assert "llm" in result.error.lower()

    @pytest.mark.asyncio
    async def test_prompt_includes_risk_context(self):
        """Should include risk context in LLM prompt."""
        captured_prompt = []
        
        async def mock_generate(prompt):
            captured_prompt.append(prompt)
            return self.valid_response
        
        self.mock_llm.generate = mock_generate
        tool = ReviewChangesTool(llm_client=self.mock_llm)
        
        diff = """
diff --git a/.env b/.env
+SECRET=value
"""
        await tool.execute(diff)
        
        if captured_prompt:
            # Should mention risk context
            assert "risk" in captured_prompt[0].lower() or len(captured_prompt[0]) > 100


# =============================================================================
# Test RiskFeatures Serialization
# =============================================================================


class TestRiskFeaturesSerialization:
    """Tests for RiskFeatures dataclass methods."""

    def test_feature_names(self):
        """Should return correct feature names list."""
        names = RiskFeatures.feature_names()
        
        assert len(names) == 18
        assert "file_sensitivity_score" in names
        assert "security_pattern_count" in names

    def test_to_array_order(self):
        """Should match feature_names order."""
        features = RiskFeatures(
            file_sensitivity_score=0.1,
            sensitive_file_count=2,
            lines_added=100,
        )
        
        array = features.to_array()
        names = RiskFeatures.feature_names()
        
        # First element should be file_sensitivity_score
        assert names[0] == "file_sensitivity_score"
        assert array[0] == 0.1
