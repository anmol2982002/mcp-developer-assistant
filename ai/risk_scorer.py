"""
Risk Scorer

Scores code changes for risk level based on:
- File sensitivity
- Change complexity
- Security patterns
- ML model predictions (when available)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from observability.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ai.risk_feature_extractor import RiskFeatureExtractor, RiskFeatures
    from ai.risk_prediction_model import RiskPredictionModel, RiskPrediction


@dataclass
class RiskScore:
    """Risk assessment result."""

    score: float  # 0.0 - 1.0 (higher = riskier)
    level: str  # LOW, MEDIUM, HIGH
    factors: List[str]  # Contributing factors
    
    # ML-enhanced fields
    ml_score: Optional[float] = None  # ML model prediction
    ml_confidence: Optional[float] = None  # Model confidence
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    model_id: Optional[str] = None


class RiskScorer:
    """
    Score code changes for risk.

    Features analyzed:
    - File type/path sensitivity
    - Change size
    - Security-related patterns
    - Credential/secret patterns
    - ML model predictions (when available)
    """

    # High-risk file patterns
    SENSITIVE_PATTERNS = [
        r"\.env",
        r"config\.py",
        r"settings\.py",
        r"credentials",
        r"secrets",
        r"auth",
        r"password",
        r"\.pem$",
        r"\.key$",
    ]

    # Security-related code patterns
    SECURITY_PATTERNS = [
        (r"eval\s*\(", "eval() usage detected"),
        (r"exec\s*\(", "exec() usage detected"),
        (r"subprocess\.(call|Popen|run)", "subprocess usage"),
        (r"os\.system", "os.system usage"),
        (r"pickle\.(load|loads)", "pickle deserialization"),
        (r"(api_key|secret_key|password)\s*=\s*['\"]", "Hardcoded credential"),
        (r"requests?\.(get|post|put|delete)", "HTTP request"),
        (r"cursor\.execute.*%s", "SQL with string formatting"),
    ]

    def __init__(
        self,
        ml_model: Optional["RiskPredictionModel"] = None,
        feature_extractor: Optional["RiskFeatureExtractor"] = None,
    ):
        """
        Initialize risk scorer.
        
        Args:
            ml_model: Optional ML model for enhanced predictions
            feature_extractor: Optional feature extractor (creates default if None)
        """
        self._compile_patterns()
        self.ml_model = ml_model
        self._feature_extractor = feature_extractor
    
    @property
    def feature_extractor(self) -> "RiskFeatureExtractor":
        """Lazy-load feature extractor."""
        if self._feature_extractor is None:
            from ai.risk_feature_extractor import RiskFeatureExtractor
            self._feature_extractor = RiskFeatureExtractor()
        return self._feature_extractor

    def _compile_patterns(self):
        """Compile regex patterns."""
        self.sensitive_re = [re.compile(p, re.I) for p in self.SENSITIVE_PATTERNS]
        self.security_re = [(re.compile(p, re.I), msg) for p, msg in self.SECURITY_PATTERNS]

    def score_file(self, path: str, content: Optional[str] = None) -> RiskScore:
        """
        Score a single file for risk.

        Args:
            path: File path
            content: File content (optional)

        Returns:
            RiskScore with assessment
        """
        factors = []
        base_score = 0.0

        # Check path sensitivity
        for pattern in self.sensitive_re:
            if pattern.search(path):
                factors.append(f"Sensitive path pattern: {pattern.pattern}")
                base_score += 0.3
                break

        # Check content if provided
        if content:
            for pattern, message in self.security_re:
                matches = pattern.findall(content)
                if matches:
                    factors.append(f"{message} ({len(matches)} occurrences)")
                    base_score += 0.1 * len(matches)

            # Check file size
            lines = len(content.splitlines())
            if lines > 500:
                factors.append(f"Large file ({lines} lines)")
                base_score += 0.1

        # Determine level
        score = min(base_score, 1.0)
        if score >= 0.7:
            level = "HIGH"
        elif score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        return RiskScore(score=score, level=level, factors=factors)

    def score_diff(self, diff: str) -> RiskScore:
        """
        Score a git diff for risk (rule-based).

        Args:
            diff: Git diff content

        Returns:
            RiskScore with assessment
        """
        factors = []
        base_score = 0.0

        # Count changes
        additions = diff.count("\n+") - diff.count("\n+++")
        deletions = diff.count("\n-") - diff.count("\n---")
        total_changes = additions + deletions

        if total_changes > 500:
            factors.append(f"Large change ({total_changes} lines)")
            base_score += 0.2
        elif total_changes > 200:
            factors.append(f"Medium change ({total_changes} lines)")
            base_score += 0.1

        # Extract changed files
        file_pattern = re.compile(r"^\+\+\+ b/(.+)$", re.M)
        changed_files = file_pattern.findall(diff)

        for file in changed_files:
            for pattern in self.sensitive_re:
                if pattern.search(file):
                    factors.append(f"Changes to sensitive file: {file}")
                    base_score += 0.3
                    break

        # Check for security patterns in added lines
        added_lines = "\n".join(
            line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
        )

        for pattern, message in self.security_re:
            matches = pattern.findall(added_lines)
            if matches:
                factors.append(f"Added: {message}")
                base_score += 0.15 * min(len(matches), 3)

        # Determine level
        score = min(base_score, 1.0)
        if score >= 0.7:
            level = "HIGH"
        elif score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        logger.info("diff_scored", level=level, score=score, factors=len(factors))
        return RiskScore(score=score, level=level, factors=factors)
    
    def score_with_features(
        self,
        diff: str,
        repo_path: str = ".",
        author: Optional[str] = None,
    ) -> RiskScore:
        """
        Score using full feature extraction + ML model.
        
        Combines rule-based scoring with ML predictions for more accurate
        risk assessment.
        
        Args:
            diff: Git diff content
            repo_path: Repository path for git history analysis
            author: Commit author for familiarity scoring
            
        Returns:
            RiskScore with ML-enhanced assessment
        """
        # Get rule-based score first
        rule_score = self.score_diff(diff)
        
        # Extract features
        try:
            features = self.feature_extractor.extract_from_diff(diff, repo_path, author)
            feature_array = features.to_array()
        except Exception as e:
            logger.warning("feature_extraction_failed", error=str(e))
            return rule_score
        
        # Get ML prediction if model available
        ml_score = None
        ml_confidence = None
        feature_contributions = {}
        model_id = None
        
        if self.ml_model:
            try:
                prediction = self.ml_model.predict_with_explanation(feature_array)
                ml_score = prediction.score
                ml_confidence = prediction.confidence
                feature_contributions = prediction.feature_contributions
                model_id = prediction.model_id
            except Exception as e:
                logger.warning("ml_prediction_failed", error=str(e))
        
        # Combine scores
        if ml_score is not None:
            # Weighted combination: 60% ML, 40% rule-based
            combined_score = 0.6 * ml_score + 0.4 * rule_score.score
        else:
            combined_score = rule_score.score
        
        # Determine level from combined score
        if combined_score >= 0.7:
            level = "HIGH"
        elif combined_score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        # Enhance factors with ML insights
        factors = rule_score.factors.copy()
        
        # Add top contributing features
        if feature_contributions:
            top_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            for name, contrib in top_features:
                if abs(contrib) > 0.05:
                    direction = "increases" if contrib > 0 else "decreases"
                    factors.append(f"ML: {name} {direction} risk ({contrib:.2f})")
        
        # Add feature-based factors
        if features.critical_file_touched:
            if "critical file" not in " ".join(factors).lower():
                factors.append("Critical file touched (config/secrets)")
        
        if features.security_patterns_found:
            patterns = features.security_patterns_found[:2]
            factors.append(f"Security patterns: {', '.join(patterns)}")
        
        if features.author_familiarity < 0.3:
            factors.append(f"Author unfamiliar with codebase ({features.author_familiarity:.1%})")
        
        logger.info(
            "diff_scored_with_ml",
            level=level,
            rule_score=rule_score.score,
            ml_score=ml_score,
            combined_score=combined_score,
        )
        
        return RiskScore(
            score=combined_score,
            level=level,
            factors=factors,
            ml_score=ml_score,
            ml_confidence=ml_confidence,
            feature_contributions=feature_contributions,
            model_id=model_id,
        )


# Singleton
risk_scorer = RiskScorer()

