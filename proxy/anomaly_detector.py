"""
Behavioral Anomaly Detector

Enhanced anomaly detection using ensemble of models:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

Features 12 behavioral signals with SHAP explanations.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ai.feature_extractor import BehavioralFeatureExtractor, FeatureVector, feature_extractor
from ai.shap_explainer import AnomalyExplanation, SHAPExplainer, shap_explainer
from observability.logging_config import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)

# Try to import sklearn models
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn_not_available", msg="scikit-learn not installed")


@dataclass
class AnomalyScore:
    """Result of anomaly detection with explanation."""

    is_anomalous: bool
    score: float  # 0.0 to 1.0 (higher = more anomalous)
    reason: str
    explanation: Optional[AnomalyExplanation] = None
    model_scores: Optional[Dict[str, float]] = None  # Per-model scores


@dataclass
class ToolRequest:
    """Tool request for anomaly analysis."""

    tool_name: str
    user_id: str
    timestamp: datetime
    ip: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class EnsembleAnomalyModel:
    """
    Ensemble of anomaly detection models.
    
    Combines Isolation Forest, LOF, and One-Class SVM with weighted voting.
    """
    
    # Model weights (can be tuned based on validation)
    DEFAULT_WEIGHTS = {
        "isolation_forest": 0.4,
        "lof": 0.35,
        "one_class_svm": 0.25,
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize ensemble.
        
        Args:
            weights: Model weights for voting (must sum to 1)
            threshold: Anomaly threshold (0-1)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        
        self.isolation_forest = None
        self.lof = None
        self.one_class_svm = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, contamination: float = 0.05) -> "EnsembleAnomalyModel":
        """
        Train all ensemble models.
        
        Args:
            X: Training data (n_samples, n_features)
            contamination: Expected fraction of anomalies
            
        Returns:
            Self for chaining
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn_not_available", msg="Cannot train models")
            return self
        
        logger.info("ensemble_training_start", samples=len(X))
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)
        
        # Train LOF (novelty detection mode)
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,  # Important for predict on new data
            n_jobs=-1,
        )
        self.lof.fit(X_scaled)
        
        # Train One-Class SVM
        self.one_class_svm = OneClassSVM(
            kernel="rbf",
            gamma="auto",
            nu=contamination,
        )
        self.one_class_svm.fit(X_scaled)
        
        self.is_fitted = True
        logger.info("ensemble_training_complete", samples=len(X))
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict anomaly scores for samples.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (combined_scores, per_model_scores)
        """
        if not self.is_fitted:
            # Return neutral scores if not fitted
            n_samples = len(X)
            return np.full(n_samples, 0.3), {}
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        model_scores = {}
        
        # Isolation Forest scores
        if self.isolation_forest is not None:
            # score_samples returns negative values, more negative = more anomalous
            if_raw = -self.isolation_forest.score_samples(X_scaled)
            # Normalize to 0-1
            if_scores = (if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-10)
            model_scores["isolation_forest"] = if_scores
        
        # LOF scores
        if self.lof is not None:
            # score_samples returns negative values
            lof_raw = -self.lof.score_samples(X_scaled)
            lof_scores = (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min() + 1e-10)
            model_scores["lof"] = lof_scores
        
        # One-Class SVM scores
        if self.one_class_svm is not None:
            # decision_function returns distance from boundary (negative = anomaly)
            svm_raw = -self.one_class_svm.decision_function(X_scaled)
            svm_scores = (svm_raw - svm_raw.min()) / (svm_raw.max() - svm_raw.min() + 1e-10)
            model_scores["one_class_svm"] = svm_scores
        
        # Weighted combination
        combined = np.zeros(len(X))
        total_weight = 0.0
        
        for model_name, scores in model_scores.items():
            weight = self.weights.get(model_name, 0.33)
            combined += weight * scores
            total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        return combined, model_scores
    
    def save(self, path: str) -> None:
        """Save ensemble to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "isolation_forest": self.isolation_forest,
            "lof": self.lof,
            "one_class_svm": self.one_class_svm,
            "scaler": self.scaler,
            "weights": self.weights,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, path)
        logger.info("ensemble_saved", path=path)
    
    @classmethod
    def load(cls, path: str) -> "EnsembleAnomalyModel":
        """Load ensemble from file."""
        try:
            data = joblib.load(path)
            
            model = cls(
                weights=data.get("weights"),
                threshold=data.get("threshold", 0.5),
            )
            model.isolation_forest = data.get("isolation_forest")
            model.lof = data.get("lof")
            model.one_class_svm = data.get("one_class_svm")
            model.scaler = data.get("scaler")
            model.is_fitted = data.get("is_fitted", False)
            
            logger.info("ensemble_loaded", path=path)
            return model
            
        except Exception as e:
            logger.error("ensemble_load_failed", path=path, error=str(e))
            return cls()


class BehavioralAnomalyDetector:
    """
    Enhanced anomaly detector with ensemble models and SHAP explanations.
    
    Features:
    - 12 behavioral features (time, rate, sequence, sensitivity, etc.)
    - Ensemble of 3 models (IF, LOF, One-Class SVM)
    - SHAP-based human-readable explanations
    - Real-time feature extraction and caching
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        enable_explanations: bool = True,
    ):
        """
        Initialize anomaly detector.

        Args:
            model_path: Path to trained ensemble model (if exists)
            threshold: Anomaly score threshold (0-1)
            enable_explanations: Whether to generate SHAP explanations
        """
        self.threshold = threshold
        self.enable_explanations = enable_explanations
        
        # Feature extractor
        self.feature_extractor = feature_extractor
        
        # SHAP explainer
        self.explainer = shap_explainer
        
        # Ensemble model
        self.ensemble = EnsembleAnomalyModel(threshold=threshold)
        
        # Legacy support: known IPs per user
        self._known_ips: Dict[str, set] = {}
        
        # Load model if path provided
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load pre-trained ensemble model."""
        try:
            model_file = Path(path)
            if model_file.exists():
                self.ensemble = EnsembleAnomalyModel.load(path)
                logger.info("anomaly_model_loaded", path=path)
            else:
                logger.warning("anomaly_model_not_found", path=path)
        except Exception as e:
            logger.error("anomaly_model_load_failed", path=path, error=str(e))

    def extract_features(
        self,
        request: ToolRequest,
        user_history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Extract behavioral features from request.
        
        This is the public API - returns a dictionary of features.
        Uses the centralized feature extractor internally.
        """
        feature_vec = self.feature_extractor.extract_features(
            tool_name=request.tool_name,
            user_id=request.user_id,
            timestamp=request.timestamp,
            ip=request.ip,
            params=request.params or {},
            history=user_history,
        )
        
        return feature_vec.to_dict()

    def check_anomaly(
        self,
        request: ToolRequest,
        user_history: List[Dict[str, Any]],
    ) -> AnomalyScore:
        """
        Check if request is anomalous.

        Args:
            request: Current tool request
            user_history: Recent requests from this user

        Returns:
            AnomalyScore with detection result and explanation
        """
        # Extract features
        features = self.extract_features(request, user_history)
        feature_vec = self.feature_extractor.extract_features(
            tool_name=request.tool_name,
            user_id=request.user_id,
            timestamp=request.timestamp,
            ip=request.ip,
            params=request.params or {},
            history=user_history,
        )
        
        # If ensemble not fitted, use rule-based
        if not self.ensemble.is_fitted:
            return self._rule_based_check(features, feature_vec)
        
        try:
            # Get ensemble prediction
            X = feature_vec.to_array().reshape(1, -1)
            scores, model_scores = self.ensemble.predict(X)
            anomaly_score = float(scores[0])
            
            is_anomalous = anomaly_score >= self.threshold
            
            # Generate explanation if needed
            explanation = None
            if is_anomalous and self.enable_explanations:
                explanation = self.explainer.explain(
                    features=features,
                    anomaly_score=anomaly_score,
                    feature_names=FeatureVector.feature_names(),
                )
                reason = explanation.summary
            else:
                reason = self._generate_reason(features, anomaly_score)
            
            # Log if anomalous
            if is_anomalous:
                self._log_anomaly(request, features, anomaly_score)
            
            return AnomalyScore(
                is_anomalous=is_anomalous,
                score=anomaly_score,
                reason=reason,
                explanation=explanation,
                model_scores={k: float(v[0]) for k, v in model_scores.items()},
            )

        except Exception as e:
            logger.error("anomaly_check_failed", error=str(e))
            # Fail open on errors
            return AnomalyScore(
                is_anomalous=False,
                score=0.0,
                reason=f"Anomaly check failed: {e}",
            )

    def _rule_based_check(
        self,
        features: Dict[str, float],
        feature_vec: FeatureVector,
    ) -> AnomalyScore:
        """
        Rule-based anomaly detection (fallback when no model).
        """
        reasons = []
        score = 0.0

        # Very rapid requests
        if features["time_since_last"] < 1:
            reasons.append("Very rapid requests (< 1 second)")
            score += 0.3

        # High request rate
        if features["request_rate_per_min"] >= 8:
            reasons.append("High request rate (>= 8/min)")
            score += 0.3

        # Unusual hour
        if features["unusual_hour"] > 0:
            reasons.append("Request at unusual hour")
            score += 0.15

        # New IP
        if features["new_ip"] > 0:
            reasons.append("Request from new IP address")
            score += 0.15
        
        # High file sensitivity
        if features["file_sensitivity_score"] >= 0.7:
            reasons.append("Accessing sensitive files")
            score += 0.25
        
        # High sequence entropy (random patterns)
        if features["sequence_entropy"] >= 0.85:
            reasons.append("Unusual tool access pattern")
            score += 0.15
        
        # Low transition probability
        if features["tool_transition_prob"] < 0.1:
            reasons.append("Unusual tool sequence")
            score += 0.1
        
        # High velocity change
        if abs(features["velocity_change"]) >= 3:
            reasons.append("Sudden change in request rate")
            score += 0.2

        is_anomalous = score >= self.threshold
        final_score = min(score, 1.0)
        
        # Generate explanation if anomalous
        explanation = None
        if is_anomalous and self.enable_explanations:
            explanation = self.explainer.explain(
                features=features,
                anomaly_score=final_score,
            )

        return AnomalyScore(
            is_anomalous=is_anomalous,
            score=final_score,
            reason="; ".join(reasons) if reasons else "Normal behavior",
            explanation=explanation,
        )

    def _generate_reason(self, features: Dict[str, float], score: float) -> str:
        """Generate concise reason string."""
        if score < 0.3:
            return "Normal behavior pattern"
        elif score < 0.5:
            return "Minor deviation from normal pattern"
        elif score < 0.7:
            return "Suspicious behavior detected"
        else:
            return "Highly anomalous behavior detected"

    def _log_anomaly(
        self,
        request: ToolRequest,
        features: Dict[str, float],
        score: float,
    ) -> None:
        """Log detected anomaly."""
        logger.warning(
            "anomaly_detected",
            user_id=request.user_id,
            tool=request.tool_name,
            features=features,
            score=score,
        )
        metrics.anomalies_detected_total.labels(user=request.user_id).inc()

    def train_from_data(
        self,
        training_data: List[Dict[str, Any]],
        output_path: str,
        contamination: float = 0.05,
    ) -> bool:
        """
        Train ensemble from feature data.
        
        Args:
            training_data: List of feature dictionaries
            output_path: Path to save trained model
            contamination: Expected fraction of anomalies
            
        Returns:
            True if training successful
        """
        if len(training_data) < 100:
            logger.warning("insufficient_training_data", count=len(training_data))
            return False
        
        # Convert to numpy array
        feature_names = FeatureVector.feature_names()
        X = np.array([
            [d.get(name, 0.0) for name in feature_names]
            for d in training_data
        ])
        
        # Train ensemble
        self.ensemble.fit(X, contamination=contamination)
        
        # Save model
        self.ensemble.save(output_path)
        
        logger.info("ensemble_trained", samples=len(X), output=output_path)
        return True
    
    def update_model(self, path: str) -> bool:
        """
        Hot-reload model from path.
        
        Args:
            path: Path to new model
            
        Returns:
            True if update successful
        """
        try:
            new_ensemble = EnsembleAnomalyModel.load(path)
            if new_ensemble.is_fitted:
                self.ensemble = new_ensemble
                logger.info("model_updated", path=path)
                return True
            return False
        except Exception as e:
            logger.error("model_update_failed", error=str(e))
            return False


# Default detector instance
anomaly_detector = BehavioralAnomalyDetector()
