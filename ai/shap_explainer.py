"""
SHAP Explainer

Provides human-readable explanations for anomaly detection decisions.
Uses SHAP (SHapley Additive exPlanations) values when available,
with rule-based fallbacks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from observability.logging_config import get_logger

logger = get_logger(__name__)

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap_not_available", msg="SHAP library not installed, using rule-based explanations")


@dataclass
class AnomalyExplanation:
    """Human-readable explanation for an anomaly."""
    
    summary: str  # One-line summary
    contributing_features: List[Tuple[str, float, str]]  # (feature, importance, reason)
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: List[str]  # Suggested actions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "contributing_features": [
                {"feature": f, "importance": i, "reason": r}
                for f, i, r in self.contributing_features
            ],
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
        }


# Feature descriptions for human-readable output
FEATURE_DESCRIPTIONS = {
    "time_since_last": {
        "name": "Request Timing",
        "low_reason": "Rapid requests (potential automation)",
        "high_reason": "Long gap after previous request",
        "threshold_low": 1.0,
        "threshold_high": 3600.0,
    },
    "unusual_hour": {
        "name": "Access Time",
        "triggered_reason": "Request at unusual hour (night/weekend)",
        "normal_reason": "Request during normal hours",
    },
    "session_duration": {
        "name": "Session Length",
        "low_reason": "Very short session",
        "high_reason": "Unusually long session",
        "threshold_low": 60.0,
        "threshold_high": 7200.0,
    },
    "request_rate_per_min": {
        "name": "Request Rate",
        "low_reason": "Normal request rate",
        "high_reason": "High request rate (potential abuse)",
        "threshold_high": 8.0,
    },
    "velocity_change": {
        "name": "Rate Acceleration",
        "triggered_reason": "Sudden change in request pattern",
        "normal_reason": "Steady request pattern",
        "threshold_abs": 3.0,
    },
    "tool_changed": {
        "name": "Tool Usage",
        "triggered_reason": "Frequent tool switching",
        "normal_reason": "Consistent tool usage",
    },
    "sequence_entropy": {
        "name": "Pattern Randomness",
        "high_reason": "Unusual tool access pattern",
        "normal_reason": "Normal usage pattern",
        "threshold_high": 0.8,
    },
    "tool_transition_prob": {
        "name": "Tool Sequence",
        "low_reason": "Unusual tool sequence",
        "normal_reason": "Expected tool sequence",
        "threshold_low": 0.2,
    },
    "file_sensitivity_score": {
        "name": "File Sensitivity",
        "high_reason": "Accessing sensitive files",
        "low_reason": "Accessing normal files",
        "threshold_high": 0.6,
    },
    "resource_depth_score": {
        "name": "Directory Depth",
        "high_reason": "Deep directory access",
        "normal_reason": "Normal directory access",
        "threshold_high": 0.7,
    },
    "new_ip": {
        "name": "IP Address",
        "triggered_reason": "Request from new IP address",
        "normal_reason": "Request from known IP",
    },
    "request_size_anomaly": {
        "name": "Request Size",
        "triggered_reason": "Unusual request size",
        "normal_reason": "Normal request size",
        "threshold_abs": 2.0,
    },
}


class SHAPExplainer:
    """
    Generate human-readable explanations for anomaly detections.
    
    Uses SHAP values when available, with rule-based fallbacks.
    """
    
    def __init__(self, model=None, background_data: Optional[np.ndarray] = None):
        """
        Initialize explainer.
        
        Args:
            model: Trained anomaly detection model (for SHAP)
            background_data: Background samples for SHAP (100-1000 samples)
        """
        self.model = model
        self.shap_explainer = None
        
        if SHAP_AVAILABLE and model is not None and background_data is not None:
            try:
                # Create SHAP explainer for tree-based models
                self.shap_explainer = shap.TreeExplainer(model)
                logger.info("shap_explainer_initialized")
            except Exception as e:
                logger.warning("shap_init_failed", error=str(e))
    
    def explain(
        self,
        features: Dict[str, float],
        anomaly_score: float,
        feature_names: Optional[List[str]] = None,
    ) -> AnomalyExplanation:
        """
        Generate explanation for an anomaly detection.
        
        Args:
            features: Feature dictionary
            anomaly_score: Overall anomaly score (0-1)
            feature_names: Ordered feature names (if using array input)
            
        Returns:
            AnomalyExplanation with human-readable details
        """
        # Try SHAP-based explanation first
        if self.shap_explainer is not None:
            try:
                return self._explain_with_shap(features, anomaly_score, feature_names)
            except Exception as e:
                logger.debug("shap_explain_failed", error=str(e))
        
        # Fall back to rule-based explanation
        return self._explain_with_rules(features, anomaly_score)
    
    def _explain_with_shap(
        self,
        features: Dict[str, float],
        anomaly_score: float,
        feature_names: Optional[List[str]] = None,
    ) -> AnomalyExplanation:
        """Generate SHAP-based explanation."""
        # Convert features to array
        if feature_names:
            X = np.array([[features.get(name, 0) for name in feature_names]])
        else:
            X = np.array([list(features.values())])
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-output models
        
        shap_values = shap_values[0]  # First sample
        
        # Get feature importances
        feature_importance = []
        names = feature_names or list(features.keys())
        
        for i, name in enumerate(names):
            importance = abs(shap_values[i]) if i < len(shap_values) else 0
            value = features.get(name, 0)
            reason = self._get_feature_reason(name, value, shap_values[i])
            feature_importance.append((name, importance, reason))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:5]  # Top 5
        
        # Generate summary and recommendations
        summary = self._generate_summary(top_features, anomaly_score)
        risk_level = self._get_risk_level(anomaly_score)
        recommendations = self._generate_recommendations(top_features, risk_level)
        
        return AnomalyExplanation(
            summary=summary,
            contributing_features=top_features,
            risk_level=risk_level,
            recommendations=recommendations,
        )
    
    def _explain_with_rules(
        self,
        features: Dict[str, float],
        anomaly_score: float,
    ) -> AnomalyExplanation:
        """Generate rule-based explanation (fallback)."""
        contributing = []
        
        for name, value in features.items():
            desc = FEATURE_DESCRIPTIONS.get(name, {})
            importance = 0.0
            reason = ""
            
            # Check thresholds
            if "threshold_low" in desc and value < desc["threshold_low"]:
                importance = 0.5
                reason = desc.get("low_reason", f"Low {name}")
            elif "threshold_high" in desc and value > desc["threshold_high"]:
                importance = 0.7
                reason = desc.get("high_reason", f"High {name}")
            elif "threshold_abs" in desc and abs(value) > desc["threshold_abs"]:
                importance = 0.6
                reason = desc.get("triggered_reason", f"Unusual {name}")
            elif name in ["unusual_hour", "new_ip", "tool_changed"] and value > 0:
                importance = 0.4
                reason = desc.get("triggered_reason", f"Triggered: {name}")
            
            if importance > 0:
                contributing.append((
                    desc.get("name", name),
                    importance * anomaly_score,
                    reason,
                ))
        
        # Sort by importance
        contributing.sort(key=lambda x: x[1], reverse=True)
        top_features = contributing[:5]
        
        # Generate outputs
        summary = self._generate_summary(top_features, anomaly_score)
        risk_level = self._get_risk_level(anomaly_score)
        recommendations = self._generate_recommendations(top_features, risk_level)
        
        return AnomalyExplanation(
            summary=summary,
            contributing_features=top_features,
            risk_level=risk_level,
            recommendations=recommendations,
        )
    
    def _get_feature_reason(self, name: str, value: float, shap_value: float) -> str:
        """Get human-readable reason for feature contribution."""
        desc = FEATURE_DESCRIPTIONS.get(name, {})
        feature_name = desc.get("name", name)
        
        if shap_value > 0:  # Contributes to anomaly
            if "high_reason" in desc:
                return desc["high_reason"]
            elif "triggered_reason" in desc:
                return desc["triggered_reason"]
            return f"{feature_name} contributing to anomaly"
        else:
            if "normal_reason" in desc:
                return desc["normal_reason"]
            return f"{feature_name} is normal"
    
    def _generate_summary(
        self,
        top_features: List[Tuple[str, float, str]],
        anomaly_score: float,
    ) -> str:
        """Generate one-line summary."""
        if not top_features:
            if anomaly_score < 0.3:
                return "Request appears normal with no concerning patterns."
            else:
                return "Anomalous request detected with subtle pattern deviations."
        
        # Get top reason
        top_name, _, top_reason = top_features[0]
        
        if anomaly_score >= 0.8:
            return f"HIGH RISK: {top_reason}. Multiple concerning patterns detected."
        elif anomaly_score >= 0.5:
            return f"SUSPICIOUS: {top_reason}. Request deviates from normal behavior."
        else:
            return f"MINOR: {top_reason}. Slight deviation from normal patterns."
    
    def _get_risk_level(self, anomaly_score: float) -> str:
        """Determine risk level from anomaly score."""
        if anomaly_score >= 0.9:
            return "CRITICAL"
        elif anomaly_score >= 0.7:
            return "HIGH"
        elif anomaly_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(
        self,
        top_features: List[Tuple[str, float, str]],
        risk_level: str,
    ) -> List[str]:
        """Generate action recommendations."""
        recommendations = []
        
        # Add risk-level-based recommendations
        if risk_level == "CRITICAL":
            recommendations.append("Block request and investigate immediately")
            recommendations.append("Review user's recent activity for patterns")
        elif risk_level == "HIGH":
            recommendations.append("Flag for manual review")
            recommendations.append("Consider requiring additional authentication")
        elif risk_level == "MEDIUM":
            recommendations.append("Log for pattern analysis")
        
        # Add feature-specific recommendations
        for name, importance, reason in top_features[:3]:
            if "IP" in name or "ip" in str(reason).lower():
                recommendations.append("Verify IP address legitimacy")
            elif "rate" in name.lower() or "rapid" in reason.lower():
                recommendations.append("Consider rate limiting this user")
            elif "sensitive" in reason.lower() or "sensitivity" in name.lower():
                recommendations.append("Review file access permissions")
            elif "hour" in name.lower() or "unusual" in reason.lower():
                recommendations.append("Verify user identity if outside normal hours")
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        
        return unique[:5]  # Max 5 recommendations
    
    def explain_batch(
        self,
        features_batch: List[Dict[str, float]],
        scores_batch: List[float],
    ) -> List[AnomalyExplanation]:
        """Explain multiple anomalies efficiently."""
        return [
            self.explain(features, score)
            for features, score in zip(features_batch, scores_batch)
        ]
    
    def get_feature_importance_summary(
        self,
        features_batch: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Get aggregate feature importance across batch."""
        if not features_batch:
            return {}
        
        importance_sums: Dict[str, float] = {}
        
        for features in features_batch:
            for name, value in features.items():
                desc = FEATURE_DESCRIPTIONS.get(name, {})
                importance = 0.0
                
                # Calculate importance based on deviation from normal
                if "threshold_low" in desc and value < desc["threshold_low"]:
                    importance = 1.0
                elif "threshold_high" in desc and value > desc["threshold_high"]:
                    importance = 1.0
                elif "threshold_abs" in desc and abs(value) > desc["threshold_abs"]:
                    importance = 0.8
                elif value > 0 and name in ["unusual_hour", "new_ip"]:
                    importance = 0.5
                
                importance_sums[name] = importance_sums.get(name, 0) + importance
        
        # Normalize
        total = sum(importance_sums.values()) or 1
        return {k: v / total for k, v in importance_sums.items()}


# Singleton (initialized without model)
shap_explainer = SHAPExplainer()
