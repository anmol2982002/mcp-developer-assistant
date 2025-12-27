"""
Risk Prediction Model

ML-based risk prediction for code reviews using sklearn ensemble:
- Random Forest for robust predictions
- Gradient Boosting for accuracy
- Feature importance for explainability
- Model registry integration for versioning
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from observability.logging_config import get_logger

logger = get_logger(__name__)

# Try to import sklearn
try:
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn_not_available", message="Risk prediction model will use rule-based fallback")


@dataclass
class RiskPrediction:
    """Risk prediction result."""
    
    score: float  # 0-1 risk score
    confidence: float  # Model confidence 0-1
    level: str  # HIGH, MEDIUM, LOW
    feature_contributions: Dict[str, float]  # Feature importance
    model_id: str  # Model version used


class RiskPredictionModel:
    """
    ML model for code review risk prediction.
    
    Uses ensemble of Random Forest and Gradient Boosting for robust predictions.
    Provides feature importance for explainability.
    """
    
    # Feature names (must match RiskFeatures.to_array() order)
    FEATURE_NAMES = [
        "file_sensitivity_score",
        "sensitive_file_count",
        "critical_file_touched",
        "lines_added",
        "lines_removed",
        "files_changed",
        "complexity_score",
        "author_familiarity",
        "author_file_ownership",
        "author_recent_activity",
        "days_since_last_review",
        "is_weekend_commit",
        "is_late_night_commit",
        "time_risk_score",
        "security_pattern_count",
        "new_dependency_count",
        "has_binary_files",
        "has_large_additions",
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model.
        
        Args:
            model_path: Path to load trained model from
        """
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.model_id = "untrained"
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def train(
        self,
        features_list: List[List[float]],
        labels: List[float],
        output_path: str,
        version: str = "1.0.0",
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train ensemble model on labeled data.
        
        Args:
            features_list: List of feature arrays
            labels: Risk scores (0-1)
            output_path: Path to save trained model
            version: Model version string
            hyperparams: Optional hyperparameters
            
        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn_required_for_training")
            return {"success": False, "error": "sklearn not available"}
        
        if len(features_list) < 10:
            logger.warning("insufficient_training_data", count=len(features_list))
        
        # Convert to numpy
        X = np.array(features_list)
        y = np.array(labels)
        
        # Hyperparameters
        hp = hyperparams or {}
        rf_estimators = hp.get("rf_n_estimators", 100)
        gb_estimators = hp.get("gb_n_estimators", 100)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=hp.get("max_depth", 10),
            min_samples_split=hp.get("min_samples_split", 5),
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_scaled, y)
        
        # Train Gradient Boosting
        self.gb_model = GradientBoostingRegressor(
            n_estimators=gb_estimators,
            max_depth=hp.get("max_depth", 5),
            learning_rate=hp.get("learning_rate", 0.1),
            random_state=42,
        )
        self.gb_model.fit(X_scaled, y)
        
        # Calculate metrics
        rf_cv_scores = cross_val_score(self.rf_model, X_scaled, y, cv=min(5, len(X)))
        gb_cv_scores = cross_val_score(self.gb_model, X_scaled, y, cv=min(5, len(X)))
        
        # Combined feature importance
        feature_importance = self._calc_feature_importance()
        
        # Save model
        self.model_id = f"risk_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_trained = True
        self.save(output_path)
        
        metrics = {
            "success": True,
            "model_id": self.model_id,
            "samples": len(features_list),
            "rf_cv_mean": float(np.mean(rf_cv_scores)),
            "rf_cv_std": float(np.std(rf_cv_scores)),
            "gb_cv_mean": float(np.mean(gb_cv_scores)),
            "gb_cv_std": float(np.std(gb_cv_scores)),
            "feature_importance": feature_importance,
        }
        
        logger.info("risk_model_trained", **metrics)
        return metrics
    
    def train_from_synthetic(
        self,
        output_path: str,
        n_samples: int = 500,
        version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """
        Train model using synthetic data for cold start.
        
        Generates realistic feature distributions based on domain knowledge.
        """
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "sklearn not available"}
        
        logger.info("generating_synthetic_data", samples=n_samples)
        
        features_list = []
        labels = []
        
        for _ in range(n_samples):
            # Generate features with realistic distributions
            features = self._generate_synthetic_sample()
            features_list.append(features)
            
            # Calculate label based on features (domain knowledge)
            label = self._synthetic_label(features)
            labels.append(label)
        
        return self.train(features_list, labels, output_path, version)
    
    def _generate_synthetic_sample(self) -> List[float]:
        """Generate a synthetic feature sample."""
        import random
        
        # File sensitivity (skewed toward low)
        file_sensitivity = random.betavariate(2, 5)
        sensitive_files = random.randint(0, 5) if random.random() > 0.7 else 0
        critical_file = 1.0 if random.random() < 0.05 else 0.0
        
        # Change complexity (log-normal distribution)
        lines_added = int(np.random.lognormal(4, 1.5))
        lines_removed = int(np.random.lognormal(3, 1.5))
        files_changed = max(1, int(np.random.lognormal(1, 1)))
        complexity = min(1.0, np.random.beta(2, 5))
        
        # Author familiarity
        author_familiarity = random.betavariate(5, 2)  # Skewed high (familiar)
        author_ownership = random.betavariate(3, 2)
        recent_activity = random.betavariate(4, 3)
        
        # Temporal
        days_since_review = random.randint(0, 90)
        is_weekend = 1.0 if random.random() < 0.15 else 0.0
        is_late_night = 1.0 if random.random() < 0.08 else 0.0
        time_risk = is_weekend * 0.3 + is_late_night * 0.4
        
        # Security
        security_patterns = random.randint(0, 3) if random.random() < 0.2 else 0
        new_deps = random.randint(0, 5) if random.random() < 0.3 else 0
        
        # Binary/large
        has_binary = 1.0 if random.random() < 0.05 else 0.0
        has_large = 1.0 if random.random() < 0.1 else 0.0
        
        return [
            file_sensitivity,
            float(sensitive_files),
            critical_file,
            float(lines_added),
            float(lines_removed),
            float(files_changed),
            complexity,
            author_familiarity,
            author_ownership,
            recent_activity,
            float(days_since_review),
            is_weekend,
            is_late_night,
            time_risk,
            float(security_patterns),
            float(new_deps),
            has_binary,
            has_large,
        ]
    
    def _synthetic_label(self, features: List[float]) -> float:
        """Calculate risk label from features (domain knowledge)."""
        # Weighted combination based on domain expertise
        weights = {
            0: 0.20,   # file_sensitivity
            2: 0.25,   # critical_file
            6: 0.10,   # complexity
            7: -0.10,  # author_familiarity (negative = reduces risk)
            8: -0.05,  # author_ownership
            13: 0.10,  # time_risk
            14: 0.15,  # security_patterns
            17: 0.05,  # has_large
        }
        
        score = 0.3  # Base risk
        for idx, weight in weights.items():
            score += features[idx] * weight
        
        # Normalize to 0-1
        return max(0.0, min(1.0, score))
    
    def predict(self, features: List[float]) -> float:
        """
        Predict risk score from features.
        
        Args:
            features: Feature array (from RiskFeatures.to_array())
            
        Returns:
            Risk score 0-1
        """
        if not self.is_trained:
            # Fallback to rule-based
            return self._rule_based_prediction(features)
        
        if not SKLEARN_AVAILABLE:
            return self._rule_based_prediction(features)
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction (average)
        rf_pred = self.rf_model.predict(X_scaled)[0]
        gb_pred = self.gb_model.predict(X_scaled)[0]
        
        score = (rf_pred + gb_pred) / 2
        return max(0.0, min(1.0, score))
    
    def predict_with_explanation(
        self,
        features: List[float],
    ) -> RiskPrediction:
        """
        Predict with feature contributions for explainability.
        
        Args:
            features: Feature array
            
        Returns:
            RiskPrediction with score, confidence, and feature contributions
        """
        score = self.predict(features)
        
        # Calculate feature contributions
        contributions = {}
        if self.is_trained and SKLEARN_AVAILABLE:
            importance = self._calc_feature_importance()
            for name, imp in importance.items():
                idx = self.FEATURE_NAMES.index(name)
                contributions[name] = features[idx] * imp
        else:
            # Rule-based contributions
            contributions = self._rule_based_contributions(features)
        
        # Determine level
        if score >= 0.7:
            level = "HIGH"
        elif score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        # Confidence based on model agreement
        confidence = self._calc_confidence(features) if self.is_trained else 0.6
        
        return RiskPrediction(
            score=score,
            confidence=confidence,
            level=level,
            feature_contributions=contributions,
            model_id=self.model_id,
        )
    
    def _calc_feature_importance(self) -> Dict[str, float]:
        """Calculate combined feature importance from ensemble."""
        if not self.is_trained:
            return {}
        
        rf_imp = self.rf_model.feature_importances_
        gb_imp = self.gb_model.feature_importances_
        
        # Average importance
        avg_imp = (rf_imp + gb_imp) / 2
        
        return {
            name: float(imp) 
            for name, imp in zip(self.FEATURE_NAMES, avg_imp)
        }
    
    def _calc_confidence(self, features: List[float]) -> float:
        """Calculate prediction confidence based on model agreement."""
        if not SKLEARN_AVAILABLE:
            return 0.6
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        rf_pred = self.rf_model.predict(X_scaled)[0]
        gb_pred = self.gb_model.predict(X_scaled)[0]
        
        # Confidence inversely proportional to disagreement
        disagreement = abs(rf_pred - gb_pred)
        confidence = 1.0 - min(1.0, disagreement * 2)
        
        return max(0.5, confidence)
    
    def _rule_based_prediction(self, features: List[float]) -> float:
        """Fallback rule-based prediction."""
        score = 0.3  # Base
        
        # File sensitivity
        score += features[0] * 0.2
        score += features[2] * 0.25  # Critical file
        
        # Complexity
        score += features[6] * 0.1
        
        # Author familiarity (reduces risk)
        score -= features[7] * 0.1
        
        # Time risk
        score += features[13] * 0.1
        
        # Security patterns
        score += min(features[14] * 0.05, 0.2)
        
        return max(0.0, min(1.0, score))
    
    def _rule_based_contributions(self, features: List[float]) -> Dict[str, float]:
        """Rule-based feature contributions for explainability."""
        return {
            "file_sensitivity_score": features[0] * 0.2,
            "critical_file_touched": features[2] * 0.25,
            "complexity_score": features[6] * 0.1,
            "author_familiarity": -features[7] * 0.1,
            "time_risk_score": features[13] * 0.1,
            "security_pattern_count": min(features[14] * 0.05, 0.2),
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        model_data = {
            "model_id": self.model_id,
            "is_trained": self.is_trained,
            "feature_names": self.FEATURE_NAMES,
        }
        
        if self.is_trained and SKLEARN_AVAILABLE:
            model_data["rf_model"] = self.rf_model
            model_data["gb_model"] = self.gb_model
            model_data["scaler"] = self.scaler
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info("risk_model_saved", path=path, model_id=self.model_id)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model_id = model_data.get("model_id", "loaded")
            self.is_trained = model_data.get("is_trained", False)
            
            if self.is_trained and SKLEARN_AVAILABLE:
                self.rf_model = model_data.get("rf_model")
                self.gb_model = model_data.get("gb_model")
                self.scaler = model_data.get("scaler")
            
            logger.info("risk_model_loaded", path=path, model_id=self.model_id)
            return True
            
        except Exception as e:
            logger.error("risk_model_load_failed", path=path, error=str(e))
            return False


# Singleton (lazy initialized)
_risk_model: Optional[RiskPredictionModel] = None


def get_risk_model(model_path: Optional[str] = None) -> RiskPredictionModel:
    """Get or create risk prediction model singleton."""
    global _risk_model
    
    if _risk_model is None:
        _risk_model = RiskPredictionModel(model_path)
    
    return _risk_model
