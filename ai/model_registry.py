"""
Model Registry

Manages model versioning, storage, and A/B testing for anomaly detection models.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    
    model_id: str
    version: str
    created_at: str
    model_type: str  # ensemble, isolation_forest, etc.
    features: List[str]
    training_samples: int
    contamination: float
    metrics: Dict[str, float]  # accuracy, precision, etc.
    description: str = ""
    is_active: bool = False
    traffic_weight: float = 0.0  # For A/B testing


class ModelRegistry:
    """
    Registry for managing anomaly detection models.
    
    Features:
    - Model versioning with metadata
    - Rollback capabilities
    - A/B testing traffic allocation
    - Performance tracking
    """
    
    METADATA_FILE = "registry.json"
    
    def __init__(self, registry_path: str = "models"):
        """
        Initialize registry.
        
        Args:
            registry_path: Base path for model storage
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self._metadata_file = self.registry_path / self.METADATA_FILE
        self._models: Dict[str, ModelMetadata] = {}
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)
                    for model_id, meta_dict in data.get("models", {}).items():
                        self._models[model_id] = ModelMetadata(**meta_dict)
                logger.info("registry_loaded", models=len(self._models))
            except Exception as e:
                logger.error("registry_load_failed", error=str(e))
    
    def _save_registry(self) -> None:
        """Save registry metadata to disk."""
        try:
            data = {
                "models": {
                    model_id: asdict(meta)
                    for model_id, meta in self._models.items()
                },
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(self._metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("registry_save_failed", error=str(e))
    
    def _generate_model_id(self, model_type: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{model_type}_{version}_{timestamp}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"{model_type}_{version}_{hash_suffix}"
    
    def register(
        self,
        model: Any,
        model_type: str,
        version: str,
        features: List[str],
        training_samples: int,
        contamination: float = 0.05,
        metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        activate: bool = False,
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: The trained model object
            model_type: Type of model (e.g., 'ensemble')
            version: Semantic version (e.g., '1.0.0')
            features: List of feature names used
            training_samples: Number of training samples
            contamination: Contamination parameter used
            metrics: Evaluation metrics
            description: Model description
            activate: Whether to activate immediately
            
        Returns:
            Model ID
        """
        model_id = self._generate_model_id(model_type, version)
        model_path = self.registry_path / f"{model_id}.pkl"
        
        # Save model file
        joblib.dump(model, model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            model_type=model_type,
            features=features,
            training_samples=training_samples,
            contamination=contamination,
            metrics=metrics or {},
            description=description,
            is_active=activate,
            traffic_weight=1.0 if activate else 0.0,
        )
        
        self._models[model_id] = metadata
        
        # If activating, deactivate others
        if activate:
            for mid in self._models:
                if mid != model_id:
                    self._models[mid].is_active = False
                    self._models[mid].traffic_weight = 0.0
        
        self._save_registry()
        
        logger.info(
            "model_registered",
            model_id=model_id,
            version=version,
            active=activate,
        )
        
        return model_id
    
    def get_active_model(self) -> Optional[tuple]:
        """
        Get the currently active model.
        
        Returns:
            Tuple of (model, metadata) or None
        """
        for model_id, metadata in self._models.items():
            if metadata.is_active:
                model_path = self.registry_path / f"{model_id}.pkl"
                if model_path.exists():
                    model = joblib.load(model_path)
                    return model, metadata
        return None
    
    def get_model(self, model_id: str) -> Optional[tuple]:
        """
        Get a specific model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, metadata) or None
        """
        if model_id not in self._models:
            return None
        
        model_path = self.registry_path / f"{model_id}.pkl"
        if not model_path.exists():
            return None
        
        model = joblib.load(model_path)
        return model, self._models[model_id]
    
    def activate(self, model_id: str) -> bool:
        """
        Activate a model for production use.
        
        Args:
            model_id: Model to activate
            
        Returns:
            Success status
        """
        if model_id not in self._models:
            logger.error("model_not_found", model_id=model_id)
            return False
        
        # Deactivate all others
        for mid in self._models:
            self._models[mid].is_active = False
            self._models[mid].traffic_weight = 0.0
        
        # Activate target
        self._models[model_id].is_active = True
        self._models[model_id].traffic_weight = 1.0
        
        self._save_registry()
        logger.info("model_activated", model_id=model_id)
        
        return True
    
    def rollback(self, to_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            to_version: Version to rollback to (or previous if None)
            
        Returns:
            Success status
        """
        # Get current active
        current_active = None
        for model_id, meta in self._models.items():
            if meta.is_active:
                current_active = model_id
                break
        
        # Find target
        candidates = []
        for model_id, meta in self._models.items():
            if model_id == current_active:
                continue
            if to_version and meta.version != to_version:
                continue
            candidates.append((model_id, meta))
        
        if not candidates:
            logger.error("no_rollback_target")
            return False
        
        # Sort by creation time (newest first)
        candidates.sort(key=lambda x: x[1].created_at, reverse=True)
        target_id = candidates[0][0]
        
        return self.activate(target_id)
    
    def setup_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        weight_a: float = 0.5,
    ) -> bool:
        """
        Setup A/B testing between two models.
        
        Args:
            model_a_id: First model (control)
            model_b_id: Second model (experiment)
            weight_a: Traffic weight for model A (0-1)
            
        Returns:
            Success status
        """
        if model_a_id not in self._models or model_b_id not in self._models:
            logger.error("ab_test_models_not_found")
            return False
        
        # Deactivate all
        for mid in self._models:
            self._models[mid].is_active = False
            self._models[mid].traffic_weight = 0.0
        
        # Setup test
        self._models[model_a_id].is_active = True
        self._models[model_a_id].traffic_weight = weight_a
        
        self._models[model_b_id].is_active = True
        self._models[model_b_id].traffic_weight = 1.0 - weight_a
        
        self._save_registry()
        
        logger.info(
            "ab_test_setup",
            model_a=model_a_id,
            model_b=model_b_id,
            weight_a=weight_a,
        )
        
        return True
    
    def get_ab_test_model(self) -> Optional[tuple]:
        """
        Get model for A/B test based on traffic weights.
        
        Returns:
            Tuple of (model, metadata) for selected model
        """
        import random
        
        active_models = [
            (model_id, meta)
            for model_id, meta in self._models.items()
            if meta.is_active and meta.traffic_weight > 0
        ]
        
        if not active_models:
            return None
        
        # Weighted random selection
        weights = [meta.traffic_weight for _, meta in active_models]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        selected = random.choices(active_models, weights=weights, k=1)[0]
        model_id, metadata = selected
        
        model_path = self.registry_path / f"{model_id}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            return model, metadata
        
        return None
    
    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float],
    ) -> bool:
        """
        Update metrics for a model (for tracking A/B test results).
        
        Args:
            model_id: Model to update
            metrics: New metrics to merge
            
        Returns:
            Success status
        """
        if model_id not in self._models:
            return False
        
        self._models[model_id].metrics.update(metrics)
        self._save_registry()
        
        return True
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        active_only: bool = False,
    ) -> List[ModelMetadata]:
        """
        List registered models.
        
        Args:
            model_type: Filter by type
            active_only: Only show active models
            
        Returns:
            List of model metadata
        """
        models = list(self._models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if active_only:
            models = [m for m in models if m.is_active]
        
        # Sort by creation time
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        return models
    
    def delete(self, model_id: str) -> bool:
        """
        Delete a model from registry.
        
        Args:
            model_id: Model to delete
            
        Returns:
            Success status
        """
        if model_id not in self._models:
            return False
        
        # Don't delete active model
        if self._models[model_id].is_active:
            logger.error("cannot_delete_active_model")
            return False
        
        # Remove file
        model_path = self.registry_path / f"{model_id}.pkl"
        if model_path.exists():
            model_path.unlink()
        
        # Remove from registry
        del self._models[model_id]
        self._save_registry()
        
        logger.info("model_deleted", model_id=model_id)
        return True
    
    def cleanup_old_models(self, keep_count: int = 5) -> int:
        """
        Remove old inactive models, keeping the most recent.
        
        Args:
            keep_count: Number of inactive models to keep
            
        Returns:
            Number of models deleted
        """
        # Get inactive models sorted by date
        inactive = [
            (model_id, meta)
            for model_id, meta in self._models.items()
            if not meta.is_active
        ]
        inactive.sort(key=lambda x: x[1].created_at, reverse=True)
        
        # Delete old ones
        to_delete = inactive[keep_count:]
        deleted = 0
        
        for model_id, _ in to_delete:
            if self.delete(model_id):
                deleted += 1
        
        return deleted


# Default registry instance
model_registry = ModelRegistry(registry_path="models")
