"""
MCP Developer Assistant - AI Module

AI and ML components for Phases 3, 4, and 5:
- Enhanced anomaly detection (ensemble models)
- Feature extraction (12 behavioral features)
- SHAP-based explanations
- Intent classification
- Model training and registry
- Synthetic data generation
- Risk scoring with ML predictions [Phase 5]
- Hybrid search (BM25 + FAISS) [Phase 4]
- Output validation with secret detection [Phase 4]
- File watching for incremental indexing [Phase 4]
- Risk feature extraction [Phase 5]
- Risk prediction model [Phase 5]
- Review routing [Phase 5]
"""

from ai.embedding_manager import EmbeddingManager
from ai.feature_extractor import BehavioralFeatureExtractor, FeatureVector, feature_extractor
from ai.shap_explainer import SHAPExplainer, AnomalyExplanation, shap_explainer
from ai.synthetic_data_generator import SyntheticDataGenerator, synthetic_generator
from ai.model_registry import ModelRegistry, ModelMetadata, model_registry
from ai.model_trainer import EnsembleModelTrainer, AnomalyModelTrainer, ensemble_trainer
from ai.intent_classifier import IntentClassifier, IntentPrediction
from ai.risk_scorer import RiskScorer, RiskScore, risk_scorer

# Phase 4: AI Code Tools
from ai.hybrid_search import (
    HybridSearchEngine,
    BM25Index,
    CodeChunker,
    CodeChunk,
)
from ai.output_validator import (
    CodeReviewOutput,
    CodeSearchResult,
    DiffSummary,
    RepoSummary,
    SecretScanner,
    validate_output,
    safe_parse_llm_response,
    ReviewIssueCategory,
)
from ai.file_watcher import (
    FileWatcher,
    IncrementalIndexer,
    IndexVersion,
    WatchedHybridSearch,
)

# Phase 5: Code Review ML
from ai.risk_feature_extractor import (
    RiskFeatureExtractor,
    RiskFeatures,
    risk_feature_extractor,
)
from ai.risk_prediction_model import (
    RiskPredictionModel,
    RiskPrediction,
    get_risk_model,
)
from ai.review_router import (
    ReviewRouter,
    ReviewRouting,
    review_router,
)

__all__ = [
    # Embedding
    "EmbeddingManager",
    
    # Feature Extraction
    "BehavioralFeatureExtractor",
    "FeatureVector",
    "feature_extractor",
    
    # SHAP Explanations
    "SHAPExplainer",
    "AnomalyExplanation",
    "shap_explainer",
    
    # Synthetic Data
    "SyntheticDataGenerator",
    "synthetic_generator",
    
    # Model Registry
    "ModelRegistry",
    "ModelMetadata",
    "model_registry",
    
    # Model Training
    "EnsembleModelTrainer",
    "AnomalyModelTrainer",
    "ensemble_trainer",
    
    # Intent Classification
    "IntentClassifier",
    "IntentPrediction",
    
    # Risk Scoring
    "RiskScorer",
    "RiskScore",
    "risk_scorer",
    
    # Phase 4: Hybrid Search
    "HybridSearchEngine",
    "BM25Index",
    "CodeChunker",
    "CodeChunk",
    
    # Phase 4: Output Validation
    "CodeReviewOutput",
    "CodeSearchResult",
    "DiffSummary",
    "RepoSummary",
    "SecretScanner",
    "validate_output",
    "safe_parse_llm_response",
    "ReviewIssueCategory",
    
    # Phase 4: File Watching
    "FileWatcher",
    "IncrementalIndexer",
    "IndexVersion",
    "WatchedHybridSearch",
    
    # Phase 5: Risk Feature Extraction
    "RiskFeatureExtractor",
    "RiskFeatures",
    "risk_feature_extractor",
    
    # Phase 5: Risk Prediction Model
    "RiskPredictionModel",
    "RiskPrediction",
    "get_risk_model",
    
    # Phase 5: Review Routing
    "ReviewRouter",
    "ReviewRouting",
    "review_router",
]

