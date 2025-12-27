"""
ML Result Models

Models for ML inference results.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """Intent check result."""

    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    matched_patterns: List[str] = Field(default_factory=list)


class AnomalyResult(BaseModel):
    """Anomaly detection result."""

    is_anomalous: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    features: Optional[dict] = None


class RiskResult(BaseModel):
    """Risk assessment result."""

    score: float = Field(ge=0.0, le=1.0)
    level: str  # LOW, MEDIUM, HIGH
    factors: List[str] = Field(default_factory=list)


class EmbeddingResult(BaseModel):
    """Embedding search result."""

    file_path: str
    chunk_index: int
    similarity: float
    preview: str


class IntentPrediction(BaseModel):
    """Intent classification prediction."""

    intent: str
    confidence: float
    alternatives: List[Tuple[str, float]] = Field(default_factory=list)


class CodeReviewResult(BaseModel):
    """Code review result from AI."""

    summary: str
    issues: List[dict] = Field(default_factory=list)
    test_suggestions: List[str] = Field(default_factory=list)
    risk_level: str
    estimated_review_time_minutes: int = Field(ge=0)
