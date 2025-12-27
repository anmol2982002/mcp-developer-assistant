"""
Intent Classifier

ML-based intent classification for tool requests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IntentPrediction:
    """Intent classification result."""

    intent: str
    confidence: float
    alternatives: List[tuple]  # [(intent, confidence), ...]


class IntentClassifier:
    """
    Classify intent of tool requests.

    Uses embedding similarity to match requests to known intents.
    """

    # Known intent categories
    INTENT_CATEGORIES = {
        "code_review": [
            "review this code",
            "check for issues",
            "analyze changes",
            "review PR",
            "code quality check",
        ],
        "code_navigation": [
            "find function",
            "locate class",
            "search for usage",
            "where is this defined",
            "find imports",
        ],
        "code_understanding": [
            "explain this code",
            "what does this do",
            "understand the logic",
            "how does this work",
            "summarize this file",
        ],
        "debugging": [
            "find bug",
            "debug this",
            "why is this failing",
            "trace the error",
            "find the issue",
        ],
        "documentation": [
            "document this",
            "add comments",
            "generate docstring",
            "explain the API",
            "write readme",
        ],
    }

    def __init__(self, embedding_model=None):
        """
        Initialize classifier.

        Args:
            embedding_model: Sentence transformer model
        """
        self.embedding_model = embedding_model
        self.intent_embeddings: Dict[str, np.ndarray] = {}

        if embedding_model:
            self._build_intent_embeddings()

    def _build_intent_embeddings(self):
        """Pre-compute embeddings for known intents."""
        for intent, examples in self.INTENT_CATEGORIES.items():
            # Average embedding of examples
            embeddings = self.embedding_model.encode(examples)
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)

        logger.info("intent_embeddings_built", count=len(self.intent_embeddings))

    def classify(self, query: str) -> IntentPrediction:
        """
        Classify intent of a query.

        Args:
            query: User query or request description

        Returns:
            IntentPrediction with classification
        """
        if not self.embedding_model or not self.intent_embeddings:
            # Fallback to keyword matching
            return self._keyword_classify(query)

        # Embed query
        query_embedding = self.embedding_model.encode(query)

        # Calculate similarities
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            similarity = np.dot(query_embedding, intent_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)
            )
            similarities[intent] = similarity

        # Rank by similarity
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        best_intent, best_score = ranked[0]
        alternatives = ranked[1:4]  # Top 3 alternatives

        logger.debug("intent_classified", intent=best_intent, confidence=best_score)

        return IntentPrediction(
            intent=best_intent,
            confidence=float(best_score),
            alternatives=[(i, float(s)) for i, s in alternatives],
        )

    def _keyword_classify(self, query: str) -> IntentPrediction:
        """Simple keyword-based classification fallback."""
        query_lower = query.lower()

        keyword_map = {
            "code_review": ["review", "check", "quality", "issue", "problem"],
            "code_navigation": ["find", "locate", "search", "where", "usage"],
            "code_understanding": ["explain", "what", "how", "understand", "why"],
            "debugging": ["bug", "debug", "error", "fail", "fix"],
            "documentation": ["document", "comment", "docstring", "readme", "api"],
        }

        scores = {}
        for intent, keywords in keyword_map.items():
            score = sum(1 for k in keywords if k in query_lower)
            scores[intent] = score / len(keywords)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if ranked[0][1] > 0:
            return IntentPrediction(
                intent=ranked[0][0],
                confidence=ranked[0][1],
                alternatives=ranked[1:4],
            )

        return IntentPrediction(
            intent="unknown",
            confidence=0.0,
            alternatives=[],
        )
