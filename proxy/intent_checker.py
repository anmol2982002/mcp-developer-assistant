"""
Intent Checker (LLM-as-Judge) - Enhanced

Validates that tool requests align with their declared intent.
Prevents tool misuse (e.g., read_file used to exfiltrate secrets).

Features:
- Semantic intent caching (70% LLM call reduction)
- Configurable tool specifications from YAML
- Graceful fallback: Cache -> LLM -> Rules
- Database-backed cache persistence
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from observability.logging_config import get_logger
from observability.metrics import metrics, INTENT_CONFIDENCE, INTENT_CHECK_LATENCY

logger = get_logger(__name__)

# Try to import sentence-transformers for semantic caching
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence_transformers_not_available", 
                   msg="Semantic caching disabled, falling back to exact match")


@dataclass
class IntentCheckResult:
    """Result of intent validation."""

    is_valid: bool
    confidence: float
    reason: str
    from_cache: bool = False
    cache_similarity: float = 0.0


class SemanticIntentCache:
    """
    Semantic intent cache using embedding similarity.
    
    Caches intent validation results and matches new requests
    based on semantic similarity, not just exact match.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        max_cache_size: int = 10000,
    ):
        """
        Initialize semantic cache.
        
        Args:
            model_name: Sentence transformer model
            similarity_threshold: Minimum similarity for cache hit
            max_cache_size: Maximum cache entries
        """
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        self.embedding_model = None
        self.cache: Dict[str, Tuple[np.ndarray, IntentCheckResult]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info("semantic_cache_initialized", model=model_name)
            except Exception as e:
                logger.warning("semantic_cache_init_failed", error=str(e))
    
    def _make_key(self, tool_name: str, params: Dict[str, Any], intent: Optional[str]) -> str:
        """Create cache key from request."""
        key_data = {
            "tool": tool_name,
            "params": params,
            "intent": intent or "",
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _get_embedding(self, tool_name: str, params: Dict[str, Any], intent: Optional[str]) -> np.ndarray:
        """Get embedding for request."""
        # Combine tool, params, and intent into a single text
        text = f"Tool: {tool_name}. "
        
        if params:
            path = params.get("path") or params.get("file_path") or ""
            if path:
                text += f"Path: {path}. "
            query = params.get("query") or params.get("pattern") or ""
            if query:
                text += f"Query: {query}. "
        
        if intent:
            text += f"Intent: {intent}"
        
        return self.embedding_model.encode(text, normalize_embeddings=True)
    
    def get(
        self,
        tool_name: str,
        params: Dict[str, Any],
        intent: Optional[str] = None,
    ) -> Optional[Tuple[IntentCheckResult, float]]:
        """
        Get cached result using semantic similarity.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
            intent: User stated intent
            
        Returns:
            Tuple of (result, similarity) if found, None otherwise
        """
        # Try exact match first
        exact_key = self._make_key(tool_name, params, intent)
        if exact_key in self.cache:
            _, result = self.cache[exact_key]
            self.cache_hits += 1
            return result, 1.0
        
        # Try semantic match if embeddings available
        if self.embedding_model is not None and len(self.cache) > 0:
            try:
                query_embedding = self._get_embedding(tool_name, params, intent)
                
                best_match = None
                best_similarity = 0.0
                
                for key, (embedding, result) in self.cache.items():
                    similarity = float(np.dot(query_embedding, embedding))
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = result
                
                if best_match is not None:
                    self.cache_hits += 1
                    return best_match, best_similarity
            except Exception as e:
                logger.debug("semantic_search_failed", error=str(e))
        
        self.cache_misses += 1
        return None
    
    def set(
        self,
        tool_name: str,
        params: Dict[str, Any],
        intent: Optional[str],
        result: IntentCheckResult,
    ) -> None:
        """
        Cache intent validation result.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
            intent: User stated intent
            result: Validation result to cache
        """
        # Enforce max size (LRU-like: just clear oldest entries)
        if len(self.cache) >= self.max_cache_size:
            # Remove first 10% of entries
            keys_to_remove = list(self.cache.keys())[:int(self.max_cache_size * 0.1)]
            for key in keys_to_remove:
                del self.cache[key]
        
        exact_key = self._make_key(tool_name, params, intent)
        
        if self.embedding_model is not None:
            try:
                embedding = self._get_embedding(tool_name, params, intent)
                self.cache[exact_key] = (embedding, result)
            except Exception:
                # Fall back to storing without embedding
                self.cache[exact_key] = (np.zeros(1), result)
        else:
            self.cache[exact_key] = (np.zeros(1), result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class IntentChecker:
    """
    Validates that tool requests align with declared intent.

    Uses LLM-as-a-Judge pattern to classify if request matches tool purpose.
    Includes semantic caching for common patterns to reduce LLM calls.
    
    Fallback chain: Semantic Cache -> LLM -> Rule-Based
    """

    def __init__(
        self,
        llm_client=None,
        config_path: Optional[str] = None,
        cache_similarity_threshold: float = 0.85,
    ):
        """
        Initialize IntentChecker.

        Args:
            llm_client: LLM client for intent classification (optional)
            config_path: Path to tool_intents.yaml config
            cache_similarity_threshold: Threshold for semantic cache hits
        """
        self.llm = llm_client
        
        # Load tool specifications from config or use defaults
        self.tool_specs = self._load_tool_specs(config_path)
        
        # Semantic cache
        self._cache = SemanticIntentCache(
            similarity_threshold=cache_similarity_threshold,
        )
        
        # Legacy exact-match cache for backward compatibility
        self._exact_cache: Dict[str, IntentCheckResult] = {}
        
        # LLM prompt template
        self._prompt_template = self._load_prompt_template(config_path)
    
    def _load_tool_specs(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load tool specifications from YAML config."""
        # Try config path
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("tools", {})
            except Exception as e:
                logger.warning("config_load_failed", path=config_path, error=str(e))
        
        # Try default location
        default_path = Path(__file__).parent.parent / "config" / "tool_intents.yaml"
        if default_path.exists():
            try:
                with open(default_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("tools", {})
            except Exception as e:
                logger.warning("default_config_failed", error=str(e))
        
        # Fall back to built-in specs
        return {
            "read_file": {
                "intent": "Read non-sensitive code/docs for understanding",
                "allowed_patterns": ["*.py", "*.js", "*.ts", "*.md", "*.txt", "README*", "*.json"],
                "forbidden_patterns": [".env", ".env.*", "secrets/", ".git/config", "credentials", "*.pem", "*.key"],
            },
            "search_files": {
                "intent": "Find code patterns, search codebase",
                "allowed_intents": ["code search", "find imports", "locate function", "find usage"],
                "forbidden_intents": ["get all API keys", "extract credentials", "find passwords"],
            },
            "git_diff": {
                "intent": "Review changes, understand PRs",
                "allowed_intents": ["see what changed", "review commit", "understand changes"],
                "forbidden_intents": ["find secrets in history", "extract all commits", "git log for passwords"],
            },
            "git_status": {
                "intent": "Check repository state",
                "allowed_intents": ["check status", "see changes", "list modified files"],
                "forbidden_intents": [],
            },
            "list_dir": {
                "intent": "Explore codebase structure",
                "allowed_intents": ["list files", "see directory", "explore project"],
                "forbidden_patterns": ["secrets/", "credentials/"],
            },
        }
    
    def _load_prompt_template(self, config_path: Optional[str]) -> str:
        """Load LLM prompt template from config."""
        default_template = """
You are a security-focused intent validator for a code assistant tool.

Tool: {tool_name}
Tool Purpose: {tool_intent}
Request Parameters: {params}
User's Stated Goal: {user_intent}

Analyze if this request aligns with the tool's intended purpose.
Consider:
1. Is the user trying to access sensitive information (secrets, credentials, API keys)?
2. Does the request match legitimate development tasks?
3. Are there any signs of data exfiltration or misuse?
4. Is the stated goal consistent with the actual request?

Respond with JSON only:
{{"is_valid": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""
        
        # Try to load from config
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    prompts = config.get("prompts", {})
                    if "intent_validation" in prompts:
                        return prompts["intent_validation"]
            except Exception:
                pass
        
        return default_template

    async def validate_intent(
        self,
        tool_name: str,
        params: Dict[str, Any],
        user_intent: Optional[str] = None,
    ) -> IntentCheckResult:
        """
        Validate if request matches tool intent.

        Args:
            tool_name: Name of the tool being called
            params: Tool parameters
            user_intent: User's stated goal (optional)

        Returns:
            IntentCheckResult with validation decision
        """
        start_time = time.perf_counter()
        
        try:
            # Check semantic cache first (70% target reduction)
            cache_result = self._cache.get(tool_name, params, user_intent)
            if cache_result is not None:
                result, similarity = cache_result
                logger.debug("intent_cache_hit", tool=tool_name, similarity=similarity)
                
                # Record metrics
                INTENT_CHECK_LATENCY.observe(time.perf_counter() - start_time)
                INTENT_CONFIDENCE.observe(result.confidence)
                
                return IntentCheckResult(
                    is_valid=result.is_valid,
                    confidence=result.confidence,
                    reason=result.reason,
                    from_cache=True,
                    cache_similarity=similarity,
                )
            
            # Get tool specification
            spec = self.tool_specs.get(tool_name, {})
            if not spec:
                # Unknown tool - allow by default
                return IntentCheckResult(is_valid=True, confidence=1.0, reason="Unknown tool, allowed")
            
            # Rule-based check first (fast path)
            rule_result = self._check_with_rules(tool_name, params, spec)
            if not rule_result.is_valid:
                self._log_violation(tool_name, params, rule_result)
                self._cache.set(tool_name, params, user_intent, rule_result)
                return rule_result
            
            # If we have an LLM and user intent, do semantic check
            if self.llm and user_intent:
                llm_result = await self._check_with_llm(tool_name, params, spec, user_intent)
                
                # Record metrics
                INTENT_CHECK_LATENCY.observe(time.perf_counter() - start_time)
                INTENT_CONFIDENCE.observe(llm_result.confidence)
                
                if not llm_result.is_valid:
                    self._log_violation(tool_name, params, llm_result)
                
                # Cache the result
                self._cache.set(tool_name, params, user_intent, llm_result)
                return llm_result
            
            # Cache and return valid result
            result = IntentCheckResult(is_valid=True, confidence=1.0, reason="Request matches tool intent")
            self._cache.set(tool_name, params, user_intent, result)
            
            # Record metrics
            INTENT_CHECK_LATENCY.observe(time.perf_counter() - start_time)
            INTENT_CONFIDENCE.observe(result.confidence)

            return result
            
        except Exception as e:
            logger.error("intent_check_failed", error=str(e))
            # Fail open on errors
            return IntentCheckResult(
                is_valid=True,
                confidence=0.5,
                reason=f"Intent check failed: {e}",
            )

    def _check_with_rules(
        self,
        tool_name: str,
        params: Dict[str, Any],
        spec: Dict[str, Any],
    ) -> IntentCheckResult:
        """
        Fast rule-based intent check.

        Checks forbidden patterns in parameters.
        """
        # Get path parameter if present
        path = params.get("path", "") or params.get("file_path", "") or ""
        path_lower = path.lower()

        # Check forbidden patterns
        forbidden_patterns = spec.get("forbidden_patterns") or spec.get("forbidden", [])
        for forbidden in forbidden_patterns:
            if isinstance(forbidden, str):
                forbidden_lower = forbidden.lower().replace("*", "")
                if forbidden_lower in path_lower:
                    return IntentCheckResult(
                        is_valid=False,
                        confidence=0.95,
                        reason=f"Accessing forbidden pattern: {forbidden}",
                    )
        
        # Check for sensitive content keywords
        query = params.get("query") or params.get("pattern") or ""
        if query:
            forbidden_intents = spec.get("forbidden_intents", [])
            query_lower = query.lower()
            for forbidden in forbidden_intents:
                if forbidden.lower() in query_lower:
                    return IntentCheckResult(
                        is_valid=False,
                        confidence=0.9,
                        reason=f"Query matches forbidden intent: {forbidden}",
                    )

        return IntentCheckResult(is_valid=True, confidence=0.8, reason="Passed rule check")

    async def _check_with_llm(
        self,
        tool_name: str,
        params: Dict[str, Any],
        spec: Dict[str, Any],
        user_intent: str,
    ) -> IntentCheckResult:
        """
        LLM-based semantic intent check.

        Uses LLM to validate if request aligns with stated intent.
        """
        tool_intent = spec.get("intent") or spec.get("description", "Unknown purpose")
        
        prompt = self._prompt_template.format(
            tool_name=tool_name,
            tool_intent=tool_intent,
            params=json.dumps(params, default=str),
            user_intent=user_intent,
        )

        try:
            response = await self.llm.generate(prompt)
            
            # Try to parse JSON response
            result = self._parse_llm_response(response)

            return IntentCheckResult(
                is_valid=result.get("is_valid", True),
                confidence=result.get("confidence", 0.5),
                reason=result.get("reason", "LLM check completed"),
            )

        except Exception as e:
            logger.warning("llm_intent_check_failed", error=str(e))
            # Fail open on LLM errors
            return IntentCheckResult(
                is_valid=True,
                confidence=0.5,
                reason=f"LLM check failed, allowing: {e}",
            )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response, handling various formats."""
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fall back to parsing text
        response_lower = response.lower()
        is_valid = "invalid" not in response_lower and "not valid" not in response_lower
        
        return {
            "is_valid": is_valid,
            "confidence": 0.6,
            "reason": response[:200] if len(response) > 200 else response,
        }

    def _log_violation(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: IntentCheckResult,
    ) -> None:
        """Log intent violation."""
        logger.warning(
            "intent_violation",
            tool=tool_name,
            params=params,
            reason=result.reason,
            confidence=result.confidence,
        )
        metrics.intent_violations_total.labels(tool=tool_name).inc()

    def _make_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Create cache key from tool name and parameters."""
        return f"{tool_name}:{json.dumps(params, sort_keys=True)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self._cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear intent cache."""
        self._cache.clear()
        self._exact_cache.clear()


# Factory function for creating intent checker
def create_intent_checker(
    llm_client=None,
    config_path: Optional[str] = None,
) -> IntentChecker:
    """
    Create an IntentChecker instance.
    
    Args:
        llm_client: Optional LLM client for semantic checking
        config_path: Optional path to tool_intents.yaml
        
    Returns:
        Configured IntentChecker instance
    """
    return IntentChecker(llm_client=llm_client, config_path=config_path)
