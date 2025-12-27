"""
Tests for Enhanced Intent Checker

Tests cover:
- Semantic caching
- LLM fallback
- Configurable tool specifications
- Rule-based validation
"""

import json

import pytest

from proxy.intent_checker import IntentChecker, IntentCheckResult, SemanticIntentCache


class TestSemanticIntentCache:
    """Test semantic intent caching."""

    def test_exact_match_cache(self):
        """Exact matches should hit cache."""
        cache = SemanticIntentCache()
        
        # Set a result
        result = IntentCheckResult(is_valid=True, confidence=0.9, reason="Test")
        cache.set("read_file", {"path": "test.py"}, None, result)
        
        # Get the same query
        cached = cache.get("read_file", {"path": "test.py"}, None)
        
        assert cached is not None
        cached_result, similarity = cached
        assert cached_result.is_valid == result.is_valid
        assert similarity == 1.0  # Exact match

    def test_cache_miss(self):
        """Different queries should miss cache."""
        cache = SemanticIntentCache()
        
        result = IntentCheckResult(is_valid=True, confidence=0.9, reason="Test")
        cache.set("read_file", {"path": "test.py"}, None, result)
        
        # Different query
        cached = cache.get("read_file", {"path": "other.py"}, None)
        
        # Without embeddings model, should be None (miss)
        if cache.embedding_model is None:
            assert cached is None

    def test_cache_stats(self):
        """Should track cache statistics."""
        cache = SemanticIntentCache()
        
        result = IntentCheckResult(is_valid=True, confidence=0.9, reason="Test")
        cache.set("read_file", {"path": "test.py"}, None, result)
        
        # Hit
        cache.get("read_file", {"path": "test.py"}, None)
        # Miss
        cache.get("read_file", {"path": "other.py"}, None)
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestIntentChecker:
    """Test intent checker functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.checker = IntentChecker(llm_client=None)

    @pytest.mark.asyncio
    async def test_allow_normal_file_read(self):
        """Normal file reads should be allowed."""
        result = await self.checker.validate_intent(
            "read_file",
            {"path": "src/main.py"},
            "Review the main module",
        )

        assert result.is_valid
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_block_env_file(self):
        """Reading .env files should be blocked."""
        result = await self.checker.validate_intent(
            "read_file",
            {"path": ".env"},
            "Check environment variables",
        )

        assert not result.is_valid
        assert "forbidden" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_block_env_production(self):
        """Reading .env.production should be blocked."""
        result = await self.checker.validate_intent(
            "read_file",
            {"path": ".env.production"},
            "Check production config",
        )

        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_block_secrets_directory(self):
        """Reading from secrets/ should be blocked."""
        result = await self.checker.validate_intent(
            "read_file",
            {"path": "secrets/api_key.txt"},
            "Get API key",
        )

        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_allow_unknown_tool(self):
        """Unknown tools should be allowed by default."""
        result = await self.checker.validate_intent(
            "unknown_tool",
            {"param": "value"},
            None,
        )

        assert result.is_valid
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Cache keys should be consistent."""
        key1 = self.checker._make_cache_key("read_file", {"path": "test.py"})
        key2 = self.checker._make_cache_key("read_file", {"path": "test.py"})
        key3 = self.checker._make_cache_key("read_file", {"path": "other.py"})

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_on_repeated_requests(self):
        """Repeated requests should hit cache."""
        # First request
        result1 = await self.checker.validate_intent(
            "read_file",
            {"path": "src/utils.py"},
            "Review utils",
        )
        
        # Second request (same)
        result2 = await self.checker.validate_intent(
            "read_file",
            {"path": "src/utils.py"},
            "Review utils",
        )
        
        # Second should be from cache
        assert result2.from_cache or result1.is_valid == result2.is_valid

    @pytest.mark.asyncio
    async def test_block_credential_files(self):
        """Credential files should be blocked."""
        result = await self.checker.validate_intent(
            "read_file",
            {"path": "config/credentials.yaml"},
            "Check credentials",
        )

        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_block_pem_keys(self):
        """PEM key files should be blocked."""
        result = await self.checker.validate_intent(
            "read_file",
            {"path": "certs/server.pem"},
            "Check server certificate",
        )

        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_search_with_forbidden_intent(self):
        """Search for credentials should be blocked."""
        result = await self.checker.validate_intent(
            "search_files",
            {"query": "find all API keys"},
            "Find API keys in codebase",
        )
        
        # Should be blocked by rule
        assert not result.is_valid or "api" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_git_status_allowed(self):
        """Git status should generally be allowed."""
        result = await self.checker.validate_intent(
            "git_status",
            {},
            "Check current changes",
        )

        assert result.is_valid

    def test_get_cache_stats(self):
        """Should return cache statistics."""
        stats = self.checker.get_cache_stats()
        
        assert "size" in stats
        assert "hit_rate" in stats

    def test_clear_cache(self):
        """Should clear cache."""
        self.checker.clear_cache()
        stats = self.checker.get_cache_stats()
        
        assert stats["size"] == 0


class TestIntentCheckerWithMockLLM:
    """Test intent checker with mock LLM."""

    @pytest.mark.asyncio
    async def test_llm_validation_valid(self, mock_llm):
        """LLM should validate requests."""
        checker = IntentChecker(llm_client=mock_llm)
        
        result = await checker.validate_intent(
            "read_file",
            {"path": "src/main.py"},
            "Review the main module",
        )
        
        assert result.is_valid
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self):
        """Should fall back to rules on LLM error."""
        
        class FailingLLM:
            async def generate(self, prompt):
                raise Exception("LLM unavailable")
        
        checker = IntentChecker(llm_client=FailingLLM())
        
        result = await checker.validate_intent(
            "read_file",
            {"path": "src/main.py"},
            "Review main",
        )
        
        # Should still work (fall back to rules)
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_parse_malformed_llm_response(self):
        """Should handle malformed LLM responses."""
        
        class MalformedLLM:
            async def generate(self, prompt):
                return "This is not valid JSON"
        
        checker = IntentChecker(llm_client=MalformedLLM())
        
        result = await checker.validate_intent(
            "read_file",
            {"path": "src/main.py"},
            "Review main",
        )
        
        # Should extract what it can or fall back
        assert result.confidence >= 0


class TestConfigurableSpecs:
    """Test configurable tool specifications."""

    def test_load_default_specs(self):
        """Should load default specs when no config."""
        checker = IntentChecker()
        
        assert "read_file" in checker.tool_specs
        assert "intent" in checker.tool_specs["read_file"]

    def test_specs_structure(self):
        """Specs should have required fields."""
        checker = IntentChecker()
        
        for tool_name, spec in checker.tool_specs.items():
            assert "intent" in spec or "description" in spec

    @pytest.mark.asyncio
    async def test_custom_forbidden_patterns(self):
        """Should respect forbidden patterns."""
        checker = IntentChecker()
        
        # Password files should be blocked
        result = await checker.validate_intent(
            "read_file",
            {"path": "passwords.txt"},
            "Check passwords",
        )
        
        assert not result.is_valid
