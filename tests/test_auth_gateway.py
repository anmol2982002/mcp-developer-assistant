"""
Tests for Auth Gateway

Integration tests for the proxy authorization gateway.
"""

import pytest
from fastapi.testclient import TestClient

# Import app after mocking dependencies


@pytest.fixture
def mock_config(monkeypatch):
    """Mock proxy configuration."""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("MCP_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_SERVER_PORT", "8000")
    monkeypatch.setenv("INTENT_CHECK_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")


@pytest.fixture
def test_token():
    """Generate a test JWT token."""
    from proxy.oauth_validator import OAuthValidator

    validator = OAuthValidator()
    return validator.create_token(user_id="test_user", client_id="test_client")


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self, mock_config):
        """Health endpoint should return 200."""
        from proxy.auth_gateway import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "initializing"]

    def test_health_check_includes_version(self, mock_config):
        """Health endpoint should include version."""
        from proxy.auth_gateway import app

        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        assert "version" in data


class TestToolCallEndpoint:
    """Tests for tool_call endpoint."""

    def test_tool_call_without_auth_returns_401(self, mock_config):
        """Tool call without auth should return 401."""
        from proxy.auth_gateway import app

        client = TestClient(app)
        response = client.post(
            "/tool_call",
            json={"tool_name": "read_file", "params": {"path": "test.py"}},
        )

        assert response.status_code == 401

    def test_tool_call_with_invalid_token_returns_401(self, mock_config):
        """Tool call with invalid token should return 401."""
        from proxy.auth_gateway import app

        client = TestClient(app)
        response = client.post(
            "/tool_call",
            json={"tool_name": "read_file", "params": {"path": "test.py"}},
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401

    def test_tool_call_with_valid_token(self, mock_config, test_token):
        """Tool call with valid token should be processed."""
        from proxy.auth_gateway import app

        client = TestClient(app)
        response = client.post(
            "/tool_call",
            json={"tool_name": "read_file", "params": {"path": "README.md"}},
            headers={"Authorization": f"Bearer {test_token}"},
        )

        # Should not return 401 (may return 500 if MCP server not running)
        assert response.status_code != 401


class TestOAuthValidator:
    """Tests for OAuth token validation."""

    def test_create_and_validate_token(self, mock_config):
        """Should create and validate tokens successfully."""
        from proxy.oauth_validator import OAuthValidator

        validator = OAuthValidator()
        token = validator.create_token(user_id="test_user")

        # Validate synchronously for testing
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            user = loop.run_until_complete(
                validator.validate_token(f"Bearer {token}")
            )
            assert user.id == "test_user"
        finally:
            loop.close()

    def test_token_contains_claims(self, mock_config):
        """Token should contain proper claims."""
        from jose import jwt

        from proxy.oauth_validator import OAuthValidator

        validator = OAuthValidator()
        token = validator.create_token(
            user_id="test_user",
            client_id="test_client",
            scopes=["read", "write"],
        )

        # Decode without verification to check claims
        claims = jwt.get_unverified_claims(token)
        assert claims["sub"] == "test_user"
        assert claims["client_id"] == "test_client"
        assert claims["scopes"] == ["read", "write"]


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_rate_limiter_allows_initial_requests(self, mock_config):
        """Rate limiter should allow initial requests."""
        from proxy.rate_limiter import RateLimiter

        limiter = RateLimiter()
        # First request should not be rate limited
        is_limited = limiter.is_rate_limited("user1", "read_file")

        assert is_limited is False

    def test_rate_limiter_returns_retry_after(self, mock_config):
        """Rate limiter should return retry after."""
        from proxy.rate_limiter import RateLimiter

        limiter = RateLimiter()
        retry_after = limiter.get_retry_after("user1", "read_file")

        assert isinstance(retry_after, float)
        assert retry_after >= 0
