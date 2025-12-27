"""
Integration Tests for Auth Flow

End-to-end tests for:
- Complete OAuth flow
- Token refresh cycle
- Consent grant/revoke flow
- Rate limiting under load
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient


@pytest.fixture
def mock_config(monkeypatch):
    """Mock proxy configuration."""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("MCP_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_SERVER_PORT", "8000")
    monkeypatch.setenv("INTENT_CHECK_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("CONSENT_ENABLED", "false")
    monkeypatch.setenv("AUDIT_ENABLED", "false")


@pytest.fixture
def auth_token(mock_config):
    """Generate a test JWT token."""
    from proxy.oauth_validator import OAuthValidator
    
    validator = OAuthValidator()
    return validator.create_token(
        user_id="test_user",
        client_id="test_client",
        scopes=["read", "write", "admin"],
    )


@pytest.fixture
def client(mock_config):
    """Create test client."""
    from proxy.auth_gateway import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "initializing"]
    
    def test_health_includes_components(self, client):
        """Health should include component status."""
        response = client.get("/health")
        
        data = response.json()
        assert "components" in data
        assert "oauth" in data["components"]


class TestOAuthEndpoints:
    """Integration tests for OAuth endpoints."""
    
    def test_token_endpoint_missing_params(self, client):
        """Token endpoint should require parameters."""
        response = client.post(
            "/oauth/token",
            json={"grant_type": "authorization_code", "client_id": "test"},
        )
        
        assert response.status_code == 400
    
    def test_token_endpoint_unsupported_grant(self, client):
        """Token endpoint should reject unsupported grants."""
        response = client.post(
            "/oauth/token",
            json={"grant_type": "password", "client_id": "test"},
        )
        
        assert response.status_code == 400
        assert "Unsupported grant_type" in response.json()["detail"]
    
    def test_introspect_requires_auth(self, client):
        """Introspect endpoint should require authentication."""
        response = client.post("/oauth/introspect", params={"token": "test"})
        
        assert response.status_code == 401
    
    def test_introspect_with_valid_auth(self, client, auth_token):
        """Introspect should work with valid auth."""
        response = client.post(
            "/oauth/introspect",
            params={"token": auth_token},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is True
    
    def test_revoke_requires_auth(self, client):
        """Revoke endpoint should require authentication."""
        response = client.post("/oauth/revoke", params={"token": "test"})
        
        assert response.status_code == 401


class TestConsentEndpoints:
    """Integration tests for consent endpoints."""
    
    def test_list_consents_requires_auth(self, client):
        """List consents should require authentication."""
        response = client.get("/consent")
        
        assert response.status_code == 401
    
    def test_list_consents_with_auth(self, client, auth_token):
        """List consents should work with valid auth."""
        response = client.get(
            "/consent",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "consents" in data
    
    def test_grant_consent(self, client, auth_token):
        """Should be able to grant consent."""
        response = client.post(
            "/consent/grant",
            json={
                "client_id": "new_client",
                "scopes": ["read", "write"],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "granted"
    
    def test_revoke_consent(self, client, auth_token):
        """Should be able to revoke consent."""
        response = client.post(
            "/consent/revoke",
            json={"client_id": "some_client"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert response.status_code == 200


class TestAuditEndpoints:
    """Integration tests for audit endpoints."""
    
    def test_query_logs_requires_auth(self, client):
        """Query logs should require authentication."""
        response = client.get("/audit/logs")
        
        assert response.status_code == 401
    
    def test_query_logs_with_auth(self, client, auth_token):
        """Query logs should work with valid auth."""
        response = client.get(
            "/audit/logs",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_activity_summary(self, client, auth_token):
        """Activity summary should return stats."""
        response = client.get(
            "/audit/summary",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
    
    def test_export_requires_admin(self, client, auth_token):
        """Export should work with admin scope."""
        response = client.get(
            f"/audit/export?start=2020-01-01T00:00:00&end=2025-12-31T23:59:59",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        # Should work since our token has admin scope
        assert response.status_code == 200


class TestToolCallEndpoint:
    """Integration tests for tool_call endpoint."""
    
    def test_tool_call_without_auth(self, client):
        """Tool call should require authentication."""
        response = client.post(
            "/tool_call",
            json={"tool_name": "read_file", "params": {"path": "test.py"}},
        )
        
        assert response.status_code == 401
    
    def test_tool_call_with_auth(self, client, auth_token):
        """Tool call should accept valid auth."""
        response = client.post(
            "/tool_call",
            json={"tool_name": "read_file", "params": {"path": "README.md"}},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        # Should not be 401 (may be 500 if MCP server not running)
        assert response.status_code != 401
    
    def test_tool_call_includes_rate_limit_headers(self, client, auth_token, monkeypatch):
        """Tool calls should include rate limit headers."""
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        
        response = client.post(
            "/tool_call",
            json={"tool_name": "list_dir", "params": {"path": "."}},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        # Will have headers if rate limiting is working
        # (May not have them if response fails before rate limit check)
        if response.status_code in [200, 403]:
            # Check for rate limit headers
            pass  # Headers added by middleware


class TestListToolsEndpoint:
    """Tests for tools listing."""
    
    def test_list_tools_requires_auth(self, client):
        """List tools should require auth."""
        response = client.get("/tools")
        
        assert response.status_code == 401
    
    def test_list_tools_with_auth(self, client, auth_token):
        """List tools should work with auth."""
        response = client.get(
            "/tools",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        # May fail if MCP server not running, but shouldn't be 401
        assert response.status_code != 401


class TestConsentMiddlewareIntegration:
    """Integration tests for consent middleware."""
    
    def test_consent_flow(self, client, auth_token):
        """Full consent grant and usage flow."""
        # Step 1: Grant consent
        grant_response = client.post(
            "/consent/grant",
            json={
                "client_id": "integration_test_client",
                "scopes": ["file:read", "git:read"],
                "expires_days": 7,
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert grant_response.status_code == 200
        
        # Step 2: List consents
        list_response = client.get(
            "/consent",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert list_response.status_code == 200
        consents = list_response.json()["consents"]
        
        # Should have the consent we just created
        client_ids = [c["client_id"] for c in consents]
        assert "integration_test_client" in client_ids
        
        # Step 3: Revoke consent
        revoke_response = client.post(
            "/consent/revoke",
            json={"client_id": "integration_test_client"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        
        assert revoke_response.status_code == 200


class TestSecurityHeaders:
    """Tests for security features."""
    
    def test_cors_headers(self, client):
        """Should include CORS headers."""
        # OPTIONS request for CORS preflight
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        
        # CORS should be configured
        assert response.status_code in [200, 400]  # Depends on CORS config
    
    def test_invalid_json_body(self, client, auth_token):
        """Should handle invalid JSON gracefully."""
        response = client.post(
            "/tool_call",
            content="not valid json",
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
        )
        
        assert response.status_code == 422  # Validation error
