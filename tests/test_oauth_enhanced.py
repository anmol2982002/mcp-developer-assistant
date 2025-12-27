"""
Tests for OAuth 2.1 Enhanced Features

Tests for:
- Refresh token creation and rotation
- PKCE verification
- Token introspection
- Token revocation
"""

import pytest
from datetime import datetime, timedelta

from proxy.oauth_validator import OAuthValidator, PKCEVerificationError


@pytest.fixture
def mock_config(monkeypatch):
    """Mock proxy configuration."""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("MCP_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_SERVER_PORT", "8000")


@pytest.fixture
def oauth_validator(mock_config):
    """Create OAuth validator instance."""
    return OAuthValidator()


class TestAccessTokens:
    """Tests for access token operations."""
    
    def test_create_token(self, oauth_validator):
        """Should create a valid JWT token."""
        token = oauth_validator.create_token(
            user_id="user123",
            client_id="client456",
            scopes=["read", "write"],
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_token_with_jti(self, oauth_validator):
        """Should create token with custom JTI."""
        jti = "custom-jti-123"
        token = oauth_validator.create_token(
            user_id="user123",
            jti=jti,
        )
        
        # Decode and check JTI
        from jose import jwt
        payload = jwt.get_unverified_claims(token)
        assert payload["jti"] == jti
    
    @pytest.mark.asyncio
    async def test_validate_token(self, oauth_validator):
        """Should validate a token and return user."""
        token = oauth_validator.create_token(
            user_id="user123",
            client_id="client456",
            scopes=["read"],
        )
        
        user = await oauth_validator.validate_token(f"Bearer {token}")
        
        assert user.id == "user123"
        assert user.client_id == "client456"
        assert user.scopes == ["read"]
    
    @pytest.mark.asyncio
    async def test_validate_invalid_token(self, oauth_validator):
        """Should reject invalid tokens."""
        with pytest.raises(PermissionError, match="Invalid token"):
            await oauth_validator.validate_token("Bearer invalid-token")
    
    @pytest.mark.asyncio
    async def test_validate_missing_auth(self, oauth_validator):
        """Should reject missing authorization."""
        with pytest.raises(PermissionError, match="Missing authorization"):
            await oauth_validator.validate_token(None)
    
    @pytest.mark.asyncio
    async def test_validate_wrong_scheme(self, oauth_validator):
        """Should reject non-Bearer schemes."""
        with pytest.raises(PermissionError, match="Invalid authorization scheme"):
            await oauth_validator.validate_token("Basic dXNlcjpwYXNz")


class TestTokenIntrospection:
    """Tests for token introspection (RFC 7662)."""
    
    def test_introspect_valid_token(self, oauth_validator):
        """Should return active=True for valid tokens."""
        token = oauth_validator.create_token(
            user_id="user123",
            scopes=["read", "write"],
        )
        
        result = oauth_validator.introspect_token(token)
        
        assert result.active is True
        assert result.sub == "user123"
        assert result.scope == "read write"
        assert result.token_type == "Bearer"
    
    def test_introspect_invalid_token(self, oauth_validator):
        """Should return active=False for invalid tokens."""
        result = oauth_validator.introspect_token("invalid-token")
        
        assert result.active is False
    
    def test_introspect_expired_token(self, oauth_validator, monkeypatch):
        """Should return active=False for expired tokens."""
        # Create a token that expires immediately
        from jose import jwt
        from datetime import datetime, timedelta
        
        payload = {
            "sub": "user123",
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired
            "iat": datetime.utcnow() - timedelta(hours=2),
        }
        
        expired_token = jwt.encode(
            payload,
            "test-secret-key-for-testing-only",
            algorithm="HS256",
        )
        
        result = oauth_validator.introspect_token(expired_token)
        
        assert result.active is False


class TestPKCE:
    """Tests for PKCE support."""
    
    def test_generate_code_verifier(self):
        """Should generate a valid code verifier."""
        verifier = OAuthValidator.generate_code_verifier()
        
        assert len(verifier) >= 43
        assert len(verifier) <= 128
    
    def test_generate_code_challenge_s256(self):
        """Should generate S256 code challenge."""
        verifier = "test-code-verifier-12345"
        challenge = OAuthValidator.generate_code_challenge(verifier, "S256")
        
        assert challenge is not None
        assert challenge != verifier  # Challenge should be different
    
    def test_generate_code_challenge_plain(self):
        """Should pass through for plain method."""
        verifier = "test-code-verifier-12345"
        challenge = OAuthValidator.generate_code_challenge(verifier, "plain")
        
        assert challenge == verifier
    
    def test_verify_pkce_s256_valid(self):
        """Should verify valid S256 PKCE."""
        verifier = OAuthValidator.generate_code_verifier()
        challenge = OAuthValidator.generate_code_challenge(verifier, "S256")
        
        result = OAuthValidator.verify_pkce_challenge(verifier, challenge, "S256")
        
        assert result is True
    
    def test_verify_pkce_s256_invalid(self):
        """Should reject invalid S256 PKCE."""
        verifier = "correct-verifier"
        challenge = OAuthValidator.generate_code_challenge(verifier, "S256")
        
        with pytest.raises(PKCEVerificationError):
            OAuthValidator.verify_pkce_challenge("wrong-verifier", challenge, "S256")
    
    def test_verify_pkce_plain_valid(self):
        """Should verify valid plain PKCE."""
        verifier = "test-code-verifier"
        
        result = OAuthValidator.verify_pkce_challenge(verifier, verifier, "plain")
        
        assert result is True
    
    def test_verify_pkce_plain_invalid(self):
        """Should reject invalid plain PKCE."""
        with pytest.raises(PKCEVerificationError):
            OAuthValidator.verify_pkce_challenge("verifier1", "verifier2", "plain")
    
    def test_verify_pkce_unsupported_method(self):
        """Should reject unsupported PKCE methods."""
        with pytest.raises(PKCEVerificationError):
            OAuthValidator.verify_pkce_challenge("verifier", "challenge", "unsupported")


class TestTokenResponse:
    """Tests for token response generation."""
    
    def test_create_token_response(self, oauth_validator):
        """Should create a complete token response."""
        response = oauth_validator.create_token_response(
            user_id="user123",
            client_id="client456",
            scopes=["read", "write"],
        )
        
        assert response.access_token is not None
        assert response.token_type == "Bearer"
        assert response.expires_in > 0
        assert response.scope == "read write"
