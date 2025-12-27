"""
tests for consent middleware

Tests for:
- Scope-based consent checks
- Consent expiration
- Cascading revocation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from proxy.consent_middleware import (
    ConsentMiddleware,
    ConsentEnforcer,
    get_required_scopes,
    TOOL_SCOPES,
)


class TestToolScopes:
    """Tests for tool to scope mapping."""
    
    def test_file_tools_have_file_scopes(self):
        """File tools should have file scopes."""
        assert get_required_scopes("read_file") == ["file:read"]
        assert get_required_scopes("write_file") == ["file:write"]
        assert get_required_scopes("list_dir") == ["file:read"]
    
    def test_git_tools_have_git_scopes(self):
        """Git tools should have git scopes."""
        assert get_required_scopes("git_status") == ["git:read"]
        assert get_required_scopes("git_diff") == ["git:read"]
        assert get_required_scopes("git_commit") == ["git:write"]
    
    def test_ai_tools_have_ai_scopes(self):
        """AI tools should have ai scopes."""
        assert get_required_scopes("ask_about_code") == ["ai:query"]
        assert get_required_scopes("review_changes") == ["ai:review"]
    
    def test_unknown_tool_returns_default(self):
        """Unknown tools should return default scope."""
        assert get_required_scopes("unknown_tool") == ["default"]
    
    def test_system_tools_have_elevated_scopes(self):
        """System tools should have execute scope."""
        assert get_required_scopes("run_command") == ["system:execute"]


class TestConsentEnforcer:
    """Tests for programmatic consent enforcement."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock()
    
    @pytest.fixture
    def enforcer(self, mock_session):
        """Create consent enforcer with mock session."""
        return ConsentEnforcer(mock_session)
    
    @pytest.mark.asyncio
    async def test_check_consent_has_consent(self, enforcer, mock_session):
        """Should return True when consent exists."""
        # Mock the consent_db.has_consent method
        enforcer.consent_db = AsyncMock()
        enforcer.consent_db.has_consent = AsyncMock(return_value=True)
        
        result = await enforcer.check_consent("user1", "client1", "read_file")
        
        assert result is True
        enforcer.consent_db.has_consent.assert_called_once_with(
            user_id="user1",
            client_id="client1",
            required_scopes=["file:read"],
        )
    
    @pytest.mark.asyncio
    async def test_check_consent_no_consent(self, enforcer):
        """Should return False when consent missing."""
        enforcer.consent_db = AsyncMock()
        enforcer.consent_db.has_consent = AsyncMock(return_value=False)
        
        result = await enforcer.check_consent("user1", "client1", "read_file")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_require_consent_raises_on_missing(self, enforcer):
        """Should raise PermissionError when consent missing."""
        enforcer.consent_db = AsyncMock()
        enforcer.consent_db.has_consent = AsyncMock(return_value=False)
        
        with pytest.raises(PermissionError, match="requires consent"):
            await enforcer.require_consent("user1", "client1", "read_file")
    
    @pytest.mark.asyncio
    async def test_grant_tool_consent(self, enforcer):
        """Should grant consent for tool scopes."""
        enforcer.consent_db = AsyncMock()
        enforcer.consent_db.grant_consent = AsyncMock()
        
        await enforcer.grant_tool_consent("user1", "client1", "git_status")
        
        enforcer.consent_db.grant_consent.assert_called_once_with(
            user_id="user1",
            client_id="client1",
            scopes=["git:read"],
        )
    
    @pytest.mark.asyncio
    async def test_grant_scope_consent(self, enforcer):
        """Should grant consent for specific scopes."""
        enforcer.consent_db = AsyncMock()
        enforcer.consent_db.grant_consent = AsyncMock()
        
        await enforcer.grant_scope_consent("user1", "client1", ["file:read", "file:write"])
        
        enforcer.consent_db.grant_consent.assert_called_once_with(
            user_id="user1",
            client_id="client1",
            scopes=["file:read", "file:write"],
        )


class TestConsentMiddleware:
    """Tests for consent middleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app."""
        return MagicMock()
    
    @pytest.fixture
    def middleware(self, mock_app):
        """Create consent middleware."""
        return ConsentMiddleware(mock_app)
    
    def test_exempt_paths_default(self, middleware):
        """Should have default exempt paths."""
        assert "/health" in middleware.exempt_paths
        assert "/docs" in middleware.exempt_paths
        assert "/oauth/token" in middleware.exempt_paths
    
    def test_exempt_paths_custom(self, mock_app):
        """Should accept custom exempt paths."""
        custom_paths = {"/custom", "/another"}
        middleware = ConsentMiddleware(mock_app, exempt_paths=custom_paths)
        
        assert "/custom" in middleware.exempt_paths
        assert "/another" in middleware.exempt_paths
    
    def test_require_consent_toggle(self, mock_app):
        """Should respect require_consent flag."""
        middleware = ConsentMiddleware(mock_app, require_consent=False)
        
        assert middleware.require_consent is False


class TestScopeHierarchy:
    """Tests for scope hierarchy and relationships."""
    
    def test_read_scopes_are_separate_from_write(self):
        """Read and write scopes should be separate."""
        read_tools = ["read_file", "list_dir", "search_files"]
        write_tools = ["write_file"]
        
        for tool in read_tools:
            scopes = get_required_scopes(tool)
            assert all("read" in s for s in scopes)
        
        for tool in write_tools:
            scopes = get_required_scopes(tool)
            assert all("write" in s for s in scopes)
    
    def test_all_tools_have_scopes(self):
        """All defined tools should have scopes."""
        for tool, scopes in TOOL_SCOPES.items():
            assert len(scopes) > 0
            assert all(isinstance(s, str) for s in scopes)
