"""
Consent Middleware

FastAPI middleware for automatic consent enforcement
in the confused deputy prevention flow.
"""

from typing import Callable, List, Optional, Set

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from database.schema import get_db
from observability.logging_config import get_logger
from proxy.consent_db import ConsentDB

logger = get_logger(__name__)


# Tool to scope mapping
TOOL_SCOPES = {
    # File operations
    "read_file": ["file:read"],
    "write_file": ["file:write"],
    "list_dir": ["file:read"],
    "search_files": ["file:read"],
    
    # Git operations
    "git_status": ["git:read"],
    "git_diff": ["git:read"],
    "git_log": ["git:read"],
    "git_commit": ["git:write"],
    "git_push": ["git:write"],
    
    # Code analysis
    "analyze_code": ["code:analyze"],
    "extract_symbols": ["code:analyze"],
    "get_dependencies": ["code:analyze"],
    
    # AI operations
    "ask_about_code": ["ai:query"],
    "review_changes": ["ai:review"],
    "summarize_code": ["ai:query"],
    
    # Admin operations
    "run_command": ["system:execute"],
}


def get_required_scopes(tool_name: str) -> List[str]:
    """Get required scopes for a tool."""
    return TOOL_SCOPES.get(tool_name, ["default"])


class ConsentMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce user consent before tool execution.
    
    This prevents confused deputy attacks by ensuring that
    the authenticated client has explicit user consent to
    act on their behalf for the requested scopes.
    """
    
    def __init__(
        self,
        app,
        require_consent: bool = True,
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.require_consent = require_consent
        self.exempt_paths = exempt_paths or {
            "/health",
            "/docs",
            "/openapi.json",
            "/oauth/token",
            "/oauth/authorize",
            "/consent/grant",
        }
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Check consent before processing request."""
        
        # Skip exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Skip if consent not required
        if not self.require_consent:
            return await call_next(request)
        
        # Only check for tool calls
        if request.url.path != "/tool_call":
            return await call_next(request)
        
        # Get user and client info from request state
        # (Set by auth middleware earlier in chain)
        user_id = getattr(request.state, "user_id", None)
        client_id = getattr(request.state, "client_id", None)
        
        if not user_id or not client_id:
            # No auth info - let the auth handler deal with it
            return await call_next(request)
        
        # Parse request body to get tool name
        try:
            body = await request.json()
            tool_name = body.get("tool_name")
        except Exception:
            return await call_next(request)
        
        if not tool_name:
            return await call_next(request)
        
        # Get required scopes
        required_scopes = get_required_scopes(tool_name)
        
        # Check consent
        try:
            async with get_db() as session:
                consent_db = ConsentDB(session)
                has_consent = await consent_db.has_consent(
                    user_id=user_id,
                    client_id=client_id,
                    required_scopes=required_scopes,
                )
                
                if not has_consent:
                    logger.warning(
                        "consent_denied_by_middleware",
                        user_id=user_id,
                        client_id=client_id,
                        tool=tool_name,
                        required_scopes=required_scopes,
                    )
                    
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "consent_required",
                            "message": f"Client '{client_id}' requires consent for scopes: {required_scopes}",
                            "consent_url": f"/consent/request?client_id={client_id}&scopes={','.join(required_scopes)}",
                            "required_scopes": required_scopes,
                        },
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("consent_check_error", error=str(e))
            # Fail open or closed based on config
            # For security, default to fail closed
            raise HTTPException(
                status_code=500,
                detail="Consent verification error",
            )
        
        return await call_next(request)


class ConsentEnforcer:
    """
    Programmatic consent enforcement for direct use in handlers.
    """
    
    def __init__(self, session):
        self.session = session
        self.consent_db = ConsentDB(session)
    
    async def check_consent(
        self,
        user_id: str,
        client_id: str,
        tool_name: str,
    ) -> bool:
        """Check if client has consent for tool."""
        required_scopes = get_required_scopes(tool_name)
        return await self.consent_db.has_consent(
            user_id=user_id,
            client_id=client_id,
            required_scopes=required_scopes,
        )
    
    async def require_consent(
        self,
        user_id: str,
        client_id: str,
        tool_name: str,
    ):
        """Require consent or raise exception."""
        if not await self.check_consent(user_id, client_id, tool_name):
            required_scopes = get_required_scopes(tool_name)
            raise PermissionError(
                f"Client '{client_id}' requires consent for: {required_scopes}"
            )
    
    async def grant_tool_consent(
        self,
        user_id: str,
        client_id: str,
        tool_name: str,
    ):
        """Grant consent for a specific tool."""
        scopes = get_required_scopes(tool_name)
        await self.consent_db.grant_consent(
            user_id=user_id,
            client_id=client_id,
            scopes=scopes,
        )
    
    async def grant_scope_consent(
        self,
        user_id: str,
        client_id: str,
        scopes: List[str],
    ):
        """Grant consent for specific scopes."""
        await self.consent_db.grant_consent(
            user_id=user_id,
            client_id=client_id,
            scopes=scopes,
        )
