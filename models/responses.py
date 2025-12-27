"""
Response Models

Pydantic models for outgoing responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """Tool execution response."""

    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {"content": "file contents..."},
                "error": None,
            }
        }


class MCPToolResponse(BaseModel):
    """MCP protocol tool response."""

    jsonrpc: str = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class AuthResponse(BaseModel):
    """Authentication response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    checks: Dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
