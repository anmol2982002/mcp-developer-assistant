"""
Request Models

Pydantic models for incoming requests.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    """Tool request from MCP client."""

    tool_name: str = Field(..., description="Name of the tool to call")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    user_stated_intent: Optional[str] = Field(None, description="User's stated purpose")
    client_id: Optional[str] = Field(None, description="Client application ID")

    class Config:
        json_schema_extra = {
            "example": {
                "tool_name": "read_file",
                "params": {"path": "src/main.py", "max_lines": 100},
                "user_stated_intent": "Review the main module",
                "client_id": "claude-desktop",
            }
        }


class MCPToolCall(BaseModel):
    """MCP protocol tool call."""

    jsonrpc: str = "2.0"
    method: str = "tools/call"
    id: str
    params: Dict[str, Any]


class AuthRequest(BaseModel):
    """Authentication request."""

    grant_type: str = "client_credentials"
    client_id: str
    client_secret: str
    scope: Optional[str] = None


class ConsentRequest(BaseModel):
    """Consent grant request."""

    client_id: str
    scopes: List[str] = Field(default_factory=list)
    redirect_uri: Optional[str] = None
