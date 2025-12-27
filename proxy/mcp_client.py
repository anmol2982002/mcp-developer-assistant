"""
MCP Client - HTTP Client for Proxy to Server Communication

Provides async HTTP client for forwarding validated requests
from the proxy to the MCP server.
"""

import asyncio
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel

from observability.logging_config import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)


class ToolCallRequest(BaseModel):
    """Request payload for tool execution."""

    tool_name: str
    params: Dict[str, Any]
    user_id: Optional[str] = None
    client_id: Optional[str] = None


class ToolCallResponse(BaseModel):
    """Response from tool execution."""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class MCPClient:
    """
    HTTP client for communicating with the MCP server.

    Features:
    - Connection pooling for performance
    - Automatic retries with exponential backoff
    - Request timeout handling
    - Structured logging and metrics
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize MCP client.

        Args:
            base_url: MCP server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> ToolCallResponse:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            user_id: Authenticated user ID
            client_id: Client application ID

        Returns:
            ToolCallResponse with execution result
        """
        import time

        start_time = time.perf_counter()
        client = await self._get_client()

        request_data = {
            "tool_name": tool_name,
            "params": params,
            "user_id": user_id,
            "client_id": client_id,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"/tool/{tool_name}",
                    json=request_data,
                )

                duration_ms = int((time.perf_counter() - start_time) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        "mcp_tool_call_success",
                        tool=tool_name,
                        duration_ms=duration_ms,
                    )
                    metrics.mcp_client_requests_total.labels(
                        tool=tool_name, status="success"
                    ).inc()
                    metrics.mcp_client_latency_ms.labels(tool=tool_name).observe(
                        duration_ms
                    )

                    return ToolCallResponse(
                        success=True,
                        result=data.get("result"),
                        duration_ms=duration_ms,
                    )
                else:
                    error_msg = response.text or f"HTTP {response.status_code}"
                    logger.warning(
                        "mcp_tool_call_failed",
                        tool=tool_name,
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    metrics.mcp_client_requests_total.labels(
                        tool=tool_name, status="error"
                    ).inc()

                    return ToolCallResponse(
                        success=False,
                        error=error_msg,
                        duration_ms=duration_ms,
                    )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "mcp_tool_call_timeout",
                    tool=tool_name,
                    attempt=attempt + 1,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    "mcp_tool_call_request_error",
                    tool=tool_name,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        # All retries exhausted
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        metrics.mcp_client_requests_total.labels(tool=tool_name, status="error").inc()

        return ToolCallResponse(
            success=False,
            error=f"Request failed after {self.max_retries} attempts: {last_error}",
            duration_ms=duration_ms,
        )

    async def list_tools(self) -> Dict[str, Any]:
        """
        Get list of available tools from MCP server.

        Returns:
            Dictionary with tool definitions
        """
        client = await self._get_client()

        try:
            response = await client.get("/tools")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error("list_tools_failed", status_code=response.status_code)
                return {"tools": [], "error": response.text}

        except Exception as e:
            logger.error("list_tools_error", error=str(e))
            return {"tools": [], "error": str(e)}

    async def health_check(self) -> bool:
        """
        Check MCP server health.

        Returns:
            True if server is healthy
        """
        client = await self._get_client()

        try:
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
mcp_client = MCPClient()
