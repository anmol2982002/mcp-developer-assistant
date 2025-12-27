"""
MCP Server - Main Application

AI-Enhanced Developer Assistant MCP Server.
Implements tool registration, dispatch, and JSON-RPC-style request handling.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from observability.logging_config import get_logger
from observability.metrics import metrics
from server.config import get_server_config
from server.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class ToolRequest(BaseModel):
    """Tool execution request."""

    tool_name: str
    params: Dict[str, Any]
    user_id: Optional[str] = None
    client_id: Optional[str] = None


class ToolResponse(BaseModel):
    """Tool execution response."""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ToolDefinition(BaseModel):
    """Tool definition for listing."""

    name: str
    description: str
    inputSchema: Dict[str, Any]


class ToolListResponse(BaseModel):
    """Response for listing tools."""

    tools: List[ToolDefinition]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"
    tools_loaded: int = 0


# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """
    Registry for MCP tools.

    Manages tool registration, lookup, and execution.
    Implements a singleton pattern for global access.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False

    def register(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class.

        Args:
            tool_class: Tool class (not instance) to register
        """
        try:
            tool = tool_class()
            self._tools[tool.name] = tool
            logger.info("tool_registered", name=tool.name)
        except Exception as e:
            logger.error("tool_registration_failed", tool=tool_class.__name__, error=str(e))

    def register_instance(self, tool: BaseTool) -> None:
        """Register a tool instance directly."""
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get all tool definitions."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_names(self) -> List[str]:
        """Get all tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, params: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            params: Tool parameters

        Returns:
            ToolResult from execution
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, result=None, error=f"Unknown tool: {name}")

        try:
            result = await tool.execute(**params)
            return result
        except Exception as e:
            logger.error("tool_execution_error", tool=name, error=str(e))
            return ToolResult(success=False, result=None, error=str(e))

    def initialize_tools(self) -> None:
        """Initialize all available tools."""
        if self._initialized:
            return

        # Import and register file tools
        from server.tools.file_tools import ListDirTool, ReadFileTool, SearchFilesTool

        self.register(ReadFileTool)
        self.register(SearchFilesTool)
        self.register(ListDirTool)

        # Import and register git tools
        from server.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool

        self.register(GitStatusTool)
        self.register(GitDiffTool)
        self.register(GitLogTool)

        # Import and register code tools
        try:
            from server.tools.code_tools import (
                AnalyzeImportsTool,
                ExtractClassesTool,
                ExtractFunctionsTool,
            )

            self.register(ExtractFunctionsTool)
            self.register(ExtractClassesTool)
            self.register(AnalyzeImportsTool)
        except Exception as e:
            logger.warning("code_tools_import_failed", error=str(e))

        # Initialize shared components for AI tools (Phase 4)
        hybrid_search = None
        llm_client = None
        
        try:
            # Initialize Hybrid Search Engine
            from ai.hybrid_search import HybridSearchEngine
            hybrid_search = HybridSearchEngine()
            logger.info("hybrid_search_engine_initialized")
        except Exception as e:
            logger.warning("hybrid_search_init_failed", error=str(e))

        try:
            # Initialize LLM Client
            from ai.llm_client import get_llm_client
            llm_client = get_llm_client()
            logger.info("llm_client_initialized")
        except Exception as e:
            logger.warning("llm_client_init_failed", error=str(e))

        # Import and register AI tools with shared components
        try:
            from server.tools.ai_tools import (
                AskAboutCodeTool,
                ReviewChangesTool,
                SummarizeRepoTool,
                SummarizeDiffTool,
                QueryExpansionTool,
            )

            # Register tools with hybrid search and LLM
            ask_tool = AskAboutCodeTool(
                hybrid_search_engine=hybrid_search,
                llm_client=llm_client,
            )
            self.register_instance(ask_tool)

            review_tool = ReviewChangesTool(llm_client=llm_client)
            self.register_instance(review_tool)

            summarize_repo_tool = SummarizeRepoTool(llm_client=llm_client)
            self.register_instance(summarize_repo_tool)

            summarize_diff_tool = SummarizeDiffTool(llm_client=llm_client)
            self.register_instance(summarize_diff_tool)

            query_expansion_tool = QueryExpansionTool()
            self.register_instance(query_expansion_tool)

            logger.info("ai_tools_registered", count=5)
        except Exception as e:
            logger.warning("ai_tools_import_failed", error=str(e))

        self._initialized = True
        logger.info("tools_initialized", count=len(self._tools), tools=self.get_names())


# Global registry
tool_registry = ToolRegistry()


# -----------------------------------------------------------------------------
# Application Lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting MCP Server...")

    # Initialize tool registry
    tool_registry.initialize_tools()

    yield

    logger.info("Shutting down MCP Server...")


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------


app = FastAPI(
    title="MCP Developer Assistant - Server",
    description="AI-Enhanced Developer Assistant with file, git, and code analysis tools",
    version="0.1.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        tools_loaded=len(tool_registry.get_names()),
    )


@app.get("/tools", response_model=ToolListResponse)
async def list_tools():
    """List all available MCP tools."""
    tools = tool_registry.list_tools()
    return ToolListResponse(tools=tools)


@app.post("/tool/{tool_name}", response_model=ToolResponse)
async def execute_tool(tool_name: str, request: ToolRequest):
    """
    Execute a specific MCP tool.

    Args:
        tool_name: Name of the tool to execute
        request: Tool request with parameters
    """
    start_time = time.perf_counter()

    logger.info(
        "tool_execution_start",
        tool=tool_name,
        user_id=request.user_id,
        client_id=request.client_id,
    )

    # Verify tool exists
    tool = tool_registry.get(tool_name)
    if not tool:
        logger.warning("tool_not_found", tool=tool_name)
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    # Execute tool
    result = await tool_registry.execute(tool_name, request.params)

    duration = time.perf_counter() - start_time
    metrics.record_request(tool_name, duration, result.success)

    logger.info(
        "tool_execution_complete",
        tool=tool_name,
        success=result.success,
        duration_ms=int(duration * 1000),
    )

    if not result.success:
        return ToolResponse(success=False, result=None, error=result.error)

    return ToolResponse(success=True, result=result.result)


@app.post("/execute", response_model=ToolResponse)
async def execute_tool_generic(request: ToolRequest):
    """
    Generic tool execution endpoint (alternative to /tool/{name}).

    Useful for JSON-RPC style requests.
    """
    return await execute_tool(request.tool_name, request)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main():
    """Main entry point."""
    import uvicorn

    config = get_server_config()
    uvicorn.run(
        "server.mcp_server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()
