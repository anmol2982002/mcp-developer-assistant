"""
MCP Server - Tools Package

Contains all tool implementations:
- Base tool class
- File tools
- Git tools
- Code analysis tools
- AI-powered tools (Phase 4: hybrid search, validated outputs)
"""

from server.tools.base import BaseTool
from server.tools.file_tools import ReadFileTool, SearchFilesTool, ListDirTool
from server.tools.git_tools import GitStatusTool, GitDiffTool, GitLogTool
from server.tools.code_tools import ExtractFunctionsTool, ExtractClassesTool, AnalyzeImportsTool
from server.tools.ai_tools import (
    AskAboutCodeTool,
    ReviewChangesTool,
    SummarizeRepoTool,
    SummarizeDiffTool,
    QueryExpansionTool,
)

__all__ = [
    # Base
    "BaseTool",
    
    # File tools
    "ReadFileTool",
    "SearchFilesTool",
    "ListDirTool",
    
    # Git tools
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    
    # Code tools
    "ExtractFunctionsTool",
    "ExtractClassesTool",
    "AnalyzeImportsTool",
    
    # AI tools (Phase 4)
    "AskAboutCodeTool",
    "ReviewChangesTool",
    "SummarizeRepoTool",
    "SummarizeDiffTool",
    "QueryExpansionTool",
]
