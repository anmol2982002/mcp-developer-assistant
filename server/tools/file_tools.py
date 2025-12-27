"""
File Tools

File system tools: read_file, search_files, list_dir.
"""

import fnmatch
import re
from pathlib import Path
from typing import List, Optional

from observability.logging_config import get_logger
from server.config import get_server_config
from server.tools.base import BaseTool, ToolParameter, ToolResult

logger = get_logger(__name__)


class ReadFileTool(BaseTool):
    """Read file contents."""

    name = "read_file"
    description = "Read the contents of a file from the codebase"
    parameters = [
        ToolParameter(name="path", description="Path to the file", type="string"),
        ToolParameter(
            name="max_lines",
            description="Maximum lines to read",
            type="integer",
            required=False,
            default=100,
        ),
    ]

    async def execute(self, path: str, max_lines: int = 100) -> ToolResult:
        """Read file contents with syntax detection."""
        config = get_server_config()

        # Security check: sensitive patterns
        for pattern in config.sensitive_patterns:
            if fnmatch.fnmatch(path, pattern):
                logger.warning("sensitive_file_blocked", path=path)
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Access denied: {path} matches sensitive pattern",
                )

        try:
            file_path = Path(path)

            if not file_path.exists():
                return ToolResult(success=False, result=None, error=f"File not found: {path}")

            if not file_path.is_file():
                return ToolResult(success=False, result=None, error=f"Not a file: {path}")

            # Detect file type and syntax language
            extension = file_path.suffix.lower()
            syntax_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".jsx": "javascript",
                ".java": "java",
                ".go": "go",
                ".rs": "rust",
                ".c": "c",
                ".cpp": "cpp",
                ".h": "c",
                ".hpp": "cpp",
                ".cs": "csharp",
                ".rb": "ruby",
                ".php": "php",
                ".swift": "swift",
                ".kt": "kotlin",
                ".scala": "scala",
                ".r": "r",
                ".sql": "sql",
                ".sh": "bash",
                ".bash": "bash",
                ".zsh": "zsh",
                ".ps1": "powershell",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".json": "json",
                ".xml": "xml",
                ".html": "html",
                ".css": "css",
                ".scss": "scss",
                ".sass": "sass",
                ".less": "less",
                ".md": "markdown",
                ".rst": "restructuredtext",
                ".tex": "latex",
                ".toml": "toml",
                ".ini": "ini",
                ".cfg": "ini",
                ".dockerfile": "dockerfile",
            }
            
            # Check for special filenames
            filename_lower = file_path.name.lower()
            if filename_lower == "dockerfile":
                syntax_lang = "dockerfile"
            elif filename_lower in ("makefile", "gnumakefile"):
                syntax_lang = "makefile"
            elif filename_lower in (".gitignore", ".dockerignore"):
                syntax_lang = "gitignore"
            else:
                syntax_lang = syntax_map.get(extension, "text")

            # Read file
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            if max_lines and len(lines) > max_lines:
                lines = lines[:max_lines]
                content = "\n".join(lines) + f"\n... (truncated, {max_lines}/{total_lines} lines shown)"

            # Get file size
            file_size = file_path.stat().st_size

            logger.info("file_read", path=path, lines=len(lines), syntax=syntax_lang)
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "content": content,
                    "lines": len(lines),
                    "total_lines": total_lines,
                    "syntax": syntax_lang,
                    "extension": extension,
                    "size_bytes": file_size,
                },
            )

        except Exception as e:
            logger.error("file_read_error", path=path, error=str(e))
            return ToolResult(success=False, result=None, error=str(e))


class SearchFilesTool(BaseTool):
    """Search files for patterns."""

    name = "search_files"
    description = "Search for patterns in files"
    parameters = [
        ToolParameter(name="pattern", description="Search pattern (regex)", type="string"),
        ToolParameter(name="directory", description="Directory to search", type="string", required=False, default="."),
        ToolParameter(name="file_pattern", description="File glob pattern", type="string", required=False, default="*"),
        ToolParameter(name="max_results", description="Maximum results", type="integer", required=False, default=50),
    ]

    async def execute(
        self,
        pattern: str,
        directory: str = ".",
        file_pattern: str = "*",
        max_results: int = 50,
    ) -> ToolResult:
        """Search files for pattern."""
        config = get_server_config()
        results = []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
            search_path = Path(directory)

            if not search_path.exists():
                return ToolResult(success=False, result=None, error=f"Directory not found: {directory}")

            for file_path in search_path.rglob(file_pattern):
                if len(results) >= max_results:
                    break

                # Skip excluded patterns
                if any(p in str(file_path) for p in config.excluded_patterns):
                    continue

                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            results.append({
                                "file": str(file_path),
                                "line": i,
                                "content": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                break
                except Exception:
                    continue

            logger.info("search_completed", pattern=pattern, results=len(results))
            return ToolResult(success=True, result={"matches": results, "count": len(results)})

        except re.error as e:
            return ToolResult(success=False, result=None, error=f"Invalid regex: {e}")
        except Exception as e:
            logger.error("search_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))


class ListDirTool(BaseTool):
    """List directory contents."""

    name = "list_dir"
    description = "List files and directories"
    parameters = [
        ToolParameter(name="path", description="Directory path", type="string", required=False, default="."),
        ToolParameter(name="recursive", description="Recursive listing", type="boolean", required=False, default=False),
    ]

    async def execute(self, path: str = ".", recursive: bool = False) -> ToolResult:
        """List directory contents."""
        config = get_server_config()

        try:
            dir_path = Path(path)

            if not dir_path.exists():
                return ToolResult(success=False, result=None, error=f"Directory not found: {path}")

            if not dir_path.is_dir():
                return ToolResult(success=False, result=None, error=f"Not a directory: {path}")

            items = []
            iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()

            for item in iterator:
                # Skip excluded
                if any(p in str(item) for p in config.excluded_patterns):
                    continue

                items.append({
                    "path": str(item.relative_to(dir_path)),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })

            logger.info("list_dir", path=path, items=len(items))
            return ToolResult(success=True, result={"path": path, "items": items, "count": len(items)})

        except Exception as e:
            logger.error("list_dir_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))
