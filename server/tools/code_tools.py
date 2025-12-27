"""
Code Analysis Tools

Code analysis: extract functions, classes, imports.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List

from observability.logging_config import get_logger
from server.tools.base import BaseTool, ToolParameter, ToolResult

logger = get_logger(__name__)


class ExtractFunctionsTool(BaseTool):
    """Extract function definitions from Python files."""

    name = "extract_functions"
    description = "Extract function definitions from a Python file"
    parameters = [
        ToolParameter(name="path", description="Path to Python file", type="string"),
    ]

    async def execute(self, path: str) -> ToolResult:
        """Extract function definitions."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(success=False, result=None, error=f"File not found: {path}")

            content = file_path.read_text()
            tree = ast.parse(content)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    func = {
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [self._get_name(d) for d in node.decorator_list],
                        "docstring": ast.get_docstring(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    }
                    functions.append(func)

            logger.info("extracted_functions", path=path, count=len(functions))
            return ToolResult(success=True, result={"path": path, "functions": functions})

        except SyntaxError as e:
            return ToolResult(success=False, result=None, error=f"Syntax error: {e}")
        except Exception as e:
            logger.error("extract_functions_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))

    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(node)


class ExtractClassesTool(BaseTool):
    """Extract class definitions from Python files."""

    name = "extract_classes"
    description = "Extract class definitions from a Python file"
    parameters = [
        ToolParameter(name="path", description="Path to Python file", type="string"),
    ]

    async def execute(self, path: str) -> ToolResult:
        """Extract class definitions."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(success=False, result=None, error=f"File not found: {path}")

            content = file_path.read_text()
            tree = ast.parse(content)

            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append(item.name)

                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [self._get_name(b) for b in node.bases],
                        "methods": methods,
                        "docstring": ast.get_docstring(node),
                    })

            logger.info("extracted_classes", path=path, count=len(classes))
            return ToolResult(success=True, result={"path": path, "classes": classes})

        except SyntaxError as e:
            return ToolResult(success=False, result=None, error=f"Syntax error: {e}")
        except Exception as e:
            logger.error("extract_classes_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))

    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)


class AnalyzeImportsTool(BaseTool):
    """Analyze imports in Python files."""

    name = "analyze_imports"
    description = "Analyze imports in a Python file"
    parameters = [
        ToolParameter(name="path", description="Path to Python file", type="string"),
    ]

    async def execute(self, path: str) -> ToolResult:
        """Analyze imports."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(success=False, result=None, error=f"File not found: {path}")

            content = file_path.read_text()
            tree = ast.parse(content)

            imports = {"standard": [], "third_party": [], "local": []}
            stdlib_modules = {"os", "sys", "re", "json", "typing", "pathlib", "datetime", "collections", "functools", "itertools", "abc", "dataclasses", "contextlib", "asyncio"}

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        category = "standard" if module in stdlib_modules else "third_party"
                        imports[category].append({"module": alias.name, "alias": alias.asname})

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_module = node.module.split(".")[0]
                        if node.level > 0:
                            category = "local"
                        elif base_module in stdlib_modules:
                            category = "standard"
                        else:
                            category = "third_party"

                        for alias in node.names:
                            imports[category].append({
                                "from": node.module,
                                "import": alias.name,
                                "alias": alias.asname,
                            })

            logger.info("analyzed_imports", path=path)
            return ToolResult(success=True, result={"path": path, "imports": imports})

        except SyntaxError as e:
            return ToolResult(success=False, result=None, error=f"Syntax error: {e}")
        except Exception as e:
            logger.error("analyze_imports_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))
