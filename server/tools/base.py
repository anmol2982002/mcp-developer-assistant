"""
Base Tool Class

Abstract base class for all MCP tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str
    description: str
    type: str  # "string", "integer", "boolean", etc.
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    result: Any
    error: Optional[str] = None


class BaseTool(ABC):
    """
    Abstract base class for MCP tools.

    All tools must implement:
    - name: Tool name
    - description: Tool description
    - parameters: List of parameters
    - execute(): Main execution method
    """

    name: str
    description: str
    parameters: List[ToolParameter]

    def __init__(self):
        self._validate_definition()

    def _validate_definition(self):
        """Validate tool definition."""
        if not hasattr(self, "name") or not self.name:
            raise ValueError("Tool must have a name")
        if not hasattr(self, "description") or not self.description:
            raise ValueError("Tool must have a description")

    @abstractmethod
    async def execute(self, **params) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **params: Tool parameters

        Returns:
            ToolResult with execution result
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for MCP."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
