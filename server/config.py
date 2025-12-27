"""
Server Configuration Module
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Configuration for the MCP Server."""

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Allowed paths
    allowed_paths: List[str] = Field(
        default=["."], description="Allowed project paths"
    )
    excluded_patterns: List[str] = Field(
        default=[".git", "node_modules", "__pycache__", "venv"],
        description="Excluded file patterns",
    )

    # Security
    sensitive_patterns: List[str] = Field(
        default=[".env", ".env.*", "*.pem", "*.key", "secrets/*"],
        description="Sensitive file patterns to block",
    )
    sanitize_output: bool = Field(default=True, description="Sanitize output")
    max_output_size_kb: int = Field(default=100, description="Max output size in KB")

    # LLM settings
    llm_provider: str = Field(default="anthropic", description="LLM provider")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    claude_model: str = Field(default="claude-3-sonnet-20240229", description="Claude model")
    claude_max_tokens: int = Field(default=4096, description="Max tokens")

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model",
    )
    faiss_index_path: str = Field(
        default="./data/embeddings/faiss.index", description="FAISS index path"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_server_config() -> ServerConfig:
    """Get cached server configuration."""
    return ServerConfig()
