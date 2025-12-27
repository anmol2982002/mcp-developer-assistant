"""
Proxy Configuration Module

Handles all configuration settings for the proxy server including
OAuth 2.1, rate limiting, consent, and audit settings.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class ProxyConfig(BaseSettings):
    """Configuration for the MCP Proxy server."""

    # Server settings
    host: str = Field(default="127.0.0.1", description="Proxy host")
    port: int = Field(default=8001, description="Proxy port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # MCP Server connection
    mcp_server_host: str = Field(default="127.0.0.1", description="MCP server host")
    mcp_server_port: int = Field(default=8000, description="MCP server port")

    # JWT settings
    jwt_secret_key: str = Field(default="change-me-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT expiration in hours")

    # Refresh token settings
    refresh_token_expiry_days: int = Field(default=7, description="Refresh token expiry in days")
    refresh_token_rotation: bool = Field(default=True, description="Enable refresh token rotation")
    refresh_token_family_tracking: bool = Field(default=True, description="Track token families for reuse detection")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=60, description="Requests per minute")
    rate_limit_burst: int = Field(default=10, description="Burst limit")
    rate_limit_algorithm: str = Field(default="sliding_window", description="Rate limit algorithm (sliding_window, token_bucket)")
    rate_limit_backoff_enabled: bool = Field(default=True, description="Enable exponential backoff")
    rate_limit_backoff_multiplier: float = Field(default=2.0, description="Backoff multiplier for repeat violations")

    # Intent checking
    intent_check_enabled: bool = Field(default=True, description="Enable intent checking")
    intent_check_cache_ttl_seconds: int = Field(default=3600, description="Intent cache TTL")

    # Anomaly detection
    anomaly_model_path: str = Field(
        default="./models/anomaly_detector.pkl", description="Anomaly model path"
    )
    anomaly_threshold: float = Field(default=0.7, description="Anomaly threshold")

    # Consent settings
    consent_enabled: bool = Field(default=True, description="Enable consent checking")
    consent_default_expiry_days: int = Field(default=30, description="Default consent expiry in days")
    consent_require_explicit: bool = Field(default=True, description="Require explicit consent for all tools")

    # Audit settings
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_retention_days: int = Field(default=90, description="Audit log retention in days")
    audit_sanitize_outputs: bool = Field(default=True, description="Sanitize sensitive data in audit logs")
    audit_hash_pii: bool = Field(default=True, description="Hash PII in audit logs")
    audit_archive_enabled: bool = Field(default=False, description="Archive logs before deletion")
    audit_archive_path: str = Field(default="./data/audit_archive", description="Path for archived logs")

    # LLM settings
    llm_provider: str = Field(default="groq", description="LLM provider (groq, anthropic)")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], description="Allowed CORS origins"
    )

    # Security
    security_headers_enabled: bool = Field(default=True, description="Add security headers to responses")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_proxy_config() -> ProxyConfig:
    """Get cached proxy configuration."""
    return ProxyConfig()
