"""
MCP Developer Assistant - Proxy Module

This module contains the OAuth 2.1 authorization gateway with:
- Token validation and JWT handling
- Confused deputy prevention
- Intent checking (LLM-as-Judge)
- Behavioral anomaly detection (ML)
- Rate limiting
- Audit logging
"""

from proxy.config import ProxyConfig

__all__ = [
    "ProxyConfig",
]

# Lazy imports for components
# from proxy.auth_gateway import app
# from proxy.oauth_validator import oauth_validator
# from proxy.intent_checker import IntentChecker
# from proxy.anomaly_detector import BehavioralAnomalyDetector
# from proxy.rate_limiter import rate_limiter
# from proxy.mcp_client import mcp_client
# from proxy.path_sanitizer import path_sanitizer
