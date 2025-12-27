"""
MCP Developer Assistant - Observability Module

Logging, metrics, and tracing:
- Structlog configuration
- Prometheus metrics
- OpenTelemetry tracing
- Health checks
"""

from observability.logging_config import get_logger

__all__ = ["get_logger"]
