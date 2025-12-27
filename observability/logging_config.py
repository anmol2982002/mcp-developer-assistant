"""
Logging Configuration

Structlog-based JSON logging for structured, machine-parseable logs.
"""

import logging
import sys
from functools import lru_cache

import structlog


def configure_logging(
    level: str = "INFO",
    format_json: bool = True,
    log_file: str = None,
):
    """
    Configure structlog logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_json: Use JSON format (True) or console (False)
        log_file: Optional log file path
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Common processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@lru_cache
def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name or "mcp")


# Configure on import
try:
    configure_logging()
except Exception:
    # Fallback if structlog not installed
    logging.basicConfig(level=logging.INFO)


# Convenience function for adding context
def bind_context(**kwargs):
    """Add context variables to all subsequent log entries."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context():
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()
