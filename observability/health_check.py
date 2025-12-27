"""
Health Check

Health check endpoint for monitoring.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health check result."""

    status: str  # healthy, degraded, unhealthy
    version: str
    timestamp: str
    checks: Dict[str, bool]
    details: Optional[Dict[str, str]] = None


async def check_health(
    db_available: bool = True,
    embedding_model_loaded: bool = True,
    anomaly_model_loaded: bool = True,
) -> HealthStatus:
    """
    Perform health check.

    Args:
        db_available: Database is accessible
        embedding_model_loaded: Embedding model is loaded
        anomaly_model_loaded: Anomaly model is loaded

    Returns:
        HealthStatus with check results
    """
    checks = {
        "database": db_available,
        "embedding_model": embedding_model_loaded,
        "anomaly_model": anomaly_model_loaded,
    }

    # Determine overall status
    all_healthy = all(checks.values())
    any_healthy = any(checks.values())

    if all_healthy:
        status = "healthy"
    elif any_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthStatus(
        status=status,
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
        checks=checks,
    )
