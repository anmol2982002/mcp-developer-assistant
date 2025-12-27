"""
Audit Event Models

Models for audit logging.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    """Audit log event."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    client_id: Optional[str] = None
    tool_name: str
    tool_input: Optional[Dict[str, Any]] = None
    tool_output_hash: Optional[str] = None
    status: str  # success, denied, error
    deny_reason: Optional[str] = None
    duration_ms: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # ML-related fields
    intent_check_passed: Optional[bool] = None
    intent_confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_detected: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-20T10:30:00Z",
                "user_id": "user_123",
                "client_id": "claude-desktop",
                "tool_name": "read_file",
                "status": "success",
                "duration_ms": 45,
                "intent_check_passed": True,
                "intent_confidence": 0.95,
            }
        }


class SecurityEvent(BaseModel):
    """Security-related event."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # intent_violation, anomaly_detected, auth_failure
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None


class AccessEvent(BaseModel):
    """Access control event."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str  # grant, revoke, check
    user_id: str
    client_id: str
    resource: Optional[str] = None
    result: str  # allowed, denied
