"""
Database Schema

SQLAlchemy models and database configuration.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from observability.logging_config import get_logger

logger = get_logger(__name__)

# Base class for models
Base = declarative_base()

# Default database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite+aiosqlite:///./data/mcp_assistant.db"
)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class AuditLog(Base):
    """Audit log table with enhanced fields for Phase 2."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    client_id = Column(String(255), index=True)
    tool_name = Column(String(255), nullable=False, index=True)
    tool_input = Column(Text)  # JSON
    tool_output_hash = Column(String(64))
    tool_output_sanitized = Column(Text)  # Sanitized output for auditing
    status = Column(String(50))  # success, denied, error
    deny_reason = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    duration_ms = Column(Integer)
    ip_address = Column(String(50))
    user_agent = Column(Text)
    
    # Correlation fields
    request_id = Column(String(36), index=True)  # UUID for request correlation
    session_id = Column(String(255))  # Session tracking

    # ML fields
    intent_check_passed = Column(Boolean)
    intent_confidence = Column(Float)
    anomaly_score = Column(Float)
    anomaly_detected = Column(Boolean)

    __table_args__ = (
        Index("idx_audit_user_time", "user_id", "timestamp"),
        Index("idx_audit_anomaly", "anomaly_detected", "timestamp"),
        Index("idx_audit_intent", "intent_check_passed", "timestamp"),
        Index("idx_audit_request", "request_id"),
    )


class MLCache(Base):
    """ML result cache table."""

    __tablename__ = "ml_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(255), unique=True, index=True)
    result_type = Column(String(50))  # intent_check, anomaly_score, embedding
    result = Column(Text)  # JSON
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, index=True)
    hits = Column(Integer, default=0)


class Embedding(Base):
    """Embeddings cache table."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(1000), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)
    embedding = Column(LargeBinary)  # numpy binary
    preview = Column(Text)
    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_embedding_file", "file_path"),)


class ConsentRecord(Base):
    """User consent records with enhanced scope management."""

    __tablename__ = "consent_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    client_id = Column(String(255), nullable=False, index=True)
    granted = Column(Boolean, default=True)
    granted_at = Column(DateTime, default=datetime.utcnow)
    revoked_at = Column(DateTime)
    expires_at = Column(DateTime)  # Consent expiration
    scopes = Column(String(1000), default="")
    scope_details = Column(Text)  # JSON: per-scope metadata

    __table_args__ = (Index("idx_consent_user_client", "user_id", "client_id"),)


class RefreshToken(Base):
    """Refresh token storage with rotation support."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jti = Column(String(255), unique=True, index=True)  # JWT ID
    token_hash = Column(String(64), unique=True, index=True)  # SHA-256 hash
    user_id = Column(String(255), nullable=False, index=True)
    client_id = Column(String(255), nullable=False, index=True)
    family_id = Column(String(255), index=True)  # For rotation tracking
    scopes = Column(String(1000))
    issued_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    revoked_at = Column(DateTime)
    replaced_by = Column(String(255))  # JTI of replacement token

    __table_args__ = (
        Index("idx_refresh_user_client", "user_id", "client_id"),
        Index("idx_refresh_family", "family_id"),
    )


class UserQuota(Base):
    """User rate limit quotas."""

    __tablename__ = "user_quotas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), unique=True, index=True)
    requests_per_minute = Column(Integer, default=60)
    requests_per_hour = Column(Integer, default=1000)
    daily_limit = Column(Integer, default=10000)
    burst_size = Column(Integer, default=10)
    tier = Column(String(50), default="default")  # default, premium, enterprise
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_quota_tier", "tier"),)


class PKCEChallenge(Base):
    """PKCE code challenges for OAuth 2.1."""

    __tablename__ = "pkce_challenges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(255), unique=True, index=True)
    code_challenge = Column(String(255), nullable=False)
    code_challenge_method = Column(String(10), default="S256")  # S256 or plain
    client_id = Column(String(255), nullable=False)
    user_id = Column(String(255))
    scopes = Column(String(1000))
    redirect_uri = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime)  # Set when code is exchanged


# -----------------------------------------------------------------------------
# Database Session
# -----------------------------------------------------------------------------

# Async engine
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Session factory
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")

