"""
Refresh Token Store

Secure refresh token storage with rotation support for OAuth 2.1 compliance.
Implements token family tracking to detect token reuse attacks.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from pydantic import BaseModel
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database.schema import Base, RefreshToken
from observability.logging_config import get_logger

logger = get_logger(__name__)


class RefreshTokenData(BaseModel):
    """Refresh token payload data."""
    
    jti: str  # Unique token ID
    user_id: str
    client_id: str
    scopes: List[str]
    family_id: str  # For rotation tracking
    issued_at: datetime
    expires_at: datetime


class RefreshTokenResult(BaseModel):
    """Result of refresh token operation."""
    
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    error: Optional[str] = None
    token_data: Optional[RefreshTokenData] = None


class RefreshTokenStore:
    """
    Secure refresh token storage with rotation support.
    
    Features:
    - Token family tracking for detecting reuse attacks
    - Automatic rotation on refresh
    - Secure token hashing (tokens are never stored in plaintext)
    - Automatic cleanup of expired tokens
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @staticmethod
    def _hash_token(token: str) -> str:
        """Hash token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    @staticmethod
    def _generate_jti() -> str:
        """Generate unique token ID."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def _generate_family_id() -> str:
        """Generate token family ID."""
        return secrets.token_urlsafe(16)
    
    async def create_refresh_token(
        self,
        user_id: str,
        client_id: str,
        scopes: List[str],
        expiry_days: int = 7,
        family_id: Optional[str] = None,
    ) -> tuple[str, RefreshTokenData]:
        """
        Create a new refresh token.
        
        Args:
            user_id: User identifier
            client_id: Client application ID
            scopes: Granted scopes
            expiry_days: Token expiration in days
            family_id: Existing family ID for rotation (None for new family)
            
        Returns:
            Tuple of (raw_token, token_data)
        """
        # Generate token
        raw_token = secrets.token_urlsafe(64)
        jti = self._generate_jti()
        family = family_id or self._generate_family_id()
        
        now = datetime.utcnow()
        expires_at = now + timedelta(days=expiry_days)
        
        # Store hashed token
        token_record = RefreshToken(
            jti=jti,
            token_hash=self._hash_token(raw_token),
            user_id=user_id,
            client_id=client_id,
            family_id=family,
            scopes=",".join(scopes),
            issued_at=now,
            expires_at=expires_at,
        )
        
        self.session.add(token_record)
        await self.session.commit()
        
        token_data = RefreshTokenData(
            jti=jti,
            user_id=user_id,
            client_id=client_id,
            scopes=scopes,
            family_id=family,
            issued_at=now,
            expires_at=expires_at,
        )
        
        logger.info(
            "refresh_token_created",
            user_id=user_id,
            client_id=client_id,
            family_id=family,
            expires_at=expires_at.isoformat(),
        )
        
        return raw_token, token_data
    
    async def validate_refresh_token(self, raw_token: str) -> Optional[RefreshTokenData]:
        """
        Validate a refresh token.
        
        Args:
            raw_token: The raw refresh token
            
        Returns:
            Token data if valid, None otherwise
        """
        token_hash = self._hash_token(raw_token)
        
        query = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > datetime.utcnow(),
        )
        
        result = await self.session.execute(query)
        token = result.scalar_one_or_none()
        
        if not token:
            logger.warning("refresh_token_invalid", reason="not_found_or_expired")
            return None
        
        return RefreshTokenData(
            jti=token.jti,
            user_id=token.user_id,
            client_id=token.client_id,
            scopes=token.scopes.split(",") if token.scopes else [],
            family_id=token.family_id,
            issued_at=token.issued_at,
            expires_at=token.expires_at,
        )
    
    async def rotate_refresh_token(
        self,
        old_token: str,
        expiry_days: int = 7,
    ) -> Optional[tuple[str, RefreshTokenData]]:
        """
        Rotate a refresh token (invalidate old, create new).
        
        This implements refresh token rotation as recommended by OAuth 2.1.
        The old token is invalidated and a new token is issued in the same family.
        
        Args:
            old_token: The refresh token to rotate
            expiry_days: New token expiration in days
            
        Returns:
            Tuple of (new_raw_token, token_data) or None if invalid
        """
        # Validate old token
        old_token_hash = self._hash_token(old_token)
        
        query = select(RefreshToken).where(
            RefreshToken.token_hash == old_token_hash,
        )
        
        result = await self.session.execute(query)
        old_record = result.scalar_one_or_none()
        
        if not old_record:
            logger.warning("refresh_token_rotation_failed", reason="token_not_found")
            return None
        
        # Check if already revoked - potential token reuse attack!
        if old_record.revoked_at is not None:
            # Revoke entire token family for security
            await self._revoke_token_family(old_record.family_id)
            logger.error(
                "token_reuse_attack_detected",
                family_id=old_record.family_id,
                user_id=old_record.user_id,
            )
            return None
        
        # Check if expired
        if old_record.expires_at <= datetime.utcnow():
            logger.warning("refresh_token_rotation_failed", reason="token_expired")
            return None
        
        # Revoke old token
        old_record.revoked_at = datetime.utcnow()
        
        # Create new token in the same family
        new_token, token_data = await self.create_refresh_token(
            user_id=old_record.user_id,
            client_id=old_record.client_id,
            scopes=old_record.scopes.split(",") if old_record.scopes else [],
            expiry_days=expiry_days,
            family_id=old_record.family_id,
        )
        
        # Link old token to new
        old_record.replaced_by = token_data.jti
        await self.session.commit()
        
        logger.info(
            "refresh_token_rotated",
            old_jti=old_record.jti,
            new_jti=token_data.jti,
            family_id=old_record.family_id,
        )
        
        return new_token, token_data
    
    async def revoke_token(self, jti: str) -> bool:
        """Revoke a specific token by its ID."""
        query = (
            update(RefreshToken)
            .where(RefreshToken.jti == jti)
            .values(revoked_at=datetime.utcnow())
        )
        
        result = await self.session.execute(query)
        await self.session.commit()
        
        if result.rowcount > 0:
            logger.info("refresh_token_revoked", jti=jti)
            return True
        return False
    
    async def revoke_user_tokens(self, user_id: str, client_id: Optional[str] = None) -> int:
        """Revoke all tokens for a user (optionally for a specific client)."""
        conditions = [
            RefreshToken.user_id == user_id,
            RefreshToken.revoked_at.is_(None),
        ]
        
        if client_id:
            conditions.append(RefreshToken.client_id == client_id)
        
        query = (
            update(RefreshToken)
            .where(and_(*conditions))
            .values(revoked_at=datetime.utcnow())
        )
        
        result = await self.session.execute(query)
        await self.session.commit()
        
        logger.info(
            "user_tokens_revoked",
            user_id=user_id,
            client_id=client_id,
            count=result.rowcount,
        )
        
        return result.rowcount
    
    async def _revoke_token_family(self, family_id: str) -> int:
        """Revoke all tokens in a family (used for security incidents)."""
        query = (
            update(RefreshToken)
            .where(
                RefreshToken.family_id == family_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.utcnow())
        )
        
        result = await self.session.execute(query)
        await self.session.commit()
        
        logger.warning(
            "token_family_revoked",
            family_id=family_id,
            count=result.rowcount,
        )
        
        return result.rowcount
    
    async def cleanup_expired_tokens(self, days_old: int = 30) -> int:
        """
        Clean up expired tokens older than specified days.
        
        Args:
            days_old: Delete tokens expired more than this many days ago
            
        Returns:
            Number of tokens deleted
        """
        from sqlalchemy import delete
        
        cutoff = datetime.utcnow() - timedelta(days=days_old)
        
        query = delete(RefreshToken).where(RefreshToken.expires_at < cutoff)
        
        result = await self.session.execute(query)
        await self.session.commit()
        
        logger.info("expired_tokens_cleaned", count=result.rowcount)
        
        return result.rowcount
