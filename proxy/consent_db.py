"""
Consent Database

Manages user consent for MCP clients (prevents confused deputy attacks).
Enhanced with scope granularity, expiration, and listing capabilities.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Integer, String, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database.schema import Base, ConsentRecord
from observability.logging_config import get_logger

logger = get_logger(__name__)


class ConsentDetails(BaseModel):
    """Detailed consent information for API responses."""
    
    id: int
    client_id: str
    scopes: List[str]
    granted_at: datetime
    expires_at: Optional[datetime]
    is_active: bool


class ConsentDB:
    """
    Consent registry for confused deputy prevention.

    Tracks which clients have user consent to act on their behalf.
    Supports scope-level granularity and consent expiration.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def has_consent(
        self,
        user_id: str,
        client_id: str,
        required_scopes: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if client has user consent.

        Args:
            user_id: User identifier
            client_id: Client application ID
            required_scopes: Scopes needed (optional)

        Returns:
            True if consent exists and is valid
        """
        now = datetime.utcnow()
        
        query = select(ConsentRecord).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.client_id == client_id,
            ConsentRecord.granted == True,
            ConsentRecord.revoked_at == None,
        )

        result = await self.session.execute(query)
        consent = result.scalar_one_or_none()

        if not consent:
            logger.warning("consent_missing", user_id=user_id, client_id=client_id)
            return False
        
        # Check expiration
        if consent.expires_at and consent.expires_at < now:
            logger.warning("consent_expired", user_id=user_id, client_id=client_id)
            return False

        # Check scopes if required
        if required_scopes:
            granted_scopes = set(consent.scopes.split(",") if consent.scopes else [])
            if not all(s in granted_scopes for s in required_scopes):
                logger.warning(
                    "consent_scope_mismatch",
                    user_id=user_id,
                    client_id=client_id,
                    required=required_scopes,
                    granted=list(granted_scopes),
                )
                return False

        return True

    async def grant_consent(
        self,
        user_id: str,
        client_id: str,
        scopes: Optional[List[str]] = None,
        expires_days: Optional[int] = None,
    ) -> ConsentRecord:
        """
        Grant consent to a client.

        Args:
            user_id: User identifier
            client_id: Client application ID
            scopes: Granted scopes
            expires_days: Consent expiration in days (None = never expires)

        Returns:
            Created consent record
        """
        # Check for existing consent
        existing_query = select(ConsentRecord).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.client_id == client_id,
            ConsentRecord.granted == True,
            ConsentRecord.revoked_at == None,
        )
        
        result = await self.session.execute(existing_query)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing consent
            existing_scopes = set(existing.scopes.split(",") if existing.scopes else [])
            new_scopes = existing_scopes.union(set(scopes or []))
            existing.scopes = ",".join(new_scopes)
            
            if expires_days:
                existing.expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
            await self.session.commit()
            logger.info(
                "consent_updated",
                user_id=user_id,
                client_id=client_id,
                scopes=list(new_scopes),
            )
            return existing
        
        # Create new consent
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        consent = ConsentRecord(
            user_id=user_id,
            client_id=client_id,
            granted=True,
            scopes=",".join(scopes) if scopes else "",
            expires_at=expires_at,
        )

        self.session.add(consent)
        await self.session.commit()

        logger.info("consent_granted", user_id=user_id, client_id=client_id, scopes=scopes)
        return consent

    async def revoke_consent(self, user_id: str, client_id: str) -> bool:
        """
        Revoke consent from a client.

        Args:
            user_id: User identifier
            client_id: Client application ID

        Returns:
            True if revoked, False if not found
        """
        query = select(ConsentRecord).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.client_id == client_id,
            ConsentRecord.granted == True,
            ConsentRecord.revoked_at == None,
        )

        result = await self.session.execute(query)
        consent = result.scalar_one_or_none()

        if consent:
            consent.revoked_at = datetime.utcnow()
            consent.granted = False
            await self.session.commit()
            logger.info("consent_revoked", user_id=user_id, client_id=client_id)
            return True

        return False

    async def list_user_consents(self, user_id: str) -> List[ConsentDetails]:
        """
        List all consents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of consent details
        """
        now = datetime.utcnow()
        
        query = select(ConsentRecord).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.revoked_at == None,
        ).order_by(ConsentRecord.granted_at.desc())
        
        result = await self.session.execute(query)
        consents = result.scalars().all()
        
        return [
            ConsentDetails(
                id=consent.id,
                client_id=consent.client_id,
                scopes=consent.scopes.split(",") if consent.scopes else [],
                granted_at=consent.granted_at,
                expires_at=consent.expires_at,
                is_active=consent.granted and (
                    consent.expires_at is None or consent.expires_at > now
                ),
            )
            for consent in consents
        ]

    async def has_scope_consent(
        self,
        user_id: str,
        client_id: str,
        scope: str,
    ) -> bool:
        """
        Check if client has consent for a specific scope.
        
        Args:
            user_id: User identifier
            client_id: Client application ID
            scope: Required scope
            
        Returns:
            True if scope is consented
        """
        return await self.has_consent(user_id, client_id, [scope])

    async def revoke_scope_consent(
        self,
        user_id: str,
        client_id: str,
        scope: str,
    ) -> bool:
        """
        Revoke consent for a specific scope.
        
        Args:
            user_id: User identifier
            client_id: Client application ID
            scope: Scope to revoke
            
        Returns:
            True if scope was removed
        """
        query = select(ConsentRecord).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.client_id == client_id,
            ConsentRecord.granted == True,
            ConsentRecord.revoked_at == None,
        )
        
        result = await self.session.execute(query)
        consent = result.scalar_one_or_none()
        
        if not consent:
            return False
        
        current_scopes = set(consent.scopes.split(",") if consent.scopes else [])
        
        if scope not in current_scopes:
            return False
        
        current_scopes.remove(scope)
        
        if not current_scopes:
            # No scopes left, revoke entire consent
            consent.revoked_at = datetime.utcnow()
            consent.granted = False
        else:
            consent.scopes = ",".join(current_scopes)
        
        await self.session.commit()
        logger.info(
            "scope_consent_revoked",
            user_id=user_id,
            client_id=client_id,
            scope=scope,
        )
        
        return True

    async def get_clients_with_consent(self, user_id: str) -> List[str]:
        """Get all client IDs that have user consent."""
        query = select(ConsentRecord.client_id).where(
            ConsentRecord.user_id == user_id,
            ConsentRecord.granted == True,
            ConsentRecord.revoked_at == None,
        ).distinct()
        
        result = await self.session.execute(query)
        return [row[0] for row in result.fetchall()]

    async def cleanup_expired_consents(self) -> int:
        """Remove expired consents. Returns count of removed records."""
        from sqlalchemy import delete
        
        now = datetime.utcnow()
        
        query = delete(ConsentRecord).where(
            ConsentRecord.expires_at < now,
        )
        
        result = await self.session.execute(query)
        await self.session.commit()
        
        logger.info("expired_consents_cleaned", count=result.rowcount)
        return result.rowcount
