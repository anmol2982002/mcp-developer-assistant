"""
Consent Registry Database Operations

Manage user consent for clients.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.schema import ConsentRecord
from observability.logging_config import get_logger

logger = get_logger(__name__)


class ConsentRegistryDB:
    """Consent registry operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def has_consent(
        self,
        user_id: str,
        client_id: str,
        required_scopes: Optional[List[str]] = None,
    ) -> bool:
        """Check if client has user consent."""
        query = (
            select(ConsentRecord)
            .where(ConsentRecord.user_id == user_id)
            .where(ConsentRecord.client_id == client_id)
            .where(ConsentRecord.granted == True)
            .where(ConsentRecord.revoked_at == None)
        )

        result = await self.session.execute(query)
        consent = result.scalar_one_or_none()

        if not consent:
            return False

        # Check scopes
        if required_scopes:
            granted_scopes = set(consent.scopes.split(",") if consent.scopes else [])
            if not all(s in granted_scopes for s in required_scopes):
                return False

        return True

    async def grant_consent(
        self,
        user_id: str,
        client_id: str,
        scopes: Optional[List[str]] = None,
    ) -> ConsentRecord:
        """Grant consent to a client."""
        consent = ConsentRecord(
            user_id=user_id,
            client_id=client_id,
            granted=True,
            scopes=",".join(scopes) if scopes else "",
        )

        self.session.add(consent)
        await self.session.commit()

        logger.info("consent_granted", user_id=user_id, client_id=client_id)
        return consent

    async def revoke_consent(self, user_id: str, client_id: str) -> bool:
        """Revoke consent from a client."""
        query = (
            select(ConsentRecord)
            .where(ConsentRecord.user_id == user_id)
            .where(ConsentRecord.client_id == client_id)
            .where(ConsentRecord.granted == True)
            .where(ConsentRecord.revoked_at == None)
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

    async def list_consents(self, user_id: str) -> List[ConsentRecord]:
        """List all active consents for a user."""
        query = (
            select(ConsentRecord)
            .where(ConsentRecord.user_id == user_id)
            .where(ConsentRecord.granted == True)
            .where(ConsentRecord.revoked_at == None)
        )

        result = await self.session.execute(query)
        return result.scalars().all()
