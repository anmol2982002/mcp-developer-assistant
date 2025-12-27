"""
ML Cache Database Operations

Cache ML results for efficiency.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.schema import MLCache
from observability.logging_config import get_logger

logger = get_logger(__name__)


class MLCacheDB:
    """ML result caching operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(
        self,
        cache_key: str,
        result_type: str,
    ) -> Optional[dict]:
        """Get cached result."""
        query = (
            select(MLCache)
            .where(MLCache.cache_key == cache_key)
            .where(MLCache.result_type == result_type)
            .where(MLCache.expires_at > datetime.utcnow())
        )

        result = await self.session.execute(query)
        cache_entry = result.scalar_one_or_none()

        if cache_entry:
            # Update hit count
            cache_entry.hits += 1
            await self.session.commit()

            logger.debug("cache_hit", key=cache_key, type=result_type)
            return json.loads(cache_entry.result)

        return None

    async def set(
        self,
        cache_key: str,
        result_type: str,
        result: Any,
        confidence: float = 1.0,
        ttl_seconds: int = 3600,
    ) -> MLCache:
        """Set cached result."""
        # Check if exists
        query = (
            select(MLCache)
            .where(MLCache.cache_key == cache_key)
            .where(MLCache.result_type == result_type)
        )
        existing = await self.session.execute(query)
        entry = existing.scalar_one_or_none()

        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        if entry:
            entry.result = json.dumps(result)
            entry.confidence = confidence
            entry.expires_at = expires_at
            entry.created_at = datetime.utcnow()
        else:
            entry = MLCache(
                cache_key=cache_key,
                result_type=result_type,
                result=json.dumps(result),
                confidence=confidence,
                expires_at=expires_at,
            )
            self.session.add(entry)

        await self.session.commit()
        logger.debug("cache_set", key=cache_key, type=result_type)
        return entry

    async def delete(
        self,
        cache_key: str,
        result_type: Optional[str] = None,
    ) -> int:
        """Delete cached result."""
        query = delete(MLCache).where(MLCache.cache_key == cache_key)

        if result_type:
            query = query.where(MLCache.result_type == result_type)

        result = await self.session.execute(query)
        await self.session.commit()
        return result.rowcount

    async def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        query = delete(MLCache).where(MLCache.expires_at < datetime.utcnow())

        result = await self.session.execute(query)
        await self.session.commit()

        logger.info("cache_cleanup", removed=result.rowcount)
        return result.rowcount
