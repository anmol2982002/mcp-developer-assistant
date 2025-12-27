"""
Log Retention

Manages log rotation, retention policies, and cleanup for audit logs.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.schema import AuditLog, get_db
from observability.logging_config import get_logger

logger = get_logger(__name__)


class RetentionPolicy:
    """Audit log retention policy configuration."""
    
    def __init__(
        self,
        retention_days: int = 90,
        archive_enabled: bool = False,
        archive_path: Optional[str] = None,
        cleanup_batch_size: int = 1000,
    ):
        """
        Initialize retention policy.
        
        Args:
            retention_days: Days to keep logs (default 90)
            archive_enabled: Whether to archive before deletion
            archive_path: Path for archived logs
            cleanup_batch_size: Number of records to delete per batch
        """
        self.retention_days = retention_days
        self.archive_enabled = archive_enabled
        self.archive_path = archive_path
        self.cleanup_batch_size = cleanup_batch_size


class LogRetentionManager:
    """
    Manages audit log retention and cleanup.
    
    Features:
    - Configurable retention periods
    - Batch deletion for performance
    - Optional archiving before deletion
    - Scheduled cleanup support
    """
    
    def __init__(self, policy: Optional[RetentionPolicy] = None):
        self.policy = policy or RetentionPolicy()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def cleanup_old_logs(
        self,
        session: AsyncSession,
        retention_days: Optional[int] = None,
    ) -> int:
        """
        Delete logs older than retention period.
        
        Args:
            session: Database session
            retention_days: Override retention period
            
        Returns:
            Number of deleted records
        """
        days = retention_days or self.policy.retention_days
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        total_deleted = 0
        
        while True:
            # Get batch of old log IDs
            query = (
                select(AuditLog.id)
                .where(AuditLog.timestamp < cutoff)
                .limit(self.policy.cleanup_batch_size)
            )
            
            result = await session.execute(query)
            ids = [row[0] for row in result.fetchall()]
            
            if not ids:
                break
            
            # Archive if enabled
            if self.policy.archive_enabled:
                await self._archive_logs(session, ids)
            
            # Delete batch
            delete_query = delete(AuditLog).where(AuditLog.id.in_(ids))
            await session.execute(delete_query)
            await session.commit()
            
            total_deleted += len(ids)
            
            logger.debug(
                "log_cleanup_batch",
                deleted=len(ids),
                total=total_deleted,
            )
        
        if total_deleted > 0:
            logger.info(
                "log_cleanup_complete",
                deleted=total_deleted,
                retention_days=days,
            )
        
        return total_deleted
    
    async def _archive_logs(
        self,
        session: AsyncSession,
        log_ids: list,
    ):
        """Archive logs before deletion."""
        if not self.policy.archive_path:
            return
        
        import json
        from pathlib import Path
        
        # Get full log records
        query = select(AuditLog).where(AuditLog.id.in_(log_ids))
        result = await session.execute(query)
        logs = result.scalars().all()
        
        # Write to archive file
        archive_file = Path(self.policy.archive_path) / f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(archive_file, "a") as f:
            for log in logs:
                record = {
                    "id": log.id,
                    "user_id": log.user_id,
                    "client_id": log.client_id,
                    "tool_name": log.tool_name,
                    "status": log.status,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "duration_ms": log.duration_ms,
                    "anomaly_score": log.anomaly_score,
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info("logs_archived", count=len(logs), file=str(archive_file))
    
    async def get_retention_stats(self, session: AsyncSession) -> dict:
        """Get statistics about log retention."""
        now = datetime.utcnow()
        
        # Total logs
        total_query = select(func.count(AuditLog.id))
        total_result = await session.execute(total_query)
        total_count = total_result.scalar()
        
        # Logs eligible for deletion
        cutoff = now - timedelta(days=self.policy.retention_days)
        old_query = select(func.count(AuditLog.id)).where(AuditLog.timestamp < cutoff)
        old_result = await session.execute(old_query)
        old_count = old_result.scalar()
        
        # Oldest log
        oldest_query = select(func.min(AuditLog.timestamp))
        oldest_result = await session.execute(oldest_query)
        oldest = oldest_result.scalar()
        
        # Newest log
        newest_query = select(func.max(AuditLog.timestamp))
        newest_result = await session.execute(newest_query)
        newest = newest_result.scalar()
        
        return {
            "total_logs": total_count,
            "logs_to_delete": old_count,
            "retention_days": self.policy.retention_days,
            "oldest_log": oldest.isoformat() if oldest else None,
            "newest_log": newest.isoformat() if newest else None,
            "archive_enabled": self.policy.archive_enabled,
        }
    
    def start_scheduled_cleanup(
        self,
        interval_hours: int = 24,
    ):
        """Start scheduled cleanup task."""
        
        async def cleanup_loop():
            while True:
                try:
                    async with get_db() as session:
                        await self.cleanup_old_logs(session)
                except Exception as e:
                    logger.error("scheduled_cleanup_error", error=str(e))
                
                await asyncio.sleep(interval_hours * 3600)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("scheduled_cleanup_started", interval_hours=interval_hours)
    
    def stop_scheduled_cleanup(self):
        """Stop scheduled cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("scheduled_cleanup_stopped")


# Default manager
log_retention_manager = LogRetentionManager()


async def cleanup_logs(retention_days: int = 90) -> int:
    """Convenience function for log cleanup."""
    async with get_db() as session:
        return await log_retention_manager.cleanup_old_logs(
            session,
            retention_days=retention_days,
        )
