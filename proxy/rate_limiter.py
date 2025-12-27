"""
Sliding Window Rate Limiter

Production-grade rate limiting with sliding window algorithm,
per-user quotas, burst handling, and proper HTTP headers.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.schema import UserQuota as UserQuotaModel
from observability.logging_config import get_logger
from proxy.config import get_proxy_config

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class UserQuota(BaseModel):
    """User rate limit quota configuration."""
    
    user_id: str
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    daily_limit: int = 10000
    burst_size: int = 10
    tier: str = "default"


class RateLimitResult(BaseModel):
    """Result of a rate limit check."""
    
    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after: Optional[int] = None
    window: str = "minute"  # minute, hour, day
    
    def get_headers(self) -> Dict[str, str]:
        """Generate rate limit headers for HTTP response."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
            "X-RateLimit-Window": self.window,
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return headers


@dataclass
class SlidingWindowState:
    """State for sliding window rate limiting."""
    
    # Requests in the current window: list of (timestamp, count)
    requests: List[Tuple[float, int]] = field(default_factory=list)
    
    # Backoff state
    consecutive_violations: int = 0
    last_violation: Optional[float] = None
    
    def cleanup(self, window_seconds: float, now: float):
        """Remove requests outside the window."""
        cutoff = now - window_seconds
        self.requests = [(ts, count) for ts, count in self.requests if ts > cutoff]
    
    def count(self) -> int:
        """Get total request count in window."""
        return sum(count for _, count in self.requests)
    
    def add_request(self, timestamp: float, count: int = 1):
        """Add a request to the window."""
        self.requests.append((timestamp, count))


# -----------------------------------------------------------------------------
# Rate Limiter
# -----------------------------------------------------------------------------


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with per-user quotas.
    
    Features:
    - Sliding window counter (more accurate than fixed windows)
    - Per-user configurable limits via database
    - Multiple time windows (minute, hour, day)
    - Burst handling with exponential backoff
    - Rate limit headers in responses
    
    The sliding window algorithm provides smoother rate limiting
    compared to fixed windows by considering requests across
    window boundaries.
    """
    
    def __init__(self):
        self.config = get_proxy_config()
        
        # In-memory state for sliding windows
        # Key: (user_id, window_type) -> SlidingWindowState
        self._minute_windows: Dict[str, SlidingWindowState] = defaultdict(SlidingWindowState)
        self._hour_windows: Dict[str, SlidingWindowState] = defaultdict(SlidingWindowState)
        self._day_windows: Dict[str, SlidingWindowState] = defaultdict(SlidingWindowState)
        
        # User quota cache
        self._quota_cache: Dict[str, Tuple[UserQuota, float]] = {}
        self._quota_cache_ttl = 300  # 5 minutes
        
        # Default quota
        self._default_quota = UserQuota(
            user_id="default",
            requests_per_minute=self.config.rate_limit_requests_per_minute,
            requests_per_hour=1000,
            daily_limit=10000,
            burst_size=self.config.rate_limit_burst,
        )
    
    async def check_rate_limit(
        self,
        user_id: str,
        tool_name: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> RateLimitResult:
        """
        Check if a request should be rate limited.
        
        Args:
            user_id: User identifier
            tool_name: Optional tool name for fine-grained limiting
            session: Database session for quota lookup
            
        Returns:
            RateLimitResult with allowed status and metadata
        """
        if not self.config.rate_limit_enabled:
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                limit=999999,
                reset_at=datetime.utcnow() + timedelta(minutes=1),
            )
        
        now = time.time()
        quota = await self._get_user_quota(user_id, session)
        
        # Check each window (minute -> hour -> day)
        # Short windows take precedence
        
        # Minute window
        minute_result = self._check_window(
            user_id=user_id,
            windows=self._minute_windows,
            window_seconds=60,
            limit=quota.requests_per_minute,
            burst_size=quota.burst_size,
            window_name="minute",
            now=now,
        )
        
        if not minute_result.allowed:
            return minute_result
        
        # Hour window
        hour_result = self._check_window(
            user_id=user_id,
            windows=self._hour_windows,
            window_seconds=3600,
            limit=quota.requests_per_hour,
            burst_size=quota.burst_size * 5,
            window_name="hour",
            now=now,
        )
        
        if not hour_result.allowed:
            return hour_result
        
        # Day window
        day_result = self._check_window(
            user_id=user_id,
            windows=self._day_windows,
            window_seconds=86400,
            limit=quota.daily_limit,
            burst_size=quota.burst_size * 10,
            window_name="day",
            now=now,
        )
        
        if not day_result.allowed:
            return day_result
        
        # All checks passed - record the request
        self._record_request(user_id, now)
        
        # Return the most restrictive remaining count
        min_remaining = min(
            minute_result.remaining,
            hour_result.remaining,
            day_result.remaining,
        )
        
        return RateLimitResult(
            allowed=True,
            remaining=min_remaining,
            limit=quota.requests_per_minute,
            reset_at=minute_result.reset_at,
            window="minute",
        )
    
    def _check_window(
        self,
        user_id: str,
        windows: Dict[str, SlidingWindowState],
        window_seconds: float,
        limit: int,
        burst_size: int,
        window_name: str,
        now: float,
    ) -> RateLimitResult:
        """Check rate limit for a specific time window."""
        state = windows[user_id]
        state.cleanup(window_seconds, now)
        
        current_count = state.count()
        remaining = limit - current_count
        reset_at = datetime.fromtimestamp(now + window_seconds)
        
        # Check if within limit
        if current_count < limit:
            # Allow with burst consideration
            effective_limit = min(limit, current_count + burst_size)
            return RateLimitResult(
                allowed=True,
                remaining=limit - current_count - 1,  # -1 for this request
                limit=limit,
                reset_at=reset_at,
                window=window_name,
            )
        
        # Rate limited - calculate retry after with backoff
        retry_after = self._calculate_retry_after(state, window_seconds, now)
        
        # Record violation
        state.consecutive_violations += 1
        state.last_violation = now
        
        logger.warning(
            "rate_limited",
            user_id=user_id,
            window=window_name,
            limit=limit,
            current=current_count,
            retry_after=retry_after,
            consecutive_violations=state.consecutive_violations,
        )
        
        return RateLimitResult(
            allowed=False,
            remaining=0,
            limit=limit,
            reset_at=reset_at,
            retry_after=retry_after,
            window=window_name,
        )
    
    def _calculate_retry_after(
        self,
        state: SlidingWindowState,
        window_seconds: float,
        now: float,
    ) -> int:
        """Calculate retry-after with exponential backoff for repeat offenders."""
        if not self.config.rate_limit_backoff_enabled:
            # Simple: wait for oldest request to expire
            if state.requests:
                oldest = min(ts for ts, _ in state.requests)
                return max(1, int(oldest + window_seconds - now))
            return int(window_seconds)
        
        # Exponential backoff based on consecutive violations
        base_wait = window_seconds / 60  # Base wait in seconds
        multiplier = self.config.rate_limit_backoff_multiplier ** min(
            state.consecutive_violations, 5  # Cap at 5 doublings
        )
        
        return max(1, int(base_wait * multiplier))
    
    def _record_request(self, user_id: str, timestamp: float):
        """Record a request in all windows."""
        self._minute_windows[user_id].add_request(timestamp)
        self._hour_windows[user_id].add_request(timestamp)
        self._day_windows[user_id].add_request(timestamp)
        
        # Reset consecutive violations on successful request
        self._minute_windows[user_id].consecutive_violations = 0
    
    async def _get_user_quota(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> UserQuota:
        """Get user quota from cache or database."""
        now = time.time()
        
        # Check cache
        if user_id in self._quota_cache:
            quota, cached_at = self._quota_cache[user_id]
            if now - cached_at < self._quota_cache_ttl:
                return quota
        
        # Try database lookup
        if session:
            try:
                query = select(UserQuotaModel).where(UserQuotaModel.user_id == user_id)
                result = await session.execute(query)
                db_quota = result.scalar_one_or_none()
                
                if db_quota:
                    quota = UserQuota(
                        user_id=user_id,
                        requests_per_minute=db_quota.requests_per_minute,
                        requests_per_hour=db_quota.requests_per_hour,
                        daily_limit=db_quota.daily_limit,
                        burst_size=db_quota.burst_size,
                        tier=db_quota.tier,
                    )
                    self._quota_cache[user_id] = (quota, now)
                    return quota
            except Exception as e:
                logger.warning("quota_lookup_failed", user_id=user_id, error=str(e))
        
        # Return default quota
        default = UserQuota(
            user_id=user_id,
            requests_per_minute=self._default_quota.requests_per_minute,
            requests_per_hour=self._default_quota.requests_per_hour,
            daily_limit=self._default_quota.daily_limit,
            burst_size=self._default_quota.burst_size,
            tier="default",
        )
        self._quota_cache[user_id] = (default, now)
        return default
    
    async def set_user_quota(
        self,
        user_id: str,
        quota: UserQuota,
        session: AsyncSession,
    ) -> UserQuotaModel:
        """Set or update user quota in database."""
        from sqlalchemy import update
        from database.schema import UserQuota as UserQuotaModel
        
        # Try update first
        query = (
            update(UserQuotaModel)
            .where(UserQuotaModel.user_id == user_id)
            .values(
                requests_per_minute=quota.requests_per_minute,
                requests_per_hour=quota.requests_per_hour,
                daily_limit=quota.daily_limit,
                burst_size=quota.burst_size,
                tier=quota.tier,
                updated_at=datetime.utcnow(),
            )
        )
        
        result = await session.execute(query)
        
        if result.rowcount == 0:
            # Insert new quota
            db_quota = UserQuotaModel(
                user_id=user_id,
                requests_per_minute=quota.requests_per_minute,
                requests_per_hour=quota.requests_per_hour,
                daily_limit=quota.daily_limit,
                burst_size=quota.burst_size,
                tier=quota.tier,
            )
            session.add(db_quota)
        
        await session.commit()
        
        # Clear cache
        if user_id in self._quota_cache:
            del self._quota_cache[user_id]
        
        logger.info("user_quota_updated", user_id=user_id, tier=quota.tier)
        return db_quota if result.rowcount == 0 else None
    
    def reset_user_limits(self, user_id: str):
        """Reset all rate limit windows for a user."""
        if user_id in self._minute_windows:
            del self._minute_windows[user_id]
        if user_id in self._hour_windows:
            del self._hour_windows[user_id]
        if user_id in self._day_windows:
            del self._day_windows[user_id]
        
        logger.info("user_limits_reset", user_id=user_id)
    
    def get_user_usage(self, user_id: str) -> Dict[str, int]:
        """Get current usage across all windows."""
        return {
            "minute": self._minute_windows[user_id].count(),
            "hour": self._hour_windows[user_id].count(),
            "day": self._day_windows[user_id].count(),
        }
    
    # -------------------------------------------------------------------------
    # Legacy API compatibility
    # -------------------------------------------------------------------------
    
    def is_rate_limited(self, user_id: str, tool_name: str) -> bool:
        """Legacy API: Check if rate limited (sync)."""
        # Simple in-memory check without async
        now = time.time()
        state = self._minute_windows[user_id]
        state.cleanup(60, now)
        
        if state.count() >= self._default_quota.requests_per_minute:
            return True
        
        # Record request
        state.add_request(now)
        return False
    
    def get_retry_after(self, user_id: str, tool_name: str) -> float:
        """Legacy API: Get retry after seconds."""
        now = time.time()
        state = self._minute_windows[user_id]
        
        if not state.requests:
            return 0
        
        oldest = min(ts for ts, _ in state.requests)
        return max(0, oldest + 60 - now)


# Singleton
rate_limiter = SlidingWindowRateLimiter()
