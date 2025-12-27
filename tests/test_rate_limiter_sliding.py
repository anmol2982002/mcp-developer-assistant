"""
Tests for Sliding Window Rate Limiter

Tests for:
- Sliding window accuracy
- Per-user quotas
- Burst handling
- Rate limit headers
"""

import pytest
import time
from datetime import datetime, timedelta

from proxy.rate_limiter import (
    SlidingWindowRateLimiter,
    UserQuota,
    RateLimitResult,
    SlidingWindowState,
)


@pytest.fixture
def mock_config(monkeypatch):
    """Mock proxy configuration."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "10")
    monkeypatch.setenv("RATE_LIMIT_BURST", "5")
    monkeypatch.setenv("RATE_LIMIT_BACKOFF_ENABLED", "true")


@pytest.fixture
def rate_limiter(mock_config):
    """Create rate limiter instance."""
    return SlidingWindowRateLimiter()


class TestSlidingWindowState:
    """Tests for sliding window state management."""
    
    def test_add_request(self):
        """Should add requests to window."""
        state = SlidingWindowState()
        state.add_request(time.time())
        
        assert state.count() == 1
    
    def test_cleanup_removes_old_requests(self):
        """Should remove requests outside the window."""
        state = SlidingWindowState()
        now = time.time()
        
        # Add old request
        state.add_request(now - 120)  # 2 minutes ago
        
        # Add recent request
        state.add_request(now)
        
        # Cleanup with 60-second window
        state.cleanup(60, now)
        
        assert state.count() == 1  # Only recent request remains
    
    def test_count_sums_all_requests(self):
        """Should sum all request counts."""
        state = SlidingWindowState()
        now = time.time()
        
        state.add_request(now, 3)
        state.add_request(now, 2)
        
        assert state.count() == 5


class TestRateLimitResult:
    """Tests for rate limit result and headers."""
    
    def test_get_headers_allowed(self):
        """Should generate correct headers for allowed request."""
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            limit=10,
            reset_at=datetime.utcnow() + timedelta(minutes=1),
        )
        
        headers = result.get_headers()
        
        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "5"
        assert "X-RateLimit-Reset" in headers
        assert "Retry-After" not in headers
    
    def test_get_headers_denied(self):
        """Should include Retry-After when denied."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=10,
            reset_at=datetime.utcnow() + timedelta(minutes=1),
            retry_after=30,
        )
        
        headers = result.get_headers()
        
        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["Retry-After"] == "30"
    
    def test_remaining_never_negative(self):
        """Remaining should never be negative in headers."""
        result = RateLimitResult(
            allowed=False,
            remaining=-5,  # Negative
            limit=10,
            reset_at=datetime.utcnow(),
        )
        
        headers = result.get_headers()
        
        assert headers["X-RateLimit-Remaining"] == "0"


class TestRateLimiter:
    """Tests for rate limiter functionality."""
    
    def test_allows_initial_requests(self, rate_limiter):
        """Should allow initial requests."""
        is_limited = rate_limiter.is_rate_limited("user1", "read_file")
        
        assert is_limited is False
    
    def test_rate_limits_after_burst(self, rate_limiter):
        """Should rate limit after burst is exhausted."""
        # Exhaust the burst limit (10 requests/min limit)
        for i in range(15):
            rate_limiter.is_rate_limited("user2", "read_file")
        
        # Next request should be limited
        is_limited = rate_limiter.is_rate_limited("user2", "read_file")
        
        assert is_limited is True
    
    def test_different_users_independent(self, rate_limiter):
        """Different users should have independent limits."""
        # Exhaust user1's limit
        for i in range(15):
            rate_limiter.is_rate_limited("user1", "read_file")
        
        # user2 should still be allowed
        is_limited = rate_limiter.is_rate_limited("user2", "read_file")
        
        assert is_limited is False
    
    def test_get_retry_after(self, rate_limiter):
        """Should return valid retry_after time."""
        retry_after = rate_limiter.get_retry_after("user1", "read_file")
        
        assert isinstance(retry_after, float)
        assert retry_after >= 0
    
    def test_get_user_usage(self, rate_limiter):
        """Should return usage across all windows."""
        # Make some requests
        rate_limiter.is_rate_limited("user1", "read_file")
        rate_limiter.is_rate_limited("user1", "read_file")
        
        usage = rate_limiter.get_user_usage("user1")
        
        assert "minute" in usage
        assert "hour" in usage
        assert "day" in usage
        assert usage["minute"] == 2
    
    def test_reset_user_limits(self, rate_limiter):
        """Should reset all user limits."""
        # Make some requests
        for i in range(5):
            rate_limiter.is_rate_limited("user1", "read_file")
        
        # Reset
        rate_limiter.reset_user_limits("user1")
        
        # Usage should be zero
        usage = rate_limiter.get_user_usage("user1")
        assert usage["minute"] == 0


class TestUserQuota:
    """Tests for user quota model."""
    
    def test_default_quota_values(self):
        """Should have sensible defaults."""
        quota = UserQuota(user_id="user1")
        
        assert quota.requests_per_minute == 60
        assert quota.requests_per_hour == 1000
        assert quota.daily_limit == 10000
        assert quota.burst_size == 10
        assert quota.tier == "default"
    
    def test_custom_quota_values(self):
        """Should accept custom values."""
        quota = UserQuota(
            user_id="user1",
            requests_per_minute=120,
            tier="premium",
        )
        
        assert quota.requests_per_minute == 120
        assert quota.tier == "premium"


class TestAsyncRateLimiter:
    """Async tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter):
        """Should allow requests within limit."""
        result = await rate_limiter.check_rate_limit("user1")
        
        assert result.allowed is True
        assert result.remaining >= 0
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_headers(self, rate_limiter):
        """Should include proper headers."""
        result = await rate_limiter.check_rate_limit("user1")
        headers = result.get_headers()
        
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
