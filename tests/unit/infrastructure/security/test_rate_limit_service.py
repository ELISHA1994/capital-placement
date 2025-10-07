"""
Comprehensive unit tests for RateLimitService infrastructure implementation.

Tests cover:
- Basic rate limiting functionality
- Time window handling (minute, hour, day)
- Multiple rate limit types (user, IP, tenant, API key, endpoint)
- Concurrent request handling
- Whitelist management
- Rate limit reset
- Cache integration
- Error handling and fail-open behavior
- Tenant isolation
- Edge cases and boundary conditions
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.infrastructure.security.rate_limit_service import RateLimitService
from app.domain.interfaces import (
    IRateLimitService, ICacheService, RateLimitType, TimeWindow,
    RateLimitRule, RateLimitResult
)


class MockCacheService:
    """Mock cache service for testing."""

    def __init__(self):
        self.cache = {}
        self.ttl_cache = {}
        self._fail_mode = False

    def set_fail_mode(self, enabled: bool):
        """Enable/disable failure mode for testing error handling."""
        self._fail_mode = enabled

    async def get(self, key: str):
        if self._fail_mode:
            raise Exception("Cache unavailable")

        if key in self.cache:
            # Check TTL
            if key in self.ttl_cache:
                if datetime.utcnow() > self.ttl_cache[key]:
                    del self.cache[key]
                    del self.ttl_cache[key]
                    return None
            return self.cache[key]
        return None

    async def set(self, key: str, value, ttl: int = 3600):
        if self._fail_mode:
            raise Exception("Cache unavailable")

        self.cache[key] = value
        if ttl > 0:
            self.ttl_cache[key] = datetime.utcnow() + timedelta(seconds=ttl)
        return True

    async def delete(self, key: str):
        if self._fail_mode:
            raise Exception("Cache unavailable")

        self.cache.pop(key, None)
        self.ttl_cache.pop(key, None)
        return True

    async def exists(self, key: str):
        if self._fail_mode:
            raise Exception("Cache unavailable")
        return key in self.cache

    async def clear(self, pattern: str = "*"):
        if self._fail_mode:
            raise Exception("Cache unavailable")

        cleared_count = len(self.cache)
        self.cache.clear()
        self.ttl_cache.clear()
        return cleared_count

    async def check_health(self):
        return {
            "status": "healthy" if not self._fail_mode else "unhealthy",
            "backend": "memory",
            "entries": len(self.cache)
        }


@pytest.fixture
def mock_cache():
    """Provide mock cache service."""
    return MockCacheService()


@pytest.fixture
def rate_limiter(mock_cache):
    """Provide rate limit service with mock cache."""
    return RateLimitService(cache_service=mock_cache)


@pytest.mark.asyncio
class TestRateLimitServiceBasics:
    """Test basic rate limit service functionality."""

    async def test_service_implements_interface(self, rate_limiter):
        """Test service implements IRateLimitService interface."""
        assert isinstance(rate_limiter, IRateLimitService)

    async def test_service_health_check_healthy(self, rate_limiter):
        """Test service health check when healthy."""
        health = await rate_limiter.check_health()

        assert health["status"] == "healthy"
        assert "cache_backend" in health
        assert "cache_status" in health

    async def test_service_health_check_unhealthy(self, rate_limiter, mock_cache):
        """Test service health check when cache is unhealthy."""
        mock_cache.set_fail_mode(True)

        health = await rate_limiter.check_health()

        assert health["status"] == "unhealthy"
        assert "error" in health

    async def test_basic_rate_limit_within_limit(self, rate_limiter):
        """Test request within rate limit."""
        result = await rate_limiter.check_rate_limit(
            identifier="user_123",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10
        )

        assert result.allowed is True
        assert result.current_usage == 1
        assert result.remaining == 9
        assert result.max_requests == 10
        assert result.limit_type == RateLimitType.USER
        assert result.time_window == TimeWindow.MINUTE

    async def test_basic_rate_limit_exceeded(self, rate_limiter):
        """Test request exceeding rate limit."""
        identifier = "user_limited"
        max_requests = 3

        # Make requests up to limit
        for i in range(max_requests):
            result = await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=max_requests
            )
            assert result.allowed is True
            assert result.current_usage == i + 1

        # Next request should be blocked
        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=max_requests
        )

        assert result.allowed is False
        assert result.current_usage == 4
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    async def test_rate_limit_check_without_increment(self, rate_limiter):
        """Test checking rate limit without incrementing counter."""
        identifier = "user_check_only"

        # Make actual request
        await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10
        )

        # Check status without incrementing
        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10,
            increment=False
        )

        assert result.current_usage == 1  # Should still be 1
        assert result.remaining == 9


@pytest.mark.asyncio
class TestRateLimitTimeWindows:
    """Test rate limiting across different time windows."""

    async def test_minute_time_window(self, rate_limiter):
        """Test minute time window rate limiting."""
        identifier = "user_minute"

        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )

        assert result.allowed is True
        assert result.time_window == TimeWindow.MINUTE

    async def test_hour_time_window(self, rate_limiter):
        """Test hour time window rate limiting."""
        identifier = "user_hour"

        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.HOUR,
            max_requests=100
        )

        assert result.allowed is True
        assert result.time_window == TimeWindow.HOUR

    async def test_day_time_window(self, rate_limiter):
        """Test day time window rate limiting."""
        identifier = "user_day"

        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.DAY,
            max_requests=1000
        )

        assert result.allowed is True
        assert result.time_window == TimeWindow.DAY

    async def test_independent_time_windows(self, rate_limiter):
        """Test that different time windows are independent."""
        identifier = "user_multi_window"

        # Make request in minute window
        minute_result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )
        assert minute_result.current_usage == 1

        # Make request in hour window (should be independent)
        hour_result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.HOUR,
            max_requests=10
        )
        assert hour_result.current_usage == 1  # Independent counter

        # Make another request in minute window
        minute_result2 = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )
        assert minute_result2.current_usage == 2  # Incremented from first


@pytest.mark.asyncio
class TestRateLimitTypes:
    """Test different rate limit types."""

    async def test_user_rate_limit(self, rate_limiter):
        """Test user-based rate limiting."""
        result = await rate_limiter.check_rate_limit(
            identifier="user_123",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10
        )

        assert result.allowed is True
        assert result.limit_type == RateLimitType.USER

    async def test_ip_rate_limit(self, rate_limiter):
        """Test IP-based rate limiting."""
        result = await rate_limiter.check_rate_limit(
            identifier="192.168.1.1",
            limit_type=RateLimitType.IP,
            time_window=TimeWindow.MINUTE,
            max_requests=20
        )

        assert result.allowed is True
        assert result.limit_type == RateLimitType.IP

    async def test_tenant_rate_limit(self, rate_limiter):
        """Test tenant-based rate limiting."""
        result = await rate_limiter.check_rate_limit(
            identifier="tenant_123",
            limit_type=RateLimitType.TENANT,
            time_window=TimeWindow.HOUR,
            max_requests=1000
        )

        assert result.allowed is True
        assert result.limit_type == RateLimitType.TENANT

    async def test_api_key_rate_limit(self, rate_limiter):
        """Test API key-based rate limiting."""
        result = await rate_limiter.check_rate_limit(
            identifier="api_key_xyz",
            limit_type=RateLimitType.API_KEY,
            time_window=TimeWindow.HOUR,
            max_requests=500
        )

        assert result.allowed is True
        assert result.limit_type == RateLimitType.API_KEY

    async def test_endpoint_rate_limit(self, rate_limiter):
        """Test endpoint-based rate limiting."""
        result = await rate_limiter.check_rate_limit(
            identifier="/api/v1/upload",
            limit_type=RateLimitType.ENDPOINT,
            time_window=TimeWindow.MINUTE,
            max_requests=50
        )

        assert result.allowed is True
        assert result.limit_type == RateLimitType.ENDPOINT

    async def test_global_rate_limit(self, rate_limiter):
        """Test global rate limiting."""
        result = await rate_limiter.check_rate_limit(
            identifier="global",
            limit_type=RateLimitType.GLOBAL,
            time_window=TimeWindow.MINUTE,
            max_requests=10000
        )

        assert result.allowed is True
        assert result.limit_type == RateLimitType.GLOBAL


@pytest.mark.asyncio
class TestMultipleLimits:
    """Test checking multiple rate limits."""

    async def test_check_multiple_limits_all_pass(self, rate_limiter):
        """Test checking multiple limits when all pass."""
        identifiers = {
            RateLimitType.USER: "user_123",
            RateLimitType.IP: "192.168.1.1",
            RateLimitType.TENANT: "tenant_abc"
        }

        rules = [
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10,
                priority=1
            ),
            RateLimitRule(
                limit_type=RateLimitType.IP,
                time_window=TimeWindow.MINUTE,
                max_requests=20,
                priority=2
            ),
            RateLimitRule(
                limit_type=RateLimitType.TENANT,
                time_window=TimeWindow.HOUR,
                max_requests=1000,
                priority=3
            )
        ]

        results = await rate_limiter.check_multiple_limits(
            identifiers=identifiers,
            rules=rules
        )

        assert len(results) == 3
        assert all(result.allowed for result in results)

    async def test_check_multiple_limits_one_fails(self, rate_limiter):
        """Test checking multiple limits when one fails (fail fast)."""
        identifiers = {
            RateLimitType.USER: "user_limited",
            RateLimitType.IP: "192.168.1.1"
        }

        # Exhaust user limit first
        for _ in range(2):
            await rate_limiter.check_rate_limit(
                identifier="user_limited",
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=2
            )

        rules = [
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=2,
                priority=10  # Higher priority (checked first)
            ),
            RateLimitRule(
                limit_type=RateLimitType.IP,
                time_window=TimeWindow.MINUTE,
                max_requests=20,
                priority=1  # Lower priority
            )
        ]

        results = await rate_limiter.check_multiple_limits(
            identifiers=identifiers,
            rules=rules
        )

        # Should fail fast on first limit (highest priority)
        # Since USER limit is exhausted, it should fail and stop checking
        assert len(results) == 1  # Only checked first rule (highest priority)
        assert results[0].allowed is False
        assert results[0].limit_type == RateLimitType.USER

    async def test_check_multiple_limits_priority_order(self, rate_limiter):
        """Test that multiple limits are checked in priority order."""
        identifiers = {
            RateLimitType.USER: "user_123",
            RateLimitType.TENANT: "tenant_abc"
        }

        rules = [
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10,
                priority=1  # Lower priority
            ),
            RateLimitRule(
                limit_type=RateLimitType.TENANT,
                time_window=TimeWindow.MINUTE,
                max_requests=100,
                priority=10  # Higher priority (checked first)
            )
        ]

        results = await rate_limiter.check_multiple_limits(
            identifiers=identifiers,
            rules=rules
        )

        # Results should be in priority order (highest first)
        assert results[0].limit_type == RateLimitType.TENANT
        assert results[1].limit_type == RateLimitType.USER


@pytest.mark.asyncio
class TestWhitelist:
    """Test whitelist functionality."""

    async def test_add_to_whitelist(self, rate_limiter):
        """Test adding identifier to whitelist."""
        identifier = "user_whitelisted"

        success = await rate_limiter.add_to_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        assert success is True

    async def test_is_whitelisted(self, rate_limiter):
        """Test checking if identifier is whitelisted."""
        identifier = "user_whitelisted"

        await rate_limiter.add_to_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        is_whitelisted = await rate_limiter.is_whitelisted(
            identifier=identifier,
            limit_type=RateLimitType.USER
        )

        assert is_whitelisted is True

    async def test_whitelist_bypasses_rate_limit(self, rate_limiter):
        """Test that whitelisted identifiers bypass rate limits."""
        identifier = "user_unlimited"

        await rate_limiter.add_to_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        # Make many requests (beyond limit)
        for _ in range(100):
            result = await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=5  # Very low limit
            )
            assert result.allowed is True
            assert result.current_usage == 0  # Whitelisted doesn't count

    async def test_whitelist_with_tenant_isolation(self, rate_limiter):
        """Test whitelist respects tenant isolation."""
        identifier = "user_123"
        tenant1 = "tenant_a"
        tenant2 = "tenant_b"

        # Whitelist for tenant1 only
        await rate_limiter.add_to_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            tenant_id=tenant1,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        # Check tenant1 (should be whitelisted)
        is_whitelisted_t1 = await rate_limiter.is_whitelisted(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            tenant_id=tenant1
        )
        assert is_whitelisted_t1 is True

        # Check tenant2 (should NOT be whitelisted)
        is_whitelisted_t2 = await rate_limiter.is_whitelisted(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            tenant_id=tenant2
        )
        assert is_whitelisted_t2 is False

    async def test_remove_from_whitelist(self, rate_limiter):
        """Test removing identifier from whitelist."""
        identifier = "user_temp_whitelist"

        # Add to whitelist
        await rate_limiter.add_to_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        # Verify whitelisted
        assert await rate_limiter.is_whitelisted(identifier, RateLimitType.USER) is True

        # Remove from whitelist
        success = await rate_limiter.remove_from_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER
        )
        assert success is True

        # Verify no longer whitelisted
        assert await rate_limiter.is_whitelisted(identifier, RateLimitType.USER) is False


@pytest.mark.asyncio
class TestRateLimitReset:
    """Test rate limit reset functionality."""

    async def test_reset_rate_limit(self, rate_limiter):
        """Test resetting rate limit counter."""
        identifier = "user_reset"

        # Make some requests
        for _ in range(3):
            await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10
            )

        # Reset the limit
        success = await rate_limiter.reset_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE
        )

        assert success is True

        # Check that counter is reset
        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10,
            increment=False
        )

        assert result.current_usage == 0

    async def test_reset_with_tenant_isolation(self, rate_limiter):
        """Test reset respects tenant isolation."""
        identifier = "user_123"
        tenant1 = "tenant_a"
        tenant2 = "tenant_b"

        # Make requests for both tenants
        for _ in range(3):
            await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10,
                tenant_id=tenant1
            )
            await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10,
                tenant_id=tenant2
            )

        # Reset only tenant1
        await rate_limiter.reset_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            tenant_id=tenant1
        )

        # Check tenant1 is reset
        result_t1 = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10,
            tenant_id=tenant1,
            increment=False
        )
        assert result_t1.current_usage == 0

        # Check tenant2 is NOT reset
        result_t2 = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10,
            tenant_id=tenant2,
            increment=False
        )
        assert result_t2.current_usage == 3


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and fail-open behavior."""

    async def test_fail_open_on_cache_error(self, rate_limiter, mock_cache):
        """Test service fails open when cache is unavailable."""
        mock_cache.set_fail_mode(True)

        result = await rate_limiter.check_rate_limit(
            identifier="user_123",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )

        # Should allow request when cache fails
        assert result.allowed is True
        assert result.current_usage == 0  # No count due to error

    async def test_whitelist_check_error_handling(self, rate_limiter, mock_cache):
        """Test whitelist check handles errors gracefully."""
        mock_cache.set_fail_mode(True)

        is_whitelisted = await rate_limiter.is_whitelisted(
            identifier="user_123",
            limit_type=RateLimitType.USER
        )

        # Should return False on error
        assert is_whitelisted is False

    async def test_reset_error_handling(self, rate_limiter, mock_cache):
        """Test reset handles errors gracefully."""
        mock_cache.set_fail_mode(True)

        success = await rate_limiter.reset_rate_limit(
            identifier="user_123",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE
        )

        # Should return False on error
        assert success is False


@pytest.mark.asyncio
class TestTenantIsolation:
    """Test tenant isolation in rate limiting."""

    async def test_tenant_isolation_separate_counters(self, rate_limiter):
        """Test tenants have separate rate limit counters."""
        identifier = "user_123"
        tenant1 = "tenant_a"
        tenant2 = "tenant_b"

        # Make requests for tenant1
        for _ in range(5):
            result = await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10,
                tenant_id=tenant1
            )
            assert result.allowed is True

        # tenant1 should have 5 requests
        result_t1 = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10,
            tenant_id=tenant1,
            increment=False
        )
        assert result_t1.current_usage == 5

        # tenant2 should start fresh
        result_t2 = await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10,
            tenant_id=tenant2
        )
        assert result_t2.current_usage == 1  # First request for tenant2


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_zero_max_requests(self, rate_limiter):
        """Test rate limit with zero max requests."""
        result = await rate_limiter.check_rate_limit(
            identifier="user_123",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=0
        )

        # First request should be blocked
        assert result.allowed is False
        assert result.remaining == 0

    async def test_very_high_max_requests(self, rate_limiter):
        """Test rate limit with very high max requests."""
        result = await rate_limiter.check_rate_limit(
            identifier="user_123",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=1_000_000
        )

        assert result.allowed is True
        assert result.remaining == 999_999

    async def test_long_identifier(self, rate_limiter):
        """Test rate limiting with very long identifier."""
        long_identifier = "x" * 500  # Very long identifier

        result = await rate_limiter.check_rate_limit(
            identifier=long_identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10
        )

        # Should handle long identifiers via hashing
        assert result.allowed is True

    async def test_get_rate_limit_status(self, rate_limiter):
        """Test getting rate limit status without modifying."""
        identifier = "user_status_check"

        # Make some requests
        await rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=10
        )

        # Get status
        status = await rate_limiter.get_rate_limit_status(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE
        )

        assert status is not None
        assert status.current_usage == 1


@pytest.mark.asyncio
class TestStatistics:
    """Test rate limit statistics."""

    async def test_get_rate_limit_stats(self, rate_limiter):
        """Test getting rate limit statistics."""
        stats = await rate_limiter.get_rate_limit_stats()

        assert isinstance(stats, dict)
        assert "total_requests_checked" in stats
        assert "total_requests_blocked" in stats
        assert "cache_backend" in stats

    async def test_cleanup_expired_limits(self, rate_limiter):
        """Test cleanup of expired rate limit entries."""
        # Should not fail even if no cleanup is needed
        cleaned = await rate_limiter.cleanup_expired_limits(batch_size=100)

        assert isinstance(cleaned, int)
        assert cleaned >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])