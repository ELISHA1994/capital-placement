"""
Tests for rate limiting implementation.

Tests comprehensive rate limiting functionality including:
- Service-level rate limiting
- Middleware integration
- Tenant-aware rate limiting
- Error handling and fallbacks
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from app.infrastructure.security.rate_limit_service import RateLimitService
from app.domain.interfaces import (
    IRateLimitService, ICacheService, RateLimitType, TimeWindow,
    RateLimitRule, RateLimitResult
)
from app.domain.exceptions import RateLimitExceededError
from app.middleware.rate_limit_middleware import RateLimitMiddleware


class MockCacheService:
    """Mock cache service for testing."""
    
    def __init__(self):
        self.cache = {}
        self.ttl_cache = {}
    
    async def get(self, key: str):
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
        self.cache[key] = value
        if ttl > 0:
            self.ttl_cache[key] = datetime.utcnow() + timedelta(seconds=ttl)
        return True
    
    async def delete(self, key: str):
        self.cache.pop(key, None)
        self.ttl_cache.pop(key, None)
        return True
    
    async def exists(self, key: str):
        return key in self.cache
    
    async def clear(self, pattern: str = "*"):
        cleared_count = len(self.cache)
        self.cache.clear()
        self.ttl_cache.clear()
        return cleared_count
    
    async def check_health(self):
        return {
            "status": "healthy",
            "backend": "memory",
            "entries": len(self.cache)
        }


@pytest.fixture
def mock_cache_service():
    """Provide mock cache service."""
    return MockCacheService()


@pytest.fixture
def rate_limit_service(mock_cache_service):
    """Provide rate limit service with mock cache."""
    return RateLimitService(cache_service=mock_cache_service)


@pytest.mark.asyncio
class TestRateLimitService:
    """Test rate limiting service functionality."""
    
    async def test_service_health_check(self, rate_limit_service):
        """Test service health check."""
        health = await rate_limit_service.check_health()
        assert health["status"] == "healthy"
        assert "cache_backend" in health
    
    async def test_basic_rate_limit_check(self, rate_limit_service):
        """Test basic rate limit checking."""
        # First request should be allowed
        result = await rate_limit_service.check_rate_limit(
            identifier="test_user",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )
        
        assert result.allowed is True
        assert result.current_usage == 1
        assert result.remaining == 4
        assert result.max_requests == 5
    
    async def test_rate_limit_exceeded(self, rate_limit_service):
        """Test rate limit exceeded scenario."""
        identifier = "test_user_limited"
        
        # Make requests up to the limit
        for i in range(3):
            result = await rate_limit_service.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=3
            )
            assert result.allowed is True
            assert result.current_usage == i + 1
        
        # Next request should be blocked
        result = await rate_limit_service.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=3
        )
        
        assert result.allowed is False
        assert result.current_usage == 4
        assert result.remaining == 0
        assert result.retry_after is not None
    
    async def test_multiple_limits_check(self, rate_limit_service):
        """Test checking multiple rate limits simultaneously."""
        identifiers = {
            RateLimitType.USER: "test_user",
            RateLimitType.IP: "192.168.1.1"
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
            )
        ]
        
        results = await rate_limit_service.check_multiple_limits(
            identifiers=identifiers,
            rules=rules
        )
        
        assert len(results) == 2
        assert all(result.allowed for result in results)
    
    async def test_whitelist_functionality(self, rate_limit_service):
        """Test whitelist functionality."""
        identifier = "whitelisted_user"
        
        # Add to whitelist
        success = await rate_limit_service.add_to_whitelist(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert success is True
        
        # Check if whitelisted
        is_whitelisted = await rate_limit_service.is_whitelisted(
            identifier=identifier,
            limit_type=RateLimitType.USER
        )
        assert is_whitelisted is True
        
        # Rate limit check should always allow
        result = await rate_limit_service.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=1  # Very low limit
        )
        assert result.allowed is True
        assert result.current_usage == 0  # Whitelisted requests don't count
    
    async def test_rate_limit_reset(self, rate_limit_service):
        """Test rate limit reset functionality."""
        identifier = "reset_test_user"
        
        # Make some requests
        for _ in range(3):
            await rate_limit_service.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=5
            )
        
        # Reset the limit
        success = await rate_limit_service.reset_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE
        )
        assert success is True
        
        # Next request should start fresh
        result = await rate_limit_service.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5,
            increment=False  # Just check without incrementing
        )
        assert result.current_usage == 0
    
    async def test_tenant_isolation(self, rate_limit_service):
        """Test tenant isolation in rate limiting."""
        identifier = "shared_user"
        tenant1 = "tenant1"
        tenant2 = "tenant2"
        
        # Make requests for tenant1
        for _ in range(3):
            result = await rate_limit_service.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=5,
                tenant_id=tenant1
            )
            assert result.allowed is True
        
        # Requests for tenant2 should start fresh
        result = await rate_limit_service.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5,
            tenant_id=tenant2
        )
        assert result.allowed is True
        assert result.current_usage == 1  # Fresh start for tenant2


@pytest.mark.asyncio
class TestRateLimitMiddleware:
    """Test rate limiting middleware functionality."""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI app."""
        async def app(request):
            return {"message": "success"}
        return app
    
    @pytest.fixture
    def mock_request(self):
        """Mock request object."""
        request = Mock()
        request.url.path = "/api/v1/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-agent"}
        request.client.host = "192.168.1.1"
        request.state = Mock()
        request.state.user_id = "test_user"
        request.state.tenant_id = "test_tenant"
        request.state.user_roles = []
        return request
    
    @pytest.fixture
    def rate_limit_middleware(self, rate_limit_service, mock_app):
        """Create rate limit middleware."""
        return RateLimitMiddleware(
            app=mock_app,
            rate_limit_service=rate_limit_service,
            audit_service=None,
            fail_open=True
        )
    
    async def test_middleware_allows_excluded_paths(self, rate_limit_middleware, mock_request):
        """Test middleware excludes health check paths."""
        mock_request.url.path = "/health"
        
        # Mock call_next
        async def call_next(request):
            return {"message": "health ok"}
        
        response = await rate_limit_middleware.dispatch(mock_request, call_next)
        assert response["message"] == "health ok"
    
    async def test_middleware_extracts_identifiers(self, rate_limit_middleware):
        """Test middleware correctly extracts rate limiting identifiers."""
        mock_request = Mock()
        mock_request.url.path = "/api/v1/test"
        mock_request.headers = {
            "user-agent": "test-agent",
            "x-forwarded-for": "203.0.113.1, 192.168.1.1"
        }
        mock_request.state = Mock()
        mock_request.state.user_id = "test_user"
        mock_request.state.tenant_id = "test_tenant"
        
        identifiers = rate_limit_middleware._get_identifiers(mock_request)
        
        assert identifiers[RateLimitType.IP] == "203.0.113.1"
        assert identifiers[RateLimitType.USER] == "test_user"
        assert identifiers[RateLimitType.TENANT] == "test_tenant"
        assert identifiers[RateLimitType.GLOBAL] == "global"
    
    async def test_middleware_admin_bypass(self, rate_limit_middleware, mock_request):
        """Test middleware bypasses rate limits for admin users."""
        mock_request.state.user_roles = ["admin"]
        
        should_bypass = rate_limit_middleware._should_bypass_rate_limiting(mock_request)
        assert should_bypass is True
    
    async def test_middleware_upload_endpoint_detection(self, rate_limit_middleware, mock_request):
        """Test middleware detects upload endpoints."""
        mock_request.url.path = "/api/v1/upload"
        mock_request.method = "POST"
        
        is_upload = rate_limit_middleware._is_upload_endpoint(mock_request)
        assert is_upload is True
        
        # Test batch upload
        mock_request.url.path = "/api/v1/upload/batch"
        is_upload = rate_limit_middleware._is_upload_endpoint(mock_request)
        assert is_upload is True


@pytest.mark.asyncio 
class TestRateLimitIntegration:
    """Integration tests for rate limiting."""
    
    async def test_concurrent_requests(self, rate_limit_service):
        """Test rate limiting under concurrent requests."""
        identifier = "concurrent_user"
        max_requests = 5
        
        async def make_request():
            return await rate_limit_service.check_rate_limit(
                identifier=identifier,
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=max_requests
            )
        
        # Make concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Some should be allowed, some should be blocked
        allowed_count = sum(1 for result in results if result.allowed)
        blocked_count = sum(1 for result in results if not result.allowed)
        
        assert allowed_count <= max_requests
        assert blocked_count > 0
        assert allowed_count + blocked_count == 10
    
    async def test_different_time_windows(self, rate_limit_service):
        """Test rate limiting across different time windows."""
        identifier = "time_window_user"
        
        # Test minute window
        minute_result = await rate_limit_service.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )
        assert minute_result.allowed is True
        assert minute_result.current_usage == 1
        
        # Test hour window (should be independent)
        hour_result = await rate_limit_service.check_rate_limit(
            identifier=identifier,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.HOUR,
            max_requests=10
        )
        assert hour_result.allowed is True
        assert hour_result.current_usage == 1  # Independent counter
    
    async def test_error_handling_fail_open(self, mock_cache_service):
        """Test rate limiting fails open when cache service fails."""
        
        # Mock cache service to raise errors
        async def failing_get(key):
            raise Exception("Cache service unavailable")
        
        mock_cache_service.get = failing_get
        rate_limit_service = RateLimitService(cache_service=mock_cache_service)
        
        # Should fail open and allow the request
        result = await rate_limit_service.check_rate_limit(
            identifier="test_user",
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=5
        )
        
        assert result.allowed is True  # Failed open
        assert result.current_usage == 0  # No increment due to error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])