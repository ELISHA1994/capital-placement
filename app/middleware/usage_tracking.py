"""
Usage Tracking Middleware

FastAPI middleware for automatic API request tracking with tenant-aware metrics.
Provides comprehensive request monitoring while maintaining sub-5ms overhead.

Features:
- Automatic API request counting
- Response time tracking
- Status code monitoring
- Tenant-aware metrics collection
- Non-blocking background processing
- Error handling that never fails requests
"""

import asyncio
import time
from typing import Optional
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.infrastructure.providers.usage_provider import get_usage_service

logger = structlog.get_logger(__name__)


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for tracking API usage metrics.
    
    Automatically tracks all API requests with tenant isolation,
    response times, and status codes. Designed to have minimal
    performance impact on request processing.
    """
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        """
        Initialize usage tracking middleware.
        
        Args:
            app: FastAPI application instance
            excluded_paths: List of path patterns to exclude from tracking
        """
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
            "/static/",
            "/metrics"  # Prometheus/monitoring endpoints
        ]
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and track usage metrics.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint in chain
            
        Returns:
            Response: HTTP response
        """
        start_time = time.time()
        
        # Skip tracking for excluded paths
        if self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Track usage in background (non-blocking)
        asyncio.create_task(
            self._track_request_usage(
                request=request,
                response=response,
                response_time_ms=response_time_ms
            )
        )
        
        return response
    
    def _should_exclude_path(self, path: str) -> bool:
        """
        Check if path should be excluded from tracking.
        
        Args:
            path: Request path
            
        Returns:
            bool: True if path should be excluded
        """
        for excluded_path in self.excluded_paths:
            if path.startswith(excluded_path):
                return True
        return False
    
    async def _track_request_usage(
        self,
        request: Request,
        response: Response,
        response_time_ms: float
    ) -> None:
        """
        Track API request usage in background.
        
        Args:
            request: HTTP request
            response: HTTP response
            response_time_ms: Response time in milliseconds
        """
        try:
            # Extract tenant ID from request
            tenant_id = self._extract_tenant_id(request)
            
            if not tenant_id:
                # Skip tracking if no tenant context
                return
            
            # Extract request details
            endpoint = self._get_endpoint_pattern(request)
            method = request.method
            status_code = response.status_code
            
            # Track API usage (get tracker from provider)
            usage_tracker = await get_usage_service()
            await usage_tracker.track_api_usage(
                tenant_id=tenant_id,
                endpoint=endpoint,
                method=method,
                response_time_ms=response_time_ms,
                status_code=status_code
            )
            
        except Exception as e:
            logger.warning(
                "Failed to track API usage",
                error=str(e),
                path=request.url.path,
                method=request.method
            )
            # NEVER re-raise - tracking failures should not affect requests
    
    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """
        Extract tenant ID from request context.
        
        Args:
            request: HTTP request
            
        Returns:
            Optional[str]: Tenant ID if available
        """
        try:
            # Try to get tenant ID from various sources
            
            # 1. From authenticated user in request state
            if hasattr(request.state, 'current_user'):
                current_user = getattr(request.state, 'current_user')
                if hasattr(current_user, 'tenant_id') and current_user.tenant_id:
                    return str(current_user.tenant_id)
            
            # 2. From path parameters (for tenant-scoped APIs)
            path_params = getattr(request, 'path_params', {})
            if 'tenant_id' in path_params:
                return str(path_params['tenant_id'])
            
            # 3. From query parameters
            if 'tenant_id' in request.query_params:
                return str(request.query_params['tenant_id'])
            
            # 4. From headers (if explicitly passed)
            if 'X-Tenant-ID' in request.headers:
                return str(request.headers['X-Tenant-ID'])
            
            return None
            
        except Exception as e:
            logger.debug(
                "Could not extract tenant ID",
                error=str(e),
                path=request.url.path
            )
            return None
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """
        Get normalized endpoint pattern for consistent tracking.
        
        Args:
            request: HTTP request
            
        Returns:
            str: Normalized endpoint pattern
        """
        try:
            # Get the route pattern if available
            if hasattr(request, 'route') and request.route:
                route = request.route
                if hasattr(route, 'path'):
                    return route.path
            
            # Fallback to raw path with parameter normalization
            path = request.url.path
            
            # Normalize common ID patterns
            import re
            
            # Replace UUIDs with {id}
            uuid_pattern = r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
            path = re.sub(uuid_pattern, '/{id}', path, flags=re.IGNORECASE)
            
            # Replace numeric IDs with {id}
            numeric_id_pattern = r'/\d+'
            path = re.sub(numeric_id_pattern, '/{id}', path)
            
            return path
            
        except Exception as e:
            logger.debug(
                "Could not get endpoint pattern",
                error=str(e),
                path=request.url.path
            )
            return request.url.path


# Configuration helper
def create_usage_tracking_middleware(
    excluded_paths: Optional[list] = None
) -> type:
    """
    Factory function to create configured usage tracking middleware.
    
    Args:
        excluded_paths: List of path patterns to exclude from tracking
        
    Returns:
        Configured middleware class
    """
    class ConfiguredUsageTrackingMiddleware(UsageTrackingMiddleware):
        def __init__(self, app):
            super().__init__(app, excluded_paths=excluded_paths)
    
    return ConfiguredUsageTrackingMiddleware


# Default middleware instance
DEFAULT_EXCLUDED_PATHS = [
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
    "/static/",
    "/metrics",
    "/_internal/",  # Internal endpoints
    "/api/v1/setup/"  # Setup endpoints (pre-tenant)
]

DefaultUsageTrackingMiddleware = create_usage_tracking_middleware(
    excluded_paths=DEFAULT_EXCLUDED_PATHS
)