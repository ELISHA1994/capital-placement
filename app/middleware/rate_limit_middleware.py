"""
FastAPI Rate Limiting Middleware

Provides comprehensive rate limiting at the API layer with:
- Multiple limit types (user, tenant, IP, endpoint)
- Tenant-aware rate limiting based on quota configuration
- Special handling for upload endpoints
- Proper HTTP headers for rate limit information
- Integration with audit logging for violations
- Admin user bypass support
"""

import time
from typing import Callable, Dict, List, Optional
import structlog
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.domain.interfaces import (
    IRateLimitService, IAuditService, RateLimitType, TimeWindow, 
    RateLimitRule, RateLimitResult, RateLimitViolation
)
from app.domain.exceptions import RateLimitExceededError
from app.models.tenant_models import QuotaLimits


logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI applications.
    
    Features:
    - Multiple simultaneous rate limits per request
    - Tenant-aware limiting based on subscription quotas
    - Special upload endpoint protection
    - Admin user bypass capabilities
    - Comprehensive audit logging
    - Proper HTTP headers for client guidance
    """
    
    def __init__(
        self,
        app,
        rate_limit_service: Optional[IRateLimitService] = None,
        audit_service: Optional[IAuditService] = None,
        *,
        enable_global_limits: bool = True,
        enable_ip_limits: bool = True,
        enable_user_limits: bool = True,
        enable_tenant_limits: bool = True,
        enable_endpoint_limits: bool = True,
        bypass_for_admins: bool = True,
        fail_open: bool = True,  # Allow requests if rate limiting fails
        settings = None  # Allow passing settings for lazy initialization
    ):
        super().__init__(app)
        self.rate_limit_service = rate_limit_service
        self.audit_service = audit_service
        self.enable_global_limits = enable_global_limits
        self.enable_ip_limits = enable_ip_limits
        self.enable_user_limits = enable_user_limits
        self.enable_tenant_limits = enable_tenant_limits
        self.enable_endpoint_limits = enable_endpoint_limits
        self.bypass_for_admins = bypass_for_admins
        self.fail_open = fail_open
        self.settings = settings
        
        # Lazy initialization support
        self._initialized = False
        self._initialization_lock = None
        
        # Default rate limiting rules
        self.default_rules = self._get_default_rules()
        
        # Special rules for upload endpoints
        self.upload_rules = self._get_upload_rules()
        
        # Paths to exclude from rate limiting
        self.excluded_paths = {
            "/health",
            "/health/detailed",
            "/health/database",
            "/health/migrations",
            "/health/transactions",
            "/",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def _initialize_services(self) -> bool:
        """
        Initialize rate limiting services on first request.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._initialized:
            return True
            
        # Use a simple flag-based approach for initialization
        if self._initialization_lock is None:
            try:
                # Initialize rate limit service if not provided
                if self.rate_limit_service is None:
                    from app.infrastructure.providers.rate_limit_provider import get_rate_limit_service
                    self.rate_limit_service = await get_rate_limit_service()
                
                # Initialize audit service if not provided (optional)
                if self.audit_service is None:
                    try:
                        from app.infrastructure.providers.audit_provider import get_audit_service
                        self.audit_service = await get_audit_service()
                    except ImportError:
                        logger.info("Audit service not available for rate limiting")
                
                self._initialized = True
                logger.info("Rate limiting middleware services initialized successfully")
                return True
                
            except Exception as e:
                logger.error("Failed to initialize rate limiting services", error=str(e))
                if self.settings and self.settings.ENVIRONMENT == "production":
                    raise  # Don't start in production without rate limiting
                return False
            finally:
                self._initialization_lock = True
        
        return self._initialized
    
    def _get_default_rules(self) -> List[RateLimitRule]:
        """Get default rate limiting rules."""
        return [
            # Global limits (applied to all requests)
            RateLimitRule(
                limit_type=RateLimitType.GLOBAL,
                time_window=TimeWindow.MINUTE,
                max_requests=1000,
                description="Global requests per minute",
                priority=10
            ),
            RateLimitRule(
                limit_type=RateLimitType.GLOBAL,
                time_window=TimeWindow.HOUR,
                max_requests=10000,
                description="Global requests per hour",
                priority=9
            ),
            
            # IP-based limits (DDoS protection)
            RateLimitRule(
                limit_type=RateLimitType.IP,
                time_window=TimeWindow.MINUTE,
                max_requests=60,
                description="IP requests per minute",
                priority=8
            ),
            RateLimitRule(
                limit_type=RateLimitType.IP,
                time_window=TimeWindow.HOUR,
                max_requests=1000,
                description="IP requests per hour",
                priority=7
            ),
            
            # User-based limits (prevent individual abuse)
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=30,
                description="User requests per minute",
                priority=6
            ),
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.HOUR,
                max_requests=500,
                description="User requests per hour",
                priority=5
            ),
        ]
    
    def _get_upload_rules(self) -> List[RateLimitRule]:
        """Get special rate limiting rules for upload endpoints."""
        return [
            # Stricter limits for uploads
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.MINUTE,
                max_requests=10,
                description="User upload requests per minute",
                priority=20
            ),
            RateLimitRule(
                limit_type=RateLimitType.USER,
                time_window=TimeWindow.HOUR,
                max_requests=100,
                description="User upload requests per hour",
                priority=19
            ),
            RateLimitRule(
                limit_type=RateLimitType.IP,
                time_window=TimeWindow.MINUTE,
                max_requests=15,
                description="IP upload requests per minute",
                priority=18
            ),
            RateLimitRule(
                limit_type=RateLimitType.TENANT,
                time_window=TimeWindow.MINUTE,
                max_requests=50,
                description="Tenant upload requests per minute",
                priority=17
            ),
        ]
    
    def _get_tenant_rules(self, quota_limits: QuotaLimits) -> List[RateLimitRule]:
        """Get tenant-specific rate limiting rules based on quota configuration."""
        rules = []
        
        if quota_limits.max_api_requests_per_minute:
            rules.append(RateLimitRule(
                limit_type=RateLimitType.TENANT,
                time_window=TimeWindow.MINUTE,
                max_requests=quota_limits.max_api_requests_per_minute,
                description="Tenant API requests per minute",
                priority=15
            ))
        
        if quota_limits.max_api_requests_per_hour:
            rules.append(RateLimitRule(
                limit_type=RateLimitType.TENANT,
                time_window=TimeWindow.HOUR,
                max_requests=quota_limits.max_api_requests_per_hour,
                description="Tenant API requests per hour",
                priority=14
            ))
        
        if quota_limits.max_api_requests_per_day:
            rules.append(RateLimitRule(
                limit_type=RateLimitType.TENANT,
                time_window=TimeWindow.DAY,
                max_requests=quota_limits.max_api_requests_per_day,
                description="Tenant API requests per day",
                priority=13
            ))
        
        return rules
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (from load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct connection IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _should_bypass_rate_limiting(self, request: Request) -> bool:
        """Check if request should bypass rate limiting."""
        
        # Check excluded paths
        path = request.url.path
        if path in self.excluded_paths:
            return True
        
        # Check if path starts with excluded prefixes
        excluded_prefixes = ["/docs", "/redoc", "/openapi"]
        if any(path.startswith(prefix) for prefix in excluded_prefixes):
            return True
        
        # Check for admin bypass
        if self.bypass_for_admins:
            # Check if user has admin role (this would be set by auth middleware)
            user_roles = getattr(request.state, "user_roles", [])
            if "admin" in user_roles or "super_admin" in user_roles:
                return True
        
        return False
    
    def _is_upload_endpoint(self, request: Request) -> bool:
        """Check if request is to an upload endpoint."""
        path = request.url.path
        return (
            path.startswith("/api/v1/upload") or
            "upload" in path.lower() or
            request.method == "POST" and any(
                keyword in path.lower() 
                for keyword in ["file", "document", "cv", "batch"]
            )
        )
    
    def _get_identifiers(self, request: Request) -> Dict[RateLimitType, str]:
        """Extract rate limiting identifiers from request."""
        identifiers = {}
        
        # IP address
        client_ip = self._get_client_ip(request)
        identifiers[RateLimitType.IP] = client_ip
        
        # Global identifier (always the same)
        identifiers[RateLimitType.GLOBAL] = "global"
        
        # User identifier (if authenticated)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            identifiers[RateLimitType.USER] = str(user_id)
        
        # Tenant identifier (if available)
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            identifiers[RateLimitType.TENANT] = str(tenant_id)
        
        # API key identifier (if using API key auth)
        api_key_id = getattr(request.state, "api_key_id", None)
        if api_key_id:
            identifiers[RateLimitType.API_KEY] = str(api_key_id)
        
        # Endpoint identifier
        endpoint_pattern = f"{request.method}:{request.url.path}"
        identifiers[RateLimitType.ENDPOINT] = endpoint_pattern
        
        return identifiers
    
    def _add_rate_limit_headers(self, response: Response, results: List[RateLimitResult]) -> None:
        """Add rate limit headers to response."""
        if not results:
            return
        
        # Find the most restrictive limit for headers
        most_restrictive = min(results, key=lambda r: r.remaining)
        
        response.headers["X-RateLimit-Limit"] = str(most_restrictive.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(most_restrictive.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(most_restrictive.reset_time.timestamp()))
        
        if most_restrictive.time_window:
            response.headers["X-RateLimit-Window"] = most_restrictive.time_window.value
        
        # Add retry-after header if request was blocked
        if not most_restrictive.allowed and most_restrictive.retry_after:
            response.headers["Retry-After"] = str(most_restrictive.retry_after)
    
    async def _log_rate_limit_violation(
        self,
        request: Request,
        blocked_result: RateLimitResult,
        identifiers: Dict[RateLimitType, str]
    ) -> None:
        """Log rate limit violation for audit purposes."""
        if not self.audit_service:
            return
        
        try:
            tenant_id = identifiers.get(RateLimitType.TENANT, "unknown")
            user_id = identifiers.get(RateLimitType.USER)
            
            violation = RateLimitViolation(
                limit_type=blocked_result.limit_type,
                time_window=blocked_result.time_window,
                max_requests=blocked_result.max_requests,
                actual_requests=blocked_result.current_usage,
                identifier=blocked_result.identifier,
                tenant_id=tenant_id,
                user_id=user_id,
                ip_address=identifiers.get(RateLimitType.IP, "unknown"),
                user_agent=request.headers.get("user-agent", "unknown"),
                endpoint=f"{request.method} {request.url.path}"
            )
            
            await self.audit_service.log_security_event(
                event_type="rate_limit_exceeded",
                tenant_id=tenant_id,
                threat_type="rate_limit_violation",
                severity="medium",
                user_id=user_id,
                ip_address=violation.ip_address,
                user_agent=violation.user_agent,
                threat_details={
                    "limit_type": violation.limit_type.value,
                    "time_window": violation.time_window.value,
                    "max_requests": violation.max_requests,
                    "actual_requests": violation.actual_requests,
                    "endpoint": violation.endpoint,
                    "identifier": violation.identifier
                }
            )
            
        except Exception as e:
            logger.error("Failed to log rate limit violation", error=str(e))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting middleware."""
        
        # Skip rate limiting for excluded paths or admin users
        if self._should_bypass_rate_limiting(request):
            return await call_next(request)
        
        # Initialize services on first request if using lazy initialization
        if not self._initialized:
            initialization_success = await self._initialize_services()
            
            # If initialization failed, handle based on fail_open setting
            if not initialization_success:
                if self.fail_open:
                    # Fail open - allow request to proceed
                    logger.warning("Rate limiting initialization failed, proceeding without rate limiting")
                    return await call_next(request)
                else:
                    # Fail closed - reject request
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "rate_limiting_initialization_failed",
                            "message": "Rate limiting service unavailable"
                        }
                    )
        
        start_time = time.time()
        
        try:
            # Extract identifiers for rate limiting
            identifiers = self._get_identifiers(request)
            
            # Determine which rules to apply
            rules = []
            
            # Add default rules based on configuration
            if self.enable_global_limits:
                rules.extend([r for r in self.default_rules if r.limit_type == RateLimitType.GLOBAL])
            
            if self.enable_ip_limits:
                rules.extend([r for r in self.default_rules if r.limit_type == RateLimitType.IP])
            
            if self.enable_user_limits and RateLimitType.USER in identifiers:
                rules.extend([r for r in self.default_rules if r.limit_type == RateLimitType.USER])
            
            # Add tenant-specific rules if tenant context available
            if self.enable_tenant_limits and RateLimitType.TENANT in identifiers:
                # Get tenant quota limits from request state if available
                quota_limits = getattr(request.state, "quota_limits", None)
                if quota_limits:
                    tenant_rules = self._get_tenant_rules(quota_limits)
                    rules.extend(tenant_rules)
            
            # Add special upload rules if this is an upload endpoint
            if self._is_upload_endpoint(request):
                rules.extend(self.upload_rules)
            
            # Check rate limits
            if rules:
                results = await self.rate_limit_service.check_multiple_limits(
                    identifiers=identifiers,
                    rules=rules,
                    tenant_id=identifiers.get(RateLimitType.TENANT),
                    increment=True
                )
                
                # Check if any limit was exceeded
                blocked_results = [r for r in results if not r.allowed]
                if blocked_results:
                    # Find the most restrictive violated limit
                    most_restrictive = min(blocked_results, key=lambda r: r.retry_after or 0)
                    
                    # Log the violation
                    await self._log_rate_limit_violation(request, most_restrictive, identifiers)
                    
                    # Return rate limit exceeded response
                    response = JSONResponse(
                        status_code=429,
                        content={
                            "error": "rate_limit_exceeded",
                            "message": f"Rate limit exceeded: {most_restrictive.max_requests} requests per {most_restrictive.time_window.value}",
                            "details": {
                                "limit_type": most_restrictive.limit_type.value,
                                "time_window": most_restrictive.time_window.value,
                                "max_requests": most_restrictive.max_requests,
                                "current_usage": most_restrictive.current_usage,
                                "remaining": most_restrictive.remaining,
                                "reset_time": most_restrictive.reset_time.isoformat(),
                                "retry_after": most_restrictive.retry_after
                            }
                        }
                    )
                    
                    # Add rate limit headers
                    self._add_rate_limit_headers(response, results)
                    
                    return response
            else:
                results = []
            
            # Process the request normally
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            if results:
                self._add_rate_limit_headers(response, results)
            
            # Log processing time
            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                "Rate limit check completed",
                processing_time_ms=processing_time,
                rules_checked=len(rules),
                path=request.url.path,
                method=request.method
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Rate limiting middleware error",
                error=str(e),
                path=request.url.path,
                method=request.method
            )
            
            # Fail open if configured to do so
            if self.fail_open:
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "rate_limiting_error",
                        "message": "Rate limiting service temporarily unavailable"
                    }
                )


class LazyRateLimitMiddleware(RateLimitMiddleware):
    """
    Convenience class for lazy-loading rate limit middleware.
    
    This is specifically designed for use during app creation where async
    initialization is not possible. Services will be initialized on first request.
    """
    
    def __init__(self, app, settings, **kwargs):
        """
        Initialize with settings for lazy loading.
        
        Args:
            app: FastAPI application
            settings: Application settings for environment-specific behavior
            **kwargs: Additional configuration options
        """
        # Set defaults for lazy initialization
        kwargs.setdefault("enable_global_limits", True)
        kwargs.setdefault("enable_ip_limits", True)
        kwargs.setdefault("enable_user_limits", True)
        kwargs.setdefault("enable_tenant_limits", True)
        kwargs.setdefault("enable_endpoint_limits", True)
        kwargs.setdefault("bypass_for_admins", True)
        kwargs.setdefault("fail_open", settings.ENVIRONMENT != "production")
        
        # Initialize with None services for lazy loading
        super().__init__(
            app=app,
            rate_limit_service=None,
            audit_service=None,
            settings=settings,
            **kwargs
        )