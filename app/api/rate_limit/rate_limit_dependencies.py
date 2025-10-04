"""
FastAPI Dependencies for Rate Limiting

Provides dependency injection for rate limiting services and utilities:
- Rate limit service access
- Tenant quota extraction
- Rate limit checking decorators
- Admin bypass detection
- Rate limit header utilities
"""

from typing import Optional, Dict, Any, Annotated
from fastapi import Depends, Request, HTTPException
from fastapi.security import HTTPBearer
import structlog

from app.domain.interfaces import IRateLimitService, RateLimitType, TimeWindow, RateLimitRule
from app.domain.exceptions import RateLimitExceededError
from app.infrastructure.providers.rate_limit_provider import get_rate_limit_service
from app.models.tenant_models import QuotaLimits, RateLimitConfiguration
from app.core.dependencies import CurrentUserDep, TenantContextDep


logger = structlog.get_logger(__name__)

# Define type aliases for dependencies
RateLimitServiceDep = Annotated[IRateLimitService, Depends(get_rate_limit_service)]


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Client IP address
    """
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


def get_user_agent(request: Request) -> str:
    """
    Extract user agent from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: User agent string
    """
    return request.headers.get("user-agent", "unknown")


def is_admin_user(current_user: CurrentUserDep) -> bool:
    """
    Check if current user is an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        bool: True if user is admin
    """
    if not current_user:
        return False
    
    user_roles = getattr(current_user, "roles", [])
    return any(role in ["admin", "super_admin"] for role in user_roles)


def get_tenant_quota_limits(tenant_context: TenantContextDep) -> Optional[QuotaLimits]:
    """
    Get tenant quota limits from context.
    
    Args:
        tenant_context: Tenant context dependency
        
    Returns:
        Optional[QuotaLimits]: Tenant quota limits if available
    """
    if not tenant_context:
        return None
    
    # Extract quota limits from tenant configuration
    if hasattr(tenant_context, "quota_limits"):
        return tenant_context.quota_limits
    
    return None


def get_tenant_rate_limit_config(tenant_context: TenantContextDep) -> Optional[RateLimitConfiguration]:
    """
    Get tenant rate limiting configuration.
    
    Args:
        tenant_context: Tenant context dependency
        
    Returns:
        Optional[RateLimitConfiguration]: Rate limit configuration if available
    """
    if not tenant_context:
        return None
    
    # Extract rate limit config from tenant processing configuration
    if hasattr(tenant_context, "processing_configuration"):
        processing_config = tenant_context.processing_configuration
        if hasattr(processing_config, "rate_limit_config"):
            return processing_config.rate_limit_config
    
    return None


async def check_user_rate_limit(
    request: Request,
    current_user: CurrentUserDep,
    tenant_context: TenantContextDep,
    rate_limit_service: RateLimitServiceDep,
    *,
    requests_per_minute: int = 30,
    requests_per_hour: int = 500
) -> None:
    """
    Check user-specific rate limits.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        tenant_context: Tenant context
        rate_limit_service: Rate limiting service
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    if not current_user:
        return  # Skip for unauthenticated requests
    
    # Skip for admin users if configured
    if is_admin_user(current_user):
        rate_limit_config = get_tenant_rate_limit_config(tenant_context)
        if rate_limit_config and rate_limit_config.bypass_for_admins:
            return
    
    user_id = str(current_user.user_id)
    tenant_id = str(current_user.tenant_id) if current_user.tenant_id else None
    
    try:
        # Check minute limit
        minute_result = await rate_limit_service.check_rate_limit(
            identifier=user_id,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.MINUTE,
            max_requests=requests_per_minute,
            tenant_id=tenant_id
        )
        
        if not minute_result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"User rate limit exceeded: {requests_per_minute} requests per minute",
                    "limit_type": "user",
                    "time_window": "minute",
                    "retry_after": minute_result.retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(minute_result.reset_time.timestamp())),
                    "Retry-After": str(minute_result.retry_after) if minute_result.retry_after else "60"
                }
            )
        
        # Check hour limit
        hour_result = await rate_limit_service.check_rate_limit(
            identifier=user_id,
            limit_type=RateLimitType.USER,
            time_window=TimeWindow.HOUR,
            max_requests=requests_per_hour,
            tenant_id=tenant_id
        )
        
        if not hour_result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"User rate limit exceeded: {requests_per_hour} requests per hour",
                    "limit_type": "user",
                    "time_window": "hour",
                    "retry_after": hour_result.retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(requests_per_hour),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(hour_result.reset_time.timestamp())),
                    "Retry-After": str(hour_result.retry_after) if hour_result.retry_after else "3600"
                }
            )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error("User rate limit check failed", user_id=user_id, error=str(e))
        # Fail open - allow request if rate limiting fails


async def check_ip_rate_limit(
    request: Request,
    tenant_context: TenantContextDep,
    rate_limit_service: RateLimitServiceDep,
    *,
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000
) -> None:
    """
    Check IP-based rate limits.
    
    Args:
        request: FastAPI request object
        tenant_context: Tenant context
        rate_limit_service: Rate limiting service
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    client_ip = get_client_ip(request)
    if client_ip == "unknown":
        return  # Skip if we can't determine IP
    
    # Get tenant ID if available
    tenant_id = None
    if tenant_context and hasattr(tenant_context, "id"):
        tenant_id = str(tenant_context.id)
    
    try:
        # Check minute limit
        minute_result = await rate_limit_service.check_rate_limit(
            identifier=client_ip,
            limit_type=RateLimitType.IP,
            time_window=TimeWindow.MINUTE,
            max_requests=requests_per_minute,
            tenant_id=tenant_id
        )
        
        if not minute_result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"IP rate limit exceeded: {requests_per_minute} requests per minute",
                    "limit_type": "ip",
                    "time_window": "minute",
                    "retry_after": minute_result.retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(minute_result.reset_time.timestamp())),
                    "Retry-After": str(minute_result.retry_after) if minute_result.retry_after else "60"
                }
            )
        
        # Check hour limit
        hour_result = await rate_limit_service.check_rate_limit(
            identifier=client_ip,
            limit_type=RateLimitType.IP,
            time_window=TimeWindow.HOUR,
            max_requests=requests_per_hour,
            tenant_id=tenant_id
        )
        
        if not hour_result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"IP rate limit exceeded: {requests_per_hour} requests per hour",
                    "limit_type": "ip",
                    "time_window": "hour",
                    "retry_after": hour_result.retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(requests_per_hour),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(hour_result.reset_time.timestamp())),
                    "Retry-After": str(hour_result.retry_after) if hour_result.retry_after else "3600"
                }
            )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error("IP rate limit check failed", ip=client_ip, error=str(e))
        # Fail open - allow request if rate limiting fails


async def check_upload_rate_limit(
    request: Request,
    current_user: CurrentUserDep,
    tenant_context: TenantContextDep,
    rate_limit_service: RateLimitServiceDep
) -> None:
    """
    Check rate limits specific to upload endpoints.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        tenant_context: Tenant context
        rate_limit_service: Rate limiting service
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Get tenant quota limits
    quota_limits = get_tenant_quota_limits(tenant_context)
    
    # Use tenant-specific limits if available, otherwise use defaults
    upload_per_minute = 10
    upload_per_hour = 100
    
    if quota_limits:
        upload_per_minute = quota_limits.max_upload_requests_per_minute
        upload_per_hour = quota_limits.max_upload_requests_per_hour
    
    # Check user upload limits
    await check_user_rate_limit(
        request=request,
        current_user=current_user,
        tenant_context=tenant_context,
        rate_limit_service=rate_limit_service,
        requests_per_minute=upload_per_minute,
        requests_per_hour=upload_per_hour
    )
    
    # Also check IP limits for upload endpoints (stricter)
    await check_ip_rate_limit(
        request=request,
        tenant_context=tenant_context,
        rate_limit_service=rate_limit_service,
        requests_per_minute=15,  # Stricter IP limit for uploads
        requests_per_hour=200
    )


def create_rate_limit_dependency(
    requests_per_minute: int,
    requests_per_hour: int,
    limit_type: RateLimitType = RateLimitType.USER
):
    """
    Create a custom rate limit dependency with specific limits.
    
    Args:
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        limit_type: Type of rate limit to apply
        
    Returns:
        Dependency function for FastAPI
    """
    async def rate_limit_dependency(
        request: Request,
        current_user: CurrentUserDep,
        tenant_context: TenantContextDep,
        rate_limit_service: RateLimitServiceDep
    ) -> None:
        if limit_type == RateLimitType.USER:
            await check_user_rate_limit(
                request=request,
                current_user=current_user,
                tenant_context=tenant_context,
                rate_limit_service=rate_limit_service,
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour
            )
        elif limit_type == RateLimitType.IP:
            await check_ip_rate_limit(
                request=request,
                tenant_context=tenant_context,
                rate_limit_service=rate_limit_service,
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour
            )
    
    return rate_limit_dependency


# Pre-configured dependencies for common use cases
UserRateLimitDep = Depends(
    create_rate_limit_dependency(30, 500, RateLimitType.USER)
)

IPRateLimitDep = Depends(
    create_rate_limit_dependency(60, 1000, RateLimitType.IP)
)

UploadRateLimitDep = Depends(check_upload_rate_limit)

StrictRateLimitDep = Depends(
    create_rate_limit_dependency(10, 100, RateLimitType.USER)
)


class RateLimitChecker:
    """Helper class for manual rate limit checking in endpoints."""
    
    def __init__(self, rate_limit_service: IRateLimitService):
        self.rate_limit_service = rate_limit_service
    
    async def check_custom_limit(
        self,
        identifier: str,
        limit_type: RateLimitType,
        time_window: TimeWindow,
        max_requests: int,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check a custom rate limit and return detailed results.
        
        Args:
            identifier: Identifier to check
            limit_type: Type of rate limit
            time_window: Time window
            max_requests: Maximum requests allowed
            tenant_id: Tenant context
            
        Returns:
            Dict[str, Any]: Rate limit check results
        """
        try:
            result = await self.rate_limit_service.check_rate_limit(
                identifier=identifier,
                limit_type=limit_type,
                time_window=time_window,
                max_requests=max_requests,
                tenant_id=tenant_id,
                increment=True
            )
            
            return {
                "allowed": result.allowed,
                "limit_type": result.limit_type.value,
                "time_window": result.time_window.value,
                "max_requests": result.max_requests,
                "current_usage": result.current_usage,
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "retry_after": result.retry_after
            }
            
        except Exception as e:
            logger.error("Custom rate limit check failed", error=str(e))
            return {
                "allowed": True,  # Fail open
                "error": str(e)
            }


def get_rate_limit_checker(
    rate_limit_service: RateLimitServiceDep
) -> RateLimitChecker:
    """Dependency to get rate limit checker instance."""
    return RateLimitChecker(rate_limit_service)


RateLimitCheckerDep = Annotated[RateLimitChecker, Depends(get_rate_limit_checker)]