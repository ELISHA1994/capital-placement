"""
Rate Limiting Provider

Provides rate limiting services following the provider pattern with:
- Singleton instance management
- Cache service integration
- Health monitoring
- Cleanup utilities
"""

from typing import Optional
import structlog

from app.domain.interfaces import IRateLimitService
from app.infrastructure.security.rate_limit_service import RateLimitService
from app.infrastructure.providers.cache_provider import get_cache_service


logger = structlog.get_logger(__name__)

# Global singleton instance
_rate_limit_service: Optional[IRateLimitService] = None


async def get_rate_limit_service() -> IRateLimitService:
    """
    Get or create the rate limiting service singleton.
    
    Returns:
        IRateLimitService: Rate limiting service instance
    """
    global _rate_limit_service
    
    if _rate_limit_service is None:
        try:
            # Get cache service for backend storage
            cache_service = await get_cache_service()
            
            # Create rate limit service
            _rate_limit_service = RateLimitService(cache_service=cache_service)
            
            # Verify service health
            health = await _rate_limit_service.check_health()
            if health.get("status") != "healthy":
                logger.warning("Rate limit service health check failed", health=health)
            else:
                logger.info("Rate limit service initialized successfully", 
                          backend=health.get("cache_backend", "unknown"))
            
        except Exception as e:
            logger.error("Failed to initialize rate limit service", error=str(e))
            raise
    
    return _rate_limit_service


async def reset_rate_limit_service() -> None:
    """Reset the rate limiting service singleton."""
    global _rate_limit_service
    
    if _rate_limit_service is not None:
        try:
            # Check if service has any cleanup methods
            if hasattr(_rate_limit_service, "cleanup"):
                await _rate_limit_service.cleanup()
                
            logger.info("Rate limit service reset completed")
            
        except Exception as e:
            logger.error("Error during rate limit service reset", error=str(e))
        finally:
            _rate_limit_service = None


async def get_rate_limit_service_health() -> dict:
    """
    Get health status of the rate limiting service.
    
    Returns:
        dict: Health status information
    """
    try:
        service = await get_rate_limit_service()
        return await service.check_health()
    except Exception as e:
        logger.error("Failed to get rate limit service health", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def cleanup_expired_rate_limits(batch_size: int = 1000) -> dict:
    """
    Clean up expired rate limit entries.
    
    Args:
        batch_size: Number of entries to clean in one batch
        
    Returns:
        dict: Cleanup statistics
    """
    try:
        service = await get_rate_limit_service()
        cleaned_count = await service.cleanup_expired_limits(batch_size=batch_size)
        
        return {
            "status": "success",
            "cleaned_entries": cleaned_count,
            "batch_size": batch_size
        }
        
    except Exception as e:
        logger.error("Failed to cleanup expired rate limits", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "cleaned_entries": 0
        }


async def get_rate_limit_stats(
    tenant_id: Optional[str] = None,
    limit_type: Optional[str] = None
) -> dict:
    """
    Get rate limiting statistics.
    
    Args:
        tenant_id: Filter by tenant ID
        limit_type: Filter by limit type
        
    Returns:
        dict: Rate limiting statistics
    """
    try:
        service = await get_rate_limit_service()
        
        # Convert string limit_type to enum if provided
        from app.domain.interfaces import RateLimitType
        limit_type_enum = None
        if limit_type:
            try:
                limit_type_enum = RateLimitType(limit_type)
            except ValueError:
                logger.warning("Invalid limit type provided", limit_type=limit_type)
        
        stats = await service.get_rate_limit_stats(
            tenant_id=tenant_id,
            limit_type=limit_type_enum
        )
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        logger.error("Failed to get rate limit stats", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "stats": {}
        }


# Helper functions for common rate limiting operations

async def whitelist_ip_address(
    ip_address: str,
    tenant_id: Optional[str] = None,
    hours: int = 24
) -> bool:
    """
    Add IP address to rate limit whitelist.
    
    Args:
        ip_address: IP address to whitelist
        tenant_id: Tenant context
        hours: Number of hours to whitelist for
        
    Returns:
        bool: True if successfully whitelisted
    """
    try:
        from datetime import datetime, timedelta
        from app.domain.interfaces import RateLimitType
        
        service = await get_rate_limit_service()
        expires_at = datetime.utcnow() + timedelta(hours=hours)
        
        return await service.add_to_whitelist(
            identifier=ip_address,
            limit_type=RateLimitType.IP,
            tenant_id=tenant_id,
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.error("Failed to whitelist IP address", ip=ip_address, error=str(e))
        return False


async def whitelist_user(
    user_id: str,
    tenant_id: Optional[str] = None,
    hours: int = 24
) -> bool:
    """
    Add user to rate limit whitelist.
    
    Args:
        user_id: User ID to whitelist
        tenant_id: Tenant context
        hours: Number of hours to whitelist for
        
    Returns:
        bool: True if successfully whitelisted
    """
    try:
        from datetime import datetime, timedelta
        from app.domain.interfaces import RateLimitType
        
        service = await get_rate_limit_service()
        expires_at = datetime.utcnow() + timedelta(hours=hours)
        
        return await service.add_to_whitelist(
            identifier=user_id,
            limit_type=RateLimitType.USER,
            tenant_id=tenant_id,
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.error("Failed to whitelist user", user_id=user_id, error=str(e))
        return False


async def reset_user_rate_limits(
    user_id: str,
    tenant_id: Optional[str] = None
) -> dict:
    """
    Reset all rate limits for a user.
    
    Args:
        user_id: User ID to reset limits for
        tenant_id: Tenant context
        
    Returns:
        dict: Reset operation results
    """
    try:
        from app.domain.interfaces import RateLimitType, TimeWindow
        
        service = await get_rate_limit_service()
        results = {}
        
        # Reset limits for different time windows
        for time_window in [TimeWindow.MINUTE, TimeWindow.HOUR, TimeWindow.DAY]:
            success = await service.reset_rate_limit(
                identifier=user_id,
                limit_type=RateLimitType.USER,
                time_window=time_window,
                tenant_id=tenant_id
            )
            results[time_window.value] = success
        
        return {
            "status": "success",
            "user_id": user_id,
            "reset_results": results
        }
        
    except Exception as e:
        logger.error("Failed to reset user rate limits", user_id=user_id, error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
            "reset_results": {}
        }


async def reset_ip_rate_limits(
    ip_address: str,
    tenant_id: Optional[str] = None
) -> dict:
    """
    Reset all rate limits for an IP address.
    
    Args:
        ip_address: IP address to reset limits for
        tenant_id: Tenant context
        
    Returns:
        dict: Reset operation results
    """
    try:
        from app.domain.interfaces import RateLimitType, TimeWindow
        
        service = await get_rate_limit_service()
        results = {}
        
        # Reset limits for different time windows
        for time_window in [TimeWindow.MINUTE, TimeWindow.HOUR, TimeWindow.DAY]:
            success = await service.reset_rate_limit(
                identifier=ip_address,
                limit_type=RateLimitType.IP,
                time_window=time_window,
                tenant_id=tenant_id
            )
            results[time_window.value] = success
        
        return {
            "status": "success",
            "ip_address": ip_address,
            "reset_results": results
        }
        
    except Exception as e:
        logger.error("Failed to reset IP rate limits", ip=ip_address, error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "ip_address": ip_address,
            "reset_results": {}
        }