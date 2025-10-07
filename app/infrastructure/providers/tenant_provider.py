"""
Tenant Provider

Provides singleton access to TenantService for tenant management operations.

This replaces the old tenant_manager_provider.py pattern with a cleaner
provider-based approach that integrates with the new hexagonal architecture.
"""

import asyncio
from typing import Optional

import structlog

from app.infrastructure.tenant.tenant_service import TenantService
from app.database.repositories.postgres import TenantRepository, UserRepository
from app.infrastructure.providers.cache_provider import get_cache_service

logger = structlog.get_logger(__name__)

_tenant_service: Optional[TenantService] = None
_lock = asyncio.Lock()


async def get_tenant_service() -> TenantService:
    """
    Get or create singleton TenantService instance.

    Provides centralized tenant management with proper dependency injection.
    Replaces the old get_tenant_manager() pattern.

    Returns:
        TenantService: Configured tenant service instance

    Example:
        ```python
        from app.infrastructure.providers.tenant_provider import get_tenant_service

        tenant_service = await get_tenant_service()
        config = await tenant_service.get_tenant(tenant_id)
        ```
    """
    global _tenant_service

    if _tenant_service is not None:
        return _tenant_service

    async with _lock:
        if _tenant_service is not None:
            return _tenant_service

        logger.debug("Initializing TenantService singleton")

        # Create repositories
        tenant_repository = TenantRepository()
        user_repository = UserRepository()

        # Get cache service
        cache_service = await get_cache_service()

        # Create tenant service
        _tenant_service = TenantService(
            tenant_repository=tenant_repository,
            user_repository=user_repository,
            cache_manager=cache_service
        )

        logger.info("TenantService initialized successfully")

        return _tenant_service


async def reset_tenant_service() -> None:
    """
    Reset the tenant service singleton.

    Used primarily for testing to ensure clean state between tests.
    """
    global _tenant_service

    async with _lock:
        _tenant_service = None
        logger.debug("TenantService singleton reset")


__all__ = [
    "get_tenant_service",
    "reset_tenant_service",
]
