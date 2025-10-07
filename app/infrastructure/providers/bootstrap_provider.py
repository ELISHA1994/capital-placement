"""Bootstrap service provider utilities."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.infrastructure.bootstrap import BootstrapService

_bootstrap_service: Optional[BootstrapService] = None
_bootstrap_lock = asyncio.Lock()


async def get_bootstrap_service() -> BootstrapService:
    """Return singleton bootstrap service."""
    global _bootstrap_service

    if _bootstrap_service is not None:
        return _bootstrap_service

    async with _bootstrap_lock:
        if _bootstrap_service is not None:
            return _bootstrap_service

        # Bootstrap service requires repositories and auth service
        # Import here to avoid circular imports
        from app.database.repositories.postgres import TenantRepository, UserRepository
        from app.infrastructure.providers.auth_provider import get_authentication_service
        
        tenant_repository = TenantRepository()
        user_repository = UserRepository()
        auth_service = await get_authentication_service()

        _bootstrap_service = BootstrapService(
            tenant_repository=tenant_repository,
            user_repository=user_repository,
            auth_service=auth_service,
        )
        return _bootstrap_service


async def reset_bootstrap_service() -> None:
    """Reset bootstrap service singleton."""
    global _bootstrap_service
    async with _bootstrap_lock:
        _bootstrap_service = None


__all__ = [
    "get_bootstrap_service",
    "reset_bootstrap_service",
]