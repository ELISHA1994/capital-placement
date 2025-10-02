"""Shared TenantManager provider wired through service factory and adapters."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.core.service_factory import get_service_factory
from app.services.core.tenant_manager import TenantManager
from app.services.adapters.tenant_database_adapter import TenantConfigDatabaseAdapter
from app.database.repositories.postgres import TenantRepository

_tenant_manager: Optional[TenantManager] = None
_lock = asyncio.Lock()


async def get_tenant_manager() -> TenantManager:
    """Return a singleton TenantManager with proper adapters."""
    global _tenant_manager

    if _tenant_manager is not None:
        return _tenant_manager

    async with _lock:
        if _tenant_manager is not None:
            return _tenant_manager

        factory = get_service_factory()
        cache_service = await factory.create_cache_service()

        tenant_repository = TenantRepository()
        database_adapter = TenantConfigDatabaseAdapter(tenant_repository)

        _tenant_manager = TenantManager(
            database=database_adapter,
            cache_service=cache_service
        )

        return _tenant_manager
