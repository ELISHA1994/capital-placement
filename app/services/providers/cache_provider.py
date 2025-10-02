"""Cache service provider built on top of the service factory."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.core.interfaces import ICacheService
from app.core.service_factory import get_service_factory

_cache_service: Optional[ICacheService] = None
_lock = asyncio.Lock()


async def get_cache_service() -> ICacheService:
    """Return a singleton cache service resolved from the service factory."""
    global _cache_service

    if _cache_service is not None:
        return _cache_service

    async with _lock:
        if _cache_service is not None:
            return _cache_service

        factory = get_service_factory()
        _cache_service = await factory.create_cache_service()
        return _cache_service


async def reset_cache_service() -> None:
    """Reset cached instance (useful for tests)."""
    global _cache_service
    async with _lock:
        _cache_service = None
