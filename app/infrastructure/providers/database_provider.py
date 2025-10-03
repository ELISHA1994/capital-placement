"""Database service provider for hexagonal adapters."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import IDatabase
from app.core.service_factory import get_service_factory

_database_service: Optional[IDatabase] = None
_lock = asyncio.Lock()


async def get_database_service() -> IDatabase:
    """Return the configured IDatabase implementation."""
    global _database_service

    if _database_service is not None:
        return _database_service

    async with _lock:
        if _database_service is not None:
            return _database_service

        factory = get_service_factory()
        _database_service = await factory.create_database()
        return _database_service


async def reset_database_service() -> None:
    global _database_service
    async with _lock:
        _database_service = None


__all__ = ["get_database_service", "reset_database_service"]

