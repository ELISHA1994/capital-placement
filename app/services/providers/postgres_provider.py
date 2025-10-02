"""Provider for Postgres adapter."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.services.adapters.postgres_adapter import PostgresAdapter

_postgres_adapter: Optional[PostgresAdapter] = None
_lock = asyncio.Lock()


async def get_postgres_adapter() -> PostgresAdapter:
    global _postgres_adapter

    if _postgres_adapter is not None:
        return _postgres_adapter

    async with _lock:
        if _postgres_adapter is not None:
            return _postgres_adapter

        _postgres_adapter = PostgresAdapter()
        return _postgres_adapter


async def reset_postgres_adapter() -> None:
    global _postgres_adapter
    async with _lock:
        _postgres_adapter = None
