"""Analytics service provider."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import IAnalyticsService

from app.infrastructure.providers.search_provider import (
    get_search_analytics_service,
    reset_search_services,
)

_analytics_service: Optional[IAnalyticsService] = None
_lock = asyncio.Lock()


async def get_analytics_service() -> IAnalyticsService:
    global _analytics_service

    if _analytics_service is not None:
        return _analytics_service

    async with _lock:
        if _analytics_service is not None:
            return _analytics_service

        _analytics_service = await get_search_analytics_service()
        return _analytics_service


async def reset_analytics_service() -> None:
    global _analytics_service
    async with _lock:
        _analytics_service = None
    await reset_search_services()


__all__ = ["get_analytics_service", "reset_analytics_service"]

