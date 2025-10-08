"""Event publisher provider."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.infrastructure.adapters.event_publisher_adapter import EventPublisherAdapter

if TYPE_CHECKING:
    from app.domain.interfaces import IEventPublisher

_event_publisher: IEventPublisher | None = None
_lock = asyncio.Lock()


async def get_event_publisher() -> IEventPublisher:
    """Return the configured event publisher implementation."""
    global _event_publisher

    if _event_publisher is not None:
        return _event_publisher

    async with _lock:
        if _event_publisher is not None:
            return _event_publisher

        _event_publisher = EventPublisherAdapter()
        return _event_publisher


async def reset_event_publisher() -> None:
    """Reset the cached event publisher instance."""
    global _event_publisher
    async with _lock:
        _event_publisher = None


__all__ = ["get_event_publisher", "reset_event_publisher"]
