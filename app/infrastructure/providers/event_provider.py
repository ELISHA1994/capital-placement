"""Event publisher provider."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import IEventPublisher
from app.infrastructure.adapters.messaging_adapters import LocalEventPublisher

_event_publisher: Optional[IEventPublisher] = None
_lock = asyncio.Lock()


async def get_event_publisher() -> IEventPublisher:
    """Return the configured event publisher implementation."""
    global _event_publisher

    if _event_publisher is not None:
        return _event_publisher

    async with _lock:
        if _event_publisher is not None:
            return _event_publisher

        _event_publisher = LocalEventPublisher()
        return _event_publisher


async def reset_event_publisher() -> None:
    """Reset the cached event publisher instance."""
    global _event_publisher
    async with _lock:
        _event_publisher = None


__all__ = ["get_event_publisher", "reset_event_publisher"]
