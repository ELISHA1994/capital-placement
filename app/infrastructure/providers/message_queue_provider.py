"""Message queue provider."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import IMessageQueue
from app.core.service_factory import get_service_factory

_message_queue: Optional[IMessageQueue] = None
_lock = asyncio.Lock()


async def get_message_queue() -> IMessageQueue:
    global _message_queue

    if _message_queue is not None:
        return _message_queue

    async with _lock:
        if _message_queue is not None:
            return _message_queue

        factory = get_service_factory()
        _message_queue = await factory.create_message_queue()
        return _message_queue


async def reset_message_queue() -> None:
    global _message_queue
    async with _lock:
        _message_queue = None


__all__ = ["get_message_queue", "reset_message_queue"]

