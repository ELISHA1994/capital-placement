"""Notification service provider."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import INotificationService
from app.core.service_factory import get_service_factory

_notification_service: Optional[INotificationService] = None
_lock = asyncio.Lock()


async def get_notification_service() -> INotificationService:
    global _notification_service

    if _notification_service is not None:
        return _notification_service

    async with _lock:
        if _notification_service is not None:
            return _notification_service

        factory = get_service_factory()
        _notification_service = await factory.create_notification_service()
        return _notification_service


async def reset_notification_service() -> None:
    global _notification_service
    async with _lock:
        _notification_service = None


__all__ = ["get_notification_service", "reset_notification_service"]

