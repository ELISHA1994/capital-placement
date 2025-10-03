"""Usage tracking provider utilities."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.services.usage.usage_tracker import UsageTracker, usage_tracker as global_usage_tracker

_usage_service: Optional[UsageTracker] = None
_usage_lock = asyncio.Lock()


async def get_usage_service() -> UsageTracker:
    """Return singleton usage service."""
    global _usage_service

    if _usage_service is not None:
        return _usage_service

    async with _usage_lock:
        if _usage_service is not None:
            return _usage_service

        # Use the global instance for consistency
        _usage_service = global_usage_tracker
        return _usage_service


async def reset_usage_service() -> None:
    """Reset usage service singleton."""
    global _usage_service
    async with _usage_lock:
        _usage_service = None


__all__ = [
    "get_usage_service",
    "reset_usage_service",
]