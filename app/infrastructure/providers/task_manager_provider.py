"""Provider for task manager service."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.domain.interfaces import ITaskManager
from app.infrastructure.task_manager import TaskManager


_task_manager: Optional[ITaskManager] = None
_lock = asyncio.Lock()


async def get_task_manager() -> ITaskManager:
    """Return the singleton task manager."""
    global _task_manager

    if _task_manager is not None:
        return _task_manager

    async with _lock:
        if _task_manager is None:
            _task_manager = TaskManager()
        return _task_manager


async def reset_task_manager() -> None:
    """Reset the cached task manager instance (for testing)."""
    global _task_manager
    async with _lock:
        if _task_manager is not None:
            await _task_manager.shutdown()
            _task_manager = None


__all__ = ["get_task_manager", "reset_task_manager"]
