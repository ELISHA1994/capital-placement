"""Background task management system for tracking and cancelling async operations."""

from __future__ import annotations

import asyncio
import weakref
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import structlog

from app.domain.interfaces import ITaskManager
from app.domain.task_types import TaskInfo, TaskStatus, TaskType


logger = structlog.get_logger(__name__)


class TaskManager(ITaskManager):
    """Manages background tasks with cancellation support."""
    
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._task_info: Dict[str, TaskInfo] = {}
        self._upload_to_tasks: Dict[str, Set[str]] = {}  # upload_id -> set of task_ids
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Start the cleanup task for completed tasks."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_completed_tasks())
    
    async def _cleanup_completed_tasks(self) -> None:
        """Periodically clean up completed tasks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error during task cleanup", error=str(e))
    
    async def _perform_cleanup(self) -> None:
        """Remove completed tasks older than 1 hour."""
        current_time = datetime.utcnow()
        tasks_to_remove = []
        
        for task_id, task_info in self._task_info.items():
            if (task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task_info.completed_at and
                (current_time - task_info.completed_at).seconds > 3600):  # 1 hour
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            await self._remove_task(task_id)
        
        if tasks_to_remove:
            logger.debug("Cleaned up completed tasks", count=len(tasks_to_remove))
    
    async def _remove_task(self, task_id: str) -> None:
        """Remove a task from tracking."""
        task_info = self._task_info.get(task_id)
        if task_info:
            # Remove from upload mapping
            upload_task_set = self._upload_to_tasks.get(task_info.upload_id)
            if upload_task_set:
                upload_task_set.discard(task_id)
                if not upload_task_set:
                    del self._upload_to_tasks[task_info.upload_id]
            
            # Remove task info
            del self._task_info[task_id]
        
        # Remove asyncio task
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if not task.done():
                task.cancel()
            del self._tasks[task_id]
    
    def create_task(
        self,
        coro: Any,
        *,
        task_id: str,
        upload_id: str,
        tenant_id: str,
        user_id: str,
        task_type: TaskType,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> asyncio.Task:
        """Create and track a background task."""
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            upload_id=upload_id,
            tenant_id=tenant_id,
            user_id=user_id,
            task_type=task_type,
            additional_data=additional_data or {}
        )
        
        # Create asyncio task with error handling wrapper
        async def wrapped_coro():
            try:
                task_info.mark_started()
                logger.info(
                    "Background task started",
                    task_id=task_id,
                    upload_id=upload_id,
                    task_type=task_type.value
                )
                
                result = await coro
                task_info.mark_completed()
                
                logger.info(
                    "Background task completed",
                    task_id=task_id,
                    upload_id=upload_id,
                    task_type=task_type.value
                )
                
                return result
                
            except asyncio.CancelledError:
                task_info.mark_cancelled("Task was cancelled")
                logger.info(
                    "Background task cancelled",
                    task_id=task_id,
                    upload_id=upload_id,
                    task_type=task_type.value
                )
                raise
                
            except Exception as e:
                error_msg = str(e)
                task_info.mark_failed(error_msg)
                logger.error(
                    "Background task failed",
                    task_id=task_id,
                    upload_id=upload_id,
                    task_type=task_type.value,
                    error=error_msg
                )
                raise
        
        task = asyncio.create_task(wrapped_coro())
        
        # Store task and info
        self._tasks[task_id] = task
        self._task_info[task_id] = task_info
        
        # Map upload_id to task_id
        if upload_id not in self._upload_to_tasks:
            self._upload_to_tasks[upload_id] = set()
        self._upload_to_tasks[upload_id].add(task_id)
        
        logger.debug(
            "Created background task",
            task_id=task_id,
            upload_id=upload_id,
            task_type=task_type.value
        )
        
        return task
    
    async def cancel_tasks_for_upload(self, upload_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel all tasks associated with an upload."""
        task_ids = self._upload_to_tasks.get(upload_id, set()).copy()
        
        if not task_ids:
            return {
                "cancelled_count": 0,
                "task_ids": [],
                "message": "No tasks found for upload"
            }
        
        cancelled_tasks = []
        failed_cancellations = []
        
        for task_id in task_ids:
            try:
                result = await self._cancel_task(task_id, reason)
                if result["cancelled"]:
                    cancelled_tasks.append(task_id)
                else:
                    failed_cancellations.append({
                        "task_id": task_id,
                        "reason": result.get("reason", "Unknown")
                    })
            except Exception as e:
                failed_cancellations.append({
                    "task_id": task_id,
                    "reason": f"Cancellation error: {str(e)}"
                })
        
        logger.info(
            "Bulk task cancellation completed",
            upload_id=upload_id,
            cancelled_count=len(cancelled_tasks),
            failed_count=len(failed_cancellations)
        )
        
        return {
            "cancelled_count": len(cancelled_tasks),
            "failed_count": len(failed_cancellations),
            "task_ids": cancelled_tasks,
            "failed_cancellations": failed_cancellations,
            "message": f"Cancelled {len(cancelled_tasks)} of {len(task_ids)} tasks"
        }
    
    async def _cancel_task(self, task_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a specific task."""
        task_info = self._task_info.get(task_id)
        if not task_info:
            return {"cancelled": False, "reason": "Task not found"}
        
        # Check if task is in a cancellable state
        if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return {
                "cancelled": False,
                "reason": f"Task already {task_info.status.value}"
            }
        
        # Cancel the asyncio task
        task = self._tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            
            # Wait a short time for cancellation to take effect
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Expected for cancelled tasks
                pass
            except Exception:
                # Task may have completed with error during cancellation
                pass
        
        # Mark as cancelled
        task_info.mark_cancelled(reason)
        
        logger.info(
            "Task cancelled",
            task_id=task_id,
            upload_id=task_info.upload_id,
            task_type=task_info.task_type.value,
            reason=reason
        )
        
        return {"cancelled": True, "reason": reason}
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get information about a task."""
        return self._task_info.get(task_id)
    
    def get_tasks_for_upload(self, upload_id: str) -> List[TaskInfo]:
        """Get all tasks for an upload."""
        task_ids = self._upload_to_tasks.get(upload_id, set())
        return [self._task_info[task_id] for task_id in task_ids if task_id in self._task_info]
    
    def get_active_task_count(self) -> int:
        """Get count of active (running/pending) tasks."""
        return sum(1 for info in self._task_info.values() 
                  if info.status in [TaskStatus.PENDING, TaskStatus.RUNNING])
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get overall task statistics."""
        stats = {
            "total_tasks": len(self._task_info),
            "active_tasks": self.get_active_task_count(),
            "by_status": {},
            "by_type": {}
        }

        for info in self._task_info.values():
            # Count by status
            status_key = info.status.value
            stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1

            # Count by type
            type_key = info.task_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

        return stats

    async def check_health(self) -> Dict[str, Any]:
        """Return health details in line with the domain interface."""
        return {
            "status": "healthy" if not self._shutdown else "shutting_down",
            "active_tasks": self.get_active_task_count(),
            "total_tasks": len(self._task_info),
        }
    
    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active tasks
        active_tasks = [task for task in self._tasks.values() if not task.done()]
        if active_tasks:
            for task in active_tasks:
                task.cancel()
            
            # Wait for all tasks to complete cancellation
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        logger.info("Task manager shutdown completed", cancelled_tasks=len(active_tasks))
__all__ = [
    "TaskManager",
    "TaskInfo",
    "TaskType",
    "TaskStatus",
]
