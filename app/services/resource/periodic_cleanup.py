"""Periodic resource cleanup service integrated with task manager."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import structlog

from app.domain.interfaces import IFileResourceManager
from app.infrastructure.task_manager import get_task_manager, TaskType

logger = structlog.get_logger(__name__)


class PeriodicResourceCleanup:
    """
    Periodic cleanup service for file resources.
    
    Integrates with the task manager to run cleanup tasks without
    interfering with ongoing operations.
    """
    
    def __init__(
        self,
        file_resource_manager: IFileResourceManager,
        cleanup_interval_minutes: int = 15,
        orphaned_resource_threshold_minutes: int = 60,
    ):
        """
        Initialize periodic resource cleanup.
        
        Args:
            file_resource_manager: File resource manager instance
            cleanup_interval_minutes: How often to run cleanup
            orphaned_resource_threshold_minutes: Age threshold for orphaned resources
        """
        self._file_resource_manager = file_resource_manager
        self._cleanup_interval = cleanup_interval_minutes
        self._orphaned_threshold = orphaned_resource_threshold_minutes
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Start periodic cleanup
        self._start_cleanup()
    
    def _start_cleanup(self) -> None:
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                "Periodic resource cleanup started",
                interval_minutes=self._cleanup_interval,
                orphaned_threshold_minutes=self._orphaned_threshold
            )
    
    async def _cleanup_loop(self) -> None:
        """Main cleanup loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._cleanup_interval * 60)
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error during periodic resource cleanup loop", error=str(e))
    
    async def _perform_cleanup(self) -> None:
        """Perform cleanup using task manager for tracking."""
        task_manager = get_task_manager()
        cleanup_id = f"resource_cleanup_{int(datetime.utcnow().timestamp())}"
        
        try:
            # Create cleanup task
            cleanup_task = task_manager.create_task(
                self._run_cleanup(),
                task_id=cleanup_id,
                upload_id="system",
                tenant_id="system",
                user_id="system",
                task_type=TaskType.RESOURCE_CLEANUP,
                additional_data={
                    "cleanup_type": "periodic_orphaned_resources",
                    "threshold_minutes": self._orphaned_threshold,
                }
            )
            
            logger.debug("Periodic resource cleanup task created", task_id=cleanup_id)
            
        except Exception as e:
            logger.error("Error creating periodic cleanup task", error=str(e))
    
    async def _run_cleanup(self) -> Dict[str, Any]:
        """Run the actual cleanup operations."""
        cleanup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "orphaned_cleanup": {},
            "stats": {},
            "errors": [],
        }
        
        try:
            # Clean up orphaned resources
            orphaned_result = await self._file_resource_manager.cleanup_orphaned_resources(
                older_than_minutes=self._orphaned_threshold,
                max_cleanup_count=100,
            )
            cleanup_results["orphaned_cleanup"] = orphaned_result
            
            # Get resource statistics
            stats = await self._file_resource_manager.get_resource_stats()
            cleanup_results["stats"] = stats
            
            # Log results if there was cleanup activity
            cleaned_count = orphaned_result.get("cleaned_count", 0)
            if cleaned_count > 0:
                logger.info(
                    "Periodic resource cleanup completed",
                    cleaned_orphaned=cleaned_count,
                    total_size_mb=orphaned_result.get("total_size_mb", 0),
                    current_resources=stats.get("total_resources", 0),
                    current_size_mb=stats.get("total_size_mb", 0)
                )
            else:
                logger.debug(
                    "Periodic resource cleanup completed (no orphaned resources)",
                    current_resources=stats.get("total_resources", 0),
                    current_size_mb=stats.get("total_size_mb", 0)
                )
            
            return cleanup_results
            
        except Exception as e:
            error_msg = f"Error during resource cleanup: {str(e)}"
            cleanup_results["errors"].append(error_msg)
            logger.error("Error during periodic resource cleanup", error=str(e))
            raise
    
    async def trigger_immediate_cleanup(
        self,
        *,
        orphaned_threshold_minutes: Optional[int] = None,
        max_cleanup_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Trigger immediate cleanup outside of the regular schedule.
        
        Args:
            orphaned_threshold_minutes: Override default threshold
            max_cleanup_count: Override default cleanup count
            
        Returns:
            Dictionary with cleanup results
        """
        threshold = orphaned_threshold_minutes or self._orphaned_threshold
        max_count = max_cleanup_count or 100
        
        task_manager = get_task_manager()
        cleanup_id = f"immediate_cleanup_{int(datetime.utcnow().timestamp())}"
        
        try:
            logger.info(
                "Triggering immediate resource cleanup",
                threshold_minutes=threshold,
                max_count=max_count
            )
            
            # Create immediate cleanup task
            cleanup_coro = self._run_immediate_cleanup(threshold, max_count)
            cleanup_task = task_manager.create_task(
                cleanup_coro,
                task_id=cleanup_id,
                upload_id="immediate",
                tenant_id="system",
                user_id="system",
                task_type=TaskType.RESOURCE_CLEANUP,
                additional_data={
                    "cleanup_type": "immediate_cleanup",
                    "threshold_minutes": threshold,
                    "max_count": max_count,
                }
            )
            
            # Wait for completion
            result = await cleanup_task
            return result
            
        except Exception as e:
            logger.error("Error during immediate cleanup", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def _run_immediate_cleanup(
        self,
        threshold_minutes: int,
        max_count: int,
    ) -> Dict[str, Any]:
        """Run immediate cleanup with specified parameters."""
        try:
            result = await self._file_resource_manager.cleanup_orphaned_resources(
                older_than_minutes=threshold_minutes,
                max_cleanup_count=max_count,
            )
            
            # Get updated stats
            stats = await self._file_resource_manager.get_resource_stats()
            
            logger.info(
                "Immediate resource cleanup completed",
                cleaned_count=result.get("cleaned_count", 0),
                failed_count=result.get("failed_count", 0),
                total_size_mb=result.get("total_size_mb", 0),
                remaining_resources=stats.get("total_resources", 0)
            )
            
            return {
                "cleanup_result": result,
                "current_stats": stats,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Error in immediate cleanup execution", error=str(e))
            raise
    
    async def get_cleanup_status(self) -> Dict[str, Any]:
        """Get status of the periodic cleanup service."""
        try:
            stats = await self._file_resource_manager.get_resource_stats()
            
            return {
                "service": "PeriodicResourceCleanup",
                "status": "running" if not self._shutdown else "stopped",
                "cleanup_interval_minutes": self._cleanup_interval,
                "orphaned_threshold_minutes": self._orphaned_threshold,
                "cleanup_task_running": self._cleanup_task and not self._cleanup_task.done(),
                "current_resources": stats.get("total_resources", 0),
                "current_size_mb": stats.get("total_size_mb", 0),
                "in_use_count": stats.get("in_use_count", 0),
            }
            
        except Exception as e:
            return {
                "service": "PeriodicResourceCleanup",
                "status": "error",
                "error": str(e),
            }
    
    async def shutdown(self) -> None:
        """Shutdown the periodic cleanup service."""
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Periodic resource cleanup service shutdown completed")


# Global instance management
_periodic_cleanup: Optional[PeriodicResourceCleanup] = None


def get_periodic_cleanup(
    file_resource_manager: IFileResourceManager,
    cleanup_interval_minutes: int = 15,
    orphaned_resource_threshold_minutes: int = 60,
) -> PeriodicResourceCleanup:
    """Get or create the global periodic cleanup instance."""
    global _periodic_cleanup
    if _periodic_cleanup is None:
        _periodic_cleanup = PeriodicResourceCleanup(
            file_resource_manager=file_resource_manager,
            cleanup_interval_minutes=cleanup_interval_minutes,
            orphaned_resource_threshold_minutes=orphaned_resource_threshold_minutes,
        )
    return _periodic_cleanup


async def shutdown_periodic_cleanup() -> None:
    """Shutdown the global periodic cleanup service."""
    global _periodic_cleanup
    if _periodic_cleanup is not None:
        await _periodic_cleanup.shutdown()
        _periodic_cleanup = None


__all__ = [
    "PeriodicResourceCleanup",
    "get_periodic_cleanup",
    "shutdown_periodic_cleanup",
]