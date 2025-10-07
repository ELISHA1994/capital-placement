"""File resource manager for comprehensive file cleanup and memory management."""

from __future__ import annotations

import asyncio
import gc
import os
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog

from app.domain.interfaces import IFileResourceManager

logger = structlog.get_logger(__name__)


@dataclass
class TrackedResource:
    """Information about a tracked resource."""

    resource_id: str
    resource_type: str
    size_bytes: int
    tenant_id: str
    upload_id: str | None
    file_path: str | None
    metadata: dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    in_use: bool = False
    auto_cleanup_after: int | None = None  # seconds
    cleanup_attempts: int = 0


class BytesContainer:
    """
    Container for bytes objects that supports weak references.

    Python's weakref module cannot create weak references to immutable types
    like bytes, str, int, etc. This container wraps bytes to enable weak
    reference tracking for memory management.
    """
    __slots__ = ('_data', '__weakref__')

    def __init__(self, data: bytes):
        """Initialize with bytes data."""
        self._data = data

    def __len__(self) -> int:
        """Return length of contained bytes."""
        return len(self._data)

    def __bytes__(self) -> bytes:
        """Return the contained bytes."""
        return self._data

    @property
    def data(self) -> bytes:
        """Get the contained bytes data."""
        return self._data


@dataclass
class FileContentResource(TrackedResource):
    """Tracked resource for file content in memory."""

    content: bytes = field(default=b'', repr=False)  # Don't print content in logs
    _content_container: BytesContainer | None = field(default=None, init=False, repr=False)
    content_ref: weakref.ref | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Set up weak reference to content for memory tracking."""
        if self.content:
            try:
                # Wrap bytes in a container that supports weak references
                self._content_container = BytesContainer(self.content)
                self.content_ref = weakref.ref(self._content_container)
            except TypeError as e:
                # If we still can't create a weak reference, log the error but continue
                logger.warning(
                    "Could not create weak reference for file content",
                    error=str(e),
                    size_bytes=len(self.content)
                )


class FileResourceManager(IFileResourceManager):
    """
    Comprehensive file resource manager with memory cleanup.
    
    Features:
    - Track file content in memory with automatic cleanup
    - Track temporary files with automatic deletion
    - Weak references to prevent memory leaks
    - Automatic cleanup of orphaned resources
    - Resource usage statistics and monitoring
    - Safe cleanup that doesn't interfere with ongoing operations
    """

    def __init__(self, cleanup_interval_minutes: int = 5):
        """
        Initialize the file resource manager.
        
        Args:
            cleanup_interval_minutes: How often to run cleanup tasks
        """
        self._resources: dict[str, TrackedResource] = {}
        self._upload_to_resources: dict[str, set[str]] = {}
        self._file_content_cache: dict[str, bytes] = {}
        self._cleanup_interval = cleanup_interval_minutes
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown = False
        self._lock = asyncio.Lock()

        # Start background cleanup
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Main cleanup loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._cleanup_interval * 60)
                await self._periodic_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error during periodic cleanup", error=str(e))

    async def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of resources."""
        try:
            # Cleanup auto-expiring resources
            await self._cleanup_expired_resources()

            # Cleanup orphaned resources
            await self.cleanup_orphaned_resources()

            # Force garbage collection for memory cleanup
            gc.collect()

            logger.debug("Periodic resource cleanup completed")

        except Exception as e:
            logger.error("Error during periodic cleanup", error=str(e))

    async def _cleanup_expired_resources(self) -> None:
        """Cleanup resources that have expired based on auto_cleanup_after."""
        current_time = datetime.utcnow()
        expired_resources = []

        async with self._lock:
            for resource_id, resource in self._resources.items():
                if (resource.auto_cleanup_after and
                    not resource.in_use and
                    (current_time - resource.created_at).total_seconds() > resource.auto_cleanup_after):
                    expired_resources.append(resource_id)

        # Cleanup expired resources
        for resource_id in expired_resources:
            await self.release_resource(resource_id)

        if expired_resources:
            logger.debug("Cleaned up expired resources", count=len(expired_resources))

    async def track_resource(
        self,
        resource_id: str,
        resource_type: str,
        size_bytes: int,
        *,
        tenant_id: str,
        upload_id: str | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Track a resource for cleanup management."""
        try:
            async with self._lock:
                if resource_id in self._resources:
                    logger.warning("Resource already tracked", resource_id=resource_id)
                    return False

                resource = TrackedResource(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    size_bytes=size_bytes,
                    tenant_id=tenant_id,
                    upload_id=upload_id,
                    file_path=file_path,
                    metadata=metadata or {},
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                )

                self._resources[resource_id] = resource

                # Track by upload_id
                if upload_id:
                    if upload_id not in self._upload_to_resources:
                        self._upload_to_resources[upload_id] = set()
                    self._upload_to_resources[upload_id].add(resource_id)

                logger.debug(
                    "Resource tracked",
                    resource_id=resource_id,
                    resource_type=resource_type,
                    size_mb=size_bytes / (1024 * 1024),
                    upload_id=upload_id
                )

                return True

        except Exception as e:
            logger.error("Error tracking resource", resource_id=resource_id, error=str(e))
            return False

    async def release_resource(
        self,
        resource_id: str,
        *,
        force: bool = False,
    ) -> bool:
        """Release and cleanup a tracked resource."""
        try:
            async with self._lock:
                resource = self._resources.get(resource_id)
                if not resource:
                    logger.debug("Resource not found for cleanup", resource_id=resource_id)
                    return False

                # Check if resource is in use
                if resource.in_use and not force:
                    logger.debug("Resource in use, skipping cleanup", resource_id=resource_id)
                    return False

                # Perform cleanup based on resource type
                cleanup_successful = await self._cleanup_resource(resource)

                if cleanup_successful or force:
                    # Remove from tracking
                    del self._resources[resource_id]

                    # Remove from upload mapping
                    if resource.upload_id:
                        upload_resources = self._upload_to_resources.get(resource.upload_id)
                        if upload_resources:
                            upload_resources.discard(resource_id)
                            if not upload_resources:
                                del self._upload_to_resources[resource.upload_id]

                    # Remove from file content cache
                    if resource_id in self._file_content_cache:
                        del self._file_content_cache[resource_id]

                    logger.debug(
                        "Resource released",
                        resource_id=resource_id,
                        resource_type=resource.resource_type,
                        size_mb=resource.size_bytes / (1024 * 1024)
                    )

                    return True
                else:
                    # Increment cleanup attempts
                    resource.cleanup_attempts += 1
                    return False

        except Exception as e:
            logger.error("Error releasing resource", resource_id=resource_id, error=str(e))
            return False

    async def _cleanup_resource(self, resource: TrackedResource) -> bool:
        """Perform actual cleanup of a resource."""
        try:
            if resource.resource_type == "file_content":
                # Clear file content from memory
                if isinstance(resource, FileContentResource):
                    # Clear the content container and weak reference
                    if hasattr(resource, '_content_container'):
                        resource._content_container = None
                    if hasattr(resource, 'content_ref'):
                        resource.content_ref = None
                    # Clear the actual content
                    if hasattr(resource, 'content'):
                        del resource.content

                # Force garbage collection for large files
                if resource.size_bytes > 10 * 1024 * 1024:  # 10MB
                    gc.collect()

                return True

            elif resource.resource_type == "temp_file":
                # Delete temporary file if it exists
                if resource.file_path and os.path.exists(resource.file_path):
                    try:
                        os.unlink(resource.file_path)
                        logger.debug("Deleted temporary file", file_path=resource.file_path)
                    except OSError as e:
                        logger.warning("Failed to delete temporary file",
                                     file_path=resource.file_path, error=str(e))
                        return False

                return True

            else:
                # Generic cleanup - just remove from tracking
                logger.debug("Generic resource cleanup", resource_type=resource.resource_type)
                return True

        except Exception as e:
            logger.error("Error during resource cleanup",
                        resource_id=resource.resource_id, error=str(e))
            return False

    async def release_upload_resources(
        self,
        upload_id: str,
        *,
        exclude_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Release all resources associated with an upload."""
        exclude_types = exclude_types or []
        released_count = 0
        failed_count = 0
        total_size = 0

        try:
            resource_ids = self._upload_to_resources.get(upload_id, set()).copy()

            for resource_id in resource_ids:
                resource = self._resources.get(resource_id)
                if resource and resource.resource_type not in exclude_types:
                    total_size += resource.size_bytes
                    if await self.release_resource(resource_id):
                        released_count += 1
                    else:
                        failed_count += 1

            result = {
                "upload_id": upload_id,
                "released_count": released_count,
                "failed_count": failed_count,
                "total_size_mb": total_size / (1024 * 1024),
                "excluded_types": exclude_types,
            }

            logger.info(
                "Upload resources cleanup completed",
                upload_id=upload_id,
                released=released_count,
                failed=failed_count,
                total_size_mb=result["total_size_mb"]
            )

            return result

        except Exception as e:
            logger.error("Error releasing upload resources", upload_id=upload_id, error=str(e))
            return {
                "upload_id": upload_id,
                "released_count": 0,
                "failed_count": 0,
                "total_size_mb": 0,
                "error": str(e),
            }

    async def cleanup_orphaned_resources(
        self,
        *,
        older_than_minutes: int = 60,
        max_cleanup_count: int = 100,
    ) -> dict[str, Any]:
        """Clean up orphaned resources older than specified time."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=older_than_minutes)

        orphaned_resources = []
        cleaned_count = 0
        failed_count = 0
        total_size = 0

        try:
            async with self._lock:
                # Find orphaned resources
                for resource_id, resource in self._resources.items():
                    if (resource.created_at < cutoff_time and
                        not resource.in_use and
                        len(orphaned_resources) < max_cleanup_count):
                        orphaned_resources.append(resource_id)

            # Clean up orphaned resources
            for resource_id in orphaned_resources:
                resource = self._resources.get(resource_id)
                if resource:
                    total_size += resource.size_bytes
                    if await self.release_resource(resource_id, force=True):
                        cleaned_count += 1
                    else:
                        failed_count += 1

            result = {
                "cleaned_count": cleaned_count,
                "failed_count": failed_count,
                "total_size_mb": total_size / (1024 * 1024),
                "cutoff_time": cutoff_time.isoformat(),
            }

            if cleaned_count > 0:
                logger.info(
                    "Orphaned resources cleanup completed",
                    cleaned=cleaned_count,
                    failed=failed_count,
                    total_size_mb=result["total_size_mb"]
                )

            return result

        except Exception as e:
            logger.error("Error cleaning orphaned resources", error=str(e))
            return {
                "cleaned_count": 0,
                "failed_count": 0,
                "total_size_mb": 0,
                "error": str(e),
            }

    async def get_resource_stats(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Get resource usage statistics."""
        try:
            async with self._lock:
                total_resources = 0
                total_size = 0
                by_type = {}
                by_tenant = {}
                in_use_count = 0

                for resource in self._resources.values():
                    if tenant_id is None or resource.tenant_id == tenant_id:
                        total_resources += 1
                        total_size += resource.size_bytes

                        # Count by type
                        resource_type = resource.resource_type
                        by_type[resource_type] = by_type.get(resource_type, 0) + 1

                        # Count by tenant (if not filtering)
                        if tenant_id is None:
                            by_tenant[resource.tenant_id] = by_tenant.get(resource.tenant_id, 0) + 1

                        if resource.in_use:
                            in_use_count += 1

                return {
                    "total_resources": total_resources,
                    "total_size_mb": total_size / (1024 * 1024),
                    "in_use_count": in_use_count,
                    "available_count": total_resources - in_use_count,
                    "by_type": by_type,
                    "by_tenant": by_tenant if tenant_id is None else {},
                    "total_uploads_tracked": len(self._upload_to_resources),
                }

        except Exception as e:
            logger.error("Error getting resource stats", error=str(e))
            return {"error": str(e)}

    async def mark_resource_in_use(self, resource_id: str) -> bool:
        """Mark a resource as currently in use."""
        try:
            async with self._lock:
                resource = self._resources.get(resource_id)
                if resource:
                    resource.in_use = True
                    resource.last_accessed = datetime.utcnow()
                    return True
                return False
        except Exception as e:
            logger.error("Error marking resource in use", resource_id=resource_id, error=str(e))
            return False

    async def mark_resource_available(self, resource_id: str) -> bool:
        """Mark a resource as available for cleanup."""
        try:
            async with self._lock:
                resource = self._resources.get(resource_id)
                if resource:
                    resource.in_use = False
                    resource.last_accessed = datetime.utcnow()
                    return True
                return False
        except Exception as e:
            logger.error("Error marking resource available", resource_id=resource_id, error=str(e))
            return False

    async def track_file_content(
        self,
        content: bytes,
        filename: str,
        *,
        upload_id: str,
        tenant_id: str,
        auto_cleanup_after: int | None = None,
    ) -> str:
        """Track file content in memory for cleanup."""
        resource_id = f"file_content_{upload_id}_{uuid4().hex[:8]}"

        try:
            async with self._lock:
                # Create file content resource
                resource = FileContentResource(
                    resource_id=resource_id,
                    resource_type="file_content",
                    size_bytes=len(content),
                    tenant_id=tenant_id,
                    upload_id=upload_id,
                    file_path=None,
                    metadata={"filename": filename},
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    auto_cleanup_after=auto_cleanup_after,
                    content=content,
                )

                self._resources[resource_id] = resource
                self._file_content_cache[resource_id] = content

                # Track by upload_id
                if upload_id not in self._upload_to_resources:
                    self._upload_to_resources[upload_id] = set()
                self._upload_to_resources[upload_id].add(resource_id)

                logger.debug(
                    "File content tracked",
                    resource_id=resource_id,
                    filename=filename,
                    size_mb=len(content) / (1024 * 1024),
                    upload_id=upload_id
                )

                return resource_id

        except Exception as e:
            logger.error("Error tracking file content",
                        filename=filename, upload_id=upload_id, error=str(e))
            return ""

    async def get_file_content(self, resource_id: str) -> bytes | None:
        """Get tracked file content by resource ID."""
        try:
            async with self._lock:
                # Update last accessed time
                resource = self._resources.get(resource_id)
                if resource:
                    resource.last_accessed = datetime.utcnow()

                # Return cached content
                return self._file_content_cache.get(resource_id)

        except Exception as e:
            logger.error("Error getting file content", resource_id=resource_id, error=str(e))
            return None

    async def track_temp_file(
        self,
        file_path: str,
        *,
        upload_id: str,
        tenant_id: str,
        auto_cleanup_after: int | None = None,
    ) -> str:
        """Track temporary file for cleanup."""
        resource_id = f"temp_file_{upload_id}_{uuid4().hex[:8]}"

        try:
            # Get file size
            file_size = 0
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)

            success = await self.track_resource(
                resource_id=resource_id,
                resource_type="temp_file",
                size_bytes=file_size,
                tenant_id=tenant_id,
                upload_id=upload_id,
                file_path=file_path,
                metadata={"file_path": file_path},
            )

            if success:
                # Set auto cleanup
                resource = self._resources.get(resource_id)
                if resource and auto_cleanup_after:
                    resource.auto_cleanup_after = auto_cleanup_after

                logger.debug(
                    "Temporary file tracked",
                    resource_id=resource_id,
                    file_path=file_path,
                    size_mb=file_size / (1024 * 1024),
                    upload_id=upload_id
                )

                return resource_id
            else:
                return ""

        except Exception as e:
            logger.error("Error tracking temp file",
                        file_path=file_path, upload_id=upload_id, error=str(e))
            return ""

    async def cleanup_file_handles(self, upload_id: str) -> int:
        """Cleanup file handles for an upload."""
        count = 0
        try:
            resource_ids = self._upload_to_resources.get(upload_id, set()).copy()

            for resource_id in resource_ids:
                resource = self._resources.get(resource_id)
                if resource and resource.resource_type in ["file_content", "temp_file"]:
                    if await self.release_resource(resource_id):
                        count += 1

            logger.debug("File handles cleanup completed", upload_id=upload_id, count=count)
            return count

        except Exception as e:
            logger.error("Error cleaning file handles", upload_id=upload_id, error=str(e))
            return 0

    async def check_health(self) -> dict[str, Any]:
        """Health check for the resource manager."""
        try:
            stats = await self.get_resource_stats()

            return {
                "service": "FileResourceManager",
                "status": "healthy",
                "total_resources": stats.get("total_resources", 0),
                "total_size_mb": stats.get("total_size_mb", 0),
                "cleanup_task_running": self._cleanup_task and not self._cleanup_task.done(),
            }

        except Exception as e:
            return {
                "service": "FileResourceManager",
                "status": "unhealthy",
                "error": str(e),
            }

    async def shutdown(self) -> None:
        """Shutdown the resource manager and cleanup all resources."""
        self._shutdown = True

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all resources
        try:
            resource_ids = list(self._resources.keys())
            for resource_id in resource_ids:
                await self.release_resource(resource_id, force=True)

            logger.info("File resource manager shutdown completed",
                       cleaned_resources=len(resource_ids))

        except Exception as e:
            logger.error("Error during resource manager shutdown", error=str(e))


# Global instance management
_file_resource_manager: FileResourceManager | None = None


def get_file_resource_manager() -> FileResourceManager:
    """Get the global file resource manager instance."""
    global _file_resource_manager
    if _file_resource_manager is None:
        _file_resource_manager = FileResourceManager()
    return _file_resource_manager


async def shutdown_file_resource_manager() -> None:
    """Shutdown the global file resource manager."""
    global _file_resource_manager
    if _file_resource_manager is not None:
        await _file_resource_manager.shutdown()
        _file_resource_manager = None


__all__ = [
    "FileResourceManager",
    "TrackedResource",
    "FileContentResource",
    "get_file_resource_manager",
    "shutdown_file_resource_manager",
]
