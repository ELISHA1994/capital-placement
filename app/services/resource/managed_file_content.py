"""Managed file content with automatic resource cleanup."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict, Any
import structlog

from app.domain.interfaces import IFileResourceManager

logger = structlog.get_logger(__name__)


class ManagedFileContent:
    """
    Managed file content that tracks resources and ensures cleanup.
    
    This class provides a safe wrapper around file content that:
    - Tracks memory usage through the resource manager
    - Automatically cleans up resources when done
    - Prevents memory leaks in file processing
    """
    
    def __init__(
        self,
        content: bytes,
        filename: str,
        upload_id: str,
        tenant_id: str,
        resource_manager: IFileResourceManager,
        auto_cleanup_after: Optional[int] = None,
    ):
        """
        Initialize managed file content.
        
        Args:
            content: File content bytes
            filename: Original filename
            upload_id: Associated upload ID
            tenant_id: Tenant identifier
            resource_manager: Resource manager for tracking
            auto_cleanup_after: Auto cleanup after N seconds
        """
        self._content = content
        self._filename = filename
        self._upload_id = upload_id
        self._tenant_id = tenant_id
        self._resource_manager = resource_manager
        self._auto_cleanup_after = auto_cleanup_after
        self._resource_id: Optional[str] = None
        self._is_tracked = False
        self._released = False
    
    async def __aenter__(self) -> ManagedFileContent:
        """Async context manager entry."""
        await self.track()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.release()
    
    async def track(self) -> bool:
        """Track the file content with the resource manager."""
        if self._is_tracked or self._released:
            return False
        
        try:
            self._resource_id = await self._resource_manager.track_file_content(
                content=self._content,
                filename=self._filename,
                upload_id=self._upload_id,
                tenant_id=self._tenant_id,
                auto_cleanup_after=self._auto_cleanup_after,
            )
            
            if self._resource_id:
                self._is_tracked = True
                logger.debug(
                    "File content tracked",
                    resource_id=self._resource_id,
                    filename=self._filename,
                    size_mb=len(self._content) / (1024 * 1024)
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error tracking file content", filename=self._filename, error=str(e))
            return False
    
    async def release(self) -> bool:
        """Release the tracked file content."""
        if not self._is_tracked or self._released:
            return False
        
        try:
            if self._resource_id:
                success = await self._resource_manager.release_resource(self._resource_id)
                if success:
                    self._released = True
                    logger.debug(
                        "File content released",
                        resource_id=self._resource_id,
                        filename=self._filename
                    )
                return success
            
            return False
            
        except Exception as e:
            logger.error("Error releasing file content", 
                        resource_id=self._resource_id, error=str(e))
            return False
    
    async def mark_in_use(self) -> bool:
        """Mark the file content as currently in use."""
        if not self._is_tracked or not self._resource_id:
            return False
        
        return await self._resource_manager.mark_resource_in_use(self._resource_id)
    
    async def mark_available(self) -> bool:
        """Mark the file content as available for cleanup."""
        if not self._is_tracked or not self._resource_id:
            return False
        
        return await self._resource_manager.mark_resource_available(self._resource_id)
    
    @property
    def content(self) -> bytes:
        """Get the file content."""
        if self._released:
            raise RuntimeError("File content has been released")
        return self._content
    
    @property
    def filename(self) -> str:
        """Get the filename."""
        return self._filename
    
    @property
    def upload_id(self) -> str:
        """Get the upload ID."""
        return self._upload_id
    
    @property
    def size_bytes(self) -> int:
        """Get the content size in bytes."""
        return len(self._content)
    
    @property
    def size_mb(self) -> float:
        """Get the content size in MB."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def is_tracked(self) -> bool:
        """Check if content is tracked."""
        return self._is_tracked
    
    @property
    def is_released(self) -> bool:
        """Check if content has been released."""
        return self._released
    
    @property
    def resource_id(self) -> Optional[str]:
        """Get the resource ID."""
        return self._resource_id


@asynccontextmanager
async def managed_file_content(
    content: bytes,
    filename: str,
    upload_id: str,
    tenant_id: str,
    resource_manager: IFileResourceManager,
    auto_cleanup_after: Optional[int] = None,
) -> AsyncGenerator[ManagedFileContent, None]:
    """
    Context manager for managed file content with automatic cleanup.
    
    Args:
        content: File content bytes
        filename: Original filename
        upload_id: Associated upload ID
        tenant_id: Tenant identifier
        resource_manager: Resource manager for tracking
        auto_cleanup_after: Auto cleanup after N seconds
        
    Yields:
        ManagedFileContent: Managed file content instance
    """
    managed_content = ManagedFileContent(
        content=content,
        filename=filename,
        upload_id=upload_id,
        tenant_id=tenant_id,
        resource_manager=resource_manager,
        auto_cleanup_after=auto_cleanup_after,
    )
    
    try:
        await managed_content.track()
        yield managed_content
    finally:
        await managed_content.release()


class BatchManagedFileContent:
    """
    Manages multiple file contents for batch operations.
    
    Provides batch tracking and cleanup for multiple files to prevent
    memory exhaustion during batch processing operations.
    """
    
    def __init__(self, resource_manager: IFileResourceManager):
        """
        Initialize batch managed file content.
        
        Args:
            resource_manager: Resource manager for tracking
        """
        self._resource_manager = resource_manager
        self._managed_files: list[ManagedFileContent] = []
        self._batch_id: Optional[str] = None
    
    async def add_file(
        self,
        content: bytes,
        filename: str,
        upload_id: str,
        tenant_id: str,
        auto_cleanup_after: Optional[int] = None,
    ) -> ManagedFileContent:
        """
        Add a file to the batch.
        
        Args:
            content: File content bytes
            filename: Original filename
            upload_id: Associated upload ID
            tenant_id: Tenant identifier
            auto_cleanup_after: Auto cleanup after N seconds
            
        Returns:
            ManagedFileContent: Managed file content instance
        """
        managed_content = ManagedFileContent(
            content=content,
            filename=filename,
            upload_id=upload_id,
            tenant_id=tenant_id,
            resource_manager=self._resource_manager,
            auto_cleanup_after=auto_cleanup_after,
        )
        
        await managed_content.track()
        self._managed_files.append(managed_content)
        
        logger.debug(
            "File added to batch",
            filename=filename,
            batch_size=len(self._managed_files),
            total_size_mb=self.total_size_mb
        )
        
        return managed_content
    
    async def release_all(self) -> Dict[str, Any]:
        """
        Release all files in the batch.
        
        Returns:
            Dictionary with release statistics
        """
        released_count = 0
        failed_count = 0
        total_size = 0
        
        for managed_file in self._managed_files:
            total_size += managed_file.size_bytes
            if await managed_file.release():
                released_count += 1
            else:
                failed_count += 1
        
        self._managed_files.clear()
        
        result = {
            "released_count": released_count,
            "failed_count": failed_count,
            "total_size_mb": total_size / (1024 * 1024),
        }
        
        logger.info(
            "Batch file content released",
            released=released_count,
            failed=failed_count,
            total_size_mb=result["total_size_mb"]
        )
        
        return result
    
    @property
    def total_size_bytes(self) -> int:
        """Get total size of all files in bytes."""
        return sum(f.size_bytes for f in self._managed_files)
    
    @property
    def total_size_mb(self) -> float:
        """Get total size of all files in MB."""
        return self.total_size_bytes / (1024 * 1024)
    
    @property
    def file_count(self) -> int:
        """Get number of files in batch."""
        return len(self._managed_files)
    
    def __len__(self) -> int:
        """Get number of files in batch."""
        return len(self._managed_files)
    
    def __iter__(self):
        """Iterate over managed files."""
        return iter(self._managed_files)
    
    async def __aenter__(self) -> BatchManagedFileContent:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.release_all()


__all__ = [
    "ManagedFileContent",
    "managed_file_content",
    "BatchManagedFileContent",
]