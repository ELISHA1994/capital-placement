"""Resource management provider for dependency injection."""

from __future__ import annotations

from typing import Optional

from app.domain.interfaces import IFileResourceManager
from app.infrastructure.resource.file_resource_manager import (
    FileResourceManager,
    get_file_resource_manager,
)
from app.infrastructure.resource.periodic_cleanup import (
    PeriodicResourceCleanup,
    get_periodic_cleanup,
)

# Global service instances
_file_resource_manager: Optional[FileResourceManager] = None
_periodic_cleanup: Optional[PeriodicResourceCleanup] = None


async def get_file_resource_service() -> IFileResourceManager:
    """
    Get file resource manager service instance.
    
    Returns:
        IFileResourceManager: File resource manager service
    """
    global _file_resource_manager
    
    if _file_resource_manager is None:
        _file_resource_manager = get_file_resource_manager()
    
    return _file_resource_manager


async def get_periodic_cleanup_service() -> PeriodicResourceCleanup:
    """
    Get periodic cleanup service instance.
    
    Returns:
        PeriodicResourceCleanup: Periodic cleanup service
    """
    global _periodic_cleanup, _file_resource_manager
    
    if _periodic_cleanup is None:
        # Ensure file resource manager is initialized
        if _file_resource_manager is None:
            _file_resource_manager = get_file_resource_manager()
        
        _periodic_cleanup = get_periodic_cleanup(
            file_resource_manager=_file_resource_manager,
            cleanup_interval_minutes=15,  # Run every 15 minutes
            orphaned_resource_threshold_minutes=60,  # Clean resources older than 1 hour
        )
    
    return _periodic_cleanup


async def shutdown_resource_services() -> None:
    """Shutdown all resource management services."""
    global _file_resource_manager, _periodic_cleanup
    
    # Shutdown periodic cleanup first
    if _periodic_cleanup is not None:
        await _periodic_cleanup.shutdown()
        _periodic_cleanup = None
    
    # Shutdown file resource manager
    if _file_resource_manager is not None:
        await _file_resource_manager.shutdown()
        _file_resource_manager = None


__all__ = [
    "get_file_resource_service",
    "get_periodic_cleanup_service",
    "shutdown_resource_services",
]