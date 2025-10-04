"""Resource management services for file cleanup and memory management."""

from .file_resource_manager import (
    FileResourceManager,
    TrackedResource,
    FileContentResource,
    get_file_resource_manager,
    shutdown_file_resource_manager,
)
from .managed_file_content import (
    ManagedFileContent,
    managed_file_content,
    BatchManagedFileContent,
)

__all__ = [
    "FileResourceManager",
    "TrackedResource", 
    "FileContentResource",
    "get_file_resource_manager",
    "shutdown_file_resource_manager",
    "ManagedFileContent",
    "managed_file_content",
    "BatchManagedFileContent",
]