"""Provider utilities for file storage services."""

from __future__ import annotations

from typing import Optional
import structlog

from app.core.config import get_settings
from app.domain.interfaces import IFileStorage
from app.infrastructure.adapters.storage_adapter import LocalFileStorageAdapter


logger = structlog.get_logger(__name__)

_file_storage: Optional[IFileStorage] = None


async def get_file_storage() -> IFileStorage:
    """
    Return singleton file storage service configured via settings.

    Initializes and returns the appropriate file storage implementation based
    on the FILE_STORAGE_TYPE setting. Currently supports:
    - 'local': LocalFileStorageAdapter for local filesystem storage
    - 's3': S3 storage (not yet implemented)

    Returns:
        IFileStorage: File storage service instance

    Raises:
        NotImplementedError: If unsupported storage type is configured
        ValueError: If configuration is invalid

    Example:
        ```python
        file_storage = await get_file_storage()
        await file_storage.save_file(tenant_id, upload_id, filename, content)
        ```
    """
    global _file_storage

    if _file_storage is not None:
        return _file_storage

    settings = get_settings()
    storage_type = settings.FILE_STORAGE_TYPE.lower()

    if storage_type == "local":
        _file_storage = LocalFileStorageAdapter(
            base_path=settings.FILE_STORAGE_PATH
        )
        logger.info(
            "File storage service initialized",
            storage_type="local",
            base_path=settings.FILE_STORAGE_PATH
        )
    elif storage_type == "s3":
        raise NotImplementedError(
            "S3 storage not yet implemented. Use storage_type='local' for now."
        )
    else:
        raise ValueError(
            f"Unsupported storage type: {storage_type}. "
            f"Supported types: 'local', 's3'"
        )

    return _file_storage


async def shutdown_file_storage() -> None:
    """
    Shutdown and cleanup file storage resources.

    This function is called during application shutdown to properly
    cleanup any resources held by the file storage service.
    """
    global _file_storage

    if _file_storage is not None:
        logger.info("File storage service shutting down")
        _file_storage = None


__all__ = [
    "get_file_storage",
    "shutdown_file_storage",
]