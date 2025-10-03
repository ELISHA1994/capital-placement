"""Storage adapter implementation."""

from __future__ import annotations

import os
import aiofiles
from pathlib import Path
from typing import Any
import structlog

logger = structlog.get_logger(__name__)


class StorageAdapter:
    """Simple file system storage adapter implementing IStorageService interface."""

    def __init__(self, base_path: str = "./storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def store_file(self, file_content: bytes, filename: str, **kwargs) -> str:
        """Store file and return storage path."""
        try:
            # Create tenant-specific directory if provided
            tenant_id = kwargs.get('tenant_id', 'default')
            tenant_dir = self.base_path / tenant_id
            tenant_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename if file exists
            file_path = tenant_dir / filename
            counter = 1
            base_name = file_path.stem
            extension = file_path.suffix
            
            while file_path.exists():
                file_path = tenant_dir / f"{base_name}_{counter}{extension}"
                counter += 1
            
            # Write file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            relative_path = str(file_path.relative_to(self.base_path))
            
            logger.debug(
                "File stored successfully",
                filename=filename,
                storage_path=relative_path,
                file_size=len(file_content)
            )
            
            return relative_path
            
        except Exception as e:
            logger.error(
                "Failed to store file",
                filename=filename,
                error=str(e)
            )
            raise Exception(f"Storage error: {str(e)}")

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        try:
            full_path = self.base_path / file_path
            if full_path.exists():
                full_path.unlink()
                logger.debug("File deleted successfully", file_path=file_path)
                return True
            else:
                logger.warning("File not found for deletion", file_path=file_path)
                return False
                
        except Exception as e:
            logger.error(
                "Failed to delete file",
                file_path=file_path,
                error=str(e)
            )
            return False

    async def get_file(self, file_path: str) -> bytes:
        """Get file content."""
        try:
            full_path = self.base_path / file_path
            async with aiofiles.open(full_path, 'rb') as f:
                content = await f.read()
            
            logger.debug("File retrieved successfully", file_path=file_path)
            return content
            
        except Exception as e:
            logger.error(
                "Failed to retrieve file",
                file_path=file_path,
                error=str(e)
            )
            raise Exception(f"File retrieval error: {str(e)}")

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        full_path = self.base_path / file_path
        return full_path.exists()


__all__ = ["StorageAdapter"]