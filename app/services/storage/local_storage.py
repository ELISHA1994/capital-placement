"""Local storage service for file storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class LocalStorageService:
    """
    Local file storage service.
    
    This is a simple implementation for development and testing.
    In production, this would be replaced with cloud storage (S3, Azure Blob, etc.)
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize local storage service.
        
        Args:
            base_path: Base directory for storage (defaults to temp directory)
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(tempfile.gettempdir()) / "capital-placement-storage"
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Local storage initialized", base_path=str(self.base_path))
    
    async def store_file(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Store file and return storage path.
        
        Args:
            file_content: File content bytes
            filename: Original filename
            tenant_id: Tenant identifier for organization
            upload_id: Upload identifier
            **kwargs: Additional storage options
            
        Returns:
            Storage path or identifier
        """
        try:
            # Create tenant-specific directory if provided
            if tenant_id:
                storage_dir = self.base_path / tenant_id
            else:
                storage_dir = self.base_path / "default"
            
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            if upload_id:
                stored_filename = f"{upload_id}_{filename}"
            else:
                import uuid
                stored_filename = f"{uuid.uuid4().hex}_{filename}"
            
            file_path = storage_dir / stored_filename
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.debug(
                "File stored locally",
                filename=filename,
                stored_path=str(file_path),
                size_bytes=len(file_content),
                tenant_id=tenant_id
            )
            
            return str(file_path)
            
        except Exception as e:
            logger.error(
                "Error storing file locally",
                filename=filename,
                error=str(e),
                tenant_id=tenant_id
            )
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if file was deleted successfully
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                logger.debug("File deleted from local storage", file_path=file_path)
                return True
            else:
                logger.warning("File not found for deletion", file_path=file_path)
                return False
                
        except Exception as e:
            logger.error("Error deleting file from local storage", 
                        file_path=file_path, error=str(e))
            return False
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Get file content from storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content bytes or None if not found
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                with open(path, 'rb') as f:
                    return f.read()
            else:
                logger.warning("File not found", file_path=file_path)
                return None
                
        except Exception as e:
            logger.error("Error reading file from local storage",
                        file_path=file_path, error=str(e))
            return None
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file exists
        """
        try:
            return Path(file_path).exists()
        except Exception:
            return False


__all__ = [
    "LocalStorageService",
]