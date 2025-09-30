"""
Local storage implementations for development.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Union, BinaryIO
from datetime import datetime, timedelta
import mimetypes
import structlog

from app.core.interfaces import IBlobStorage, BlobMetadata

logger = structlog.get_logger(__name__)


class FileSystemBlobStorage(IBlobStorage):
    """File system-based blob storage for local development."""
    
    def __init__(self, base_path: str = "./local_storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("Local blob storage initialized", base_path=str(self.base_path))
    
    def _get_container_path(self, container: str) -> Path:
        """Get path for container directory."""
        container_path = self.base_path / container
        container_path.mkdir(parents=True, exist_ok=True)
        return container_path
    
    def _get_blob_path(self, container: str, blob_name: str) -> Path:
        """Get full path for blob file."""
        container_path = self._get_container_path(container)
        return container_path / blob_name
    
    def _get_metadata_path(self, container: str, blob_name: str) -> Path:
        """Get path for blob metadata file."""
        blob_path = self._get_blob_path(container, blob_name)
        return blob_path.with_suffix(blob_path.suffix + '.meta')
    
    def _save_metadata(self, container: str, blob_name: str, metadata: dict):
        """Save blob metadata."""
        try:
            metadata_path = self._get_metadata_path(container, blob_name)
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save blob metadata", error=str(e))
    
    def _load_metadata(self, container: str, blob_name: str) -> dict:
        """Load blob metadata."""
        try:
            metadata_path = self._get_metadata_path(container, blob_name)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    import json
                    return json.load(f)
        except Exception as e:
            logger.warning("Failed to load blob metadata", error=str(e))
        
        return {}
    
    async def upload_blob(
        self, 
        container: str, 
        blob_name: str, 
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """Upload blob to local file system."""
        try:
            blob_path = self._get_blob_path(container, blob_name)
            
            # Write blob data
            if isinstance(data, bytes):
                with open(blob_path, 'wb') as f:
                    f.write(data)
            else:
                with open(blob_path, 'wb') as f:
                    shutil.copyfileobj(data, f)
            
            # Determine content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(blob_name)
                content_type = content_type or "application/octet-stream"
            
            # Save metadata
            blob_metadata = {
                "content_type": content_type,
                "size": blob_path.stat().st_size,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            self._save_metadata(container, blob_name, blob_metadata)
            
            # Generate URL (local file path)
            blob_url = f"file://{blob_path.absolute()}"
            
            logger.debug("Blob uploaded locally",
                        container=container,
                        blob_name=blob_name,
                        size=blob_metadata["size"],
                        content_type=content_type)
            
            return blob_url
        except Exception as e:
            logger.error("Failed to upload blob locally", 
                        container=container, 
                        blob_name=blob_name, 
                        error=str(e))
            raise
    
    async def download_blob(self, container: str, blob_name: str) -> bytes:
        """Download blob from local file system."""
        try:
            blob_path = self._get_blob_path(container, blob_name)
            
            if not blob_path.exists():
                raise FileNotFoundError(f"Blob not found: {blob_name}")
            
            with open(blob_path, 'rb') as f:
                data = f.read()
            
            logger.debug("Blob downloaded locally",
                        container=container,
                        blob_name=blob_name,
                        size=len(data))
            
            return data
        except Exception as e:
            logger.error("Failed to download blob locally",
                        container=container,
                        blob_name=blob_name,
                        error=str(e))
            raise
    
    async def delete_blob(self, container: str, blob_name: str) -> bool:
        """Delete blob from local file system."""
        try:
            blob_path = self._get_blob_path(container, blob_name)
            metadata_path = self._get_metadata_path(container, blob_name)
            
            deleted = False
            
            if blob_path.exists():
                blob_path.unlink()
                deleted = True
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            if deleted:
                logger.debug("Blob deleted locally",
                            container=container,
                            blob_name=blob_name)
            
            return deleted
        except Exception as e:
            logger.error("Failed to delete blob locally",
                        container=container,
                        blob_name=blob_name,
                        error=str(e))
            return False
    
    async def list_blobs(self, container: str, prefix: Optional[str] = None) -> List[BlobMetadata]:
        """List blobs in container."""
        try:
            container_path = self._get_container_path(container)
            blobs = []
            
            for file_path in container_path.iterdir():
                if file_path.is_file() and not file_path.name.endswith('.meta'):
                    blob_name = file_path.name
                    
                    # Apply prefix filter
                    if prefix and not blob_name.startswith(prefix):
                        continue
                    
                    # Get file stats
                    stat = file_path.stat()
                    
                    # Load metadata
                    saved_metadata = self._load_metadata(container, blob_name)
                    
                    content_type = saved_metadata.get('content_type')
                    if not content_type:
                        content_type, _ = mimetypes.guess_type(blob_name)
                        content_type = content_type or "application/octet-stream"
                    
                    blob_metadata = BlobMetadata(
                        name=blob_name,
                        size=stat.st_size,
                        content_type=content_type,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        etag=f'"{stat.st_mtime}-{stat.st_size}"',
                        metadata=saved_metadata.get('metadata', {})
                    )
                    
                    blobs.append(blob_metadata)
            
            logger.debug("Listed local blobs",
                        container=container,
                        count=len(blobs),
                        prefix=prefix)
            
            return blobs
        except Exception as e:
            logger.error("Failed to list local blobs",
                        container=container,
                        error=str(e))
            raise
    
    async def get_blob_url(
        self, 
        container: str, 
        blob_name: str, 
        expires_in: Optional[timedelta] = None
    ) -> str:
        """Get blob URL (local file path)."""
        try:
            blob_path = self._get_blob_path(container, blob_name)
            
            if not blob_path.exists():
                raise FileNotFoundError(f"Blob not found: {blob_name}")
            
            # For local development, return file:// URL
            blob_url = f"file://{blob_path.absolute()}"
            
            logger.debug("Generated local blob URL",
                        container=container,
                        blob_name=blob_name,
                        url=blob_url)
            
            return blob_url
        except Exception as e:
            logger.error("Failed to generate local blob URL",
                        container=container,
                        blob_name=blob_name,
                        error=str(e))
            raise
    
    async def blob_exists(self, container: str, blob_name: str) -> bool:
        """Check if blob exists."""
        try:
            blob_path = self._get_blob_path(container, blob_name)
            exists = blob_path.exists()
            
            logger.debug("Checked local blob existence",
                        container=container,
                        blob_name=blob_name,
                        exists=exists)
            
            return exists
        except Exception as e:
            logger.error("Failed to check local blob existence",
                        container=container,
                        blob_name=blob_name,
                        error=str(e))
            return False