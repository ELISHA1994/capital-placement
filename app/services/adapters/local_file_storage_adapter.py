"""
Local file storage adapter implementation for multi-tenant file storage.

This adapter implements IFileStorage interface for local filesystem storage,
providing complete tenant isolation and following hexagonal architecture patterns.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import structlog
import os
import shutil

from app.domain.interfaces import IFileStorage


logger = structlog.get_logger(__name__)


class LocalFileStorageAdapter(IFileStorage):
    """
    Local filesystem storage adapter with multi-tenant isolation.

    Implements IFileStorage interface for local development and testing.
    All files are organized by tenant_id/upload_id/filename structure
    to ensure complete data isolation between tenants.

    Storage Path Structure:
        {base_path}/{tenant_id}/{upload_id}/{filename}

    Args:
        base_path: Base directory for file storage (default: ./storage/uploads)
    """

    def __init__(self, base_path: str = "./storage/uploads"):
        """
        Initialize local file storage adapter.

        Args:
            base_path: Base directory path for file storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Local file storage adapter initialized",
            base_path=str(self.base_path.absolute())
        )

    def _validate_identifier(self, identifier: str, name: str) -> None:
        """
        Validate identifier for security (prevent path traversal).

        Args:
            identifier: Identifier to validate (tenant_id or upload_id)
            name: Name of the identifier for error messages

        Raises:
            ValueError: If identifier contains invalid characters
        """
        if not identifier or not identifier.strip():
            raise ValueError(f"{name} cannot be empty")

        # Prevent path traversal attacks
        if ".." in identifier or "/" in identifier or "\\" in identifier:
            raise ValueError(f"{name} contains invalid characters")

    def _get_file_path(self, tenant_id: str, upload_id: str, filename: str = "file") -> Path:
        """
        Get full file path for storage.

        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier
            filename: Optional filename (defaults to 'file')

        Returns:
            Path object for the file
        """
        self._validate_identifier(tenant_id, "tenant_id")
        self._validate_identifier(upload_id, "upload_id")

        # Create tenant directory
        tenant_path = self.base_path / tenant_id
        tenant_path.mkdir(parents=True, exist_ok=True)

        # Create upload directory
        upload_path = tenant_path / upload_id
        upload_path.mkdir(parents=True, exist_ok=True)

        # Return file path
        return upload_path / filename

    async def save_file(
        self,
        tenant_id: str,
        upload_id: str,
        filename: str,
        content: bytes
    ) -> str:
        """
        Save file content to local storage.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation
            upload_id: Unique upload identifier for grouping related files
            filename: Original filename with extension
            content: Raw file content as bytes

        Returns:
            Storage path where file was saved

        Raises:
            ValueError: If tenant_id, upload_id, or filename is invalid
            IOError: If file cannot be saved
        """
        try:
            # Validate filename
            if not filename or not filename.strip():
                raise ValueError("filename cannot be empty")

            # Get file path and ensure directories exist
            file_path = self._get_file_path(tenant_id, upload_id, filename)

            # Write file content
            with open(file_path, 'wb') as f:
                f.write(content)

            # Calculate relative storage path for return
            storage_path = f"{tenant_id}/{upload_id}/{filename}"

            logger.info(
                "File saved successfully",
                tenant_id=tenant_id,
                upload_id=upload_id,
                filename=filename,
                size_bytes=len(content),
                storage_path=storage_path
            )

            return storage_path

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "Failed to save file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                filename=filename,
                error=str(e)
            )
            raise IOError(f"Failed to save file: {str(e)}") from e

    async def retrieve_file(
        self,
        tenant_id: str,
        upload_id: str
    ) -> bytes:
        """
        Retrieve file content from storage.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation
            upload_id: Upload identifier to retrieve file for

        Returns:
            Raw file content as bytes

        Raises:
            ValueError: If tenant_id or upload_id is invalid
            FileNotFoundError: If file does not exist
            IOError: If file cannot be retrieved
        """
        try:
            # Get upload directory path
            upload_path = self.base_path / tenant_id / upload_id

            if not upload_path.exists():
                raise FileNotFoundError(
                    f"No files found for tenant_id={tenant_id}, upload_id={upload_id}"
                )

            # Find the first file in the upload directory (excluding metadata files)
            files = [f for f in upload_path.iterdir() if f.is_file() and not f.name.endswith('.meta')]

            if not files:
                raise FileNotFoundError(
                    f"No files found for tenant_id={tenant_id}, upload_id={upload_id}"
                )

            # Read the first file found
            file_path = files[0]

            with open(file_path, 'rb') as f:
                content = f.read()

            logger.debug(
                "File retrieved successfully",
                tenant_id=tenant_id,
                upload_id=upload_id,
                filename=file_path.name,
                size_bytes=len(content)
            )

            return content

        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error(
                "Failed to retrieve file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e)
            )
            raise IOError(f"Failed to retrieve file: {str(e)}") from e

    async def delete_file(
        self,
        tenant_id: str,
        upload_id: str
    ) -> bool:
        """
        Delete file from storage.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation
            upload_id: Upload identifier to delete file for

        Returns:
            True if file was deleted, False if file did not exist

        Raises:
            ValueError: If tenant_id or upload_id is invalid
            IOError: If file cannot be deleted
        """
        try:
            # Get upload directory path
            upload_path = self.base_path / tenant_id / upload_id

            if not upload_path.exists():
                logger.debug(
                    "File does not exist for deletion",
                    tenant_id=tenant_id,
                    upload_id=upload_id
                )
                return False

            # Delete entire upload directory (including all files and metadata)
            shutil.rmtree(upload_path)

            # Try to clean up empty tenant directory
            tenant_path = self.base_path / tenant_id
            if tenant_path.exists() and not any(tenant_path.iterdir()):
                tenant_path.rmdir()
                logger.debug(
                    "Removed empty tenant directory",
                    tenant_id=tenant_id
                )

            logger.info(
                "File deleted successfully",
                tenant_id=tenant_id,
                upload_id=upload_id
            )

            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "Failed to delete file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e)
            )
            raise IOError(f"Failed to delete file: {str(e)}") from e

    async def exists(
        self,
        tenant_id: str,
        upload_id: str
    ) -> bool:
        """
        Check if file exists in storage.

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation
            upload_id: Upload identifier to check existence for

        Returns:
            True if file exists, False otherwise

        Raises:
            ValueError: If tenant_id or upload_id is invalid
        """
        try:
            # Get upload directory path
            upload_path = self.base_path / tenant_id / upload_id

            # Check if directory exists and contains files
            if not upload_path.exists():
                return False

            # Check if there are any files in the directory
            files = [f for f in upload_path.iterdir() if f.is_file() and not f.name.endswith('.meta')]
            exists = len(files) > 0

            logger.debug(
                "File existence check",
                tenant_id=tenant_id,
                upload_id=upload_id,
                exists=exists
            )

            return exists

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "Failed to check file existence",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e)
            )
            raise IOError(f"Failed to check file existence: {str(e)}") from e

    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of local file storage.

        Returns:
            Health check details including storage availability and disk space
        """
        try:
            # Check if base path exists and is writable
            is_available = self.base_path.exists() and os.access(self.base_path, os.W_OK)

            # Get disk space information
            stat = os.statvfs(self.base_path)
            free_space_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            total_space_mb = (stat.f_blocks * stat.f_frsize) / (1024 * 1024)
            used_space_percent = ((total_space_mb - free_space_mb) / total_space_mb) * 100

            health_status = "healthy" if is_available and free_space_mb > 100 else "degraded"

            return {
                "service": "local_file_storage",
                "status": health_status,
                "available": is_available,
                "base_path": str(self.base_path.absolute()),
                "disk_space": {
                    "free_mb": round(free_space_mb, 2),
                    "total_mb": round(total_space_mb, 2),
                    "used_percent": round(used_space_percent, 2)
                }
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "service": "local_file_storage",
                "status": "unhealthy",
                "error": str(e)
            }


__all__ = ["LocalFileStorageAdapter"]