"""Local filesystem storage adapter implementing IFileStorage interface.

This adapter provides secure, multi-tenant file storage on the local filesystem
with comprehensive validation, error handling, and security measures.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles
import structlog

from app.domain.interfaces import IFileStorage

logger = structlog.get_logger(__name__)


class LocalFileStorageAdapter(IFileStorage):
    """
    Local filesystem storage adapter with multi-tenant isolation.

    Implements IFileStorage interface for secure local file storage with:
    - Complete tenant isolation using directory structure
    - Path traversal attack prevention
    - UUID validation for tenant_id and upload_id
    - Automatic directory creation
    - Comprehensive logging and error handling
    - Atomic file operations

    Storage Structure:
        {base_path}/{tenant_id}/{upload_id}/original_file.{ext}

    Security Features:
        - UUID validation prevents path traversal attacks
        - Filename sanitization removes dangerous characters
        - Directory permissions are set securely
        - All operations are logged for audit trails

    Example:
        - >>> adapter = LocalFileStorageAdapter(base_path="/var/uploads")
        - >>> path = await adapter.save_file(
             tenant_id="550e8400-e29b-41d4-a716-446655440000",
             upload_id="660e8400-e29b-41d4-a716-446655440001",
             filename="resume.pdf",
             content=pdf_bytes)
    """

    def __init__(self, base_path: str = "./storage/uploads"):
        """
        Initialize local file storage adapter.

        Args:
            base_path: Base directory for file storage. Created if doesn't exist.
        """
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "LocalFileStorageAdapter initialized",
            base_path=str(self.base_path),
            exists=self.base_path.exists()
        )

    def _validate_uuid(self, value: str, field_name: str) -> None:
        """
        Validate UUID format for security.

        Ensures tenant_id and upload_id are valid UUIDs to prevent
        path traversal attacks and directory manipulation.

        Args:
            value: UUID string to validate
            field_name: Name of field being validated (for error messages)

        Raises:
            ValueError: If UUID format is invalid
        """
        try:
            UUID(value)
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(
                "Invalid UUID format",
                field_name=field_name,
                value=value,
                error=str(e)
            )
            raise ValueError(f"Invalid {field_name}: must be a valid UUID format") from e

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.

        Removes dangerous characters and path traversal attempts while
        preserving the original extension.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for filesystem use

        Raises:
            ValueError: If filename is empty or invalid after sanitization
        """
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")

        # Remove any directory components (path traversal protection)
        filename = os.path.basename(filename)

        # Remove or replace dangerous characters
        # Allow: alphanumeric, dash, underscore, dot
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

        # Prevent hidden files
        if sanitized.startswith('.'):
            sanitized = 'file_' + sanitized

        # Ensure we still have a filename
        if not sanitized or sanitized in ('.', '..'):
            raise ValueError(f"Invalid filename after sanitization: {filename}")

        if sanitized != filename:
            logger.debug(
                "Filename sanitized",
                original=filename,
                sanitized=sanitized
            )

        return sanitized

    def _get_file_path(self, tenant_id: str, upload_id: str) -> Path:
        """
        Construct secure file path with multi-tenant isolation.

        Args:
            tenant_id: Validated tenant UUID
            upload_id: Validated upload UUID

        Returns:
            Path object for the upload directory
        """
        return self.base_path / tenant_id / upload_id

    async def save_file(
        self,
        tenant_id: str,
        upload_id: str,
        filename: str,
        content: bytes
    ) -> str:
        """
        Save file content to local filesystem with tenant isolation.

        Creates directory structure automatically and stores file atomically.
        The file is saved as 'original_file.{ext}' within the upload directory.

        Args:
            tenant_id: Tenant identifier (must be valid UUID)
            upload_id: Unique upload identifier (must be valid UUID)
            filename: Original filename with extension
            content: Raw file content as bytes

        Returns:
            Storage path in format: "tenant_id/upload_id/original_file.ext"

        Raises:
            ValueError: If tenant_id, upload_id, or filename is invalid
            OSError: If file cannot be written (permissions, disk space, etc.)
            Exception: For other storage errors
        """
        try:
            # Security: Validate UUIDs to prevent path traversal
            self._validate_uuid(tenant_id, "tenant_id")
            self._validate_uuid(upload_id, "upload_id")

            # Security: Sanitize filename
            sanitized_filename = self._sanitize_filename(filename)

            # Extract extension and create standardized filename
            extension = Path(sanitized_filename).suffix
            storage_filename = f"original_file{extension}"

            # Create directory structure with proper permissions
            upload_dir = self._get_file_path(tenant_id, upload_id)
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Full file path
            file_path = upload_dir / storage_filename

            # Write file atomically
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)

            # Construct relative storage path
            storage_path = f"{tenant_id}/{upload_id}/{storage_filename}"

            logger.info(
                "File saved successfully",
                tenant_id=tenant_id,
                upload_id=upload_id,
                original_filename=filename,
                storage_filename=storage_filename,
                storage_path=storage_path,
                file_size=len(content)
            )

            return storage_path

        except ValueError:
            # Re-raise validation errors
            raise
        except OSError as e:
            logger.error(
                "Filesystem error saving file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                filename=filename,
                error=str(e),
                error_type=type(e).__name__
            )
            raise OSError(f"Failed to save file: {str(e)}") from e
        except Exception as e:
            logger.error(
                "Unexpected error saving file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                filename=filename,
                error=str(e),
                error_type=type(e).__name__
            )
            raise Exception(f"Storage error: {str(e)}") from e

    async def retrieve_file(
        self,
        tenant_id: str,
        upload_id: str
    ) -> bytes:
        """
        Retrieve file content from local filesystem.

        Reads the stored file and returns its content. The file is expected
        to be stored with the naming convention 'original_file.{ext}'.

        Args:
            tenant_id: Tenant identifier (must be valid UUID)
            upload_id: Upload identifier (must be valid UUID)

        Returns:
            Raw file content as bytes

        Raises:
            ValueError: If tenant_id or upload_id is invalid
            FileNotFoundError: If file does not exist
            OSError: If file cannot be read
            Exception: For other retrieval errors
        """
        try:
            # Security: Validate UUIDs
            self._validate_uuid(tenant_id, "tenant_id")
            self._validate_uuid(upload_id, "upload_id")

            # Get upload directory
            upload_dir = self._get_file_path(tenant_id, upload_id)

            # Check if directory exists
            if not upload_dir.exists():
                logger.warning(
                    "Upload directory not found",
                    tenant_id=tenant_id,
                    upload_id=upload_id,
                    path=str(upload_dir)
                )
                raise FileNotFoundError(
                    f"No file found for tenant_id={tenant_id}, upload_id={upload_id}"
                )

            # Find the original file (should be named original_file.*)
            files = list(upload_dir.glob("original_file.*"))

            if not files:
                logger.warning(
                    "No original file found in upload directory",
                    tenant_id=tenant_id,
                    upload_id=upload_id,
                    directory=str(upload_dir)
                )
                raise FileNotFoundError(
                    f"No file found for tenant_id={tenant_id}, upload_id={upload_id}"
                )

            # Use the first match (should only be one)
            file_path = files[0]

            # Read file content
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()

            logger.info(
                "File retrieved successfully",
                tenant_id=tenant_id,
                upload_id=upload_id,
                file_path=str(file_path),
                file_size=len(content)
            )

            return content

        except ValueError:
            # Re-raise validation errors
            raise
        except FileNotFoundError:
            # Re-raise file not found errors
            raise
        except OSError as e:
            logger.error(
                "Filesystem error retrieving file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise OSError(f"Failed to retrieve file: {str(e)}") from e
        except Exception as e:
            logger.error(
                "Unexpected error retrieving file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise Exception(f"File retrieval error: {str(e)}") from e

    async def delete_file(
        self,
        tenant_id: str,
        upload_id: str
    ) -> bool:
        """
        Delete file and its containing directory from local filesystem.

        Removes both the file and its upload directory for complete cleanup.
        Safe to call even if file doesn't exist.

        Args:
            tenant_id: Tenant identifier (must be valid UUID)
            upload_id: Upload identifier (must be valid UUID)

        Returns:
            True if file was deleted, False if file did not exist

        Raises:
            ValueError: If tenant_id or upload_id is invalid
            OSError: If file cannot be deleted (permissions, etc.)
            Exception: For other deletion errors
        """
        try:
            # Security: Validate UUIDs
            self._validate_uuid(tenant_id, "tenant_id")
            self._validate_uuid(upload_id, "upload_id")

            # Get upload directory
            upload_dir = self._get_file_path(tenant_id, upload_id)

            # Check if directory exists
            if not upload_dir.exists():
                logger.debug(
                    "Upload directory does not exist for deletion",
                    tenant_id=tenant_id,
                    upload_id=upload_id,
                    path=str(upload_dir)
                )
                return False

            # Delete all files in the directory
            deleted_files = []
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(file_path.name)

            # Remove the directory
            upload_dir.rmdir()

            logger.info(
                "File and directory deleted successfully",
                tenant_id=tenant_id,
                upload_id=upload_id,
                deleted_files=deleted_files,
                directory=str(upload_dir)
            )

            return True

        except ValueError:
            # Re-raise validation errors
            raise
        except OSError as e:
            logger.error(
                "Filesystem error deleting file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise OSError(f"Failed to delete file: {str(e)}") from e
        except Exception as e:
            logger.error(
                "Unexpected error deleting file",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise Exception(f"File deletion error: {str(e)}") from e

    async def exists(
        self,
        tenant_id: str,
        upload_id: str
    ) -> bool:
        """
        Check if file exists in local filesystem.

        Args:
            tenant_id: Tenant identifier (must be valid UUID)
            upload_id: Upload identifier (must be valid UUID)

        Returns:
            True if file exists, False otherwise

        Raises:
            ValueError: If tenant_id or upload_id is invalid
            Exception: For other errors during existence check
        """
        try:
            # Security: Validate UUIDs
            self._validate_uuid(tenant_id, "tenant_id")
            self._validate_uuid(upload_id, "upload_id")

            # Get upload directory
            upload_dir = self._get_file_path(tenant_id, upload_id)

            # Check if directory exists and contains original file
            if not upload_dir.exists():
                return False

            # Check for original_file.* pattern
            files = list(upload_dir.glob("original_file.*"))
            exists = len(files) > 0

            logger.debug(
                "File existence check",
                tenant_id=tenant_id,
                upload_id=upload_id,
                exists=exists,
                directory=str(upload_dir)
            )

            return exists

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(
                "Error checking file existence",
                tenant_id=tenant_id,
                upload_id=upload_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise Exception(f"Existence check error: {str(e)}") from e

    async def check_health(self) -> dict[str, Any]:
        """
        Check health of local filesystem storage.

        Verifies that:
        - Base storage directory exists
        - Base storage directory is writable
        - Sufficient disk space is available

        Returns:
            Health check details including status and metrics
        """
        try:
            # Check if base path exists
            path_exists = self.base_path.exists()

            # Check if writable (try to create a test directory)
            is_writable = False
            test_dir = self.base_path / ".health_check"
            try:
                test_dir.mkdir(exist_ok=True)
                is_writable = True
                # Clean up test directory
                test_dir.rmdir()
            except (OSError, PermissionError):
                pass

            # Get disk space info
            try:
                stat = os.statvfs(self.base_path)
                available_bytes = stat.f_bavail * stat.f_frsize
                total_bytes = stat.f_blocks * stat.f_frsize
                used_bytes = total_bytes - available_bytes
                usage_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
            except Exception:
                available_bytes = None
                total_bytes = None
                usage_percent = None

            # Determine health status
            is_healthy = path_exists and is_writable

            health_data = {
                "status": "healthy" if is_healthy else "unhealthy",
                "storage_type": "local_filesystem",
                "base_path": str(self.base_path),
                "path_exists": path_exists,
                "is_writable": is_writable,
                "disk_space": {
                    "available_bytes": available_bytes,
                    "total_bytes": total_bytes,
                    "usage_percent": round(usage_percent, 2) if usage_percent is not None else None
                }
            }

            if is_healthy:
                logger.debug("Storage health check passed", **health_data)
            else:
                logger.warning("Storage health check failed", **health_data)

            return health_data

        except Exception as e:
            logger.error(
                "Health check failed with error",
                error=str(e),
                error_type=type(e).__name__
            )
            return {
                "status": "unhealthy",
                "storage_type": "local_filesystem",
                "error": str(e)
            }


__all__ = ["LocalFileStorageAdapter"]
