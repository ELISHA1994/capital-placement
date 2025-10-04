"""
File size validation utilities for preventing memory exhaustion and security issues.

This module provides stream-based file size validation that works even when 
file.size attribute is not available, preventing memory exhaustion attacks.
"""

import asyncio
from typing import Dict, Any, Optional
import structlog
from fastapi import UploadFile

from app.domain.exceptions import FileSizeExceededError, InvalidFileError

logger = structlog.get_logger(__name__)


class FileSizeValidator:
    """
    Stream-based file size validator that prevents memory exhaustion.
    
    This validator performs size checks without loading the entire file into memory,
    making it safe against large file attacks and working reliably even when
    file.size attribute is not available.
    """
    
    # Default buffer size for streaming validation (64KB)
    DEFAULT_BUFFER_SIZE = 64 * 1024
    
    # Maximum reasonable file size to prevent obvious attacks (500MB)
    ABSOLUTE_MAX_SIZE = 500 * 1024 * 1024
    
    @classmethod
    async def validate_file_size(
        cls,
        file: UploadFile,
        max_size_bytes: int,
        tenant_config: Optional[Dict[str, Any]] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE
    ) -> Dict[str, Any]:
        """
        Validate file size using stream-based approach.
        
        This method validates file size without loading the entire file into memory,
        providing protection against memory exhaustion attacks.
        
        Args:
            file: FastAPI UploadFile instance
            max_size_bytes: Maximum allowed file size in bytes
            tenant_config: Optional tenant configuration for additional limits
            buffer_size: Buffer size for streaming validation
            
        Returns:
            Dict with validation results: {"valid": bool, "size": int, "reason": str}
            
        Raises:
            FileSizeExceededError: If file exceeds size limits
            InvalidFileError: If file is invalid or corrupted
        """
        filename = getattr(file, 'filename', 'unknown')
        
        logger.debug(
            "Starting stream-based file size validation",
            filename=filename,
            max_size_mb=max_size_bytes / (1024 * 1024),
            has_size_attr=hasattr(file, 'size')
        )
        
        # First check: Use file.size if available and reliable
        if hasattr(file, 'size') and file.size is not None:
            if file.size > cls.ABSOLUTE_MAX_SIZE:
                logger.warning(
                    "File size exceeds absolute maximum",
                    filename=filename,
                    file_size=file.size,
                    absolute_max=cls.ABSOLUTE_MAX_SIZE
                )
                raise FileSizeExceededError(
                    actual_size=file.size,
                    max_size=cls.ABSOLUTE_MAX_SIZE,
                    filename=filename
                )
            
            if file.size > max_size_bytes:
                logger.warning(
                    "File size exceeds tenant limit (via size attribute)",
                    filename=filename,
                    file_size=file.size,
                    max_size=max_size_bytes
                )
                raise FileSizeExceededError(
                    actual_size=file.size,
                    max_size=max_size_bytes,
                    filename=filename
                )
            
            # Size attribute indicates file is within limits
            logger.debug(
                "File size validation passed (via size attribute)",
                filename=filename,
                file_size=file.size
            )
            return {
                "valid": True,
                "size": file.size,
                "reason": None,
                "validation_method": "size_attribute"
            }
        
        # Second check: Stream-based validation for when size is not available
        logger.debug(
            "Performing stream-based size validation",
            filename=filename,
            buffer_size=buffer_size
        )
        
        try:
            # Ensure we're at the beginning of the file
            await file.seek(0)
            
            total_size = 0
            chunk_count = 0
            
            while True:
                # Read a chunk without loading entire file
                chunk = await file.read(buffer_size)
                
                if not chunk:
                    # End of file reached
                    break
                
                chunk_size = len(chunk)
                total_size += chunk_size
                chunk_count += 1
                
                # Check against absolute maximum to prevent obvious attacks
                if total_size > cls.ABSOLUTE_MAX_SIZE:
                    logger.warning(
                        "File size exceeds absolute maximum during streaming",
                        filename=filename,
                        bytes_read=total_size,
                        absolute_max=cls.ABSOLUTE_MAX_SIZE,
                        chunks_read=chunk_count
                    )
                    raise FileSizeExceededError(
                        actual_size=total_size,
                        max_size=cls.ABSOLUTE_MAX_SIZE,
                        filename=filename
                    )
                
                # Check against tenant-specific limit
                if total_size > max_size_bytes:
                    logger.warning(
                        "File size exceeds tenant limit during streaming",
                        filename=filename,
                        bytes_read=total_size,
                        max_size=max_size_bytes,
                        chunks_read=chunk_count
                    )
                    raise FileSizeExceededError(
                        actual_size=total_size,
                        max_size=max_size_bytes,
                        filename=filename
                    )
                
                # Yield control to allow other operations (prevent blocking)
                if chunk_count % 50 == 0:  # Every ~3MB at 64KB chunks
                    await asyncio.sleep(0)
            
            # Reset file pointer to beginning for subsequent operations
            await file.seek(0)
            
            logger.info(
                "Stream-based file size validation completed",
                filename=filename,
                total_size=total_size,
                size_mb=total_size / (1024 * 1024),
                chunks_read=chunk_count
            )
            
            return {
                "valid": True,
                "size": total_size,
                "reason": None,
                "validation_method": "stream_based"
            }
            
        except (OSError, IOError) as e:
            logger.error(
                "File I/O error during size validation",
                filename=filename,
                error=str(e)
            )
            raise InvalidFileError(
                filename=filename,
                reason=f"File I/O error: {str(e)}"
            )
        except Exception as e:
            # Ensure file pointer is reset even on unexpected errors
            try:
                await file.seek(0)
            except:
                pass
            
            logger.error(
                "Unexpected error during file size validation",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__
            )
            raise InvalidFileError(
                filename=filename,
                reason=f"Validation error: {str(e)}"
            )
    
    @classmethod
    def get_tenant_max_file_size(
        cls,
        tenant_config: Dict[str, Any],
        default_mb: int = 10
    ) -> int:
        """
        Extract maximum file size from tenant configuration.
        
        Args:
            tenant_config: Tenant configuration dictionary
            default_mb: Default size limit in MB if not configured
            
        Returns:
            Maximum file size in bytes
        """
        # Check quota_limits first (new structure)
        quota_limits = tenant_config.get("quota_limits", {})
        if "max_document_size_mb" in quota_limits:
            max_mb = quota_limits["max_document_size_mb"]
        else:
            # Fallback to direct configuration (legacy)
            max_mb = tenant_config.get("max_file_size_mb", default_mb)
        
        # Ensure we have a reasonable limit
        if max_mb is None or max_mb <= 0:
            max_mb = default_mb
        
        # Convert to bytes
        max_bytes = max_mb * 1024 * 1024
        
        # Cap at absolute maximum for security
        return min(max_bytes, cls.ABSOLUTE_MAX_SIZE)
    
    @classmethod
    def format_file_size(cls, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string (e.g., "2.5 MB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"