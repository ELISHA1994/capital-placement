"""
Comprehensive file content validation service with multi-layered security checks.

This service provides thorough validation including:
- MIME type validation from Content-Type headers
- File signature validation using magic bytes
- Cross-validation of extension vs content consistency  
- Security threat detection and malicious file pattern scanning
- Tenant-configurable validation rules and policies
"""

import asyncio
import mimetypes
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import structlog
from fastapi import UploadFile

from app.domain.interfaces import IFileContentValidator, FileValidationResult, FileTypeConfig
from app.domain.exceptions import FileSizeExceededError, InvalidFileError
from app.domain.utils import FileSizeValidator

logger = structlog.get_logger(__name__)


class FileContentValidator(IFileContentValidator):
    """
    Comprehensive file content validation with security-first approach.
    
    Implements multi-layered validation:
    1. Basic file properties (name, size, extension)
    2. MIME type validation from Content-Type header
    3. File signature validation using magic bytes
    4. Cross-validation for consistency between extension/MIME/content
    5. Security threat scanning for malicious patterns
    6. Tenant-specific policy enforcement
    """
    
    # Magic byte signatures for supported file types
    MAGIC_BYTES_SIGNATURES = {
        '.pdf': [
            b'%PDF-',  # Standard PDF signature
        ],
        '.doc': [
            b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1',  # MS Office compound document
            b'\x0D\x44\x4F\x43',  # Older DOC format
        ],
        '.docx': [
            b'PK\x03\x04',  # ZIP-based format (first check)
            b'[Content_Types].xml',  # Secondary validation for DOCX
        ],
        '.txt': [
            # Text files don't have consistent magic bytes, validate by content
        ],
    }
    
    # MIME type mappings for each supported extension
    MIME_TYPE_MAPPINGS = {
        '.pdf': [
            'application/pdf',
        ],
        '.doc': [
            'application/msword',
            'application/vnd.ms-word',
        ],
        '.docx': [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-word.document.macroEnabled.12',
        ],
        '.txt': [
            'text/plain',
            'text/plain; charset=utf-8',
            'text/plain; charset=iso-8859-1',
        ],
    }
    
    # Suspicious patterns that could indicate malicious content
    SECURITY_PATTERNS = [
        # Embedded executables
        (rb'MZ\x90\x00', 'Embedded Windows executable detected'),
        (rb'\x7fELF', 'Embedded Linux executable detected'),
        
        # Script patterns
        (rb'<script[^>]*>', 'Embedded JavaScript detected'),
        (rb'javascript:', 'JavaScript URL scheme detected'),
        (rb'vbscript:', 'VBScript URL scheme detected'),
        
        # Macro patterns
        (rb'Microsoft Office Macro', 'Office macro detected'),
        (rb'VBA\x00', 'VBA macro detected'),
        
        # Suspicious strings
        (rb'cmd\.exe', 'Command execution reference detected'),
        (rb'powershell', 'PowerShell reference detected'),
        (rb'/bin/sh', 'Shell reference detected'),
        
        # File system manipulation
        (rb'\.\./', 'Directory traversal pattern detected'),
        (rb'..\\', 'Directory traversal pattern detected'),
    ]
    
    # Maximum content sample size for analysis (to prevent memory issues)
    MAX_ANALYSIS_SIZE = 1024 * 1024  # 1MB
    
    def __init__(self):
        """Initialize the file content validator."""
        self._logger = structlog.get_logger(__name__)
        
        # Build reverse MIME type lookup
        self._mime_to_extension = {}
        for ext, mimes in self.MIME_TYPE_MAPPINGS.items():
            for mime in mimes:
                self._mime_to_extension[mime.lower()] = ext
    
    async def check_health(self) -> Dict[str, Any]:
        """Health check for the file content validator."""
        return {
            "service": "FileContentValidator",
            "status": "healthy",
            "supported_types": list(self.MAGIC_BYTES_SIGNATURES.keys()),
            "security_patterns": len(self.SECURITY_PATTERNS),
        }
    
    async def validate_file_content(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> FileValidationResult:
        """
        Perform comprehensive file content validation.
        
        This method validates file content through multiple layers:
        1. Basic validation (filename, size)
        2. Extension validation against allowed types
        3. MIME type validation from Content-Type header
        4. File signature validation using magic bytes
        5. Cross-validation for consistency
        6. Security threat scanning
        """
        self._logger.info(
            "Starting comprehensive file content validation",
            filename=filename,
            file_size=len(file_content),
            content_type=content_type,
            tenant_config_present=tenant_config is not None
        )
        
        result = FileValidationResult(
            is_valid=False,
            filename=filename,
            file_size=len(file_content),
            validation_details={
                "validation_layers": [],
                "checks_performed": [],
            }
        )
        
        try:
            # Layer 1: Basic validation
            basic_errors = await self._validate_basic_properties(
                filename, len(file_content), tenant_config
            )
            result.validation_details["checks_performed"].append("basic_properties")
            
            if basic_errors:
                result.validation_errors.extend(basic_errors)
                result.validation_details["validation_layers"].append({
                    "layer": "basic_properties",
                    "status": "failed",
                    "errors": basic_errors
                })
                return result
            
            result.validation_details["validation_layers"].append({
                "layer": "basic_properties", 
                "status": "passed"
            })
            
            # Layer 2: Extension validation
            file_extension = self._extract_file_extension(filename)
            if not self.is_file_type_allowed(file_extension, tenant_config):
                error_msg = f"File type '{file_extension}' is not allowed"
                result.validation_errors.append(error_msg)
                result.validation_details["validation_layers"].append({
                    "layer": "extension_validation",
                    "status": "failed", 
                    "errors": [error_msg]
                })
                return result
            
            result.validation_details["checks_performed"].append("extension_validation")
            result.validation_details["validation_layers"].append({
                "layer": "extension_validation",
                "status": "passed",
                "detected_extension": file_extension
            })
            
            # Layer 3: MIME type validation
            mime_errors = self._validate_mime_type(content_type, file_extension)
            result.validation_details["checks_performed"].append("mime_validation")
            
            if mime_errors:
                result.validation_errors.extend(mime_errors)
                result.validation_details["validation_layers"].append({
                    "layer": "mime_validation",
                    "status": "failed",
                    "errors": mime_errors
                })
                # Don't return here - continue with other validations
            else:
                result.validation_details["validation_layers"].append({
                    "layer": "mime_validation",
                    "status": "passed",
                    "provided_mime": content_type
                })
            
            # Layer 4: File signature validation
            detected_type = self.detect_file_type_from_content(file_content)
            detected_mime = self.detect_mime_type_from_content(file_content)
            
            result.detected_extension = detected_type
            result.detected_mime_type = detected_mime
            
            signature_valid = self.validate_file_signature(file_content, file_extension)
            result.validation_details["checks_performed"].append("signature_validation")
            
            if not signature_valid:
                error_msg = f"File signature does not match extension '{file_extension}'"
                result.validation_errors.append(error_msg)
                result.validation_details["validation_layers"].append({
                    "layer": "signature_validation",
                    "status": "failed",
                    "errors": [error_msg],
                    "detected_type": detected_type
                })
            else:
                result.validation_details["validation_layers"].append({
                    "layer": "signature_validation",
                    "status": "passed",
                    "detected_type": detected_type
                })
            
            # Layer 5: Cross-validation
            cross_errors = self.cross_validate_file_properties(
                filename, content_type, detected_type, detected_mime
            )
            result.validation_details["checks_performed"].append("cross_validation")
            
            if cross_errors:
                result.validation_errors.extend(cross_errors)
                result.validation_details["validation_layers"].append({
                    "layer": "cross_validation",
                    "status": "failed",
                    "errors": cross_errors
                })
            else:
                result.validation_details["validation_layers"].append({
                    "layer": "cross_validation",
                    "status": "passed"
                })
            
            # Layer 6: Security threat scanning
            security_warnings = self.scan_for_security_threats(file_content, filename)
            result.security_warnings = security_warnings
            result.validation_details["checks_performed"].append("security_scan")
            
            if security_warnings:
                result.validation_details["validation_layers"].append({
                    "layer": "security_scan",
                    "status": "warnings",
                    "warnings": security_warnings
                })
            else:
                result.validation_details["validation_layers"].append({
                    "layer": "security_scan",
                    "status": "passed"
                })
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence_score(result)
            
            # Determine overall validity
            # File is valid if no validation errors (warnings are okay)
            result.is_valid = len(result.validation_errors) == 0
            
            self._logger.info(
                "File content validation completed",
                filename=filename,
                is_valid=result.is_valid,
                confidence_score=result.confidence_score,
                validation_errors=len(result.validation_errors),
                security_warnings=len(result.security_warnings)
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                "Error during file content validation",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__
            )
            result.validation_errors.append(f"Validation failed: {str(e)}")
            return result
    
    async def validate_upload_file(
        self,
        file: UploadFile,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> FileValidationResult:
        """
        Validate uploaded file with memory-efficient content sampling.
        
        For large files, only analyzes a sample to prevent memory exhaustion
        while still performing comprehensive validation.
        """
        filename = getattr(file, 'filename', 'unknown')
        content_type = getattr(file, 'content_type', None)
        
        self._logger.debug(
            "Starting upload file validation",
            filename=filename,
            content_type=content_type
        )
        
        try:
            # First validate size using existing stream-based validator
            size_validation = await FileSizeValidator.validate_file_size(
                file=file,
                max_size_bytes=FileSizeValidator.get_tenant_max_file_size(
                    tenant_config or {}, default_mb=10
                ),
                tenant_config=tenant_config
            )
            
            # Read content sample for analysis
            await file.seek(0)
            content_sample = await file.read(self.MAX_ANALYSIS_SIZE)
            await file.seek(0)  # Reset for later use
            
            # Perform validation on content sample
            result = await self.validate_file_content(
                content_sample, filename, content_type, tenant_config
            )
            
            # Update file size to actual size from size validation
            result.file_size = size_validation["size"]
            
            # Add note if content was sampled
            if len(content_sample) >= self.MAX_ANALYSIS_SIZE:
                result.validation_details["content_sampled"] = True
                result.validation_details["sample_size"] = len(content_sample)
            
            return result
            
        except FileSizeExceededError as e:
            return FileValidationResult(
                is_valid=False,
                filename=filename,
                file_size=e.actual_size,
                validation_errors=[str(e)]
            )
        except Exception as e:
            self._logger.error(
                "Error during upload file validation",
                filename=filename,
                error=str(e)
            )
            return FileValidationResult(
                is_valid=False,
                filename=filename,
                file_size=0,
                validation_errors=[f"Upload validation failed: {str(e)}"]
            )
    
    def detect_file_type_from_content(self, file_content: bytes) -> Optional[str]:
        """
        Detect file type from magic bytes/file signature.
        
        Analyzes the beginning of file content to identify the actual file type
        based on known magic byte patterns.
        """
        if not file_content:
            return None
        
        # Check against known magic byte signatures
        for extension, signatures in self.MAGIC_BYTES_SIGNATURES.items():
            for signature in signatures:
                if file_content.startswith(signature):
                    return extension
        
        # Special handling for DOCX (ZIP-based format)
        if file_content.startswith(b'PK\x03\x04'):
            # Check if it contains Office-specific files
            if b'[Content_Types].xml' in file_content[:2048]:
                return '.docx'
        
        # Special handling for text files (no magic bytes)
        if self._appears_to_be_text(file_content[:1024]):
            return '.txt'
        
        return None
    
    def detect_mime_type_from_content(self, file_content: bytes) -> Optional[str]:
        """
        Detect MIME type from file content analysis.
        
        Uses magic bytes and content analysis to determine the most likely
        MIME type for the given file content.
        """
        detected_extension = self.detect_file_type_from_content(file_content)
        
        if detected_extension and detected_extension in self.MIME_TYPE_MAPPINGS:
            # Return the primary MIME type for detected extension
            return self.MIME_TYPE_MAPPINGS[detected_extension][0]
        
        return None
    
    def validate_file_signature(
        self,
        file_content: bytes,
        expected_extension: str,
    ) -> bool:
        """
        Validate file signature matches expected extension.
        
        Checks if the magic bytes at the beginning of the file content
        match the expected file type based on the extension.
        """
        if not file_content or not expected_extension:
            return False
        
        detected_type = self.detect_file_type_from_content(file_content)
        
        # If we couldn't detect the type, it might be a text file
        if detected_type is None and expected_extension == '.txt':
            return self._appears_to_be_text(file_content[:1024])
        
        return detected_type == expected_extension
    
    def cross_validate_file_properties(
        self,
        filename: str,
        content_type: Optional[str],
        detected_type: Optional[str],
        detected_mime: Optional[str],
    ) -> List[str]:
        """
        Cross-validate filename extension, MIME type, and detected type for consistency.
        
        Checks for mismatches between what the file claims to be (via filename and
        Content-Type header) vs what it actually is (via content analysis).
        """
        errors = []
        
        file_extension = self._extract_file_extension(filename)
        
        # Validate extension vs detected type
        if detected_type and detected_type != file_extension:
            errors.append(
                f"File extension '{file_extension}' does not match detected type '{detected_type}'"
            )
        
        # Validate MIME type vs extension
        if content_type and file_extension in self.MIME_TYPE_MAPPINGS:
            expected_mimes = self.MIME_TYPE_MAPPINGS[file_extension]
            if not any(content_type.lower().startswith(mime.lower()) for mime in expected_mimes):
                errors.append(
                    f"Content-Type '{content_type}' does not match extension '{file_extension}'"
                )
        
        # Validate MIME type vs detected MIME
        if content_type and detected_mime and content_type.lower() != detected_mime.lower():
            errors.append(
                f"Provided MIME type '{content_type}' does not match detected MIME '{detected_mime}'"
            )
        
        return errors
    
    def scan_for_security_threats(
        self,
        file_content: bytes,
        filename: str,
    ) -> List[str]:
        """
        Scan file content for potential security threats.
        
        Looks for patterns that could indicate malicious content such as:
        - Embedded executables
        - Script injections  
        - Macro viruses
        - Directory traversal attempts
        """
        warnings = []
        
        # Limit scanning to reasonable size to prevent performance issues
        scan_content = file_content[:self.MAX_ANALYSIS_SIZE]
        
        # Check against known suspicious patterns
        for pattern, warning_msg in self.SECURITY_PATTERNS:
            if pattern in scan_content:
                warnings.append(warning_msg)
        
        # Check for suspiciously long strings (potential buffer overflow)
        lines = scan_content.split(b'\n')[:100]  # Check first 100 lines
        for line in lines:
            if len(line) > 10000:  # Very long line
                warnings.append("Suspiciously long line detected (potential buffer overflow)")
                break
        
        # Check for NULL bytes in text files (potential binary injection)
        file_extension = self._extract_file_extension(filename)
        if file_extension == '.txt' and b'\x00' in scan_content:
            warnings.append("NULL bytes detected in text file")
        
        return warnings
    
    def get_supported_file_types(
        self,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> List[FileTypeConfig]:
        """
        Get list of supported file types with validation rules.
        
        Returns comprehensive configuration for each supported file type
        including MIME types, magic bytes, and security settings.
        """
        configs = []
        
        for extension in self.MAGIC_BYTES_SIGNATURES.keys():
            config = FileTypeConfig(
                extension=extension,
                mime_types=self.MIME_TYPE_MAPPINGS.get(extension, []),
                magic_bytes_patterns=self.MAGIC_BYTES_SIGNATURES.get(extension, []),
                max_size_mb=self._get_max_size_for_type(extension, tenant_config),
                description=self._get_type_description(extension),
                security_level="standard",
                allow_binary_content=(extension != '.txt')
            )
            configs.append(config)
        
        return configs
    
    def is_file_type_allowed(
        self,
        extension: str,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if file type is allowed based on extension.

        Checks against tenant configuration or default allowed types.
        """
        if not extension:
            return False

        if tenant_config:
            # Handle both dict and domain entity (TenantConfiguration)
            if hasattr(tenant_config, 'get'):
                # It's a dictionary
                allowed_extensions = tenant_config.get(
                    "allowed_file_extensions",
                    [".pdf", ".doc", ".docx", ".txt"]
                )
            else:
                # It's a domain entity - use default for now
                # TODO: Access tenant entity's configuration attributes
                allowed_extensions = [".pdf", ".doc", ".docx", ".txt"]
        else:
            allowed_extensions = [".pdf", ".doc", ".docx", ".txt"]

        return extension.lower() in [ext.lower() for ext in allowed_extensions]
    
    # Private helper methods
    
    async def _validate_basic_properties(
        self,
        filename: str,
        file_size: int,
        tenant_config: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Validate basic file properties like name and size."""
        errors = []
        
        # Validate filename
        if not filename or len(filename.strip()) == 0:
            errors.append("Filename cannot be empty")
        elif len(filename) > 255:
            errors.append("Filename too long (max 255 characters)")
        elif not re.match(r'^[a-zA-Z0-9._\-\s]+$', filename):
            errors.append("Filename contains invalid characters")
        
        # Validate file size
        max_size = FileSizeValidator.get_tenant_max_file_size(
            tenant_config or {}, default_mb=10
        )
        if file_size > max_size:
            errors.append(f"File size exceeds limit ({file_size} > {max_size} bytes)")
        
        return errors
    
    def _extract_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        if not filename or '.' not in filename:
            return ""
        return '.' + filename.split('.')[-1].lower()
    
    def _validate_mime_type(
        self,
        content_type: Optional[str],
        file_extension: str,
    ) -> List[str]:
        """Validate Content-Type header against expected MIME types."""
        errors = []
        
        if not content_type:
            # Missing Content-Type is not necessarily an error
            return errors
        
        if file_extension in self.MIME_TYPE_MAPPINGS:
            expected_mimes = self.MIME_TYPE_MAPPINGS[file_extension]
            if not any(content_type.lower().startswith(mime.lower()) for mime in expected_mimes):
                errors.append(
                    f"Invalid MIME type '{content_type}' for extension '{file_extension}'. "
                    f"Expected one of: {', '.join(expected_mimes)}"
                )
        
        return errors
    
    def _appears_to_be_text(self, content: bytes) -> bool:
        """Check if content appears to be text (for .txt file validation)."""
        if not content:
            return True  # Empty file is valid text
        
        # Check for NULL bytes (strong indicator of binary content)
        if b'\x00' in content:
            return False
        
        # Count printable characters
        try:
            decoded = content.decode('utf-8', errors='ignore')
            printable_chars = sum(1 for c in decoded if c.isprintable() or c.isspace())
            return printable_chars / len(decoded) > 0.95  # 95% printable
        except:
            return False
    
    def _calculate_confidence_score(self, result: FileValidationResult) -> float:
        """Calculate confidence score based on validation results."""
        score = 1.0
        
        # Reduce score for each validation error
        score -= len(result.validation_errors) * 0.2
        
        # Reduce score for security warnings
        score -= len(result.security_warnings) * 0.1
        
        # Bonus for successful cross-validation
        if (result.detected_extension and 
            result.detected_extension == self._extract_file_extension(result.filename)):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_max_size_for_type(
        self,
        extension: str,
        tenant_config: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Get maximum size limit for specific file type."""
        # Handle both dict and domain entity
        if tenant_config:
            if hasattr(tenant_config, 'get'):
                # It's a dictionary
                if "file_type_limits" in tenant_config:
                    type_limits = tenant_config["file_type_limits"]
                    return type_limits.get(extension)
            # If it's a domain entity, use defaults for now

        # Default limits by type
        defaults = {
            '.pdf': 25,   # 25MB for PDFs
            '.doc': 10,   # 10MB for DOC
            '.docx': 10,  # 10MB for DOCX
            '.txt': 1,    # 1MB for text files
        }
        return defaults.get(extension)
    
    def _get_type_description(self, extension: str) -> str:
        """Get human-readable description for file type."""
        descriptions = {
            '.pdf': 'Adobe PDF Document',
            '.doc': 'Microsoft Word Document (Legacy)',
            '.docx': 'Microsoft Word Document',
            '.txt': 'Plain Text File',
        }
        return descriptions.get(extension, f'Unknown file type: {extension}')


__all__ = ["FileContentValidator"]