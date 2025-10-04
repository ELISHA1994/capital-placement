"""
Unit tests for comprehensive file content validation.

Tests cover:
- Basic file property validation
- MIME type validation
- File signature/magic bytes validation
- Cross-validation between extension, MIME type, and content
- Security threat scanning
- Tenant configuration handling
"""

import pytest
from typing import Dict, Any
from io import BytesIO

from app.services.validation.file_content_validator import FileContentValidator
from app.domain.interfaces import FileValidationResult


class MockUploadFile:
    """Mock UploadFile for testing."""
    
    def __init__(self, content: bytes, filename: str, content_type: str = None):
        self.content = content
        self.filename = filename
        self.content_type = content_type
        self._position = 0
    
    async def read(self, size: int = -1) -> bytes:
        """Read content from the mock file."""
        if size == -1:
            content = self.content[self._position:]
            self._position = len(self.content)
        else:
            content = self.content[self._position:self._position + size]
            self._position += len(content)
        return content
    
    async def seek(self, position: int) -> None:
        """Seek to position in the mock file."""
        self._position = position


@pytest.fixture
def validator():
    """Create a FileContentValidator instance."""
    return FileContentValidator()


@pytest.fixture
def tenant_config():
    """Default tenant configuration for testing."""
    return {
        "processing_configuration": {
            "file_validation": {
                "allowed_file_extensions": [".pdf", ".doc", ".docx", ".txt"],
                "file_type_limits": {
                    ".pdf": 25,
                    ".doc": 10,
                    ".docx": 10,
                    ".txt": 1
                },
                "require_mime_validation": True,
                "require_signature_validation": True,
                "enable_content_scanning": True,
                "block_executable_content": True,
                "block_macro_documents": True,
                "block_script_content": True,
                "validation_mode": "strict",
                "min_confidence_score": 0.8,
                "reject_on_validation_errors": True,
                "reject_on_security_warnings": False,
                "log_validation_details": True
            }
        }
    }


class TestBasicValidation:
    """Test basic file property validation."""
    
    @pytest.mark.asyncio
    async def test_valid_pdf_file(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation of a valid PDF file."""
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj'
        
        result = await validator.validate_file_content(
            file_content=pdf_content,
            filename="test.pdf",
            content_type="application/pdf",
            tenant_config=tenant_config
        )
        
        assert result.is_valid
        assert result.detected_extension == ".pdf"
        assert result.detected_mime_type == "application/pdf"
        assert len(result.validation_errors) == 0
        assert result.confidence_score > 0.8
    
    @pytest.mark.asyncio
    async def test_valid_text_file(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation of a valid text file."""
        text_content = b"This is a plain text file with some content.\nMultiple lines are supported."
        
        result = await validator.validate_file_content(
            file_content=text_content,
            filename="test.txt",
            content_type="text/plain",
            tenant_config=tenant_config
        )
        
        assert result.is_valid
        assert result.detected_extension == ".txt"
        assert result.detected_mime_type == "text/plain"
        assert len(result.validation_errors) == 0
    
    @pytest.mark.asyncio
    async def test_empty_filename(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation fails for empty filename."""
        result = await validator.validate_file_content(
            file_content=b"some content",
            filename="",
            content_type="text/plain",
            tenant_config=tenant_config
        )
        
        assert not result.is_valid
        assert any("Filename cannot be empty" in error for error in result.validation_errors)
    
    @pytest.mark.asyncio
    async def test_long_filename(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation fails for excessively long filename."""
        long_filename = "a" * 300 + ".txt"
        
        result = await validator.validate_file_content(
            file_content=b"some content",
            filename=long_filename,
            content_type="text/plain",
            tenant_config=tenant_config
        )
        
        assert not result.is_valid
        assert any("Filename too long" in error for error in result.validation_errors)
    
    @pytest.mark.asyncio
    async def test_invalid_filename_characters(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation fails for invalid filename characters."""
        result = await validator.validate_file_content(
            file_content=b"some content",
            filename="test<>file.txt",
            content_type="text/plain",
            tenant_config=tenant_config
        )
        
        assert not result.is_valid
        assert any("invalid characters" in error for error in result.validation_errors)


class TestMimeTypeValidation:
    """Test MIME type validation."""
    
    @pytest.mark.asyncio
    async def test_correct_mime_type(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation passes with correct MIME type."""
        pdf_content = b'%PDF-1.4\n'
        
        result = await validator.validate_file_content(
            file_content=pdf_content,
            filename="test.pdf",
            content_type="application/pdf",
            tenant_config=tenant_config
        )
        
        assert result.is_valid
        assert len(result.validation_errors) == 0
    
    @pytest.mark.asyncio
    async def test_incorrect_mime_type(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation fails with incorrect MIME type."""
        pdf_content = b'%PDF-1.4\n'
        
        result = await validator.validate_file_content(
            file_content=pdf_content,
            filename="test.pdf",
            content_type="text/plain",  # Wrong MIME type
            tenant_config=tenant_config
        )
        
        assert not result.is_valid
        assert any("Invalid MIME type" in error for error in result.validation_errors)
    
    @pytest.mark.asyncio
    async def test_missing_mime_type(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation still works with missing MIME type."""
        pdf_content = b'%PDF-1.4\n'
        
        result = await validator.validate_file_content(
            file_content=pdf_content,
            filename="test.pdf",
            content_type=None,
            tenant_config=tenant_config
        )
        
        assert result.is_valid  # Should still pass based on content
        assert result.detected_extension == ".pdf"


class TestFileSignatureValidation:
    """Test file signature/magic bytes validation."""
    
    @pytest.mark.asyncio
    async def test_valid_pdf_signature(self, validator: FileContentValidator):
        """Test PDF file signature validation."""
        pdf_content = b'%PDF-1.4\n1 0 obj'
        
        assert validator.validate_file_signature(pdf_content, ".pdf")
        assert validator.detect_file_type_from_content(pdf_content) == ".pdf"
    
    @pytest.mark.asyncio
    async def test_invalid_pdf_signature(self, validator: FileContentValidator):
        """Test PDF file with wrong signature."""
        fake_pdf_content = b'This is not a PDF file'
        
        assert not validator.validate_file_signature(fake_pdf_content, ".pdf")
        assert validator.detect_file_type_from_content(fake_pdf_content) != ".pdf"
    
    @pytest.mark.asyncio
    async def test_text_file_content(self, validator: FileContentValidator):
        """Test text file content validation."""
        text_content = b"This is plain text content."
        
        assert validator.validate_file_signature(text_content, ".txt")
        assert validator.detect_file_type_from_content(text_content) == ".txt"
    
    @pytest.mark.asyncio
    async def test_binary_content_as_text(self, validator: FileContentValidator):
        """Test binary content incorrectly labeled as text."""
        binary_content = b"\x00\x01\x02\x03\xFF\xFE"
        
        assert not validator.validate_file_signature(binary_content, ".txt")


class TestCrossValidation:
    """Test cross-validation between extension, MIME type, and content."""
    
    @pytest.mark.asyncio
    async def test_consistent_file_properties(self, validator: FileContentValidator):
        """Test file with consistent extension, MIME type, and content."""
        errors = validator.cross_validate_file_properties(
            filename="test.pdf",
            content_type="application/pdf",
            detected_type=".pdf",
            detected_mime="application/pdf"
        )
        
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_extension_content_mismatch(self, validator: FileContentValidator):
        """Test file with mismatched extension and content."""
        errors = validator.cross_validate_file_properties(
            filename="test.pdf",
            content_type="application/pdf",
            detected_type=".txt",  # Content says it's text
            detected_mime="text/plain"
        )
        
        assert len(errors) > 0
        assert any("extension" in error and "detected type" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_mime_content_mismatch(self, validator: FileContentValidator):
        """Test file with mismatched MIME type and content."""
        errors = validator.cross_validate_file_properties(
            filename="test.txt",
            content_type="application/pdf",  # Wrong MIME type
            detected_type=".txt",
            detected_mime="text/plain"
        )
        
        assert len(errors) > 0
        assert any("Content-Type" in error and "extension" in error for error in errors)


class TestSecurityScanning:
    """Test security threat scanning."""
    
    @pytest.mark.asyncio
    async def test_embedded_executable(self, validator: FileContentValidator):
        """Test detection of embedded executable content."""
        malicious_content = b"Normal text content" + b"MZ\x90\x00" + b"more content"
        
        warnings = validator.scan_for_security_threats(malicious_content, "test.txt")
        
        assert len(warnings) > 0
        assert any("executable" in warning.lower() for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_script_injection(self, validator: FileContentValidator):
        """Test detection of script injection attempts."""
        script_content = b"<script>alert('xss')</script>"
        
        warnings = validator.scan_for_security_threats(script_content, "test.txt")
        
        assert len(warnings) > 0
        assert any("script" in warning.lower() for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_directory_traversal(self, validator: FileContentValidator):
        """Test detection of directory traversal attempts."""
        traversal_content = b"../../../etc/passwd"
        
        warnings = validator.scan_for_security_threats(traversal_content, "test.txt")
        
        assert len(warnings) > 0
        assert any("traversal" in warning.lower() for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_null_bytes_in_text(self, validator: FileContentValidator):
        """Test detection of NULL bytes in text files."""
        null_content = b"Normal text\x00hidden content"
        
        warnings = validator.scan_for_security_threats(null_content, "test.txt")
        
        assert len(warnings) > 0
        assert any("NULL bytes" in warning for warning in warnings)
    
    @pytest.mark.asyncio
    async def test_clean_content(self, validator: FileContentValidator):
        """Test clean content passes security scanning."""
        clean_content = b"This is completely normal text content with no security issues."
        
        warnings = validator.scan_for_security_threats(clean_content, "test.txt")
        
        assert len(warnings) == 0


class TestUploadFileValidation:
    """Test upload file validation with memory efficiency."""
    
    @pytest.mark.asyncio
    async def test_valid_upload_file(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation of a valid upload file."""
        content = b"This is a test text file."
        mock_file = MockUploadFile(content, "test.txt", "text/plain")
        
        result = await validator.validate_upload_file(mock_file, tenant_config)
        
        assert result.is_valid
        assert result.filename == "test.txt"
        assert result.file_size == len(content)
    
    @pytest.mark.asyncio
    async def test_large_file_sampling(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test that large files are sampled for analysis."""
        # Create content larger than MAX_ANALYSIS_SIZE
        large_content = b"A" * (validator.MAX_ANALYSIS_SIZE + 1000)
        mock_file = MockUploadFile(large_content, "large.txt", "text/plain")
        
        result = await validator.validate_upload_file(mock_file, tenant_config)
        
        assert result.is_valid
        assert result.validation_details.get("content_sampled") is True
        assert result.validation_details.get("sample_size") == validator.MAX_ANALYSIS_SIZE


class TestFileTypeConfiguration:
    """Test file type configuration and tenant settings."""
    
    @pytest.mark.asyncio
    async def test_allowed_file_types(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test file type allowlist functionality."""
        assert validator.is_file_type_allowed(".pdf", tenant_config)
        assert validator.is_file_type_allowed(".txt", tenant_config)
        assert not validator.is_file_type_allowed(".exe", tenant_config)
        assert not validator.is_file_type_allowed(".bat", tenant_config)
    
    @pytest.mark.asyncio
    async def test_file_type_configs(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test file type configuration retrieval."""
        configs = validator.get_supported_file_types(tenant_config)
        
        assert len(configs) > 0
        
        # Check that PDF config exists and has expected properties
        pdf_config = next((c for c in configs if c.extension == ".pdf"), None)
        assert pdf_config is not None
        assert "application/pdf" in pdf_config.mime_types
        assert len(pdf_config.magic_bytes_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_custom_tenant_extensions(self, validator: FileContentValidator):
        """Test custom tenant file extension configuration."""
        custom_config = {
            "processing_configuration": {
                "file_validation": {
                    "allowed_file_extensions": [".pdf", ".txt"]  # Limited set
                }
            }
        }
        
        assert validator.is_file_type_allowed(".pdf", custom_config)
        assert validator.is_file_type_allowed(".txt", custom_config)
        assert not validator.is_file_type_allowed(".doc", custom_config)  # Not allowed
        assert not validator.is_file_type_allowed(".docx", custom_config)  # Not allowed


class TestConfidenceScoring:
    """Test confidence score calculation."""
    
    @pytest.mark.asyncio
    async def test_high_confidence_file(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test high confidence score for perfect file."""
        pdf_content = b'%PDF-1.4\n1 0 obj'
        
        result = await validator.validate_file_content(
            file_content=pdf_content,
            filename="test.pdf",
            content_type="application/pdf",
            tenant_config=tenant_config
        )
        
        assert result.confidence_score >= 0.9  # Should be very high
    
    @pytest.mark.asyncio
    async def test_low_confidence_file(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test low confidence score for problematic file."""
        # File with mismatched extension and content
        text_content = b"This is text content"
        
        result = await validator.validate_file_content(
            file_content=text_content,
            filename="fake.pdf",  # Wrong extension
            content_type="application/pdf",  # Wrong MIME type
            tenant_config=tenant_config
        )
        
        assert result.confidence_score < 0.5  # Should be low


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_file_content(self, validator: FileContentValidator, tenant_config: Dict[str, Any]):
        """Test validation of empty file content."""
        result = await validator.validate_file_content(
            file_content=b"",
            filename="empty.txt",
            content_type="text/plain",
            tenant_config=tenant_config
        )
        
        # Empty file should be valid for text files
        assert result.is_valid
    
    @pytest.mark.asyncio
    async def test_malformed_tenant_config(self, validator: FileContentValidator):
        """Test handling of malformed tenant configuration."""
        malformed_config = {}  # Missing required keys
        
        result = await validator.validate_file_content(
            file_content=b"test content",
            filename="test.txt",
            content_type="text/plain",
            tenant_config=malformed_config
        )
        
        # Should still work with defaults
        assert result.is_valid
    
    @pytest.mark.asyncio
    async def test_health_check(self, validator: FileContentValidator):
        """Test service health check."""
        health = await validator.check_health()
        
        assert health["service"] == "FileContentValidator"
        assert health["status"] == "healthy"
        assert "supported_types" in health
        assert "security_patterns" in health


if __name__ == "__main__":
    pytest.main([__file__])