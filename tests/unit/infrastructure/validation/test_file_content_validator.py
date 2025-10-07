"""Comprehensive tests for FileContentValidator in infrastructure layer."""

import pytest
from io import BytesIO
from unittest.mock import Mock, AsyncMock, patch
from fastapi import UploadFile

from app.infrastructure.validation.file_content_validator import FileContentValidator
from app.domain.interfaces import FileValidationResult
from app.domain.exceptions import InvalidFileError, FileSizeExceededError


@pytest.fixture
def validator():
    """Create file content validator instance."""
    return FileContentValidator()


@pytest.fixture
def tenant_config():
    """Create tenant configuration."""
    return {
        "allowed_file_extensions": [".pdf", ".docx", ".txt"],
        "max_file_size_mb": 10
    }


# ==================== File Type Validation Tests (5 tests) ====================

@pytest.mark.asyncio
async def test_valid_pdf_file(validator):
    """Test validation of valid PDF file."""
    # PDF magic bytes
    content = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'
    filename = "resume.pdf"
    content_type = "application/pdf"

    result = await validator.validate_file_content(content, filename, content_type)

    assert result.is_valid is True
    assert result.detected_extension == '.pdf'
    assert result.detected_mime_type == 'application/pdf'
    assert len(result.validation_errors) == 0


@pytest.mark.asyncio
async def test_valid_docx_file(validator):
    """Test validation of valid DOCX file."""
    # DOCX magic bytes (ZIP format with Office-specific files)
    content = b'PK\x03\x04' + b'\x00' * 100 + b'[Content_Types].xml' + b'\x00' * 100
    filename = "document.docx"
    content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    result = await validator.validate_file_content(content, filename, content_type)

    assert result.is_valid is True
    assert result.detected_extension == '.docx'


@pytest.mark.asyncio
async def test_valid_text_file(validator):
    """Test validation of valid text file."""
    content = b'This is a plain text file with some content.\nLine 2\nLine 3'
    filename = "notes.txt"
    content_type = "text/plain"

    result = await validator.validate_file_content(content, filename, content_type)

    assert result.is_valid is True
    assert result.detected_extension == '.txt'


@pytest.mark.asyncio
async def test_invalid_file_extension(validator, tenant_config):
    """Test validation rejects invalid file extension."""
    content = b'MZ\x90\x00\x03'  # Executable magic bytes
    filename = "malware.exe"

    result = await validator.validate_file_content(content, filename, None, tenant_config)

    assert result.is_valid is False
    assert "not allowed" in str(result.validation_errors[0]).lower()


@pytest.mark.asyncio
async def test_case_insensitive_extension_validation(validator):
    """Test that extension validation is case-insensitive."""
    content = b'%PDF-1.4\n'
    filename = "document.PDF"  # Uppercase extension

    result = await validator.validate_file_content(content, filename)

    assert result.is_valid is True
    assert result.detected_extension == '.pdf'


# ==================== File Size Validation Tests (5 tests) ====================

@pytest.mark.asyncio
async def test_file_size_within_limit(validator, tenant_config):
    """Test validation of file within size limit."""
    content = b'Small file content'
    filename = "small.pdf"

    result = await validator.validate_file_content(content, filename, None, tenant_config)

    assert result.file_size == len(content)
    # Should not fail size validation (size check is in basic validation)


@pytest.mark.asyncio
async def test_file_size_exceeds_limit(validator):
    """Test validation rejects file exceeding size limit."""
    # Create 11MB content (exceeds default 10MB limit)
    large_content = b'x' * (11 * 1024 * 1024)
    filename = "large.pdf"
    tenant_config = {"max_file_size_mb": 10}

    result = await validator.validate_file_content(large_content, filename, None, tenant_config)

    assert result.is_valid is False
    assert any("size" in str(err).lower() for err in result.validation_errors)


@pytest.mark.asyncio
async def test_zero_byte_file(validator):
    """Test validation of zero-byte file."""
    content = b''
    filename = "empty.txt"

    result = await validator.validate_file_content(content, filename)

    # Empty file should still be validated
    assert result.file_size == 0


@pytest.mark.asyncio
async def test_file_size_exactly_at_limit(validator):
    """Test file exactly at size limit passes validation."""
    # Create exactly 1MB for txt file (default limit)
    content = b'x' * (1 * 1024 * 1024)
    filename = "exact_limit.txt"

    # Should pass basic validation
    result = await validator.validate_file_content(content, filename)
    assert result.file_size == len(content)


@pytest.mark.asyncio
async def test_upload_file_memory_efficient_sampling(validator):
    """Test that large upload files are sampled efficiently."""
    # Create large content
    large_content = b'%PDF-1.4\n' + b'x' * (2 * 1024 * 1024)  # 2MB

    # Mock UploadFile
    upload_file = Mock(spec=UploadFile)
    upload_file.filename = "large.pdf"
    upload_file.content_type = "application/pdf"
    upload_file.seek = AsyncMock()
    upload_file.read = AsyncMock(return_value=large_content[:validator.MAX_ANALYSIS_SIZE])

    with patch('app.infrastructure.validation.file_content_validator.FileSizeValidator.validate_file_size',
               return_value={"size": len(large_content)}):
        result = await validator.validate_upload_file(upload_file)

    # Should sample content
    assert result.validation_details.get("content_sampled") is True


# ==================== Content Validation Tests (5 tests) ====================

@pytest.mark.asyncio
async def test_pdf_signature_validation(validator):
    """Test PDF file signature validation."""
    valid_pdf_content = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'

    is_valid = validator.validate_file_signature(valid_pdf_content, '.pdf')

    assert is_valid is True


@pytest.mark.asyncio
async def test_invalid_pdf_signature(validator):
    """Test rejection of invalid PDF signature."""
    invalid_content = b'This is not a PDF'
    filename = "fake.pdf"

    result = await validator.validate_file_content(invalid_content, filename)

    assert result.is_valid is False
    assert any("signature" in str(err).lower() for err in result.validation_errors)


@pytest.mark.asyncio
async def test_mime_type_mismatch(validator):
    """Test detection of MIME type mismatch."""
    pdf_content = b'%PDF-1.4\n'
    filename = "document.pdf"
    wrong_mime = "text/plain"  # Wrong MIME type for PDF

    result = await validator.validate_file_content(pdf_content, filename, wrong_mime)

    # Should have validation error about MIME mismatch
    assert any("mime" in str(err).lower() or "content-type" in str(err).lower()
               for err in result.validation_errors)


@pytest.mark.asyncio
async def test_extension_content_mismatch(validator):
    """Test detection of extension/content mismatch."""
    pdf_content = b'%PDF-1.4\n'
    filename = "document.txt"  # Wrong extension

    result = await validator.validate_file_content(pdf_content, filename)

    assert result.is_valid is False
    assert result.detected_extension == '.pdf'
    # Should detect mismatch
    assert any("does not match" in str(err).lower() for err in result.validation_errors)


@pytest.mark.asyncio
async def test_text_file_binary_content_detection(validator):
    """Test detection of binary content in text file."""
    binary_content = b'Text with null bytes\x00\x00\x00'
    filename = "suspicious.txt"

    result = await validator.validate_file_content(binary_content, filename)

    # Should have security warning about NULL bytes
    assert any("null" in str(warn).lower() for warn in result.security_warnings)


# ==================== Security Scanning Tests (6 tests) ====================

@pytest.mark.asyncio
async def test_detect_embedded_executable(validator):
    """Test detection of embedded Windows executable."""
    # MZ header with exact bytes pattern
    content = b'Some content with MZ\x90\x00 executable header'
    filename = "malicious.pdf"

    warnings = validator.scan_for_security_threats(content, filename)

    assert len(warnings) > 0
    assert any("executable" in warn.lower() for warn in warnings)


@pytest.mark.asyncio
async def test_detect_embedded_script(validator):
    """Test detection of embedded JavaScript."""
    # Script tag with proper pattern
    content = b'PDF content <script type="text/javascript">alert("XSS")</script> more content'
    filename = "suspicious.pdf"

    warnings = validator.scan_for_security_threats(content, filename)

    assert len(warnings) > 0
    assert any("javascript" in warn.lower() or "script" in warn.lower() for warn in warnings)


@pytest.mark.asyncio
async def test_detect_directory_traversal(validator):
    """Test detection of directory traversal patterns."""
    # Directory traversal pattern
    content = b'File path: ../../../etc/passwd contains sensitive data'
    filename = "suspicious.txt"

    warnings = validator.scan_for_security_threats(content, filename)

    assert len(warnings) > 0
    assert any("traversal" in warn.lower() for warn in warnings)


@pytest.mark.asyncio
async def test_detect_command_execution_reference(validator):
    """Test detection of command execution references."""
    # Command execution pattern
    content = b'Run this command: cmd.exe /c dir to list files'
    filename = "suspicious.txt"

    warnings = validator.scan_for_security_threats(content, filename)

    assert len(warnings) > 0
    assert any("command" in warn.lower() or "execution" in warn.lower() for warn in warnings)


@pytest.mark.asyncio
async def test_detect_suspiciously_long_line(validator):
    """Test detection of buffer overflow attempts."""
    # Create very long line
    long_line = b'x' * 15000
    content = long_line + b'\nNormal line'
    filename = "suspicious.txt"

    warnings = validator.scan_for_security_threats(content, filename)

    assert len(warnings) > 0
    assert any("long line" in warn.lower() or "buffer" in warn.lower() for warn in warnings)


@pytest.mark.asyncio
async def test_clean_file_no_security_warnings(validator):
    """Test that clean file has no security warnings."""
    clean_content = b'This is a clean text file.\nNothing suspicious here.'
    filename = "clean.txt"

    warnings = validator.scan_for_security_threats(clean_content, filename)

    assert len(warnings) == 0


# ==================== Multi-tenant Configuration Tests (3 tests) ====================

@pytest.mark.asyncio
async def test_tenant_specific_allowed_extensions(validator):
    """Test tenant-specific extension configuration."""
    tenant_config = {
        "allowed_file_extensions": [".pdf"]  # Only PDF allowed
    }

    # PDF should pass
    assert validator.is_file_type_allowed(".pdf", tenant_config) is True

    # DOCX should fail
    assert validator.is_file_type_allowed(".docx", tenant_config) is False


@pytest.mark.asyncio
async def test_tenant_specific_file_size_limits(validator):
    """Test tenant-specific file size limits."""
    tenant_config = {
        "file_type_limits": {
            ".pdf": 5,  # 5MB limit for PDFs
            ".txt": 1   # 1MB limit for text
        }
    }

    pdf_limit = validator._get_max_size_for_type(".pdf", tenant_config)
    txt_limit = validator._get_max_size_for_type(".txt", tenant_config)

    assert pdf_limit == 5
    assert txt_limit == 1


@pytest.mark.asyncio
async def test_default_configuration_fallback(validator):
    """Test fallback to default configuration."""
    # No tenant config provided
    result = await validator.validate_file_content(
        b'%PDF-1.4\n',
        "test.pdf",
        "application/pdf",
        None
    )

    # Should use defaults
    assert result.is_valid is True


# ==================== Error Handling Tests (2 tests) ====================

@pytest.mark.asyncio
async def test_invalid_filename_validation(validator):
    """Test validation of invalid filename."""
    content = b'Valid content'
    invalid_filename = ""  # Empty filename

    result = await validator.validate_file_content(content, invalid_filename)

    assert result.is_valid is False
    assert any("filename" in str(err).lower() for err in result.validation_errors)


@pytest.mark.asyncio
async def test_filename_with_invalid_characters(validator):
    """Test rejection of filename with invalid characters."""
    content = b'Valid content'
    invalid_filename = "file<>:name.pdf"  # Invalid characters

    result = await validator.validate_file_content(content, invalid_filename)

    assert result.is_valid is False
    assert any("invalid" in str(err).lower() or "character" in str(err).lower()
               for err in result.validation_errors)


# ==================== Additional Tests (4 tests) ====================

@pytest.mark.asyncio
async def test_get_supported_file_types(validator):
    """Test retrieval of supported file types."""
    supported_types = validator.get_supported_file_types()

    assert len(supported_types) > 0
    assert any(ft.extension == '.pdf' for ft in supported_types)
    assert any(ft.extension == '.docx' for ft in supported_types)
    assert any(ft.extension == '.txt' for ft in supported_types)


@pytest.mark.asyncio
async def test_confidence_score_calculation(validator):
    """Test confidence score calculation."""
    # Perfect validation
    pdf_content = b'%PDF-1.4\n' + b'x' * 1000
    result = await validator.validate_file_content(
        pdf_content,
        "good.pdf",
        "application/pdf"
    )

    # Should have high confidence
    assert result.confidence_score > 0.8


@pytest.mark.asyncio
async def test_cross_validation_consistency(validator):
    """Test cross-validation ensures consistency."""
    # All properties match
    pdf_content = b'%PDF-1.4\n' + b'x' * 1000
    errors = validator.cross_validate_file_properties(
        "document.pdf",
        "application/pdf",
        ".pdf",
        "application/pdf"
    )

    # Should have no errors
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_health_check(validator):
    """Test health check endpoint."""
    health = await validator.check_health()

    assert health["service"] == "FileContentValidator"
    assert health["status"] == "healthy"
    assert "supported_types" in health
    assert "security_patterns" in health


# ==================== Integration-style Tests (1 test) ====================

@pytest.mark.asyncio
async def test_full_validation_workflow(validator):
    """Test complete validation workflow with all layers."""
    # Create valid PDF
    pdf_content = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n' + b'x' * 5000
    filename = "complete_test.pdf"
    content_type = "application/pdf"
    tenant_config = {
        "allowed_file_extensions": [".pdf", ".docx", ".txt"],
        "max_file_size_mb": 10
    }

    result = await validator.validate_file_content(
        pdf_content,
        filename,
        content_type,
        tenant_config
    )

    # Verify all validation layers passed
    assert result.is_valid is True
    assert result.filename == filename
    assert result.file_size == len(pdf_content)
    assert result.detected_extension == '.pdf'
    assert result.detected_mime_type == 'application/pdf'
    assert len(result.validation_errors) == 0
    assert result.confidence_score > 0.0

    # Verify validation details
    validation_details = result.validation_details
    assert "validation_layers" in validation_details
    assert "checks_performed" in validation_details

    # Verify all layers were executed
    checks = validation_details["checks_performed"]
    assert "basic_properties" in checks
    assert "extension_validation" in checks
    assert "mime_validation" in checks
    assert "signature_validation" in checks
    assert "cross_validation" in checks
    assert "security_scan" in checks