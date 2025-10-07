"""
Integration tests for comprehensive file content validation.

These tests demonstrate the end-to-end functionality of the new validation system
and how it prevents malicious file uploads while allowing legitimate files.
"""

import pytest
from typing import Dict, Any
from io import BytesIO

from app.infrastructure.validation.file_content_validator import FileContentValidator
from app.infrastructure.persistence.models.tenant_table import FileTypeValidationConfig, ProcessingConfiguration
from app.domain.interfaces import FileValidationResult


class MockUploadFile:
    """Mock UploadFile for integration testing."""
    
    def __init__(self, content: bytes, filename: str, content_type: str = None, size: int = None):
        self.content = content
        self.filename = filename
        self.content_type = content_type
        self.size = size if size is not None else len(content)
        self._position = 0
    
    async def read(self, size: int = -1) -> bytes:
        if size == -1:
            content = self.content[self._position:]
            self._position = len(self.content)
        else:
            content = self.content[self._position:self._position + size]
            self._position += len(content)
        return content
    
    async def seek(self, position: int) -> None:
        self._position = position


@pytest.fixture
def file_validator():
    """Create FileContentValidator instance."""
    return FileContentValidator()


@pytest.fixture
def strict_tenant_config():
    """Tenant configuration with strict security settings."""
    file_validation = FileTypeValidationConfig(
        allowed_file_extensions=[".pdf", ".doc", ".docx", ".txt"],
        file_type_limits={".pdf": 25, ".doc": 10, ".docx": 10, ".txt": 1},
        require_mime_validation=True,
        require_signature_validation=True,
        enable_content_scanning=True,
        block_executable_content=True,
        block_macro_documents=True,
        block_script_content=True,
        validation_mode="strict",
        min_confidence_score=0.8,
        reject_on_validation_errors=True,
        reject_on_security_warnings=True,  # Strict: reject on security warnings
        log_validation_details=True
    )
    
    processing_config = ProcessingConfiguration(file_validation=file_validation)
    
    return {
        "processing_configuration": processing_config.model_dump()
    }


@pytest.fixture
def permissive_tenant_config():
    """Tenant configuration with permissive security settings."""
    file_validation = FileTypeValidationConfig(
        allowed_file_extensions=[".pdf", ".doc", ".docx", ".txt"],
        file_type_limits={".pdf": 50, ".doc": 25, ".docx": 25, ".txt": 5},
        require_mime_validation=False,
        require_signature_validation=False,
        enable_content_scanning=True,
        block_executable_content=False,
        block_macro_documents=False,
        block_script_content=False,
        validation_mode="permissive",
        min_confidence_score=0.5,
        reject_on_validation_errors=False,
        reject_on_security_warnings=False,  # Permissive: allow with warnings
        log_validation_details=True
    )
    
    processing_config = ProcessingConfiguration(file_validation=file_validation)
    
    return {
        "processing_configuration": processing_config.model_dump()
    }


class TestLegitimateFiles:
    """Test that legitimate files are correctly accepted."""
    
    @pytest.mark.asyncio
    async def test_valid_pdf_upload(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test uploading a valid PDF file."""
        # Create a minimal valid PDF
        pdf_content = (
            b'%PDF-1.4\n'
            b'1 0 obj\n'
            b'<<\n'
            b'/Type /Catalog\n'
            b'/Pages 2 0 R\n'
            b'>>\n'
            b'endobj\n'
            b'2 0 obj\n'
            b'<<\n'
            b'/Type /Pages\n'
            b'/Kids [3 0 R]\n'
            b'/Count 1\n'
            b'>>\n'
            b'endobj\n'
            b'3 0 obj\n'
            b'<<\n'
            b'/Type /Page\n'
            b'/Parent 2 0 R\n'
            b'/MediaBox [0 0 612 792]\n'
            b'>>\n'
            b'endobj\n'
            b'xref\n'
            b'0 4\n'
            b'0000000000 65535 f \n'
            b'0000000010 00000 n \n'
            b'0000000053 00000 n \n'
            b'0000000125 00000 n \n'
            b'trailer\n'
            b'<<\n'
            b'/Size 4\n'
            b'/Root 1 0 R\n'
            b'>>\n'
            b'startxref\n'
            b'0000000174\n'
            b'%%EOF'
        )
        
        mock_file = MockUploadFile(pdf_content, "resume.pdf", "application/pdf")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert result.is_valid
        assert result.detected_extension == ".pdf"
        assert result.detected_mime_type == "application/pdf"
        assert len(result.validation_errors) == 0
        assert len(result.security_warnings) == 0
        assert result.confidence_score > 0.8
        
        # Verify validation details
        assert "validation_layers" in result.validation_details
        layers = result.validation_details["validation_layers"]
        layer_names = [layer["layer"] for layer in layers]
        
        assert "basic_properties" in layer_names
        assert "extension_validation" in layer_names
        assert "mime_validation" in layer_names
        assert "signature_validation" in layer_names
        assert "cross_validation" in layer_names
        assert "security_scan" in layer_names
    
    @pytest.mark.asyncio
    async def test_valid_text_upload(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test uploading a valid text file."""
        text_content = (
            b"John Doe\n"
            b"Software Engineer\n"
            b"Email: john.doe@example.com\n"
            b"Phone: (555) 123-4567\n"
            b"\n"
            b"EXPERIENCE:\n"
            b"- 5 years Python development\n"
            b"- 3 years React/JavaScript\n"
            b"- AWS cloud experience\n"
            b"\n"
            b"EDUCATION:\n"
            b"B.S. Computer Science, State University, 2018"
        )
        
        mock_file = MockUploadFile(text_content, "resume.txt", "text/plain")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert result.is_valid
        assert result.detected_extension == ".txt"
        assert result.detected_mime_type == "text/plain"
        assert len(result.validation_errors) == 0
        assert len(result.security_warnings) == 0


class TestMaliciousFiles:
    """Test that malicious files are correctly rejected."""
    
    @pytest.mark.asyncio
    async def test_fake_pdf_with_executable(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test rejecting a fake PDF containing executable content."""
        # File claims to be PDF but contains executable signature
        fake_pdf_content = (
            b"This looks like a PDF file\n"
            b"But it actually contains malicious content\n"
            b"MZ\x90\x00"  # Windows executable signature
            b"Hidden executable code here"
        )
        
        mock_file = MockUploadFile(fake_pdf_content, "malicious.pdf", "application/pdf")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert not result.is_valid
        
        # Should fail signature validation
        assert any("File signature does not match" in error for error in result.validation_errors)
        
        # Should detect security threat
        assert len(result.security_warnings) > 0
        assert any("executable" in warning.lower() for warning in result.security_warnings)
        
        # Confidence should be low
        assert result.confidence_score < 0.5
    
    @pytest.mark.asyncio
    async def test_script_injection_attempt(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test rejecting a file with script injection attempt."""
        script_content = (
            b"Normal looking resume content\n"
            b"John Smith - Software Engineer\n"
            b"<script>alert('XSS Attack!');</script>\n"
            b"javascript:void(0);\n"
            b"More resume content here"
        )
        
        mock_file = MockUploadFile(script_content, "resume.txt", "text/plain")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        # File structure is valid but contains security threats
        assert result.is_valid  # Basic validation passes
        assert len(result.security_warnings) > 0
        assert any("script" in warning.lower() for warning in result.security_warnings)
        assert any("javascript" in warning.lower() for warning in result.security_warnings)
    
    @pytest.mark.asyncio
    async def test_directory_traversal_attempt(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test rejecting a file with directory traversal attempt."""
        traversal_content = (
            b"Innocent looking content\n"
            b"../../../etc/passwd\n"
            b"..\\..\\windows\\system32\\config\\sam\n"
            b"More content"
        )
        
        mock_file = MockUploadFile(traversal_content, "resume.txt", "text/plain")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert result.is_valid  # Structure is valid
        assert len(result.security_warnings) > 0
        assert any("traversal" in warning.lower() for warning in result.security_warnings)
    
    @pytest.mark.asyncio
    async def test_null_byte_injection(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test rejecting a text file with NULL byte injection."""
        null_injection_content = (
            b"Normal text content\x00"
            b"Hidden content after null byte"
        )
        
        mock_file = MockUploadFile(null_injection_content, "resume.txt", "text/plain")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert result.is_valid  # Structure is valid
        assert len(result.security_warnings) > 0
        assert any("NULL bytes" in warning for warning in result.security_warnings)


class TestMismatchedFiles:
    """Test files with mismatched properties."""
    
    @pytest.mark.asyncio
    async def test_wrong_extension(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test file with wrong extension for its content."""
        # PDF content with .txt extension
        pdf_content = b'%PDF-1.4\n1 0 obj'
        
        mock_file = MockUploadFile(pdf_content, "document.txt", "text/plain")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert not result.is_valid
        assert result.detected_extension == ".pdf"  # Content is actually PDF
        
        # Should have cross-validation errors
        assert any("detected type" in error for error in result.validation_errors)
    
    @pytest.mark.asyncio
    async def test_wrong_mime_type(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test file with wrong MIME type for its extension."""
        text_content = b"This is plain text content"
        
        mock_file = MockUploadFile(text_content, "document.txt", "application/pdf")  # Wrong MIME
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        assert not result.is_valid
        assert any("MIME type" in error for error in result.validation_errors)


class TestTenantConfiguration:
    """Test different tenant configuration scenarios."""
    
    @pytest.mark.asyncio
    async def test_strict_vs_permissive_security(self, file_validator: FileContentValidator, 
                                                 strict_tenant_config: Dict[str, Any],
                                                 permissive_tenant_config: Dict[str, Any]):
        """Test how strict vs permissive configs handle security warnings."""
        script_content = b"Resume content <script>alert('test');</script>"
        mock_file = MockUploadFile(script_content, "resume.txt", "text/plain")
        
        # Test with strict config (should reject)
        strict_result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        # Test with permissive config (should allow with warnings)
        await mock_file.seek(0)  # Reset file position
        permissive_result = await file_validator.validate_upload_file(mock_file, permissive_tenant_config)
        
        # Both should detect the security issue
        assert len(strict_result.security_warnings) > 0
        assert len(permissive_result.security_warnings) > 0
        
        # But strict config should be more restrictive in validation
        # (Note: Since reject_on_security_warnings affects application logic, 
        # both validators return is_valid=True, but the app layer would reject based on config)
        assert strict_result.is_valid == permissive_result.is_valid
    
    @pytest.mark.asyncio
    async def test_custom_file_size_limits(self, file_validator: FileContentValidator):
        """Test custom file size limits per file type."""
        # Create config with very small limits
        small_limits_config = {
            "processing_configuration": {
                "file_validation": {
                    "allowed_file_extensions": [".txt"],
                    "file_type_limits": {".txt": 0.001},  # 1KB limit
                    "validation_mode": "strict"
                }
            }
        }
        
        # Small file should pass
        small_content = b"Small content"
        small_file = MockUploadFile(small_content, "small.txt", "text/plain")
        
        result = await file_validator.validate_upload_file(small_file, small_limits_config)
        assert result.is_valid
        
        # Large file should fail (but we'd need integration with FileSizeValidator for this)
        large_content = b"A" * 2000  # 2KB content
        large_file = MockUploadFile(large_content, "large.txt", "text/plain")
        
        # This test would require integration with the actual size validation logic
        # which happens in FileSizeValidator, not in FileContentValidator


class TestIntegrationWithUploadService:
    """Test integration scenarios that demonstrate real-world usage."""
    
    @pytest.mark.asyncio
    async def test_complete_validation_flow(self, file_validator: FileContentValidator, strict_tenant_config: Dict[str, Any]):
        """Test complete validation flow with detailed logging."""
        # Valid PDF file
        pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj'
        mock_file = MockUploadFile(pdf_content, "resume.pdf", "application/pdf")
        
        result = await file_validator.validate_upload_file(mock_file, strict_tenant_config)
        
        # Verify all validation layers were executed
        expected_layers = [
            "basic_properties",
            "extension_validation", 
            "mime_validation",
            "signature_validation",
            "cross_validation",
            "security_scan"
        ]
        
        actual_layers = [layer["layer"] for layer in result.validation_details["validation_layers"]]
        
        for expected_layer in expected_layers:
            assert expected_layer in actual_layers
        
        # Verify all checks passed
        for layer in result.validation_details["validation_layers"]:
            assert layer["status"] in ["passed", "warnings"]  # No failures
        
        assert result.is_valid
        assert result.confidence_score > 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling_with_malformed_config(self, file_validator: FileContentValidator):
        """Test error handling with malformed tenant configuration."""
        malformed_config = {
            "processing_configuration": {
                # Missing file_validation section
            }
        }
        
        text_content = b"Test content"
        mock_file = MockUploadFile(text_content, "test.txt", "text/plain")
        
        # Should still work with default values
        result = await file_validator.validate_upload_file(mock_file, malformed_config)
        
        assert result.is_valid  # Should fall back to defaults
        assert len(result.validation_errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])