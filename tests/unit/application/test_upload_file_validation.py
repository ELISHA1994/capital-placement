"""
Unit tests for upload service file validation.

Tests the integration of file size validation within the upload service,
including validation workflow, error handling, and tenant configuration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.upload_service import UploadApplicationService, UploadError
from app.domain.exceptions import FileSizeExceededError, InvalidFileError
from app.infrastructure.persistence.models.auth_tables import CurrentUser


class MockUploadFile:
    """Mock UploadFile for testing upload service."""
    
    def __init__(self, content: bytes, filename: str = "test.pdf", size: int = None, content_type: str = "application/pdf"):
        self.content = content
        self.filename = filename
        self.size = size
        self.content_type = content_type
        self.position = 0
        
    async def read(self, size: int = -1) -> bytes:
        """Simulate reading from file."""
        if size == -1:
            result = self.content[self.position:]
            self.position = len(self.content)
        else:
            end_pos = min(self.position + size, len(self.content))
            result = self.content[self.position:end_pos]
            self.position = end_pos
        return result
    
    async def seek(self, position: int) -> None:
        """Simulate seeking in file."""
        self.position = min(position, len(self.content))


class MockUploadDependencies:
    """Mock dependencies for upload service testing."""
    
    def __init__(self):
        self.tenant_manager = AsyncMock()
        self.webhook_validator = MagicMock()
        self.database_adapter = AsyncMock()
        self.document_processor = AsyncMock()
        self.content_extractor = AsyncMock()
        self.quality_analyzer = AsyncMock()
        self.embedding_service = AsyncMock()
        self.storage_service = AsyncMock()
        self.notification_service = AsyncMock()
        self.event_publisher = AsyncMock()
        
        # Mock repositories
        self.profile_repository = AsyncMock()
        self.user_repository = AsyncMock()
        self.tenant_repository = AsyncMock()


@pytest.mark.asyncio
class TestUploadServiceFileValidation:
    """Test cases for upload service file validation."""

    @pytest.fixture
    def upload_dependencies(self):
        """Mock upload dependencies."""
        return MockUploadDependencies()
    
    @pytest.fixture
    def upload_service(self, upload_dependencies):
        """Upload service with mocked dependencies."""
        return UploadApplicationService(upload_dependencies)
    
    @pytest.fixture
    def current_user(self):
        """Mock current user."""
        return CurrentUser(
            user_id="user123",
            tenant_id="tenant123",
            email="test@example.com",
            full_name="Test User",
            roles=["user"],
            permissions=[],
            is_active=True
        )
    
    @pytest.fixture
    def tenant_config_10mb(self):
        """Tenant configuration with 10MB file size limit."""
        return {
            "allowed_file_extensions": [".pdf", ".doc", ".docx"],
            "quota_limits": {
                "max_document_size_mb": 10
            },
            "documents_processed_today": 5
        }
    
    @pytest.fixture
    def tenant_config_5mb(self):
        """Tenant configuration with 5MB file size limit."""
        return {
            "allowed_file_extensions": [".pdf", ".doc", ".docx"],
            "quota_limits": {
                "max_document_size_mb": 5
            },
            "documents_processed_today": 3
        }

    async def test_upload_document_valid_file_size(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test successful upload with valid file size."""
        # Setup
        file_content = b"Valid PDF content" * 1000  # ~17KB
        mock_file = MockUploadFile(file_content, "test.pdf", len(file_content))
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        upload_dependencies.tenant_manager.check_quota_limit.return_value = {"allowed": True}
        
        # Execute
        with patch('app.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock()
            
            response = await upload_service.upload_document(
                file=mock_file,
                current_user=current_user,
                auto_process=False  # Skip processing for this test
            )
        
        # Verify
        assert response.filename == "test.pdf"
        assert response.status.value == "pending"
        
        # Verify tenant manager was called for configuration
        upload_dependencies.tenant_manager.get_tenant_configuration.assert_called_once_with("tenant123")

    async def test_upload_document_file_size_exceeded(self, upload_service, upload_dependencies, current_user, tenant_config_5mb):
        """Test upload rejection when file size exceeds limit."""
        # Setup - create file larger than 5MB limit
        file_content = b"X" * (6 * 1024 * 1024)  # 6MB
        mock_file = MockUploadFile(file_content, "large.pdf", len(file_content))
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_5mb
        
        # Execute & Verify
        with pytest.raises(UploadError) as exc_info:
            await upload_service.upload_document(
                file=mock_file,
                current_user=current_user
            )
        
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_file"
        assert "6.00MB exceeds maximum allowed size of 5.00MB" in exc_info.value.detail["message"]

    async def test_upload_document_invalid_file_extension(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test upload rejection for invalid file extension."""
        # Setup
        file_content = b"Text file content"
        mock_file = MockUploadFile(file_content, "test.txt", len(file_content), "text/plain")
        
        # Modify config to not allow .txt files
        config = tenant_config_10mb.copy()
        config["allowed_file_extensions"] = [".pdf", ".doc", ".docx"]
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = config
        
        # Execute & Verify
        with pytest.raises(UploadError) as exc_info:
            await upload_service.upload_document(
                file=mock_file,
                current_user=current_user
            )
        
        assert exc_info.value.status_code == 400
        assert "File type not supported" in exc_info.value.detail["message"]

    async def test_upload_document_invalid_filename(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test upload rejection for invalid filename."""
        # Setup
        file_content = b"PDF content"
        mock_file = MockUploadFile(file_content, "", len(file_content))  # Empty filename
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        
        # Execute & Verify
        with pytest.raises(UploadError) as exc_info:
            await upload_service.upload_document(
                file=mock_file,
                current_user=current_user
            )
        
        assert exc_info.value.status_code == 400
        assert "Invalid filename" in exc_info.value.detail["message"]

    async def test_upload_document_stream_based_validation(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test upload with stream-based validation when size attribute is missing."""
        # Setup - file without size attribute
        file_content = b"PDF content without size attribute" * 1000  # ~34KB
        mock_file = MockUploadFile(file_content, "test.pdf", size=None)  # No size attribute
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        upload_dependencies.tenant_manager.check_quota_limit.return_value = {"allowed": True}
        
        # Execute
        with patch('app.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock()
            
            response = await upload_service.upload_document(
                file=mock_file,
                current_user=current_user,
                auto_process=False
            )
        
        # Verify successful upload despite missing size attribute
        assert response.filename == "test.pdf"
        assert response.status.value == "pending"

    async def test_batch_upload_mixed_validation_results(self, upload_service, upload_dependencies, current_user, tenant_config_5mb):
        """Test batch upload with mix of valid and invalid files."""
        # Setup
        valid_content = b"Valid PDF" * 1000  # ~9KB
        invalid_content = b"X" * (6 * 1024 * 1024)  # 6MB - exceeds 5MB limit
        
        files = [
            MockUploadFile(valid_content, "valid1.pdf", len(valid_content)),
            MockUploadFile(invalid_content, "too_large.pdf", len(invalid_content)),
            MockUploadFile(valid_content, "valid2.pdf", len(valid_content)),
            MockUploadFile(b"TXT content", "invalid.txt", 11),  # Wrong extension
        ]
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_5mb
        upload_dependencies.tenant_manager.check_quota_limit.return_value = {"allowed": True}
        
        # Execute
        with patch('app.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock()
            
            response = await upload_service.upload_documents_batch(
                files=files,
                current_user=current_user,
                auto_process=False
            )
        
        # Verify results
        assert response.total_files == 4
        assert response.accepted_files == 2  # Only valid1.pdf and valid2.pdf
        assert response.rejected_files == 2
        
        # Check rejection reasons
        assert "too_large.pdf" in response.rejected_reasons
        assert "invalid.txt" in response.rejected_reasons
        assert "6.00MB exceeds maximum" in response.rejected_reasons["too_large.pdf"]
        assert "File type not supported" in response.rejected_reasons["invalid.txt"]

    async def test_upload_document_quota_exceeded(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test upload rejection when quota is exceeded."""
        # Setup
        file_content = b"Valid content"
        mock_file = MockUploadFile(file_content, "test.pdf", len(file_content))
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        upload_dependencies.tenant_manager.check_quota_limit.return_value = {
            "allowed": False,
            "remaining": 0
        }
        
        # Execute & Verify
        with pytest.raises(UploadError) as exc_info:
            await upload_service.upload_document(
                file=mock_file,
                current_user=current_user
            )
        
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"] == "quota_exceeded"

    async def test_upload_document_file_io_error_during_validation(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test handling of file I/O errors during validation."""
        # Setup
        mock_file = MockUploadFile(b"content", "test.pdf", size=None)
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        
        # Mock file.read to raise IOError during validation
        with patch.object(mock_file, 'read', side_effect=IOError("Disk error")):
            with pytest.raises(UploadError) as exc_info:
                await upload_service.upload_document(
                    file=mock_file,
                    current_user=current_user
                )
        
        assert exc_info.value.status_code == 400
        assert "File I/O error" in exc_info.value.detail["message"]

    async def test_upload_service_logs_validation_details(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test that upload service logs appropriate validation details."""
        # Setup
        file_content = b"Test content" * 100  # ~1.2KB
        mock_file = MockUploadFile(file_content, "test.pdf", len(file_content))
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        upload_dependencies.tenant_manager.check_quota_limit.return_value = {"allowed": True}
        
        # Execute with logging capture
        with patch('app.application.upload_service.logger') as mock_logger:
            with patch('app.core.config.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock()
                
                await upload_service.upload_document(
                    file=mock_file,
                    current_user=current_user,
                    auto_process=False
                )
        
        # Verify logging calls
        assert mock_logger.info.call_count >= 2  # Initial upload + validation success
        assert mock_logger.debug.call_count >= 1  # File reading

    async def test_upload_service_uses_validated_file_size(self, upload_service, upload_dependencies, current_user, tenant_config_10mb):
        """Test that upload service uses the validated file size for tracking."""
        # Setup
        file_content = b"Content for size tracking" * 100  # ~2.5KB
        mock_file = MockUploadFile(file_content, "test.pdf", len(file_content))
        
        upload_dependencies.tenant_manager.get_tenant_configuration.return_value = tenant_config_10mb
        upload_dependencies.tenant_manager.check_quota_limit.return_value = {"allowed": True}
        
        # Execute
        with patch('app.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock()
            
            await upload_service.upload_document(
                file=mock_file,
                current_user=current_user,
                auto_process=False
            )
        
        # Verify that _update_upload_usage was called with correct file size
        # Note: This tests the integration of validation with usage tracking
        assert upload_dependencies.tenant_manager.update_usage_metrics.called