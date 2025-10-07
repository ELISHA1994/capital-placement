"""Comprehensive unit tests for LocalFileStorageAdapter."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.infrastructure.adapters.storage_adapter import LocalFileStorageAdapter


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage directory for tests."""
    storage_path = tmp_path / "test_storage"
    storage_path.mkdir(exist_ok=True)
    return storage_path


@pytest.fixture
def storage_adapter(temp_storage_path):
    """Create LocalFileStorageAdapter instance with temp directory."""
    return LocalFileStorageAdapter(base_path=str(temp_storage_path))


@pytest.fixture
def sample_tenant_id():
    """Generate a sample tenant UUID."""
    return str(uuid4())


@pytest.fixture
def sample_upload_id():
    """Generate a sample upload UUID."""
    return str(uuid4())


@pytest.fixture
def sample_file_content():
    """Generate sample file content."""
    return b"This is test file content with some data."


@pytest.fixture
def sample_pdf_content():
    """Generate sample PDF-like content with magic bytes."""
    # PDF magic bytes: %PDF-1.
    return b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\nTest PDF content"


class TestLocalFileStorageAdapterInitialization:
    """Test initialization and setup of LocalFileStorageAdapter."""

    def test_initialization_creates_base_directory(self, tmp_path):
        """Test that initialization creates base directory if it doesn't exist."""
        base_path = tmp_path / "new_storage"
        assert not base_path.exists()

        adapter = LocalFileStorageAdapter(base_path=str(base_path))

        assert base_path.exists()
        assert base_path.is_dir()
        assert adapter.base_path == base_path.resolve()

    def test_initialization_with_existing_directory(self, temp_storage_path):
        """Test initialization with existing directory."""
        adapter = LocalFileStorageAdapter(base_path=str(temp_storage_path))

        assert adapter.base_path.exists()
        assert adapter.base_path == temp_storage_path.resolve()

    def test_initialization_with_default_path(self, monkeypatch, tmp_path):
        """Test initialization uses default path when not specified."""
        # Change to temp directory to avoid creating files in project root
        monkeypatch.chdir(tmp_path)

        adapter = LocalFileStorageAdapter()

        assert adapter.base_path.exists()
        assert adapter.base_path.name == "uploads"
        assert "storage" in str(adapter.base_path)


class TestSaveFile:
    """Test save_file functionality."""

    @pytest.mark.asyncio
    async def test_save_file_success(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test successful file save operation."""
        filename = "test_document.pdf"

        storage_path = await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename=filename,
            content=sample_file_content
        )

        # Verify return value format
        expected_path = f"{sample_tenant_id}/{sample_upload_id}/original_file.pdf"
        assert storage_path == expected_path

        # Verify file exists on filesystem
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.pdf"
        assert file_path.exists()
        assert file_path.read_bytes() == sample_file_content

    @pytest.mark.asyncio
    async def test_save_file_creates_tenant_directory(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that save_file creates tenant directory structure."""
        tenant_dir = storage_adapter.base_path / sample_tenant_id
        assert not tenant_dir.exists()

        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        assert tenant_dir.exists()
        assert tenant_dir.is_dir()

    @pytest.mark.asyncio
    async def test_save_file_creates_upload_directory(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that save_file creates upload directory structure."""
        upload_dir = storage_adapter.base_path / sample_tenant_id / sample_upload_id
        assert not upload_dir.exists()

        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        assert upload_dir.exists()
        assert upload_dir.is_dir()

    @pytest.mark.asyncio
    async def test_save_file_preserves_extension(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that file extension is preserved."""
        extensions = [".pdf", ".docx", ".txt", ".csv", ".json"]

        for ext in extensions:
            upload_id = str(uuid4())
            filename = f"test_file{ext}"

            storage_path = await storage_adapter.save_file(
                tenant_id=sample_tenant_id,
                upload_id=upload_id,
                filename=filename,
                content=sample_file_content
            )

            # Verify extension is preserved
            assert storage_path.endswith(f"original_file{ext}")

            # Verify file exists with correct extension
            file_path = storage_adapter.base_path / sample_tenant_id / upload_id / f"original_file{ext}"
            assert file_path.exists()

    @pytest.mark.asyncio
    async def test_save_file_sanitizes_filename(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that dangerous filenames are sanitized."""
        dangerous_filename = "../../../etc/passwd.txt"

        storage_path = await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename=dangerous_filename,
            content=sample_file_content
        )

        # Verify path traversal was prevented
        assert ".." not in storage_path
        assert "/etc/" not in storage_path
        assert storage_path == f"{sample_tenant_id}/{sample_upload_id}/original_file.txt"

        # Verify file is in correct location
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.txt"
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_save_file_handles_special_characters_in_filename(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that special characters in filename are handled."""
        special_filenames = [
            "file with spaces.pdf",
            "file@#$%^&*.txt",
            "файл.pdf",  # Cyrillic
            "文件.txt",  # Chinese
        ]

        for filename in special_filenames:
            upload_id = str(uuid4())

            storage_path = await storage_adapter.save_file(
                tenant_id=sample_tenant_id,
                upload_id=upload_id,
                filename=filename,
                content=sample_file_content
            )

            # Should save successfully (filename sanitized)
            assert storage_path is not None
            assert sample_tenant_id in storage_path
            assert upload_id in storage_path

    @pytest.mark.asyncio
    async def test_save_file_overwrites_existing(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test that saving to same location overwrites existing file."""
        original_content = b"Original content"
        new_content = b"New content that is different"

        # Save first file
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=original_content
        )

        # Save second file (should overwrite)
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=new_content
        )

        # Verify new content
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.txt"
        assert file_path.read_bytes() == new_content

    @pytest.mark.asyncio
    async def test_save_file_with_empty_content(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test saving file with empty content."""
        empty_content = b""

        storage_path = await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="empty.txt",
            content=empty_content
        )

        # Should save successfully
        assert storage_path is not None

        # Verify empty file exists
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.txt"
        assert file_path.exists()
        assert file_path.read_bytes() == empty_content

    @pytest.mark.asyncio
    async def test_save_file_with_large_content(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test saving large file (10MB)."""
        large_content = b"X" * (10 * 1024 * 1024)  # 10MB

        storage_path = await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="large_file.bin",
            content=large_content
        )

        # Verify file saved successfully
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.bin"
        assert file_path.exists()
        assert file_path.stat().st_size == len(large_content)


class TestSaveFileValidation:
    """Test validation in save_file."""

    @pytest.mark.asyncio
    async def test_save_file_invalid_tenant_id(
        self, storage_adapter, sample_upload_id, sample_file_content
    ):
        """Test that invalid tenant_id raises ValueError."""
        invalid_tenant_ids = [
            "not-a-uuid",
            "12345",
            "",
            "../../etc/passwd",
            "../tenant",
        ]

        for invalid_id in invalid_tenant_ids:
            with pytest.raises(ValueError, match="Invalid tenant_id"):
                await storage_adapter.save_file(
                    tenant_id=invalid_id,
                    upload_id=sample_upload_id,
                    filename="test.txt",
                    content=sample_file_content
                )

    @pytest.mark.asyncio
    async def test_save_file_invalid_upload_id(
        self, storage_adapter, sample_tenant_id, sample_file_content
    ):
        """Test that invalid upload_id raises ValueError."""
        invalid_upload_ids = [
            "not-a-uuid",
            "67890",
            "",
            "../../uploads",
        ]

        for invalid_id in invalid_upload_ids:
            with pytest.raises(ValueError, match="Invalid upload_id"):
                await storage_adapter.save_file(
                    tenant_id=sample_tenant_id,
                    upload_id=invalid_id,
                    filename="test.txt",
                    content=sample_file_content
                )

    @pytest.mark.asyncio
    async def test_save_file_empty_filename(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that empty filename raises ValueError."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            await storage_adapter.save_file(
                tenant_id=sample_tenant_id,
                upload_id=sample_upload_id,
                filename="",
                content=sample_file_content
            )

    @pytest.mark.asyncio
    async def test_save_file_whitespace_filename(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that whitespace-only filename raises ValueError."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            await storage_adapter.save_file(
                tenant_id=sample_tenant_id,
                upload_id=sample_upload_id,
                filename="   ",
                content=sample_file_content
            )


class TestRetrieveFile:
    """Test retrieve_file functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_file_success(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test successful file retrieval."""
        # First save a file
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        # Then retrieve it
        retrieved_content = await storage_adapter.retrieve_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        assert retrieved_content == sample_file_content

    @pytest.mark.asyncio
    async def test_retrieve_file_not_found_no_directory(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test retrieving file when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="No file found"):
            await storage_adapter.retrieve_file(
                tenant_id=sample_tenant_id,
                upload_id=sample_upload_id
            )

    @pytest.mark.asyncio
    async def test_retrieve_file_not_found_empty_directory(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test retrieving file from empty directory."""
        # Create directory but no file
        upload_dir = storage_adapter.base_path / sample_tenant_id / sample_upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="No file found"):
            await storage_adapter.retrieve_file(
                tenant_id=sample_tenant_id,
                upload_id=sample_upload_id
            )

    @pytest.mark.asyncio
    async def test_retrieve_file_preserves_binary_content(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_pdf_content
    ):
        """Test that binary content (like PDF) is preserved during retrieval."""
        # Save PDF-like binary content
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.pdf",
            content=sample_pdf_content
        )

        # Retrieve and verify
        retrieved_content = await storage_adapter.retrieve_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        assert retrieved_content == sample_pdf_content
        assert retrieved_content.startswith(b"%PDF")

    @pytest.mark.asyncio
    async def test_retrieve_file_with_different_extensions(
        self, storage_adapter, sample_tenant_id, sample_file_content
    ):
        """Test retrieving files with different extensions."""
        extensions = [".pdf", ".docx", ".txt", ".csv"]

        for ext in extensions:
            upload_id = str(uuid4())

            # Save file
            await storage_adapter.save_file(
                tenant_id=sample_tenant_id,
                upload_id=upload_id,
                filename=f"test{ext}",
                content=sample_file_content
            )

            # Retrieve file
            retrieved_content = await storage_adapter.retrieve_file(
                tenant_id=sample_tenant_id,
                upload_id=upload_id
            )

            assert retrieved_content == sample_file_content


class TestRetrieveFileValidation:
    """Test validation in retrieve_file."""

    @pytest.mark.asyncio
    async def test_retrieve_file_invalid_tenant_id(self, storage_adapter, sample_upload_id):
        """Test that invalid tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tenant_id"):
            await storage_adapter.retrieve_file(
                tenant_id="not-a-uuid",
                upload_id=sample_upload_id
            )

    @pytest.mark.asyncio
    async def test_retrieve_file_invalid_upload_id(self, storage_adapter, sample_tenant_id):
        """Test that invalid upload_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid upload_id"):
            await storage_adapter.retrieve_file(
                tenant_id=sample_tenant_id,
                upload_id="not-a-uuid"
            )


class TestDeleteFile:
    """Test delete_file functionality."""

    @pytest.mark.asyncio
    async def test_delete_file_success(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test successful file deletion."""
        # First save a file
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        # Verify file exists
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.txt"
        assert file_path.exists()

        # Delete file
        result = await storage_adapter.delete_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        assert result is True
        assert not file_path.exists()
        assert not file_path.parent.exists()  # Directory should also be removed

    @pytest.mark.asyncio
    async def test_delete_file_not_found(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test deleting non-existent file returns False."""
        result = await storage_adapter.delete_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_file_removes_directory(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that delete_file removes the upload directory."""
        # Save file
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        upload_dir = storage_adapter.base_path / sample_tenant_id / sample_upload_id
        assert upload_dir.exists()

        # Delete file
        await storage_adapter.delete_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        # Directory should be removed
        assert not upload_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_file_tenant_isolation(
        self, storage_adapter, sample_file_content
    ):
        """Test that deleting one tenant's file doesn't affect another tenant."""
        tenant_id_1 = str(uuid4())
        tenant_id_2 = str(uuid4())
        upload_id = str(uuid4())

        # Save files for both tenants with same upload_id
        await storage_adapter.save_file(
            tenant_id=tenant_id_1,
            upload_id=upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        await storage_adapter.save_file(
            tenant_id=tenant_id_2,
            upload_id=upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        # Delete first tenant's file
        await storage_adapter.delete_file(
            tenant_id=tenant_id_1,
            upload_id=upload_id
        )

        # Verify first tenant's file is deleted
        assert not await storage_adapter.exists(tenant_id_1, upload_id)

        # Verify second tenant's file still exists
        assert await storage_adapter.exists(tenant_id_2, upload_id)


class TestDeleteFileValidation:
    """Test validation in delete_file."""

    @pytest.mark.asyncio
    async def test_delete_file_invalid_tenant_id(self, storage_adapter, sample_upload_id):
        """Test that invalid tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tenant_id"):
            await storage_adapter.delete_file(
                tenant_id="not-a-uuid",
                upload_id=sample_upload_id
            )

    @pytest.mark.asyncio
    async def test_delete_file_invalid_upload_id(self, storage_adapter, sample_tenant_id):
        """Test that invalid upload_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid upload_id"):
            await storage_adapter.delete_file(
                tenant_id=sample_tenant_id,
                upload_id="not-a-uuid"
            )


class TestExists:
    """Test exists functionality."""

    @pytest.mark.asyncio
    async def test_exists_returns_true_when_file_exists(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that exists returns True when file exists."""
        # Save file
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        # Check existence
        exists = await storage_adapter.exists(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        assert exists is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_file_not_exists(
        self, storage_adapter, sample_tenant_id, sample_upload_id
    ):
        """Test that exists returns False when file doesn't exist."""
        exists = await storage_adapter.exists(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id
        )

        assert exists is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_after_deletion(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that exists returns False after file deletion."""
        # Save file
        await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        # Verify exists
        assert await storage_adapter.exists(sample_tenant_id, sample_upload_id) is True

        # Delete file
        await storage_adapter.delete_file(sample_tenant_id, sample_upload_id)

        # Verify doesn't exist
        assert await storage_adapter.exists(sample_tenant_id, sample_upload_id) is False


class TestExistsValidation:
    """Test validation in exists."""

    @pytest.mark.asyncio
    async def test_exists_invalid_tenant_id(self, storage_adapter, sample_upload_id):
        """Test that invalid tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tenant_id"):
            await storage_adapter.exists(
                tenant_id="not-a-uuid",
                upload_id=sample_upload_id
            )

    @pytest.mark.asyncio
    async def test_exists_invalid_upload_id(self, storage_adapter, sample_tenant_id):
        """Test that invalid upload_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid upload_id"):
            await storage_adapter.exists(
                tenant_id=sample_tenant_id,
                upload_id="not-a-uuid"
            )


class TestMultiTenantIsolation:
    """Test multi-tenant data isolation."""

    @pytest.mark.asyncio
    async def test_tenant_isolation_separate_storage(
        self, storage_adapter, sample_file_content
    ):
        """Test that different tenants have isolated storage."""
        tenant_id_1 = str(uuid4())
        tenant_id_2 = str(uuid4())
        upload_id = str(uuid4())

        content_1 = b"Tenant 1 content"
        content_2 = b"Tenant 2 content"

        # Save files for both tenants with same upload_id
        await storage_adapter.save_file(
            tenant_id=tenant_id_1,
            upload_id=upload_id,
            filename="test.txt",
            content=content_1
        )

        await storage_adapter.save_file(
            tenant_id=tenant_id_2,
            upload_id=upload_id,
            filename="test.txt",
            content=content_2
        )

        # Retrieve and verify each tenant's content
        retrieved_1 = await storage_adapter.retrieve_file(tenant_id_1, upload_id)
        retrieved_2 = await storage_adapter.retrieve_file(tenant_id_2, upload_id)

        assert retrieved_1 == content_1
        assert retrieved_2 == content_2
        assert retrieved_1 != retrieved_2

    @pytest.mark.asyncio
    async def test_tenant_isolation_separate_directories(
        self, storage_adapter, sample_file_content
    ):
        """Test that tenants have separate directories."""
        tenant_id_1 = str(uuid4())
        tenant_id_2 = str(uuid4())
        upload_id = str(uuid4())

        # Save files for both tenants
        await storage_adapter.save_file(
            tenant_id=tenant_id_1,
            upload_id=upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        await storage_adapter.save_file(
            tenant_id=tenant_id_2,
            upload_id=upload_id,
            filename="test.txt",
            content=sample_file_content
        )

        # Verify separate directories exist
        dir_1 = storage_adapter.base_path / tenant_id_1 / upload_id
        dir_2 = storage_adapter.base_path / tenant_id_2 / upload_id

        assert dir_1.exists()
        assert dir_2.exists()
        assert dir_1 != dir_2


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, storage_adapter):
        """Test health check when storage is healthy."""
        health_data = await storage_adapter.check_health()

        assert health_data["status"] == "healthy"
        assert health_data["storage_type"] == "local_filesystem"
        assert health_data["path_exists"] is True
        assert health_data["is_writable"] is True
        assert "disk_space" in health_data

    @pytest.mark.asyncio
    async def test_health_check_includes_disk_space(self, storage_adapter):
        """Test that health check includes disk space information."""
        health_data = await storage_adapter.check_health()

        disk_space = health_data["disk_space"]
        assert "available_bytes" in disk_space
        assert "total_bytes" in disk_space
        assert "usage_percent" in disk_space

        # Verify values are reasonable
        if disk_space["available_bytes"] is not None:
            assert disk_space["available_bytes"] >= 0
            assert disk_space["total_bytes"] > 0
            assert 0 <= disk_space["usage_percent"] <= 100

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_not_writable(self, tmp_path):
        """Test health check when directory is not writable."""
        # Create read-only directory (this test might not work on all systems)
        base_path = tmp_path / "readonly_storage"
        base_path.mkdir()

        adapter = LocalFileStorageAdapter(base_path=str(base_path))

        # Make directory read-only
        try:
            os.chmod(base_path, 0o444)

            health_data = await adapter.check_health()

            # Should detect unhealthy state
            assert health_data["status"] == "unhealthy"
            assert health_data["is_writable"] is False

        finally:
            # Restore permissions for cleanup
            os.chmod(base_path, 0o755)


class TestSecurityFeatures:
    """Test security features of LocalFileStorageAdapter."""

    @pytest.mark.asyncio
    async def test_prevents_path_traversal_in_tenant_id(
        self, storage_adapter, sample_upload_id, sample_file_content
    ):
        """Test that path traversal in tenant_id is prevented."""
        # These should all fail validation
        malicious_tenant_ids = [
            "../../../etc/passwd",
            "../../secret",
            "../tenant",
        ]

        for malicious_id in malicious_tenant_ids:
            with pytest.raises(ValueError):
                await storage_adapter.save_file(
                    tenant_id=malicious_id,
                    upload_id=sample_upload_id,
                    filename="test.txt",
                    content=sample_file_content
                )

    @pytest.mark.asyncio
    async def test_prevents_path_traversal_in_upload_id(
        self, storage_adapter, sample_tenant_id, sample_file_content
    ):
        """Test that path traversal in upload_id is prevented."""
        malicious_upload_ids = [
            "../../../uploads",
            "../../data",
            "../upload",
        ]

        for malicious_id in malicious_upload_ids:
            with pytest.raises(ValueError):
                await storage_adapter.save_file(
                    tenant_id=sample_tenant_id,
                    upload_id=malicious_id,
                    filename="test.txt",
                    content=sample_file_content
                )

    @pytest.mark.asyncio
    async def test_prevents_path_traversal_in_filename(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that path traversal in filename is prevented."""
        malicious_filename = "../../../etc/passwd"

        # Should succeed but filename should be sanitized
        storage_path = await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename=malicious_filename,
            content=sample_file_content
        )

        # Verify file is stored in correct location
        assert ".." not in storage_path
        assert "/etc/" not in storage_path

        # Verify file doesn't exist outside tenant directory
        etc_path = Path("/etc/passwd")
        if etc_path.exists():
            # Ensure we didn't overwrite system files
            assert etc_path.read_bytes() != sample_file_content

    @pytest.mark.asyncio
    async def test_sanitizes_hidden_files(
        self, storage_adapter, sample_tenant_id, sample_upload_id, sample_file_content
    ):
        """Test that hidden files (starting with dot) are handled safely."""
        hidden_filename = ".hidden_file.txt"

        storage_path = await storage_adapter.save_file(
            tenant_id=sample_tenant_id,
            upload_id=sample_upload_id,
            filename=hidden_filename,
            content=sample_file_content
        )

        # Filename should be sanitized to prevent hidden file
        assert storage_path is not None

        # File should be saved successfully
        file_path = storage_adapter.base_path / sample_tenant_id / sample_upload_id / "original_file.txt"
        assert file_path.exists()