"""
Comprehensive tests for LocalFileStorageAdapter.

Tests cover:
- Basic CRUD operations (save, retrieve, delete, exists)
- Security validations (UUID validation, path traversal prevention)
- Tenant isolation
- Error handling and edge cases
- Health checks
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4

from app.infrastructure.adapters.storage_adapter import LocalFileStorageAdapter


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_storage_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def storage_adapter(temp_storage_dir):
    """Create LocalFileStorageAdapter instance with temp directory."""
    return LocalFileStorageAdapter(base_path=temp_storage_dir)


@pytest.fixture
def tenant_id():
    """Generate valid tenant UUID."""
    return str(uuid4())


@pytest.fixture
def upload_id():
    """Generate valid upload UUID."""
    return str(uuid4())


# === Basic Operations Tests ===


@pytest.mark.asyncio
async def test_save_file_creates_file_successfully(storage_adapter, tenant_id, upload_id):
    """Test that save_file successfully saves a file."""
    filename = "test_document.pdf"
    content = b"This is test PDF content"

    storage_path = await storage_adapter.save_file(
        tenant_id=tenant_id,
        upload_id=upload_id,
        filename=filename,
        content=content
    )

    # Verify storage path format
    assert storage_path == f"{tenant_id}/{upload_id}/original_file.pdf"

    # Verify file exists on disk
    expected_file = Path(storage_adapter.base_path) / tenant_id / upload_id / "original_file.pdf"
    assert expected_file.exists()
    assert expected_file.read_bytes() == content


@pytest.mark.asyncio
async def test_retrieve_file_returns_correct_content(storage_adapter, tenant_id, upload_id):
    """Test that retrieve_file returns correct file content."""
    filename = "report.docx"
    content = b"Binary document content here"

    # Save file first
    await storage_adapter.save_file(tenant_id, upload_id, filename, content)

    # Retrieve file
    retrieved_content = await storage_adapter.retrieve_file(tenant_id, upload_id)

    assert retrieved_content == content


@pytest.mark.asyncio
async def test_delete_file_removes_file_and_directory(storage_adapter, tenant_id, upload_id):
    """Test that delete_file removes both file and directory."""
    content = b"content to delete"

    # Save file first
    await storage_adapter.save_file(tenant_id, upload_id, "file.txt", content)

    # Verify file exists
    upload_dir = Path(storage_adapter.base_path) / tenant_id / upload_id
    assert upload_dir.exists()

    # Delete file
    result = await storage_adapter.delete_file(tenant_id, upload_id)

    assert result is True
    assert not upload_dir.exists()


@pytest.mark.asyncio
async def test_delete_nonexistent_file_returns_false(storage_adapter, tenant_id, upload_id):
    """Test that deleting non-existent file returns False."""
    result = await storage_adapter.delete_file(tenant_id, upload_id)
    assert result is False


@pytest.mark.asyncio
async def test_exists_returns_true_when_file_exists(storage_adapter, tenant_id, upload_id):
    """Test that exists returns True when file exists."""
    # Save file
    await storage_adapter.save_file(tenant_id, upload_id, "test.pdf", b"content")

    # Check existence
    exists = await storage_adapter.exists(tenant_id, upload_id)
    assert exists is True


@pytest.mark.asyncio
async def test_exists_returns_false_when_file_missing(storage_adapter, tenant_id, upload_id):
    """Test that exists returns False when file doesn't exist."""
    exists = await storage_adapter.exists(tenant_id, upload_id)
    assert exists is False


# === Security & Validation Tests ===


@pytest.mark.asyncio
async def test_invalid_tenant_id_raises_value_error(storage_adapter, upload_id):
    """Test that invalid tenant_id raises ValueError."""
    invalid_tenant_ids = [
        "not-a-uuid",
        "../../../etc/passwd",
        "tenant/123",
        "",
        None
    ]

    for invalid_id in invalid_tenant_ids:
        if invalid_id is not None:  # None will raise different error
            with pytest.raises(ValueError, match="Invalid tenant_id"):
                await storage_adapter.save_file(
                    tenant_id=invalid_id,
                    upload_id=upload_id,
                    filename="test.pdf",
                    content=b"content"
                )


@pytest.mark.asyncio
async def test_invalid_upload_id_raises_value_error(storage_adapter, tenant_id):
    """Test that invalid upload_id raises ValueError."""
    invalid_upload_ids = [
        "not-a-uuid",
        "../../../etc/passwd",
        "upload/123",
        ""
    ]

    for invalid_id in invalid_upload_ids:
        with pytest.raises(ValueError, match="Invalid upload_id"):
            await storage_adapter.save_file(
                tenant_id=tenant_id,
                upload_id=invalid_id,
                filename="test.pdf",
                content=b"content"
            )


@pytest.mark.asyncio
async def test_path_traversal_attack_prevented(storage_adapter, tenant_id, upload_id):
    """Test that path traversal attacks in filename are prevented."""
    dangerous_filenames = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "/etc/passwd",
        "C:\\Windows\\System32\\config"
    ]

    for dangerous_filename in dangerous_filenames:
        # Should not raise error but sanitize the filename
        storage_path = await storage_adapter.save_file(
            tenant_id=tenant_id,
            upload_id=upload_id,
            filename=dangerous_filename,
            content=b"malicious content"
        )

        # Verify file is saved within the correct directory
        assert storage_path.startswith(f"{tenant_id}/{upload_id}/")

        # Verify file doesn't escape the upload directory
        saved_file = Path(storage_adapter.base_path) / tenant_id / upload_id
        assert saved_file.exists()


@pytest.mark.asyncio
async def test_empty_filename_raises_value_error(storage_adapter, tenant_id, upload_id):
    """Test that empty filename raises ValueError."""
    with pytest.raises(ValueError, match="Filename cannot be empty"):
        await storage_adapter.save_file(
            tenant_id=tenant_id,
            upload_id=upload_id,
            filename="",
            content=b"content"
        )

    with pytest.raises(ValueError, match="Filename cannot be empty"):
        await storage_adapter.save_file(
            tenant_id=tenant_id,
            upload_id=upload_id,
            filename="   ",  # Whitespace only
            content=b"content"
        )


# === Tenant Isolation Tests ===


@pytest.mark.asyncio
async def test_tenant_isolation_separate_directories(storage_adapter, upload_id):
    """Test that different tenants have completely isolated storage."""
    tenant_1 = str(uuid4())
    tenant_2 = str(uuid4())
    content_1 = b"Tenant 1 content"
    content_2 = b"Tenant 2 content"

    # Save files for both tenants with same upload_id
    await storage_adapter.save_file(tenant_1, upload_id, "file.pdf", content_1)
    await storage_adapter.save_file(tenant_2, upload_id, "file.pdf", content_2)

    # Retrieve files
    retrieved_1 = await storage_adapter.retrieve_file(tenant_1, upload_id)
    retrieved_2 = await storage_adapter.retrieve_file(tenant_2, upload_id)

    # Verify complete isolation
    assert retrieved_1 == content_1
    assert retrieved_2 == content_2
    assert retrieved_1 != retrieved_2


@pytest.mark.asyncio
async def test_delete_one_tenant_does_not_affect_other(storage_adapter, upload_id):
    """Test that deleting one tenant's file doesn't affect another."""
    tenant_1 = str(uuid4())
    tenant_2 = str(uuid4())

    # Save files for both tenants
    await storage_adapter.save_file(tenant_1, upload_id, "file.pdf", b"T1 content")
    await storage_adapter.save_file(tenant_2, upload_id, "file.pdf", b"T2 content")

    # Delete tenant 1's file
    await storage_adapter.delete_file(tenant_1, upload_id)

    # Verify tenant 1 file is gone
    assert not await storage_adapter.exists(tenant_1, upload_id)

    # Verify tenant 2 file still exists
    assert await storage_adapter.exists(tenant_2, upload_id)


# === Error Handling Tests ===


@pytest.mark.asyncio
async def test_retrieve_nonexistent_file_raises_file_not_found(storage_adapter, tenant_id, upload_id):
    """Test that retrieving non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No file found"):
        await storage_adapter.retrieve_file(tenant_id, upload_id)


@pytest.mark.asyncio
async def test_retrieve_with_invalid_tenant_raises_value_error(storage_adapter, upload_id):
    """Test that retrieve with invalid tenant_id raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tenant_id"):
        await storage_adapter.retrieve_file("invalid-uuid", upload_id)


@pytest.mark.asyncio
async def test_save_preserves_file_extension(storage_adapter, tenant_id, upload_id):
    """Test that file extension is preserved during save."""
    test_cases = [
        ("document.pdf", ".pdf"),
        ("image.png", ".png"),
        ("archive.tar.gz", ".gz"),
        ("file", ""),  # No extension
        ("test.DOCX", ".DOCX")  # Uppercase extension
    ]

    for filename, expected_ext in test_cases:
        upload_id_case = str(uuid4())
        storage_path = await storage_adapter.save_file(
            tenant_id=tenant_id,
            upload_id=upload_id_case,
            filename=filename,
            content=b"content"
        )

        # Verify extension is preserved in storage filename
        assert storage_path.endswith(f"original_file{expected_ext}")


# === Health Check Tests ===


@pytest.mark.asyncio
async def test_health_check_returns_healthy_status(storage_adapter):
    """Test that health check returns healthy status."""
    health = await storage_adapter.check_health()

    assert health["status"] == "healthy"
    assert health["storage_type"] == "local_filesystem"
    assert health["path_exists"] is True
    assert health["is_writable"] is True
    assert "disk_space" in health


@pytest.mark.asyncio
async def test_health_check_includes_disk_space_info(storage_adapter):
    """Test that health check includes disk space information."""
    health = await storage_adapter.check_health()

    disk_space = health.get("disk_space", {})
    assert "available_bytes" in disk_space
    assert "total_bytes" in disk_space
    assert "usage_percent" in disk_space

    # Verify reasonable values
    if disk_space["available_bytes"] is not None:
        assert disk_space["available_bytes"] >= 0
        assert disk_space["total_bytes"] > 0


# === Edge Cases ===


@pytest.mark.asyncio
async def test_save_large_file(storage_adapter, tenant_id, upload_id):
    """Test saving a large file (10 MB)."""
    # Create 10 MB of data
    large_content = b"x" * (10 * 1024 * 1024)

    storage_path = await storage_adapter.save_file(
        tenant_id=tenant_id,
        upload_id=upload_id,
        filename="large_file.bin",
        content=large_content
    )

    # Verify file was saved
    retrieved = await storage_adapter.retrieve_file(tenant_id, upload_id)
    assert len(retrieved) == len(large_content)


@pytest.mark.asyncio
async def test_save_empty_file(storage_adapter, tenant_id, upload_id):
    """Test saving an empty file."""
    empty_content = b""

    storage_path = await storage_adapter.save_file(
        tenant_id=tenant_id,
        upload_id=upload_id,
        filename="empty.txt",
        content=empty_content
    )

    # Verify empty file was saved
    retrieved = await storage_adapter.retrieve_file(tenant_id, upload_id)
    assert retrieved == b""


@pytest.mark.asyncio
async def test_filename_with_special_characters_sanitized(storage_adapter, tenant_id, upload_id):
    """Test that special characters in filename are sanitized."""
    dangerous_filename = "file@#$%^&*()+=[]{}|\\:;<>?,name.pdf"

    storage_path = await storage_adapter.save_file(
        tenant_id=tenant_id,
        upload_id=upload_id,
        filename=dangerous_filename,
        content=b"content"
    )

    # Filename should be sanitized but still work
    assert storage_path.startswith(f"{tenant_id}/{upload_id}/")

    # File should be retrievable
    retrieved = await storage_adapter.retrieve_file(tenant_id, upload_id)
    assert retrieved == b"content"


@pytest.mark.asyncio
async def test_concurrent_saves_different_uploads(storage_adapter, tenant_id):
    """Test concurrent saves for different uploads."""
    import asyncio

    upload_ids = [str(uuid4()) for _ in range(5)]
    contents = [f"Content {i}".encode() for i in range(5)]

    # Save all files concurrently
    tasks = [
        storage_adapter.save_file(tenant_id, upload_ids[i], f"file{i}.txt", contents[i])
        for i in range(5)
    ]

    await asyncio.gather(*tasks)

    # Verify all files were saved correctly
    for i in range(5):
        retrieved = await storage_adapter.retrieve_file(tenant_id, upload_ids[i])
        assert retrieved == contents[i]


@pytest.mark.asyncio
async def test_overwrite_existing_file(storage_adapter, tenant_id, upload_id):
    """Test that saving to same location overwrites existing file."""
    # Save first version
    await storage_adapter.save_file(
        tenant_id, upload_id, "file.txt", b"Version 1"
    )

    # Overwrite with second version
    await storage_adapter.save_file(
        tenant_id, upload_id, "file.txt", b"Version 2"
    )

    # Verify latest version is retrieved
    retrieved = await storage_adapter.retrieve_file(tenant_id, upload_id)
    assert retrieved == b"Version 2"