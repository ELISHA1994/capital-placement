"""Tests for ManagedFileContent."""

import pytest
import pytest_asyncio

from app.infrastructure.resource.file_resource_manager import FileResourceManager
from app.infrastructure.resource.managed_file_content import (
    ManagedFileContent,
    managed_file_content,
    BatchManagedFileContent,
)


@pytest_asyncio.fixture
async def resource_manager():
    """Create a file resource manager for testing."""
    manager = FileResourceManager(cleanup_interval_minutes=60)
    yield manager
    await manager.shutdown()


@pytest.fixture
def sample_content():
    """Sample file content for testing."""
    return b"This is test file content for managed resources"


class TestManagedFileContentContextManager:
    """Tests for ManagedFileContent context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self, resource_manager, sample_content):
        """Test async context manager enter and exit."""
        async with ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        ) as managed:
            assert managed.is_tracked is True
            assert managed.resource_id is not None
            assert managed.content == sample_content

        # After exit, resource should be released
        assert managed.is_released is True

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, resource_manager, sample_content):
        """Test context manager cleans up on exception."""
        try:
            async with ManagedFileContent(
                content=sample_content,
                filename="test.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
                resource_manager=resource_manager,
            ) as managed:
                resource_id = managed.resource_id
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Resource should still be released
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0

    @pytest.mark.asyncio
    async def test_context_manager_auto_cleanup(self, resource_manager, sample_content):
        """Test auto cleanup after timeout."""
        async with ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
            auto_cleanup_after=60,
        ) as managed:
            assert managed.is_tracked is True
            assert managed._auto_cleanup_after == 60

    @pytest.mark.asyncio
    async def test_nested_context_managers(self, resource_manager, sample_content):
        """Test nested context managers."""
        async with ManagedFileContent(
            content=sample_content,
            filename="file1.txt",
            upload_id="upload1",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        ) as managed1:
            async with ManagedFileContent(
                content=sample_content,
                filename="file2.txt",
                upload_id="upload2",
                tenant_id="test_tenant",
                resource_manager=resource_manager,
            ) as managed2:
                stats = await resource_manager.get_resource_stats()
                assert stats["total_resources"] == 2

            # Inner context released
            stats = await resource_manager.get_resource_stats()
            assert stats["total_resources"] == 1

        # Outer context released
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0


class TestManagedFileContentAccess:
    """Tests for content access."""

    @pytest.mark.asyncio
    async def test_read_content(self, resource_manager, sample_content):
        """Test reading content."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        await managed.track()
        assert managed.content == sample_content
        await managed.release()

    @pytest.mark.asyncio
    async def test_content_metadata(self, resource_manager, sample_content):
        """Test content metadata properties."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        await managed.track()

        assert managed.filename == "test.txt"
        assert managed.upload_id == "test_upload"
        assert managed.size_bytes == len(sample_content)
        assert managed.size_mb == len(sample_content) / (1024 * 1024)

        await managed.release()

    @pytest.mark.asyncio
    async def test_access_released_content(self, resource_manager, sample_content):
        """Test accessing content after release raises error."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        await managed.track()
        await managed.release()

        with pytest.raises(RuntimeError, match="File content has been released"):
            _ = managed.content

    @pytest.mark.asyncio
    async def test_content_size_calculation(self, resource_manager):
        """Test content size calculations."""
        content = b"X" * 1024 * 1024  # 1MB
        managed = ManagedFileContent(
            content=content,
            filename="test.bin",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        await managed.track()

        assert managed.size_bytes == 1024 * 1024
        assert abs(managed.size_mb - 1.0) < 0.01

        await managed.release()


class TestManagedFileContentLifecycle:
    """Tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_manual_track_and_release(self, resource_manager, sample_content):
        """Test manual track and release."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        # Initially not tracked
        assert managed.is_tracked is False
        assert managed.is_released is False

        # Track
        success = await managed.track()
        assert success is True
        assert managed.is_tracked is True

        # Release
        success = await managed.release()
        assert success is True
        assert managed.is_released is True

    @pytest.mark.asyncio
    async def test_double_track_prevention(self, resource_manager, sample_content):
        """Test that double tracking is prevented."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        success1 = await managed.track()
        success2 = await managed.track()

        assert success1 is True
        assert success2 is False

        await managed.release()

    @pytest.mark.asyncio
    async def test_double_release_prevention(self, resource_manager, sample_content):
        """Test that double release is prevented."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        await managed.track()
        success1 = await managed.release()
        success2 = await managed.release()

        assert success1 is True
        assert success2 is False

    @pytest.mark.asyncio
    async def test_mark_in_use_and_available(self, resource_manager, sample_content):
        """Test marking resource as in use and available."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        await managed.track()

        # Mark in use
        success = await managed.mark_in_use()
        assert success is True

        stats = await resource_manager.get_resource_stats()
        assert stats["in_use_count"] == 1

        # Mark available
        success = await managed.mark_available()
        assert success is True

        stats = await resource_manager.get_resource_stats()
        assert stats["available_count"] == 1

        await managed.release()


class TestManagedFileContentHelper:
    """Tests for managed_file_content helper function."""

    @pytest.mark.asyncio
    async def test_helper_context_manager(self, resource_manager, sample_content):
        """Test helper function creates proper context manager."""
        async with managed_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        ) as managed:
            assert isinstance(managed, ManagedFileContent)
            assert managed.is_tracked is True

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0

    @pytest.mark.asyncio
    async def test_helper_with_auto_cleanup(self, resource_manager, sample_content):
        """Test helper with auto cleanup."""
        async with managed_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
            auto_cleanup_after=300,
        ) as managed:
            assert managed._auto_cleanup_after == 300


class TestBatchManagedFileContent:
    """Tests for batch file management."""

    @pytest.mark.asyncio
    async def test_batch_add_files(self, resource_manager, sample_content):
        """Test adding multiple files to batch."""
        batch = BatchManagedFileContent(resource_manager=resource_manager)

        for i in range(3):
            await batch.add_file(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
            )

        assert batch.file_count == 3
        assert len(batch) == 3

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 3

        await batch.release_all()

    @pytest.mark.asyncio
    async def test_batch_total_size(self, resource_manager, sample_content):
        """Test batch total size calculation."""
        batch = BatchManagedFileContent(resource_manager=resource_manager)

        total_expected = 0
        for i in range(5):
            await batch.add_file(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
            )
            total_expected += len(sample_content)

        assert batch.total_size_bytes == total_expected
        assert batch.total_size_mb == total_expected / (1024 * 1024)

        await batch.release_all()

    @pytest.mark.asyncio
    async def test_batch_release_all(self, resource_manager, sample_content):
        """Test releasing all files in batch."""
        batch = BatchManagedFileContent(resource_manager=resource_manager)

        for i in range(4):
            await batch.add_file(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
            )

        result = await batch.release_all()

        assert result["released_count"] == 4
        assert result["failed_count"] == 0
        assert batch.file_count == 0

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0

    @pytest.mark.asyncio
    async def test_batch_context_manager(self, resource_manager, sample_content):
        """Test batch as context manager."""
        async with BatchManagedFileContent(resource_manager=resource_manager) as batch:
            for i in range(3):
                await batch.add_file(
                    content=sample_content,
                    filename=f"file_{i}.txt",
                    upload_id="test_upload",
                    tenant_id="test_tenant",
                )

            assert batch.file_count == 3

        # After exit, all files should be released
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0

    @pytest.mark.asyncio
    async def test_batch_iteration(self, resource_manager, sample_content):
        """Test iterating over batch files."""
        batch = BatchManagedFileContent(resource_manager=resource_manager)

        filenames = []
        for i in range(3):
            await batch.add_file(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
            )
            filenames.append(f"file_{i}.txt")

        # Iterate and check filenames
        batch_filenames = [m.filename for m in batch]
        assert batch_filenames == filenames

        await batch.release_all()


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_track_without_manager(self, sample_content):
        """Test tracking behavior without resource manager."""
        # This tests graceful degradation if manager is None or unavailable
        manager = FileResourceManager(cleanup_interval_minutes=60)
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=manager,
        )

        await manager.shutdown()

        # Track should handle manager errors gracefully
        success = await managed.track()
        # May succeed or fail depending on implementation
        # but should not crash

    @pytest.mark.asyncio
    async def test_release_untracked_content(self, resource_manager, sample_content):
        """Test releasing untracked content."""
        managed = ManagedFileContent(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            resource_manager=resource_manager,
        )

        # Try to release without tracking
        success = await managed.release()
        assert success is False