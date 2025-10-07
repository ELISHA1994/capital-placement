"""Tests for FileResourceManager."""

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from app.infrastructure.resource.file_resource_manager import (
    FileResourceManager,
    TrackedResource,
    FileContentResource,
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
    return b"This is test file content for resource management"


@pytest.fixture
def large_content():
    """Large file content for testing (>10MB)."""
    return b"X" * (11 * 1024 * 1024)  # 11MB


class TestResourceCreation:
    """Tests for resource creation and tracking."""

    @pytest.mark.asyncio
    async def test_track_resource_basic(self, resource_manager):
        """Test basic resource tracking."""
        resource_id = f"test_{uuid4().hex}"

        success = await resource_manager.track_resource(
            resource_id=resource_id,
            resource_type="file_content",
            size_bytes=1024,
            tenant_id="test_tenant",
            upload_id="test_upload",
        )

        assert success is True
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 1
        assert stats["total_size_mb"] == 1024 / (1024 * 1024)

    @pytest.mark.asyncio
    async def test_track_resource_with_metadata(self, resource_manager):
        """Test tracking resource with metadata."""
        resource_id = f"test_{uuid4().hex}"
        metadata = {"filename": "test.pdf", "mime_type": "application/pdf"}

        success = await resource_manager.track_resource(
            resource_id=resource_id,
            resource_type="file_content",
            size_bytes=2048,
            tenant_id="test_tenant",
            metadata=metadata,
        )

        assert success is True
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 1

    @pytest.mark.asyncio
    async def test_track_duplicate_resource(self, resource_manager):
        """Test tracking duplicate resource ID."""
        resource_id = f"test_{uuid4().hex}"

        # Track first time
        success1 = await resource_manager.track_resource(
            resource_id=resource_id,
            resource_type="file_content",
            size_bytes=1024,
            tenant_id="test_tenant",
        )

        # Track same ID again
        success2 = await resource_manager.track_resource(
            resource_id=resource_id,
            resource_type="file_content",
            size_bytes=2048,
            tenant_id="test_tenant",
        )

        assert success1 is True
        assert success2 is False
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 1

    @pytest.mark.asyncio
    async def test_track_file_content(self, resource_manager, sample_content):
        """Test tracking file content."""
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
        )

        assert resource_id != ""
        assert resource_id.startswith("file_content_")

        # Verify content can be retrieved
        retrieved_content = await resource_manager.get_file_content(resource_id)
        assert retrieved_content == sample_content


class TestResourceLifecycle:
    """Tests for resource lifecycle management."""

    @pytest.mark.asyncio
    async def test_resource_acquisition_and_release(self, resource_manager):
        """Test complete resource lifecycle."""
        resource_id = f"test_{uuid4().hex}"

        # Track resource
        await resource_manager.track_resource(
            resource_id=resource_id,
            resource_type="file_content",
            size_bytes=1024,
            tenant_id="test_tenant",
        )

        # Mark in use
        await resource_manager.mark_resource_in_use(resource_id)
        stats = await resource_manager.get_resource_stats()
        assert stats["in_use_count"] == 1

        # Mark available
        await resource_manager.mark_resource_available(resource_id)
        stats = await resource_manager.get_resource_stats()
        assert stats["available_count"] == 1

        # Release
        success = await resource_manager.release_resource(resource_id)
        assert success is True
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0

    @pytest.mark.asyncio
    async def test_release_resource_in_use(self, resource_manager):
        """Test that in-use resources are not released without force."""
        resource_id = f"test_{uuid4().hex}"

        await resource_manager.track_resource(
            resource_id=resource_id,
            resource_type="file_content",
            size_bytes=1024,
            tenant_id="test_tenant",
        )

        await resource_manager.mark_resource_in_use(resource_id)

        # Try to release without force
        success = await resource_manager.release_resource(resource_id)
        assert success is False

        # Release with force
        success = await resource_manager.release_resource(resource_id, force=True)
        assert success is True

    @pytest.mark.asyncio
    async def test_auto_cleanup_after_timeout(self, resource_manager, sample_content):
        """Test auto cleanup after timeout."""
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
            auto_cleanup_after=1,  # 1 second
        )

        # Wait for auto cleanup
        await asyncio.sleep(2)

        # Trigger cleanup
        await resource_manager._cleanup_expired_resources()

        # Resource should be cleaned up
        content = await resource_manager.get_file_content(resource_id)
        assert content is None

    @pytest.mark.asyncio
    async def test_release_nonexistent_resource(self, resource_manager):
        """Test releasing a nonexistent resource."""
        success = await resource_manager.release_resource("nonexistent_id")
        assert success is False


class TestCleanupOperations:
    """Tests for cleanup operations."""

    @pytest.mark.asyncio
    async def test_cleanup_file_content_resource(self, resource_manager, sample_content):
        """Test cleanup of file content resource."""
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
        )

        # Verify content exists
        content = await resource_manager.get_file_content(resource_id)
        assert content is not None

        # Release resource
        success = await resource_manager.release_resource(resource_id)
        assert success is True

        # Content should be gone
        content = await resource_manager.get_file_content(resource_id)
        assert content is None

    @pytest.mark.asyncio
    async def test_cleanup_large_file(self, resource_manager, large_content):
        """Test cleanup of large file triggers garbage collection."""
        resource_id = await resource_manager.track_file_content(
            content=large_content,
            filename="large.bin",
            upload_id="test_upload",
            tenant_id="test_tenant",
        )

        # Release large file
        success = await resource_manager.release_resource(resource_id)
        assert success is True

    @pytest.mark.asyncio
    async def test_cleanup_upload_resources(self, resource_manager, sample_content):
        """Test cleanup of all resources for an upload."""
        upload_id = "test_upload"

        # Track multiple resources for same upload
        resource_ids = []
        for i in range(3):
            rid = await resource_manager.track_file_content(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id=upload_id,
                tenant_id="test_tenant",
            )
            resource_ids.append(rid)

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 3

        # Cleanup all upload resources
        result = await resource_manager.release_upload_resources(upload_id)

        assert result["released_count"] == 3
        assert result["failed_count"] == 0

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_resources(self, resource_manager, sample_content):
        """Test cleanup of orphaned resources."""
        # Track old resource
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="old.txt",
            upload_id="old_upload",
            tenant_id="test_tenant",
        )

        # Manually set created_at to old time
        resource = resource_manager._resources.get(resource_id)
        if resource:
            resource.created_at = datetime.utcnow() - timedelta(minutes=120)

        # Cleanup orphaned resources
        result = await resource_manager.cleanup_orphaned_resources(
            older_than_minutes=60,
            max_cleanup_count=100,
        )

        assert result["cleaned_count"] == 1
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0


class TestResourceTracking:
    """Tests for resource tracking and monitoring."""

    @pytest.mark.asyncio
    async def test_resource_stats_by_type(self, resource_manager, sample_content):
        """Test resource statistics by type."""
        # Track different types
        await resource_manager.track_file_content(
            content=sample_content,
            filename="test1.txt",
            upload_id="upload1",
            tenant_id="tenant1",
        )

        await resource_manager.track_resource(
            resource_id=f"temp_{uuid4().hex}",
            resource_type="temp_file",
            size_bytes=2048,
            tenant_id="tenant1",
        )

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 2
        assert "file_content" in stats["by_type"]
        assert "temp_file" in stats["by_type"]
        assert stats["by_type"]["file_content"] == 1
        assert stats["by_type"]["temp_file"] == 1

    @pytest.mark.asyncio
    async def test_resource_stats_by_tenant(self, resource_manager, sample_content):
        """Test resource statistics by tenant."""
        await resource_manager.track_file_content(
            content=sample_content,
            filename="test1.txt",
            upload_id="upload1",
            tenant_id="tenant1",
        )

        await resource_manager.track_file_content(
            content=sample_content,
            filename="test2.txt",
            upload_id="upload2",
            tenant_id="tenant2",
        )

        # Stats for all tenants
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 2
        assert len(stats["by_tenant"]) == 2
        assert stats["by_tenant"]["tenant1"] == 1
        assert stats["by_tenant"]["tenant2"] == 1

        # Stats for specific tenant
        tenant_stats = await resource_manager.get_resource_stats(tenant_id="tenant1")
        assert tenant_stats["total_resources"] == 1

    @pytest.mark.asyncio
    async def test_track_upload_to_resources_mapping(self, resource_manager, sample_content):
        """Test upload to resources mapping."""
        upload_id = "test_upload"

        # Track multiple resources
        for i in range(3):
            await resource_manager.track_file_content(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id=upload_id,
                tenant_id="test_tenant",
            )

        stats = await resource_manager.get_resource_stats()
        assert stats["total_uploads_tracked"] == 1

    @pytest.mark.asyncio
    async def test_memory_tracking(self, resource_manager, sample_content):
        """Test memory usage tracking."""
        total_size = 0
        for i in range(5):
            await resource_manager.track_file_content(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id=f"upload_{i}",
                tenant_id="test_tenant",
            )
            total_size += len(sample_content)

        stats = await resource_manager.get_resource_stats()
        expected_mb = total_size / (1024 * 1024)
        assert abs(stats["total_size_mb"] - expected_mb) < 0.01


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_content_invalid_resource(self, resource_manager):
        """Test getting content for invalid resource."""
        content = await resource_manager.get_file_content("invalid_id")
        assert content is None

    @pytest.mark.asyncio
    async def test_mark_invalid_resource_in_use(self, resource_manager):
        """Test marking invalid resource as in use."""
        success = await resource_manager.mark_resource_in_use("invalid_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_mark_invalid_resource_available(self, resource_manager):
        """Test marking invalid resource as available."""
        success = await resource_manager.mark_resource_available("invalid_id")
        assert success is False


class TestConcurrentAccess:
    """Tests for concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_resource_tracking(self, resource_manager, sample_content):
        """Test concurrent resource tracking."""
        async def track_resource(index: int):
            return await resource_manager.track_file_content(
                content=sample_content,
                filename=f"file_{index}.txt",
                upload_id=f"upload_{index}",
                tenant_id="test_tenant",
            )

        # Track multiple resources concurrently
        resource_ids = await asyncio.gather(*[
            track_resource(i) for i in range(10)
        ])

        assert len(resource_ids) == 10
        assert all(rid != "" for rid in resource_ids)

        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_cleanup(self, resource_manager, sample_content):
        """Test concurrent cleanup operations."""
        # Track resources
        resource_ids = []
        for i in range(5):
            rid = await resource_manager.track_file_content(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
            )
            resource_ids.append(rid)

        # Concurrent cleanup
        results = await asyncio.gather(*[
            resource_manager.release_resource(rid) for rid in resource_ids
        ])

        assert all(results)
        stats = await resource_manager.get_resource_stats()
        assert stats["total_resources"] == 0


class TestHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, resource_manager, sample_content):
        """Test health check returns correct status."""
        await resource_manager.track_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
        )

        health = await resource_manager.check_health()

        assert health["service"] == "FileResourceManager"
        assert health["status"] == "healthy"
        assert health["total_resources"] == 1
        assert health["total_size_mb"] > 0


class TestShutdown:
    """Tests for shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, sample_content):
        """Test shutdown cleans up all resources."""
        manager = FileResourceManager(cleanup_interval_minutes=60)

        # Track resources
        for i in range(3):
            await manager.track_file_content(
                content=sample_content,
                filename=f"file_{i}.txt",
                upload_id="test_upload",
                tenant_id="test_tenant",
            )

        stats = await manager.get_resource_stats()
        assert stats["total_resources"] == 3

        # Shutdown
        await manager.shutdown()

        # All resources should be cleaned up
        stats = await manager.get_resource_stats()
        assert stats["total_resources"] == 0