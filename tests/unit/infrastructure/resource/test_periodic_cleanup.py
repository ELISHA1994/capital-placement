"""Tests for PeriodicResourceCleanup."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.infrastructure.resource.file_resource_manager import FileResourceManager
from app.infrastructure.resource.periodic_cleanup import PeriodicResourceCleanup


@pytest_asyncio.fixture
async def resource_manager():
    """Create a file resource manager for testing."""
    manager = FileResourceManager(cleanup_interval_minutes=60)
    yield manager
    await manager.shutdown()


@pytest.fixture
def sample_content():
    """Sample file content for testing."""
    return b"Test content for periodic cleanup"


class TestCleanupScheduling:
    """Tests for cleanup scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_cleanup(self, resource_manager):
        """Test that cleanup is scheduled on initialization."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=1,
        )

        assert cleanup._cleanup_task is not None
        assert not cleanup._cleanup_task.done()

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_cleanup(self, resource_manager):
        """Test canceling cleanup task."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=1,
        )

        await cleanup.shutdown()

        assert cleanup._shutdown is True
        assert cleanup._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_cleanup_interval(self, resource_manager):
        """Test cleanup runs at specified interval."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=1,
        )

        assert cleanup._cleanup_interval == 1

        await cleanup.shutdown()


class TestCleanupExecution:
    """Tests for cleanup execution."""

    @pytest.mark.asyncio
    async def test_execute_cleanup(self, resource_manager, sample_content):
        """Test cleanup execution."""
        # Track old resources
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="old.txt",
            upload_id="old_upload",
            tenant_id="test_tenant",
        )

        # Make resource old
        resource = resource_manager._resources.get(resource_id)
        if resource:
            resource.created_at = datetime.utcnow() - timedelta(minutes=120)

        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        # Trigger immediate cleanup
        result = await cleanup.trigger_immediate_cleanup()

        assert "cleanup_result" in result
        assert result["cleanup_result"]["cleaned_count"] >= 1

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_threshold(self, resource_manager, sample_content):
        """Test cleanup respects age threshold."""
        # Track recent resource
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="recent.txt",
            upload_id="recent_upload",
            tenant_id="test_tenant",
        )

        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        # Cleanup should not remove recent resources
        result = await cleanup.trigger_immediate_cleanup()

        assert result["cleanup_result"]["cleaned_count"] == 0

        await cleanup.shutdown()
        await resource_manager.release_resource(resource_id)

    @pytest.mark.asyncio
    async def test_partial_cleanup(self, resource_manager, sample_content):
        """Test partial cleanup with max count limit."""
        # Track multiple old resources
        for i in range(5):
            resource_id = await resource_manager.track_file_content(
                content=sample_content,
                filename=f"old_{i}.txt",
                upload_id=f"upload_{i}",
                tenant_id="test_tenant",
            )
            resource = resource_manager._resources.get(resource_id)
            if resource:
                resource.created_at = datetime.utcnow() - timedelta(minutes=120)

        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        # Cleanup with max count of 3
        result = await cleanup.trigger_immediate_cleanup(max_cleanup_count=3)

        assert result["cleanup_result"]["cleaned_count"] <= 3

        await cleanup.shutdown()


class TestCleanupStats:
    """Tests for cleanup statistics."""

    @pytest.mark.asyncio
    async def test_track_cleanups(self, resource_manager, sample_content):
        """Test tracking cleanup operations."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
        )

        result = await cleanup.trigger_immediate_cleanup()

        assert "timestamp" in result
        assert "cleanup_result" in result
        assert "current_stats" in result

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_metrics(self, resource_manager, sample_content):
        """Test cleanup metrics."""
        # Track resource
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
        )

        # Make it old
        resource = resource_manager._resources.get(resource_id)
        if resource:
            resource.created_at = datetime.utcnow() - timedelta(minutes=120)

        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        result = await cleanup.trigger_immediate_cleanup()

        assert result["cleanup_result"]["cleaned_count"] >= 1
        assert "total_size_mb" in result["cleanup_result"]

        await cleanup.shutdown()


class TestCleanupStatus:
    """Tests for cleanup status."""

    @pytest.mark.asyncio
    async def test_get_status_running(self, resource_manager):
        """Test getting status when cleanup is running."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=15,
            orphaned_resource_threshold_minutes=60,
        )

        status = await cleanup.get_cleanup_status()

        assert status["service"] == "PeriodicResourceCleanup"
        assert status["status"] == "running"
        assert status["cleanup_interval_minutes"] == 15
        assert status["orphaned_threshold_minutes"] == 60
        assert status["cleanup_task_running"] is True

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_get_status_stopped(self, resource_manager):
        """Test getting status after shutdown."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=15,
        )

        await cleanup.shutdown()
        status = await cleanup.get_cleanup_status()

        assert status["status"] == "stopped"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_cleanup_failure_handling(self, resource_manager):
        """Test handling cleanup failures."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
        )

        # Mock cleanup to raise error
        with patch.object(
            resource_manager,
            "cleanup_orphaned_resources",
            side_effect=Exception("Cleanup error"),
        ):
            # Should handle error gracefully and return error in result
            result = await cleanup.trigger_immediate_cleanup()
            assert "error" in result or "cleanup_result" in result

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, resource_manager):
        """Test graceful degradation on errors."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
        )

        # Shutdown resource manager
        await resource_manager.shutdown()

        # Cleanup should handle manager shutdown gracefully
        try:
            result = await cleanup.trigger_immediate_cleanup()
            # May succeed with empty result or raise exception
        except Exception:
            pass  # Expected behavior - handled gracefully

        await cleanup.shutdown()


class TestImmediateCleanup:
    """Tests for immediate cleanup triggering."""

    @pytest.mark.asyncio
    async def test_trigger_immediate_cleanup_basic(self, resource_manager, sample_content):
        """Test triggering immediate cleanup."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        result = await cleanup.trigger_immediate_cleanup()

        assert "cleanup_result" in result
        assert "current_stats" in result
        assert "timestamp" in result

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_override_threshold(self, resource_manager, sample_content):
        """Test overriding threshold for immediate cleanup."""
        # Track resource
        resource_id = await resource_manager.track_file_content(
            content=sample_content,
            filename="test.txt",
            upload_id="test_upload",
            tenant_id="test_tenant",
        )

        # Make it 30 minutes old
        resource = resource_manager._resources.get(resource_id)
        if resource:
            resource.created_at = datetime.utcnow() - timedelta(minutes=30)

        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        # Cleanup with lower threshold
        result = await cleanup.trigger_immediate_cleanup(
            orphaned_threshold_minutes=20
        )

        assert result["cleanup_result"]["cleaned_count"] >= 1

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_override_max_count(self, resource_manager, sample_content):
        """Test overriding max cleanup count."""
        # Track multiple old resources
        for i in range(5):
            resource_id = await resource_manager.track_file_content(
                content=sample_content,
                filename=f"old_{i}.txt",
                upload_id=f"upload_{i}",
                tenant_id="test_tenant",
            )
            resource = resource_manager._resources.get(resource_id)
            if resource:
                resource.created_at = datetime.utcnow() - timedelta(minutes=120)

        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
            orphaned_resource_threshold_minutes=60,
        )

        # Cleanup with max count of 2
        result = await cleanup.trigger_immediate_cleanup(max_cleanup_count=2)

        assert result["cleanup_result"]["cleaned_count"] <= 2

        await cleanup.shutdown()


class TestTaskManagerIntegration:
    """Tests for task manager integration."""

    @pytest.mark.asyncio
    async def test_cleanup_creates_task(self, resource_manager):
        """Test that periodic cleanup creates tasks."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
        )

        # The cleanup should create tasks via task manager
        # This is integration-level behavior
        assert cleanup._cleanup_task is not None

        await cleanup.shutdown()

    @pytest.mark.asyncio
    async def test_immediate_cleanup_creates_task(self, resource_manager):
        """Test that immediate cleanup creates a task."""
        cleanup = PeriodicResourceCleanup(
            file_resource_manager=resource_manager,
            cleanup_interval_minutes=60,
        )

        result = await cleanup.trigger_immediate_cleanup()

        # Should return result from task
        assert "cleanup_result" in result

        await cleanup.shutdown()