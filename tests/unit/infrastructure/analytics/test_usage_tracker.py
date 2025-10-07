"""
Comprehensive tests for UsageTracker infrastructure implementation.

Tests cover:
- Usage tracking for various operations
- Background task tracking
- Error handling and resilience
- Metric aggregation
- Performance monitoring
- Tenant-aware tracking
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.analytics.usage_tracker import (
    UsageTracker,
    usage_tracker,
    track_search_usage,
    track_upload_usage,
    track_operation_task,
    track_search_task,
    track_upload_task,
    track_profile_task,
)


class TestUsageTrackerInitialization:
    """Test UsageTracker initialization and basic operations."""

    def test_usage_tracker_initialization(self):
        """Test UsageTracker initializes correctly."""
        tracker = UsageTracker()
        assert tracker._tenant_manager is None

    def test_global_usage_tracker_exists(self):
        """Test global usage_tracker instance exists."""
        assert usage_tracker is not None
        assert isinstance(usage_tracker, UsageTracker)


class TestOperationTracking:
    """Test general operation tracking functionality."""

    @pytest.fixture
    def mock_tenant_manager(self):
        """Create mock tenant manager."""
        manager = AsyncMock()
        manager.update_usage_metrics = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_track_operation_usage_success(self, tracker, mock_tenant_manager):
        """Test successful operation tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_operation_usage(
            tenant_id="tenant-123",
            operation_type="test_operation",
            metrics={"test_metric": 1}
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once_with(
            tenant_id="tenant-123",
            metrics_update={"test_metric": 1}
        )

    @pytest.mark.asyncio

    async def test_track_operation_usage_with_metadata(self, tracker, mock_tenant_manager):
        """Test operation tracking with metadata."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_operation_usage(
            tenant_id="tenant-123",
            operation_type="test_operation",
            metrics={"test_metric": 1},
            metadata={"extra_info": "test"}
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()

    @pytest.mark.asyncio

    async def test_track_operation_usage_handles_errors(self, tracker, mock_tenant_manager):
        """Test operation tracking handles errors gracefully."""
        tracker._tenant_manager = mock_tenant_manager
        mock_tenant_manager.update_usage_metrics.side_effect = Exception("Database error")

        # Should not raise exception
        await tracker.track_operation_usage(
            tenant_id="tenant-123",
            operation_type="test_operation",
            metrics={"test_metric": 1}
        )

    @pytest.mark.asyncio

    async def test_track_operation_background(self, tracker, mock_tenant_manager):
        """Test background operation tracking."""
        tracker._tenant_manager = mock_tenant_manager

        task = tracker.track_operation_background(
            tenant_id="tenant-123",
            operation_type="test_operation",
            metrics={"test_metric": 1}
        )

        assert isinstance(task, asyncio.Task)
        await task  # Wait for completion


class TestSearchTracking:
    """Test search-specific tracking functionality."""

    @pytest.fixture
    def mock_tenant_manager(self):
        """Create mock tenant manager."""
        manager = AsyncMock()
        manager.update_usage_metrics = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_track_search_usage_basic(self, tracker, mock_tenant_manager):
        """Test basic search tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_search_usage(
            tenant_id="tenant-123",
            search_count=1
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        assert call_args[1]["metrics_update"]["searches_today"] == 1
        assert call_args[1]["metrics_update"]["total_searches"] == 1

    @pytest.mark.asyncio

    async def test_track_search_usage_with_details(self, tracker, mock_tenant_manager):
        """Test search tracking with search mode and results."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_search_usage(
            tenant_id="tenant-123",
            search_count=1,
            search_mode="semantic",
            results_count=10
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()

    @pytest.mark.asyncio

    async def test_track_search_task(self, mock_tenant_manager):
        """Test background search tracking task."""
        with patch('app.infrastructure.analytics.usage_tracker.usage_tracker') as mock_tracker:
            mock_tracker.track_search_usage = AsyncMock()

            task = track_search_task(
                tenant_id="tenant-123",
                search_count=1,
                search_mode="hybrid",
                results_count=5
            )

            assert isinstance(task, asyncio.Task)
            await task

    @pytest.mark.asyncio

    async def test_track_search_usage_convenience_function(self):
        """Test convenience function for search tracking."""
        with patch('app.infrastructure.analytics.usage_tracker.usage_tracker') as mock_tracker:
            mock_tracker.track_search_usage = AsyncMock()

            await track_search_usage(
                tenant_id="tenant-123",
                search_count=2
            )

            mock_tracker.track_search_usage.assert_called_once_with(
                "tenant-123", 2
            )


class TestUploadTracking:
    """Test upload-specific tracking functionality."""

    @pytest.fixture
    def mock_tenant_manager(self):
        """Create mock tenant manager."""
        manager = AsyncMock()
        manager.update_usage_metrics = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_track_upload_usage_basic(self, tracker, mock_tenant_manager):
        """Test basic upload tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_upload_usage(
            tenant_id="tenant-123",
            document_count=1
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        assert call_args[1]["metrics_update"]["documents_processed"] == 1
        assert call_args[1]["metrics_update"]["documents_uploaded"] == 1

    @pytest.mark.asyncio

    async def test_track_upload_usage_with_file_size(self, tracker, mock_tenant_manager):
        """Test upload tracking with file size."""
        tracker._tenant_manager = mock_tenant_manager
        file_size_bytes = 10 * 1024 * 1024  # 10 MB

        await tracker.track_upload_usage(
            tenant_id="tenant-123",
            document_count=1,
            file_size_bytes=file_size_bytes
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        storage_gb = call_args[1]["metrics_update"]["storage_used_gb"]
        assert storage_gb > 0

    @pytest.mark.asyncio

    async def test_track_upload_usage_with_processing_type(self, tracker, mock_tenant_manager):
        """Test upload tracking with processing type."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_upload_usage(
            tenant_id="tenant-123",
            document_count=1,
            file_size_bytes=1024,
            processing_type="pdf"
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()

    @pytest.mark.asyncio

    async def test_track_upload_task(self):
        """Test background upload tracking task."""
        with patch('app.infrastructure.analytics.usage_tracker.usage_tracker') as mock_tracker:
            mock_tracker.track_upload_usage = AsyncMock()

            task = track_upload_task(
                tenant_id="tenant-123",
                document_count=1,
                file_size_bytes=1024,
                processing_type="docx"
            )

            assert isinstance(task, asyncio.Task)
            await task

    @pytest.mark.asyncio

    async def test_track_upload_usage_convenience_function(self):
        """Test convenience function for upload tracking."""
        with patch('app.infrastructure.analytics.usage_tracker.usage_tracker') as mock_tracker:
            mock_tracker.track_upload_usage = AsyncMock()

            await track_upload_usage(
                tenant_id="tenant-123",
                document_count=1,
                file_size_bytes=2048
            )

            mock_tracker.track_upload_usage.assert_called_once()


class TestProfileTracking:
    """Test profile-specific tracking functionality."""

    @pytest.fixture
    def mock_tenant_manager(self):
        """Create mock tenant manager."""
        manager = AsyncMock()
        manager.update_usage_metrics = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_track_profile_create(self, tracker, mock_tenant_manager):
        """Test profile creation tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_profile_usage(
            tenant_id="tenant-123",
            operation="create",
            profile_count=1
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        metrics = call_args[1]["metrics_update"]
        assert "total_profiles" in metrics
        assert "active_profiles" in metrics

    @pytest.mark.asyncio

    async def test_track_profile_update(self, tracker, mock_tenant_manager):
        """Test profile update tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_profile_usage(
            tenant_id="tenant-123",
            operation="update",
            profile_count=1,
            fields_updated=5
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        assert "profiles_updated" in call_args[1]["metrics_update"]

    @pytest.mark.asyncio

    async def test_track_profile_delete(self, tracker, mock_tenant_manager):
        """Test profile deletion tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_profile_usage(
            tenant_id="tenant-123",
            operation="delete",
            profile_count=1
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        assert "profiles_deleted" in call_args[1]["metrics_update"]

    @pytest.mark.asyncio

    async def test_track_profile_view(self, tracker, mock_tenant_manager):
        """Test profile view tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_profile_usage(
            tenant_id="tenant-123",
            operation="view",
            profile_count=1
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        assert "profile_views" in call_args[1]["metrics_update"]

    @pytest.mark.asyncio

    async def test_track_profile_task(self):
        """Test background profile tracking task."""
        with patch('app.infrastructure.analytics.usage_tracker.usage_tracker') as mock_tracker:
            mock_tracker.track_profile_usage = AsyncMock()

            task = track_profile_task(
                tenant_id="tenant-123",
                operation="create",
                profile_count=1
            )

            assert isinstance(task, asyncio.Task)
            await task


class TestAPITracking:
    """Test API request tracking functionality."""

    @pytest.fixture
    def mock_tenant_manager(self):
        """Create mock tenant manager."""
        manager = AsyncMock()
        manager.update_usage_metrics = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_track_api_usage_basic(self, tracker, mock_tenant_manager):
        """Test basic API request tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_api_usage(
            tenant_id="tenant-123",
            endpoint="/api/v1/profiles",
            method="GET"
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        metrics = call_args[1]["metrics_update"]
        assert metrics["api_requests_today"] == 1
        assert metrics["api_requests_this_month"] == 1

    @pytest.mark.asyncio

    async def test_track_api_usage_with_performance(self, tracker, mock_tenant_manager):
        """Test API tracking with performance metrics."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_api_usage(
            tenant_id="tenant-123",
            endpoint="/api/v1/search",
            method="POST",
            response_time_ms=125.5,
            status_code=200
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()


class TestUserActivityTracking:
    """Test user activity tracking functionality."""

    @pytest.fixture
    def mock_tenant_manager(self):
        """Create mock tenant manager."""
        manager = AsyncMock()
        manager.update_usage_metrics = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_track_user_login(self, tracker, mock_tenant_manager):
        """Test user login tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_user_activity(
            tenant_id="tenant-123",
            user_id="user-456",
            activity_type="login"
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        metrics = call_args[1]["metrics_update"]
        assert metrics["user_sessions"] == 1
        assert metrics["active_users_today"] == 1

    @pytest.mark.asyncio

    async def test_track_user_session_duration(self, tracker, mock_tenant_manager):
        """Test user session duration tracking."""
        tracker._tenant_manager = mock_tenant_manager

        await tracker.track_user_activity(
            tenant_id="tenant-123",
            user_id="user-456",
            activity_type="session_end",
            session_duration_minutes=30
        )

        mock_tenant_manager.update_usage_metrics.assert_called_once()
        call_args = mock_tenant_manager.update_usage_metrics.call_args
        metrics = call_args[1]["metrics_update"]
        assert metrics["total_session_minutes"] == 30


class TestBackgroundTaskHelpers:
    """Test background task helper functions."""

    @pytest.mark.asyncio

    async def test_track_operation_task_creates_task(self):
        """Test creating background operation tracking task."""
        with patch('app.infrastructure.analytics.usage_tracker.usage_tracker') as mock_tracker:
            mock_tracker.track_operation_background = MagicMock(return_value=asyncio.create_task(asyncio.sleep(0)))

            task = track_operation_task(
                tenant_id="tenant-123",
                operation_type="test",
                metrics={"test": 1}
            )

            assert isinstance(task, asyncio.Task)
            await task


class TestPerformanceAndResilience:
    """Test performance monitoring and error resilience."""

    @pytest.fixture
    def tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()

    @pytest.mark.asyncio

    async def test_tracking_performance_overhead(self, tracker):
        """Test that tracking has minimal overhead."""
        with patch.object(tracker, '_get_tenant_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.update_usage_metrics = AsyncMock()
            mock_get_manager.return_value = mock_manager

            start = datetime.now()
            await tracker.track_operation_usage(
                tenant_id="tenant-123",
                operation_type="test",
                metrics={"test": 1}
            )
            duration = (datetime.now() - start).total_seconds() * 1000

            # Tracking should complete quickly (< 100ms in most cases)
            assert duration < 1000  # 1 second max (generous for CI)

    @pytest.mark.asyncio

    async def test_concurrent_tracking_operations(self, tracker):
        """Test handling concurrent tracking operations."""
        with patch.object(tracker, '_get_tenant_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.update_usage_metrics = AsyncMock()
            mock_get_manager.return_value = mock_manager

            # Track multiple operations concurrently
            tasks = [
                tracker.track_operation_usage(
                    tenant_id=f"tenant-{i}",
                    operation_type="test",
                    metrics={"test": 1}
                )
                for i in range(10)
            ]

            await asyncio.gather(*tasks)

            assert mock_manager.update_usage_metrics.call_count == 10

    @pytest.mark.asyncio

    async def test_error_resilience_never_raises(self, tracker):
        """Test that tracking errors never propagate."""
        with patch.object(tracker, '_get_tenant_manager') as mock_get_manager:
            mock_get_manager.side_effect = Exception("Critical error")

            # Should not raise exception
            await tracker.track_operation_usage(
                tenant_id="tenant-123",
                operation_type="test",
                metrics={"test": 1}
            )