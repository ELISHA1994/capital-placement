"""Retry monitoring tests disabled pending infrastructure restoration."""

import pytest

pytest.skip(
    "Retry monitoring stack removed during migration; tests pending rewrite.",
    allow_module_level=True,
)


@pytest.fixture
def monitoring_service():
    """Create a fresh monitoring service instance."""
    return RetryMonitoringService()


@pytest.fixture
def mock_retry_service():
    """Mock retry service."""
    service = AsyncMock()
    service.get_retry_statistics = AsyncMock(return_value={
        "total_retry_states": 100,
        "failed_retries": 10,
        "by_operation_type": {}
    })
    service.get_ready_retries = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_dead_letter_service():
    """Mock dead letter service."""
    service = AsyncMock()
    service.get_dead_letter_queue = AsyncMock(return_value={"total_count": 0, "items": []})
    service.get_dead_letter_statistics = AsyncMock(return_value={
        "total_entries": 0,
        "resolved_entries": 0
    })
    return service


@pytest.mark.asyncio
class TestRetryMonitoringService:
    """Test RetryMonitoringService functionality."""

    async def test_initialization(self, monitoring_service):
        """Test monitoring service initialization."""
        assert monitoring_service._monitoring_task is None
        assert monitoring_service._shutdown is False
        assert monitoring_service._monitoring_interval == 60
        assert len(monitoring_service._thresholds) > 0
        assert monitoring_service._alert_history == []

    async def test_start_monitoring(self, monitoring_service):
        """Test starting monitoring task."""
        await monitoring_service.start_monitoring()

        assert monitoring_service._monitoring_task is not None
        assert not monitoring_service._monitoring_task.done()

        # Clean up
        await monitoring_service.stop_monitoring()

    async def test_stop_monitoring(self, monitoring_service):
        """Test stopping monitoring task."""
        await monitoring_service.start_monitoring()
        await monitoring_service.stop_monitoring()

        assert monitoring_service._shutdown is True

    async def test_create_alert(self, monitoring_service):
        """Test alert creation."""
        await monitoring_service._create_alert(
            alert_type=AlertType.HIGH_FAILURE_RATE,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            description="Test description",
            metadata={"test": "value"}
        )

        assert len(monitoring_service._alert_history) == 1
        alert = monitoring_service._alert_history[0]
        assert alert["type"] == AlertType.HIGH_FAILURE_RATE.value
        assert alert["severity"] == AlertSeverity.CRITICAL.value
        assert alert["title"] == "Test Alert"
        assert alert["description"] == "Test description"
        assert alert["metadata"]["test"] == "value"

    async def test_alert_suppression(self, monitoring_service):
        """Test alert suppression for duplicate alerts."""
        # Create first alert
        await monitoring_service._create_alert(
            alert_type=AlertType.HIGH_FAILURE_RATE,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            description="Test description"
        )

        # Try to create duplicate alert immediately
        await monitoring_service._create_alert(
            alert_type=AlertType.HIGH_FAILURE_RATE,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            description="Test description"
        )

        # Should only have 1 alert due to suppression
        assert len(monitoring_service._alert_history) == 1

    async def test_should_suppress_alert_different_types(self, monitoring_service):
        """Test that different alert types are not suppressed."""
        alert1 = {
            "type": AlertType.HIGH_FAILURE_RATE.value,
            "title": "Test Alert",
            "created_at": datetime.utcnow(),
            "suppressed": False
        }
        alert2 = {
            "type": AlertType.DEAD_LETTER_BACKLOG.value,
            "title": "Test Alert",
            "created_at": datetime.utcnow(),
            "suppressed": False
        }

        monitoring_service._alert_history.append(alert1)

        # Different type should not be suppressed
        assert not monitoring_service._should_suppress_alert(alert2)

    @patch("app.infrastructure.retry.retry_monitor.health_check_retry_services")
    async def test_check_retry_system_health_healthy(
        self, mock_health_check, monitoring_service
    ):
        """Test health check with healthy services."""
        mock_health_check.return_value = {
            "retry_service": {"status": "healthy"},
            "dead_letter_service": {"status": "healthy"}
        }

        await monitoring_service._check_retry_system_health()

        # No alerts should be created
        assert len(monitoring_service._alert_history) == 0

    @patch("app.infrastructure.retry.retry_monitor.health_check_retry_services")
    async def test_check_retry_system_health_unhealthy(
        self, mock_health_check, monitoring_service
    ):
        """Test health check with unhealthy services."""
        mock_health_check.return_value = {
            "retry_service": {"status": "unhealthy", "error": "Connection failed"},
            "dead_letter_service": {"status": "healthy"}
        }

        await monitoring_service._check_retry_system_health()

        # Alert should be created
        assert len(monitoring_service._alert_history) == 1
        assert monitoring_service._alert_history[0]["type"] == AlertType.SYSTEM_DEGRADATION.value

    @patch("app.infrastructure.retry.retry_monitor.get_retry_service")
    async def test_check_failure_rates_normal(
        self, mock_get_retry_service, monitoring_service, mock_retry_service
    ):
        """Test failure rate check with normal rates."""
        mock_get_retry_service.return_value = mock_retry_service
        mock_retry_service.get_retry_statistics.return_value = {
            "total_retry_states": 100,
            "failed_retries": 5,  # 5% failure rate
            "by_operation_type": {}
        }

        await monitoring_service._check_failure_rates()

        # No alerts for normal failure rate
        assert len(monitoring_service._alert_history) == 0

    @patch("app.infrastructure.retry.retry_monitor.get_retry_service")
    async def test_check_failure_rates_warning(
        self, mock_get_retry_service, monitoring_service, mock_retry_service
    ):
        """Test failure rate check with warning threshold."""
        mock_get_retry_service.return_value = mock_retry_service
        mock_retry_service.get_retry_statistics.return_value = {
            "total_retry_states": 100,
            "failed_retries": 20,  # 20% failure rate (above 15% warning)
            "by_operation_type": {}
        }

        await monitoring_service._check_failure_rates()

        # Alert should be created
        assert len(monitoring_service._alert_history) == 1
        assert monitoring_service._alert_history[0]["severity"] == AlertSeverity.MEDIUM.value

    @patch("app.infrastructure.retry.retry_monitor.get_retry_service")
    async def test_check_failure_rates_critical(
        self, mock_get_retry_service, monitoring_service, mock_retry_service
    ):
        """Test failure rate check with critical threshold."""
        mock_get_retry_service.return_value = mock_retry_service
        mock_retry_service.get_retry_statistics.return_value = {
            "total_retry_states": 100,
            "failed_retries": 35,  # 35% failure rate (above 30% critical)
            "by_operation_type": {}
        }

        await monitoring_service._check_failure_rates()

        # Critical alert should be created
        assert len(monitoring_service._alert_history) == 1
        assert monitoring_service._alert_history[0]["severity"] == AlertSeverity.CRITICAL.value

    @patch("app.infrastructure.retry.retry_monitor.get_dead_letter_service")
    async def test_check_dead_letter_queue_normal(
        self, mock_get_dls, monitoring_service, mock_dead_letter_service
    ):
        """Test dead letter queue check with normal count."""
        mock_get_dls.return_value = mock_dead_letter_service
        mock_dead_letter_service.get_dead_letter_queue.return_value = {
            "total_count": 10,  # Below warning threshold
            "items": []
        }

        await monitoring_service._check_dead_letter_queue()

        # No alerts for normal count
        assert len(monitoring_service._alert_history) == 0

    @patch("app.infrastructure.retry.retry_monitor.get_dead_letter_service")
    async def test_check_dead_letter_queue_warning(
        self, mock_get_dls, monitoring_service, mock_dead_letter_service
    ):
        """Test dead letter queue check with warning threshold."""
        mock_get_dls.return_value = mock_dead_letter_service
        mock_dead_letter_service.get_dead_letter_queue.return_value = {
            "total_count": 60,  # Above warning threshold (50)
            "items": []
        }

        await monitoring_service._check_dead_letter_queue()

        # Alert should be created
        assert len(monitoring_service._alert_history) == 1
        assert monitoring_service._alert_history[0]["type"] == AlertType.DEAD_LETTER_BACKLOG.value

    @patch("app.infrastructure.retry.retry_monitor.get_retry_service")
    async def test_check_retry_queue_buildup_normal(
        self, mock_get_retry_service, monitoring_service, mock_retry_service
    ):
        """Test retry queue buildup check with normal count."""
        mock_get_retry_service.return_value = mock_retry_service
        mock_retry_service.get_ready_retries.return_value = [
            MagicMock(created_at=datetime.utcnow()) for _ in range(50)
        ]

        await monitoring_service._check_retry_queue_buildup()

        # No alerts for normal count
        assert len(monitoring_service._alert_history) == 0

    @patch("app.infrastructure.retry.retry_monitor.get_retry_service")
    async def test_check_retry_queue_buildup_critical(
        self, mock_get_retry_service, monitoring_service, mock_retry_service
    ):
        """Test retry queue buildup check with critical threshold."""
        mock_get_retry_service.return_value = mock_retry_service
        mock_retry_service.get_ready_retries.return_value = [
            MagicMock(created_at=datetime.utcnow()) for _ in range(600)
        ]

        await monitoring_service._check_retry_queue_buildup()

        # Critical alert should be created
        assert len(monitoring_service._alert_history) == 1
        assert monitoring_service._alert_history[0]["severity"] == AlertSeverity.CRITICAL.value

    async def test_cleanup_old_alerts(self, monitoring_service):
        """Test cleanup of old alerts."""
        # Add old alert
        old_alert = {
            "id": "old_1",
            "type": AlertType.HIGH_FAILURE_RATE.value,
            "severity": AlertSeverity.LOW.value,
            "title": "Old Alert",
            "description": "Old",
            "created_at": datetime.utcnow() - timedelta(hours=25),
            "suppressed": False
        }

        # Add recent alert
        recent_alert = {
            "id": "recent_1",
            "type": AlertType.HIGH_FAILURE_RATE.value,
            "severity": AlertSeverity.LOW.value,
            "title": "Recent Alert",
            "description": "Recent",
            "created_at": datetime.utcnow(),
            "suppressed": False
        }

        monitoring_service._alert_history = [old_alert, recent_alert]

        await monitoring_service._cleanup_old_alerts()

        # Only recent alert should remain
        assert len(monitoring_service._alert_history) == 1
        assert monitoring_service._alert_history[0]["id"] == "recent_1"

    @patch("app.infrastructure.retry.retry_monitor.get_retry_service")
    @patch("app.infrastructure.retry.retry_monitor.get_dead_letter_service")
    @patch("app.infrastructure.retry.retry_monitor.health_check_retry_services")
    async def test_get_monitoring_dashboard(
        self, mock_health_check, mock_get_dls, mock_get_retry,
        monitoring_service, mock_retry_service, mock_dead_letter_service
    ):
        """Test getting monitoring dashboard data."""
        mock_get_retry.return_value = mock_retry_service
        mock_get_dls.return_value = mock_dead_letter_service
        mock_health_check.return_value = {"overall_status": "healthy"}

        dashboard = await monitoring_service.get_monitoring_dashboard()

        assert "monitoring_status" in dashboard
        assert "retry_statistics" in dashboard
        assert "dead_letter_statistics" in dashboard
        assert "recent_alerts" in dashboard
        assert "alert_summary" in dashboard
        assert "health_status" in dashboard
        assert "thresholds" in dashboard

    async def test_update_thresholds(self, monitoring_service):
        """Test updating monitoring thresholds."""
        old_value = monitoring_service._thresholds["failure_rate_warning"]
        new_value = 0.25

        await monitoring_service.update_thresholds({
            "failure_rate_warning": new_value
        })

        assert monitoring_service._thresholds["failure_rate_warning"] == new_value
        assert monitoring_service._thresholds["failure_rate_warning"] != old_value

    async def test_get_alert_history_all(self, monitoring_service):
        """Test getting all alert history."""
        # Add some alerts
        for i in range(5):
            monitoring_service._alert_history.append({
                "id": f"alert_{i}",
                "type": AlertType.HIGH_FAILURE_RATE.value,
                "severity": AlertSeverity.MEDIUM.value,
                "title": f"Alert {i}",
                "description": "Test",
                "created_at": datetime.utcnow() - timedelta(hours=i),
                "suppressed": False
            })

        history = await monitoring_service.get_alert_history(hours=24)

        assert len(history) == 5

    async def test_get_alert_history_filtered_by_severity(self, monitoring_service):
        """Test getting alert history filtered by severity."""
        # Add alerts with different severities
        monitoring_service._alert_history = [
            {
                "id": "alert_1",
                "type": AlertType.HIGH_FAILURE_RATE.value,
                "severity": AlertSeverity.CRITICAL.value,
                "title": "Critical Alert",
                "description": "Test",
                "created_at": datetime.utcnow(),
                "suppressed": False
            },
            {
                "id": "alert_2",
                "type": AlertType.HIGH_FAILURE_RATE.value,
                "severity": AlertSeverity.LOW.value,
                "title": "Low Alert",
                "description": "Test",
                "created_at": datetime.utcnow(),
                "suppressed": False
            }
        ]

        history = await monitoring_service.get_alert_history(
            hours=24,
            severity=AlertSeverity.CRITICAL
        )

        assert len(history) == 1
        assert history[0]["severity"] == AlertSeverity.CRITICAL.value

    async def test_check_health(self, monitoring_service):
        """Test health check of monitoring service."""
        health = await monitoring_service.check_health()

        assert "status" in health
        assert "monitoring_active" in health
        assert "alert_history_size" in health
        assert health["status"] == "healthy"


@pytest.mark.asyncio
class TestMonitoringServiceHelpers:
    """Test monitoring service helper functions."""

    async def test_get_retry_monitoring_service(self):
        """Test getting monitoring service instance."""
        service1 = await get_retry_monitoring_service()
        service2 = await get_retry_monitoring_service()

        # Should return same instance
        assert service1 is service2

    async def test_start_retry_monitoring(self):
        """Test starting retry monitoring."""
        await start_retry_monitoring()

        service = await get_retry_monitoring_service()
        assert service._monitoring_task is not None

        # Clean up
        await stop_retry_monitoring()

    async def test_stop_retry_monitoring(self):
        """Test stopping retry monitoring."""
        await start_retry_monitoring()
        await stop_retry_monitoring()

        service = await get_retry_monitoring_service()
        assert service._shutdown is True
