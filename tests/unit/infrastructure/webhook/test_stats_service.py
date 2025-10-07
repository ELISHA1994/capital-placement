"""
Comprehensive tests for WebhookStatsService.

This test module covers webhook delivery statistics, performance metrics,
and monitoring functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from app.infrastructure.webhook.stats_service import WebhookStatsService


class TestWebhookStatsService:
    """Test webhook stats service comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_database = AsyncMock()
        self.service = WebhookStatsService(self.mock_database)
        self.tenant_id = str(uuid4())
        self.endpoint_id = str(uuid4())

    # Delivery Stats Tests (4 tests)

    @pytest.mark.asyncio
    async def test_get_delivery_stats_overall(self):
        """Test getting overall delivery statistics."""
        # Mock database query responses
        self.mock_database.query_items.side_effect = [
            [{  # Overall stats
                "total_deliveries": 1000,
                "successful_deliveries": 950,
                "failed_deliveries": 50,
                "dead_letter_deliveries": 10,
                "first_attempt_successes": 920,
                "avg_response_time_ms": 150,
                "avg_attempts_per_delivery": 1.1,
                "unique_endpoints": 5,
                "unique_event_types": 3
            }],
            [],  # Status breakdown
            [],  # Failure breakdown
            [{}],  # Performance metrics
            []  # Daily trends
        ]

        stats = await self.service.get_delivery_stats(
            tenant_id=self.tenant_id
        )

        assert stats["overall_stats"]["total_deliveries"] == 1000
        assert stats["overall_stats"]["successful_deliveries"] == 950
        assert stats["overall_stats"]["success_rate"] == 95.0

    @pytest.mark.asyncio
    async def test_get_delivery_stats_with_filters(self):
        """Test getting delivery statistics with filters."""
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()

        self.mock_database.query_items.side_effect = [
            [{"total_deliveries": 100}],
            [],
            [],
            [{}],
            []
        ]

        stats = await self.service.get_delivery_stats(
            tenant_id=self.tenant_id,
            endpoint_id=self.endpoint_id,
            event_type="document.processed",
            start_date=start_date,
            end_date=end_date
        )

        assert stats["filters"]["tenant_id"] == self.tenant_id
        assert stats["filters"]["endpoint_id"] == self.endpoint_id
        assert stats["filters"]["event_type"] == "document.processed"

    @pytest.mark.asyncio
    async def test_get_delivery_stats_success_rates(self):
        """Test success rate calculations."""
        self.mock_database.query_items.side_effect = [
            [{
                "total_deliveries": 100,
                "successful_deliveries": 85,
                "first_attempt_successes": 80,
                "failed_deliveries": 15
            }],
            [],
            [],
            [{}],
            []
        ]

        stats = await self.service.get_delivery_stats()

        overall = stats["overall_stats"]
        assert overall["success_rate"] == 85.0
        assert overall["first_attempt_success_rate"] == 80.0

    @pytest.mark.asyncio
    async def test_get_delivery_stats_with_time_range(self):
        """Test delivery stats with custom time range."""
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()

        self.mock_database.query_items.side_effect = [
            [{"total_deliveries": 500}],
            [],
            [],
            [{}],
            []
        ]

        stats = await self.service.get_delivery_stats(
            start_date=start_date,
            end_date=end_date
        )

        assert stats["period"]["days"] == 30

    # Endpoint Health Tests (3 tests)

    @pytest.mark.asyncio
    async def test_get_endpoint_health(self):
        """Test getting endpoint health metrics."""
        # Mock endpoint config
        self.service._get_endpoint_config = AsyncMock(return_value={
            "url": "https://example.com/webhook",
            "name": "Test Endpoint",
            "enabled": True,
            "total_deliveries": 1000,
            "successful_deliveries": 950
        })

        # Mock circuit breaker info
        self.service._get_circuit_breaker_info = AsyncMock(return_value={
            "state": "closed",
            "failure_count": 0
        })

        # Mock stats
        self.mock_database.query_items.side_effect = [
            [{"total_deliveries": 100, "successful_deliveries": 95}],
            [],
            [],
            [{"avg_response_time_ms": 150}],
            []
        ]

        # Mock recent deliveries
        self.service._get_recent_deliveries = AsyncMock(return_value=[])

        # Mock health score calculation
        self.service._calculate_endpoint_health_score = AsyncMock(return_value=95.0)

        health = await self.service.get_endpoint_health(
            self.endpoint_id,
            time_window_hours=24
        )

        assert health["endpoint_id"] == self.endpoint_id
        assert health["health_score"] == 95.0
        assert "circuit_breaker" in health

    @pytest.mark.asyncio
    async def test_calculate_endpoint_health_score(self):
        """Test endpoint health score calculation."""
        stats = {
            "total_deliveries": 100,
            "success_rate": 95.0,
            "avg_response_time_ms": 200
        }

        score = await self.service._calculate_endpoint_health_score(
            self.endpoint_id, stats
        )

        assert 0 <= score <= 100
        assert score >= 90  # High success rate

    @pytest.mark.asyncio
    async def test_calculate_endpoint_health_score_low_volume(self):
        """Test health score with low delivery volume."""
        stats = {
            "total_deliveries": 5,
            "success_rate": 100.0,
            "avg_response_time_ms": 150
        }

        score = await self.service._calculate_endpoint_health_score(
            self.endpoint_id, stats
        )

        # Score should be penalized for low volume
        assert score < 100

    # Tenant Summary Tests (2 tests)

    @pytest.mark.asyncio
    async def test_get_tenant_webhook_summary(self):
        """Test getting tenant webhook summary."""
        # Mock stats
        self.mock_database.query_items.side_effect = [
            [{"total_deliveries": 500}],
            [],
            [],
            [{}],
            [],
            [{"total_endpoints": 5, "enabled_endpoints": 4}],  # Endpoints summary
            [],  # Event breakdown
            []   # Failing endpoints
        ]

        # Mock health calculation
        self.service._calculate_tenant_health = AsyncMock(return_value=92.0)

        summary = await self.service.get_tenant_webhook_summary(
            self.tenant_id,
            time_window_hours=24
        )

        assert summary["tenant_id"] == self.tenant_id
        assert summary["health_score"] == 92.0

    @pytest.mark.asyncio
    async def test_get_tenant_endpoints_summary(self):
        """Test getting tenant endpoints summary."""
        self.mock_database.query_items.return_value = [{
            "total_endpoints": 10,
            "enabled_endpoints": 8,
            "circuit_open_endpoints": 1
        }]

        summary = await self.service._get_tenant_endpoints_summary(self.tenant_id)

        assert summary["total_endpoints"] == 10
        assert summary["enabled_endpoints"] == 8
        assert summary["circuit_open_endpoints"] == 1

    # Performance Metrics Tests (2 tests)

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        self.mock_database.query_items.return_value = [{
            "min_response_time": 50,
            "max_response_time": 500,
            "avg_response_time": 150,
            "median_response_time": 140,
            "p95_response_time": 300,
            "p99_response_time": 450
        }]

        metrics = await self.service._get_performance_metrics(
            "WHERE tenant_id = %s",
            [{"name": "@tenant_id", "value": self.tenant_id}]
        )

        assert metrics["avg_response_time_ms"] == 150
        assert metrics["p95_response_time_ms"] == 300
        assert metrics["p99_response_time_ms"] == 450

    @pytest.mark.asyncio
    async def test_get_daily_trends(self):
        """Test getting daily trends."""
        self.mock_database.query_items.return_value = [
            {
                "date": datetime.utcnow().date(),
                "total_deliveries": 100,
                "successful_deliveries": 95,
                "failed_deliveries": 5,
                "avg_response_time": 150
            }
        ]

        trends = await self.service._get_daily_trends(
            "WHERE tenant_id = %s",
            [{"name": "@tenant_id", "value": self.tenant_id}],
            datetime.utcnow() - timedelta(days=7),
            datetime.utcnow()
        )

        assert len(trends) == 1
        assert trends[0]["success_rate"] == 95.0

    # Digest Generation Tests (2 tests)

    @pytest.mark.asyncio
    async def test_generate_stats_digest(self):
        """Test generating statistics digest."""
        # Mock current and previous period stats
        self.mock_database.query_items.side_effect = [
            # Current period
            [{"total_deliveries": 100, "successful_deliveries": 95}],
            [], [], [{}], [],
            # Previous period
            [{"total_deliveries": 80, "successful_deliveries": 72}],
            [], [], [{}], []
        ]

        # Mock alerts and recommendations
        self.service._generate_alerts = AsyncMock(return_value=[])
        self.service._generate_recommendations = AsyncMock(return_value=[])

        digest = await self.service.generate_stats_digest(
            self.tenant_id,
            period_days=1
        )

        assert digest["tenant_id"] == self.tenant_id
        assert "current_period" in digest
        assert "previous_period" in digest
        assert "changes" in digest

    @pytest.mark.asyncio
    async def test_generate_alerts(self):
        """Test alert generation based on statistics."""
        stats = {
            "overall_stats": {
                "success_rate": 75.0,  # Low success rate
                "avg_response_time_ms": 6000,  # High response time
                "dead_letter_deliveries": 10  # Some dead letters
            }
        }

        alerts = await self.service._generate_alerts(self.tenant_id, stats)

        # Should have alerts for low success rate, high response time, and dead letters
        assert len(alerts) == 3
        assert any(a["type"] == "low_success_rate" for a in alerts)
        assert any(a["type"] == "high_response_time" for a in alerts)
        assert any(a["type"] == "dead_letters" for a in alerts)

    # Health Check Test (1 test)

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Test health check returns service status."""
        self.service._get_total_deliveries = AsyncMock(return_value=10000)
        self.service._get_active_endpoints_count = AsyncMock(return_value=15)

        health = await self.service.check_health()

        assert health["status"] == "healthy"
        assert health["service"] == "WebhookStatsService"
        assert health["total_deliveries_tracked"] == 10000
        assert health["active_endpoints"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])