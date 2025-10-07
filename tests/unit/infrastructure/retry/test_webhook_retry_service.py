"""Unit tests for WebhookRetryService."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.infrastructure.retry.webhook_retry_service import (
    CircuitBreakerState,
    EnhancedWebhookDeliveryService as WebhookRetryService,
)


@pytest.fixture
def webhook_retry_service():
    """Create webhook retry service instance."""
    return WebhookRetryService()


@pytest.mark.asyncio
class TestWebhookRetryService:
    """Test WebhookRetryService functionality."""

    @patch("app.infrastructure.retry.webhook_retry_service.get_retry_service")
    async def test_send_webhook_with_retry_success(
        self, mock_get_retry_service, webhook_retry_service
    ):
        """Test successful webhook send."""
        mock_retry_service = AsyncMock()
        mock_get_retry_service.return_value = mock_retry_service

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="OK")

            mock_session.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await webhook_retry_service.send_webhook_with_retry(
                url="https://example.com/webhook",
                payload={"test": "data"},
                tenant_id=str(uuid4())
            )

            assert result["success"] is True
            assert result["status_code"] == 200

    @patch("app.infrastructure.retry.webhook_retry_service.get_retry_service")
    async def test_circuit_breaker_opens_on_failures(
        self, mock_get_retry_service, webhook_retry_service
    ):
        """Test circuit breaker opens after failures."""
        endpoint_url = "https://example.com/webhook"

        # Simulate multiple failures
        for _ in range(6):  # More than failure_threshold (5)
            await webhook_retry_service._record_failure(endpoint_url)

        circuit_state = await webhook_retry_service.get_circuit_state(endpoint_url)

        assert circuit_state["circuit_state"] == CircuitBreakerState.OPEN.value

    @patch("app.infrastructure.retry.webhook_retry_service.get_retry_service")
    async def test_circuit_breaker_half_open_after_timeout(
        self, mock_get_retry_service, webhook_retry_service
    ):
        """Test circuit breaker transitions to half-open."""
        endpoint_url = "https://example.com/webhook"

        # Open the circuit
        for _ in range(6):
            await webhook_retry_service._record_failure(endpoint_url)

        # Manually set last failure time to past
        if endpoint_url in webhook_retry_service._circuit_breakers:
            webhook_retry_service._circuit_breakers[endpoint_url]["last_failure_time"] = (
                datetime.utcnow() - timedelta(seconds=61)
            )

        circuit_state = await webhook_retry_service.get_circuit_state(endpoint_url)

        assert circuit_state["circuit_state"] == CircuitBreakerState.HALF_OPEN.value

    @patch("app.infrastructure.retry.webhook_retry_service.get_retry_service")
    async def test_circuit_breaker_closes_on_success(
        self, mock_get_retry_service, webhook_retry_service
    ):
        """Test circuit breaker closes after successful requests."""
        endpoint_url = "https://example.com/webhook"

        # Record some failures
        for _ in range(3):
            await webhook_retry_service._record_failure(endpoint_url)

        # Record successes
        for _ in range(2):  # success_threshold is 2
            await webhook_retry_service._record_success(endpoint_url)

        circuit_state = await webhook_retry_service.get_circuit_state(endpoint_url)

        assert circuit_state["circuit_state"] == CircuitBreakerState.CLOSED.value

    @patch("app.infrastructure.retry.webhook_retry_service.get_retry_service")
    async def test_send_webhook_circuit_breaker_open(
        self, mock_get_retry_service, webhook_retry_service
    ):
        """Test webhook send when circuit breaker is open."""
        endpoint_url = "https://example.com/webhook"

        # Open circuit breaker
        for _ in range(6):
            await webhook_retry_service._record_failure(endpoint_url)

        result = await webhook_retry_service.send_webhook_with_retry(
            url=endpoint_url,
            payload={"test": "data"},
            tenant_id=str(uuid4())
        )

        assert result["success"] is False
        assert "circuit breaker" in result["error"].lower()

    @patch("app.infrastructure.retry.webhook_retry_service.get_retry_service")
    async def test_get_webhook_history(
        self, mock_get_retry_service, webhook_retry_service
    ):
        """Test getting webhook send history."""
        mock_retry_service = AsyncMock()
        mock_get_retry_service.return_value = mock_retry_service

        mock_history = [
            MagicMock(
                id=str(uuid4()),
                created_at=datetime.utcnow(),
                status="success"
            )
            for _ in range(5)
        ]

        with patch.object(
            webhook_retry_service,
            "_get_webhook_history_from_db",
            return_value=mock_history
        ):
            history = await webhook_retry_service.get_webhook_history(
                endpoint_url="https://example.com/webhook",
                tenant_id=str(uuid4())
            )

            assert len(history) == 5

    async def test_reset_circuit_breaker(self, webhook_retry_service):
        """Test resetting circuit breaker."""
        endpoint_url = "https://example.com/webhook"

        # Open circuit breaker
        for _ in range(6):
            await webhook_retry_service._record_failure(endpoint_url)

        # Reset circuit breaker
        await webhook_retry_service.reset_circuit_breaker(endpoint_url)

        circuit_state = await webhook_retry_service.get_circuit_state(endpoint_url)

        assert circuit_state["circuit_state"] == CircuitBreakerState.CLOSED.value
        assert circuit_state["failure_count"] == 0

    async def test_get_all_circuit_states(self, webhook_retry_service):
        """Test getting all circuit breaker states."""
        # Create circuit breakers for multiple endpoints
        endpoints = [
            "https://example.com/webhook1",
            "https://example.com/webhook2",
            "https://example.com/webhook3"
        ]

        for endpoint in endpoints:
            await webhook_retry_service._record_failure(endpoint)

        states = await webhook_retry_service.get_all_circuit_states()

        assert len(states) >= 3
        assert all(endpoint in states for endpoint in endpoints)

    async def test_check_health(self, webhook_retry_service):
        """Test health check."""
        health = await webhook_retry_service.check_health()

        assert "status" in health
        assert "circuit_breakers" in health
        assert health["status"] == "healthy"