"""
Comprehensive tests for WebhookCircuitBreakerService.

This test module covers circuit breaker state transitions, failure tracking,
and automatic recovery mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.infrastructure.webhook.circuit_breaker_service import WebhookCircuitBreakerService
from app.api.schemas.webhook_schemas import CircuitBreakerState


class TestWebhookCircuitBreakerService:
    """Test webhook circuit breaker service comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_database = AsyncMock()
        self.service = WebhookCircuitBreakerService(self.mock_database)
        self.endpoint_id = str(uuid4())

    # State Transition Tests (6 tests)

    @pytest.mark.asyncio
    async def test_circuit_closed_to_open_on_threshold(self):
        """Test circuit opens when failure threshold is exceeded."""
        # Mock endpoint with closed circuit near threshold
        self.mock_database.get_item.return_value = {
            "circuit_state": "closed",
            "failure_count": 4,
            "retry_policy": {"failure_threshold": 5}
        }

        # Load initial state
        state = await self.service._get_circuit_state(self.endpoint_id)
        assert state["state"] == CircuitBreakerState.CLOSED

        # Record one more failure to exceed threshold
        await self.service.record_failure(self.endpoint_id, "timeout")

        # Circuit should now be open
        state = self.service._circuit_states[self.endpoint_id]
        assert state["state"] == CircuitBreakerState.OPEN
        assert state["failure_count"] == 5

    @pytest.mark.asyncio
    async def test_circuit_open_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after recovery timeout."""
        # Set up open circuit with recovery time in past
        past_time = datetime.utcnow() - timedelta(seconds=120)

        self.mock_database.get_item.return_value = {
            "circuit_state": "open",
            "circuit_opened_at": past_time,
            "retry_policy": {"recovery_timeout_seconds": 60}
        }

        # Load state
        await self.service._get_circuit_state(self.endpoint_id)

        # Check if request should be allowed
        should_allow = await self.service.should_allow_request(self.endpoint_id)

        # Should transition to half-open and allow request
        assert should_allow is True
        state = self.service._circuit_states[self.endpoint_id]
        assert state["state"] == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_half_open_to_closed_on_success(self):
        """Test circuit closes after successful half-open tests."""
        # Set up half-open circuit
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.HALF_OPEN,
            "half_open_successes": 1,
            "min_half_open_successes": 2,
            "total_calls": 10,
            "successful_calls": 8,
            "failure_count": 2
        }

        # Record another success
        await self.service.record_success(self.endpoint_id, 150)

        # Circuit should now be closed
        state = self.service._circuit_states[self.endpoint_id]
        assert state["state"] == CircuitBreakerState.CLOSED
        assert state["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_circuit_half_open_to_open_on_failure(self):
        """Test circuit reopens on failure during half-open state."""
        # Set up half-open circuit
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.HALF_OPEN,
            "half_open_attempts": 1,
            "half_open_successes": 0,
            "total_calls": 10,
            "failed_calls": 5,
            "failure_count": 3
        }

        # Record failure
        await self.service.record_failure(self.endpoint_id, "timeout")

        # Circuit should reopen
        state = self.service._circuit_states[self.endpoint_id]
        assert state["state"] == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_remains_closed_below_threshold(self):
        """Test circuit remains closed when failures are below threshold."""
        # Mock endpoint with some failures but below threshold
        self.mock_database.get_item.return_value = {
            "circuit_state": "closed",
            "failure_count": 2,
            "retry_policy": {"failure_threshold": 5}
        }

        await self.service._get_circuit_state(self.endpoint_id)

        # Record another failure
        await self.service.record_failure(self.endpoint_id, "timeout")

        # Circuit should remain closed
        state = self.service._circuit_states[self.endpoint_id]
        assert state["state"] == CircuitBreakerState.CLOSED
        assert state["failure_count"] == 3

    @pytest.mark.asyncio
    async def test_force_circuit_state_manual_override(self):
        """Test manually forcing circuit state."""
        # Set up initial closed state
        self.mock_database.get_item.return_value = {
            "circuit_state": "closed"
        }

        await self.service._get_circuit_state(self.endpoint_id)

        # Force circuit open
        success = await self.service.force_circuit_state(
            self.endpoint_id,
            "open",
            admin_user_id="admin_123",
            reason="Manual maintenance"
        )

        assert success is True
        state = self.service._circuit_states[self.endpoint_id]
        assert state["state"] == CircuitBreakerState.OPEN
        assert state["manual_override"] is True

    # Request Allowing Tests (4 tests)

    @pytest.mark.asyncio
    async def test_should_allow_request_when_closed(self):
        """Test requests allowed when circuit is closed."""
        self.mock_database.get_item.return_value = {
            "circuit_state": "closed",
            "failure_count": 0
        }

        should_allow = await self.service.should_allow_request(self.endpoint_id)
        assert should_allow is True

    @pytest.mark.asyncio
    async def test_should_deny_request_when_open(self):
        """Test requests denied when circuit is open."""
        self.mock_database.get_item.return_value = {
            "circuit_state": "open",
            "circuit_opened_at": datetime.utcnow(),
            "retry_policy": {"recovery_timeout_seconds": 60}
        }

        should_allow = await self.service.should_allow_request(self.endpoint_id)
        assert should_allow is False

    @pytest.mark.asyncio
    async def test_should_allow_limited_requests_when_half_open(self):
        """Test limited requests allowed when circuit is half-open."""
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.HALF_OPEN,
            "half_open_attempts": 0,
            "max_half_open_calls": 3
        }

        # First few requests should be allowed
        should_allow1 = await self.service.should_allow_request(self.endpoint_id)
        should_allow2 = await self.service.should_allow_request(self.endpoint_id)
        should_allow3 = await self.service.should_allow_request(self.endpoint_id)

        assert should_allow1 is True
        assert should_allow2 is True
        assert should_allow3 is True

        # Additional requests should be denied
        should_allow4 = await self.service.should_allow_request(self.endpoint_id)
        assert should_allow4 is False

    @pytest.mark.asyncio
    async def test_should_allow_request_on_database_error(self):
        """Test fail-open behavior when database has errors."""
        # Mock database error
        self.mock_database.get_item.side_effect = Exception("Database error")

        # Should fail open (allow requests)
        should_allow = await self.service.should_allow_request(self.endpoint_id)
        assert should_allow is True

    # Success/Failure Recording Tests (4 tests)

    @pytest.mark.asyncio
    async def test_record_success_updates_metrics(self):
        """Test success recording updates metrics correctly."""
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.CLOSED,
            "total_calls": 10,
            "successful_calls": 8,
            "failure_count": 2,
            "avg_response_time_ms": 100
        }

        await self.service.record_success(self.endpoint_id, 150)

        state = self.service._circuit_states[self.endpoint_id]
        assert state["total_calls"] == 11
        assert state["successful_calls"] == 9
        assert state["failure_count"] == 0  # Should reset

    @pytest.mark.asyncio
    async def test_record_failure_updates_metrics(self):
        """Test failure recording updates metrics correctly."""
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.CLOSED,
            "total_calls": 10,
            "failed_calls": 2,
            "failure_count": 2,
            "failure_threshold": 5
        }

        await self.service.record_failure(self.endpoint_id, "timeout")

        state = self.service._circuit_states[self.endpoint_id]
        assert state["total_calls"] == 11
        assert state["failed_calls"] == 3
        assert state["failure_count"] == 3

    @pytest.mark.asyncio
    async def test_record_success_calculates_avg_response_time(self):
        """Test success recording calculates average response time."""
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.CLOSED,
            "total_calls": 0,
            "successful_calls": 0,
            "avg_response_time_ms": 0
        }

        # Record multiple successes
        await self.service.record_success(self.endpoint_id, 100)
        await self.service.record_success(self.endpoint_id, 200)
        await self.service.record_success(self.endpoint_id, 300)

        state = self.service._circuit_states[self.endpoint_id]
        assert state["successful_calls"] == 3
        # Average should be approximately 200
        assert 190 <= state["avg_response_time_ms"] <= 210

    @pytest.mark.asyncio
    async def test_record_failure_stores_last_failure_reason(self):
        """Test failure recording stores last failure reason."""
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.CLOSED,
            "total_calls": 0,
            "failed_calls": 0,
            "failure_count": 0,
            "failure_threshold": 5
        }

        await self.service.record_failure(self.endpoint_id, "connection_error")

        state = self.service._circuit_states[self.endpoint_id]
        assert state["last_failure_reason"] == "connection_error"

    # Health Check Tests (1 test)

    @pytest.mark.asyncio
    async def test_check_health_returns_status(self):
        """Test health check returns service status."""
        # Add some circuits to cache
        self.service._circuit_states["endpoint1"] = {
            "state": CircuitBreakerState.CLOSED
        }
        self.service._circuit_states["endpoint2"] = {
            "state": CircuitBreakerState.OPEN
        }

        health = await self.service.check_health()

        assert health["status"] == "healthy"
        assert health["service"] == "WebhookCircuitBreakerService"
        assert health["active_circuits"] == 2
        assert health["open_circuits"] == 1

    # Get Circuit State Test (1 test)

    @pytest.mark.asyncio
    async def test_get_circuit_state_returns_current_state(self):
        """Test getting circuit state returns current information."""
        self.service._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.OPEN,
            "failure_count": 5,
            "total_calls": 20,
            "failed_calls": 7
        }

        state = await self.service.get_circuit_state(self.endpoint_id)

        assert state["state"] == CircuitBreakerState.OPEN
        assert state["failure_count"] == 5
        assert state["total_calls"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])