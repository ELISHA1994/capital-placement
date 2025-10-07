"""Unit tests for RetryService."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.domain.retry import BackoffStrategy, ErrorCategory, RetryPolicy, RetryResult
from app.infrastructure.retry.retry_service import RetryService


@pytest.fixture
def mock_error_classifier():
    """Mock error classifier."""
    classifier = AsyncMock()
    classifier.classify_error = AsyncMock(return_value=ErrorCategory.TRANSIENT)
    classifier.check_health = AsyncMock(return_value={"status": "healthy"})
    return classifier


@pytest.fixture
def retry_service(mock_error_classifier):
    """Create retry service instance."""
    return RetryService(error_classifier=mock_error_classifier)


@pytest.fixture
def default_policy():
    """Create default retry policy."""
    return RetryPolicy(
        max_attempts=3,
        initial_delay_seconds=1.0,
        max_delay_seconds=60.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        backoff_multiplier=2.0,
        jitter_enabled=True,
        dead_letter_enabled=True
    )


@pytest.mark.asyncio
class TestRetryService:
    """Test RetryService functionality."""

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_create_retry_state(
        self, mock_get_adapter, retry_service, default_policy
    ):
        """Test creating retry state."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.execute = AsyncMock(return_value=str(uuid4()))

        retry_id = await retry_service.create_retry_state(
            operation_id="test_op",
            operation_type="test_type",
            tenant_id=str(uuid4()),
            policy=default_policy
        )

        assert retry_id is not None
        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_record_attempt_success(
        self, mock_get_adapter, retry_service
    ):
        """Test recording successful attempt."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        retry_state = MagicMock(
            id=str(uuid4()),
            current_attempt=1,
            max_attempts=3,
            status=RetryResult.FAILED
        )
        mock_adapter.fetch_one = AsyncMock(return_value=retry_state)
        mock_adapter.execute = AsyncMock()

        await retry_service.record_attempt(
            retry_id=retry_state.id,
            success=True,
            operation_duration_ms=100
        )

        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_record_attempt_failure(
        self, mock_get_adapter, retry_service, mock_error_classifier
    ):
        """Test recording failed attempt."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        retry_state = MagicMock(
            id=str(uuid4()),
            current_attempt=1,
            max_attempts=3,
            status=RetryResult.FAILED,
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL
        )
        mock_adapter.fetch_one = AsyncMock(return_value=retry_state)
        mock_adapter.execute = AsyncMock()

        error = Exception("Test error")
        await retry_service.record_attempt(
            retry_id=retry_state.id,
            success=False,
            error=error,
            operation_duration_ms=100
        )

        assert mock_error_classifier.classify_error.called
        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_get_retry_state(
        self, mock_get_adapter, retry_service
    ):
        """Test getting retry state."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        retry_id = str(uuid4())
        mock_state = MagicMock(
            id=retry_id,
            operation_id="test_op",
            status=RetryResult.FAILED
        )
        mock_adapter.fetch_one = AsyncMock(return_value=mock_state)

        state = await retry_service.get_retry_state(retry_id)

        assert state is not None
        assert mock_adapter.fetch_one.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_calculate_next_attempt_exponential(
        self, mock_get_adapter, retry_service
    ):
        """Test exponential backoff calculation."""
        delay = retry_service._calculate_next_attempt_delay(
            attempt_number=2,
            initial_delay=1.0,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter_enabled=False
        )

        # 2nd attempt: 1.0 * 2^1 = 2.0 seconds
        assert delay == 2.0

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_calculate_next_attempt_linear(
        self, mock_get_adapter, retry_service
    ):
        """Test linear backoff calculation."""
        delay = retry_service._calculate_next_attempt_delay(
            attempt_number=3,
            initial_delay=1.0,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.LINEAR,
            backoff_multiplier=2.0,
            jitter_enabled=False
        )

        # 3rd attempt: 1.0 * 2.0 * 2 = 4.0 seconds
        assert delay == 4.0

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_calculate_next_attempt_fixed(
        self, mock_get_adapter, retry_service
    ):
        """Test fixed delay calculation."""
        delay = retry_service._calculate_next_attempt_delay(
            attempt_number=5,
            initial_delay=2.0,
            max_delay=60.0,
            backoff_strategy=BackoffStrategy.FIXED,
            backoff_multiplier=2.0,
            jitter_enabled=False
        )

        # Fixed delay is always initial_delay
        assert delay == 2.0

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_calculate_next_attempt_max_delay(
        self, mock_get_adapter, retry_service
    ):
        """Test that delay respects max_delay."""
        delay = retry_service._calculate_next_attempt_delay(
            attempt_number=10,  # Would calculate very large delay
            initial_delay=1.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter_enabled=False
        )

        # Should be capped at max_delay
        assert delay <= 30.0

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_get_ready_retries(
        self, mock_get_adapter, retry_service
    ):
        """Test getting ready retries."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        mock_retries = [
            MagicMock(id=str(uuid4()), next_attempt_at=datetime.utcnow() - timedelta(seconds=10))
            for _ in range(5)
        ]
        mock_adapter.fetch_all = AsyncMock(return_value=mock_retries)

        retries = await retry_service.get_ready_retries(limit=10)

        assert len(retries) == 5
        assert mock_adapter.fetch_all.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_get_retry_statistics(
        self, mock_get_adapter, retry_service
    ):
        """Test getting retry statistics."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter

        mock_stats = {
            "total_retry_states": 100,
            "successful_retries": 80,
            "failed_retries": 15,
            "max_attempts_exceeded": 5
        }
        mock_adapter.fetch_one = AsyncMock(return_value=mock_stats)

        stats = await retry_service.get_retry_statistics(
            start_time=datetime.utcnow() - timedelta(hours=24),
            end_time=datetime.utcnow()
        )

        assert stats is not None
        assert mock_adapter.fetch_one.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_cancel_retry(
        self, mock_get_adapter, retry_service
    ):
        """Test cancelling retry."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.execute = AsyncMock()

        retry_id = str(uuid4())
        await retry_service.cancel_retry(retry_id, reason="User cancelled")

        assert mock_adapter.execute.called

    @patch("app.infrastructure.retry.retry_service.get_postgres_adapter")
    async def test_check_health(
        self, mock_get_adapter, retry_service, mock_error_classifier
    ):
        """Test health check."""
        mock_adapter = AsyncMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.fetch_one = AsyncMock(return_value={"count": 10})

        health = await retry_service.check_health()

        assert "status" in health
        assert health["status"] == "healthy"