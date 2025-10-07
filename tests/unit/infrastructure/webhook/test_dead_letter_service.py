"""
Comprehensive tests for WebhookDeadLetterService.

This test module covers dead letter queue management, retry functionality,
and cleanup operations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from app.infrastructure.webhook.dead_letter_service import WebhookDeadLetterService
from app.api.schemas.webhook_schemas import WebhookFailureReason


class TestWebhookDeadLetterService:
    """Test webhook dead letter service comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_database = AsyncMock()
        self.service = WebhookDeadLetterService(self.mock_database)
        self.delivery_id = str(uuid4())
        self.tenant_id = str(uuid4())

    # Move to Dead Letter Tests (3 tests)

    @pytest.mark.asyncio
    async def test_move_to_dead_letter_success(self):
        """Test moving delivery to dead letter queue."""
        endpoint_id = str(uuid4())

        self.mock_database.get_item.return_value = {
            "id": self.delivery_id,
            "tenant_id": self.tenant_id,
            "endpoint_id": endpoint_id,
            "event_type": "document.processed",
            "event_id": "event_123",
            "payload": {"test": "data"},
            "attempt_number": 5
        }

        dead_letter_id = await self.service.move_to_dead_letter(
            delivery_id=self.delivery_id,
            final_failure_reason=WebhookFailureReason.MAX_RETRIES_EXCEEDED.value,
            final_error_message="Max retries exceeded"
        )

        assert dead_letter_id is not None
        self.mock_database.create_item.assert_called_once()
        self.mock_database.update_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_to_dead_letter_with_metadata(self):
        """Test moving to dead letter preserves all metadata."""
        endpoint_id = str(uuid4())
        first_attempted_at = datetime.utcnow() - timedelta(hours=1)

        self.mock_database.get_item.return_value = {
            "id": self.delivery_id,
            "tenant_id": self.tenant_id,
            "endpoint_id": endpoint_id,
            "event_type": "document.processed",
            "event_id": "event_123",
            "payload": {"test": "data"},
            "attempt_number": 5,
            "first_attempted_at": first_attempted_at,
            "scheduled_at": first_attempted_at - timedelta(minutes=30)
        }

        dead_letter_id = await self.service.move_to_dead_letter(
            delivery_id=self.delivery_id,
            final_failure_reason=WebhookFailureReason.MAX_RETRIES_EXCEEDED.value,
            moved_by="system"
        )

        assert dead_letter_id is not None

        # Check that dead letter record was created with correct data
        create_call = self.mock_database.create_item.call_args
        dead_letter_record = create_call[0][1]
        assert dead_letter_record["total_attempts"] == 5
        assert dead_letter_record["can_retry"] is True

    @pytest.mark.asyncio
    async def test_move_to_dead_letter_delivery_not_found(self):
        """Test moving to dead letter when delivery not found."""
        self.mock_database.get_item.return_value = None

        with pytest.raises(ValueError) as exc_info:
            await self.service.move_to_dead_letter(
                delivery_id=self.delivery_id,
                final_failure_reason=WebhookFailureReason.MAX_RETRIES_EXCEEDED.value
            )

        assert "not found" in str(exc_info.value).lower()

    # Retry Dead Letter Tests (2 tests)

    @pytest.mark.asyncio
    async def test_retry_dead_letter_success(self):
        """Test retrying dead letter delivery."""
        dead_letter_id = str(uuid4())
        endpoint_id = str(uuid4())

        self.mock_database.get_item.return_value = {
            "id": dead_letter_id,
            "tenant_id": self.tenant_id,
            "endpoint_id": endpoint_id,
            "event_type": "document.processed",
            "event_id": "event_123",
            "payload": {"test": "data"},
            "can_retry": True,
            "retry_count": 0
        }

        new_delivery_id = await self.service.retry_dead_letter(
            dead_letter_id=dead_letter_id,
            admin_user_id="admin_123",
            notes="Manual retry"
        )

        assert new_delivery_id is not None
        self.mock_database.create_item.assert_called_once()
        self.mock_database.update_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_dead_letter_cannot_retry(self):
        """Test retrying dead letter that cannot be retried."""
        dead_letter_id = str(uuid4())

        self.mock_database.get_item.return_value = {
            "id": dead_letter_id,
            "can_retry": False
        }

        with pytest.raises(ValueError) as exc_info:
            await self.service.retry_dead_letter(
                dead_letter_id=dead_letter_id,
                admin_user_id="admin_123"
            )

        assert "cannot be retried" in str(exc_info.value).lower()

    # Resolve Dead Letter Tests (2 tests)

    @pytest.mark.asyncio
    async def test_resolve_dead_letter_success(self):
        """Test resolving dead letter."""
        dead_letter_id = str(uuid4())

        self.mock_database.get_item.return_value = {
            "id": dead_letter_id,
            "tenant_id": self.tenant_id
        }

        success = await self.service.resolve_dead_letter(
            dead_letter_id=dead_letter_id,
            resolution_action="fixed_endpoint",
            admin_user_id="admin_123",
            notes="Fixed endpoint URL"
        )

        assert success is True
        self.mock_database.update_item.assert_called_once()

        # Check that can_retry was set to False
        update_call = self.mock_database.update_item.call_args
        updates = update_call[0][2]
        assert updates["can_retry"] is False

    @pytest.mark.asyncio
    async def test_resolve_dead_letter_not_found(self):
        """Test resolving non-existent dead letter."""
        dead_letter_id = str(uuid4())

        self.mock_database.get_item.return_value = None

        success = await self.service.resolve_dead_letter(
            dead_letter_id=dead_letter_id,
            resolution_action="fixed_endpoint",
            admin_user_id="admin_123"
        )

        assert success is False

    # Get Dead Letters Tests (2 tests)

    @pytest.mark.asyncio
    async def test_get_dead_letters_with_filters(self):
        """Test getting dead letters with filters."""
        endpoint_id = str(uuid4())

        # Mock count query
        self.mock_database.query_items.side_effect = [
            [{"total": 10}],  # Count query
            [{"id": "dl1"}, {"id": "dl2"}]  # Data query
        ]

        result = await self.service.get_dead_letters(
            tenant_id=self.tenant_id,
            endpoint_id=endpoint_id,
            limit=50,
            offset=0
        )

        assert result["pagination"]["total_count"] == 10
        assert len(result["dead_letters"]) == 2
        assert result["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_get_dead_letters_pagination(self):
        """Test dead letters pagination."""
        self.mock_database.query_items.side_effect = [
            [{"total": 100}],  # Count query
            [{"id": f"dl{i}"} for i in range(50)]  # Data query (50 items)
        ]

        result = await self.service.get_dead_letters(
            limit=50,
            offset=0
        )

        assert result["pagination"]["total_count"] == 100
        assert result["pagination"]["returned_count"] == 50
        assert result["pagination"]["has_more"] is True
        assert result["pagination"]["next_offset"] == 50

    # Cleanup Tests (2 tests)

    @pytest.mark.asyncio
    async def test_cleanup_old_dead_letters(self):
        """Test cleanup of old dead letters."""
        # Mock old dead letters
        self.mock_database.query_items.side_effect = [
            [{"id": "dl1"}, {"id": "dl2"}],  # First batch
            []  # No more batches
        ]

        cleaned_count = await self.service.cleanup_old_dead_letters(
            older_than_days=30,
            batch_size=100
        )

        assert cleaned_count == 2
        assert self.mock_database.delete_item.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_empty_queue(self):
        """Test cleanup when no old entries exist."""
        self.mock_database.query_items.return_value = []

        cleaned_count = await self.service.cleanup_old_dead_letters(
            older_than_days=30
        )

        assert cleaned_count == 0
        self.mock_database.delete_item.assert_not_called()

    # Summary Tests (1 test)

    @pytest.mark.asyncio
    async def test_get_dead_letter_summary(self):
        """Test getting dead letter summary statistics."""
        self.mock_database.query_items.side_effect = [
            [{
                "total_dead_letters": 50,
                "resolved_count": 30,
                "retryable_count": 15,
                "retry_attempted_count": 10,
                "avg_attempts_before_dead_letter": 4.5,
                "max_attempts_before_dead_letter": 5
            }],
            [{"final_failure_reason": "timeout", "count": 20}],
            [{"endpoint_id": "ep1", "count": 15}]
        ]

        summary = await self.service.get_dead_letter_summary(
            tenant_id=self.tenant_id,
            days_back=7
        )

        assert summary["summary"]["total_dead_letters"] == 50
        assert summary["summary"]["resolved_count"] == 30
        assert len(summary["failure_reason_breakdown"]) > 0

    # Health Check Test (1 test)

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Test health check returns service status."""
        self.service._get_dead_letter_count = AsyncMock(return_value=50)
        self.service._get_unresolved_dead_letter_count = AsyncMock(return_value=20)

        health = await self.service.check_health()

        assert health["status"] == "healthy"
        assert health["service"] == "WebhookDeadLetterService"
        assert health["total_dead_letters"] == 50
        assert health["unresolved_dead_letters"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])