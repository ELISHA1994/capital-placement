"""
Comprehensive tests for ReliableWebhookNotificationService.

This test module covers reliable webhook notification delivery,
audit integration, and notification management.
"""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.infrastructure.webhook.reliable_notification_adapter import (
    ReliableWebhookNotificationService,
    WebhookNotificationMixin
)
from app.api.schemas.webhook_schemas import WebhookEventType
from app.domain.interfaces import WebhookDeliveryResult


class TestReliableWebhookNotificationService:
    """Test reliable webhook notification service comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_delivery_service = AsyncMock()
        self.mock_audit_service = AsyncMock()

        self.service = ReliableWebhookNotificationService(
            webhook_delivery_service=self.mock_delivery_service,
            audit_service=self.mock_audit_service
        )

        self.test_url = "https://example.com/webhook"
        self.test_payload = {"event": "test", "data": {"id": 123}}

    # Email Notification Tests (2 tests)

    @pytest.mark.asyncio
    async def test_send_email_success(self):
        """Test email notification sending."""
        result = await self.service.send_email(
            to="test@example.com",
            subject="Test Subject",
            body="Test body content",
            is_html=False
        )

        assert result is True
        self.mock_audit_service.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_logs_audit(self):
        """Test email sending logs audit event."""
        await self.service.send_email(
            to="test@example.com",
            subject="Test",
            body="Body"
        )

        # Verify audit log was created
        assert self.mock_audit_service.log_event.called
        call_args = self.mock_audit_service.log_event.call_args
        assert call_args[1]["event_type"] == "notification.email.sent"

    # Webhook Notification Tests (3 tests)

    @pytest.mark.asyncio
    async def test_send_webhook_success(self):
        """Test successful webhook notification."""
        # Mock successful delivery
        self.mock_delivery_service.deliver_webhook_immediate.return_value = (
            WebhookDeliveryResult(
                success=True,
                status_code=200,
                response_time_ms=150,
                signature_verified=True
            )
        )

        result = await self.service.send_webhook(
            url=self.test_url,
            payload=self.test_payload,
            secret="test_secret"
        )

        assert result is True
        self.mock_delivery_service.deliver_webhook_immediate.assert_called_once()
        self.mock_audit_service.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_failure(self):
        """Test webhook notification failure."""
        # Mock failed delivery
        self.mock_delivery_service.deliver_webhook_immediate.return_value = (
            WebhookDeliveryResult(
                success=False,
                status_code=500,
                error_message="Internal Server Error",
                failure_reason="http_error"
            )
        )

        result = await self.service.send_webhook(
            url=self.test_url,
            payload=self.test_payload
        )

        assert result is False

        # Verify audit log for failure
        assert self.mock_audit_service.log_event.called
        call_args = self.mock_audit_service.log_event.call_args
        assert call_args[1]["event_type"] == "notification.webhook.failed"

    @pytest.mark.asyncio
    async def test_send_webhook_with_signature(self):
        """Test webhook sending includes signature."""
        self.mock_delivery_service.deliver_webhook_immediate.return_value = (
            WebhookDeliveryResult(
                success=True,
                status_code=200,
                response_time_ms=150,
                signature_verified=True
            )
        )

        await self.service.send_webhook(
            url=self.test_url,
            payload=self.test_payload,
            secret="webhook_secret"
        )

        # Verify secret was passed to delivery service
        call_args = self.mock_delivery_service.deliver_webhook_immediate.call_args
        assert call_args[1]["secret"] == "webhook_secret"

    # Reliable Webhook Tests (3 tests)

    @pytest.mark.asyncio
    async def test_send_reliable_webhook_queues_delivery(self):
        """Test reliable webhook queues delivery correctly."""
        delivery_id = str(uuid4())
        endpoint_id = str(uuid4())
        tenant_id = str(uuid4())

        self.mock_delivery_service.deliver_webhook.return_value = delivery_id

        result = await self.service.send_reliable_webhook(
            endpoint_id=endpoint_id,
            event_type=WebhookEventType.DOCUMENT_PROCESSED,
            payload=self.test_payload,
            tenant_id=tenant_id
        )

        assert result == delivery_id
        self.mock_delivery_service.deliver_webhook.assert_called_once()
        self.mock_audit_service.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_reliable_webhook_with_priority(self):
        """Test reliable webhook with custom priority."""
        delivery_id = str(uuid4())
        endpoint_id = str(uuid4())
        tenant_id = str(uuid4())

        self.mock_delivery_service.deliver_webhook.return_value = delivery_id

        await self.service.send_reliable_webhook(
            endpoint_id=endpoint_id,
            event_type=WebhookEventType.DOCUMENT_PROCESSED,
            payload=self.test_payload,
            tenant_id=tenant_id,
            priority=10
        )

        # Verify priority was passed
        call_args = self.mock_delivery_service.deliver_webhook.call_args
        assert call_args[1]["priority"] == 10

    @pytest.mark.asyncio
    async def test_send_reliable_webhook_logs_audit(self):
        """Test reliable webhook logs audit event."""
        delivery_id = str(uuid4())
        endpoint_id = str(uuid4())
        tenant_id = str(uuid4())

        self.mock_delivery_service.deliver_webhook.return_value = delivery_id

        await self.service.send_reliable_webhook(
            endpoint_id=endpoint_id,
            event_type=WebhookEventType.DOCUMENT_PROCESSED,
            payload=self.test_payload,
            tenant_id=tenant_id
        )

        # Verify audit log
        assert self.mock_audit_service.log_event.called
        call_args = self.mock_audit_service.log_event.call_args
        assert call_args[1]["event_type"] == "notification.webhook.queued"

    # Push Notification Tests (1 test)

    @pytest.mark.asyncio
    async def test_send_push_notification(self):
        """Test push notification sending."""
        result = await self.service.send_push_notification(
            user_id="user_123",
            title="Test Title",
            message="Test message",
            data={"extra": "data"}
        )

        assert result is True
        self.mock_audit_service.log_event.assert_called_once()

    # Delivery Status Tests (1 test)

    @pytest.mark.asyncio
    async def test_get_webhook_delivery_status(self):
        """Test getting webhook delivery status."""
        delivery_id = str(uuid4())

        self.mock_delivery_service.get_delivery_status.return_value = {
            "delivery_id": delivery_id,
            "status": "delivered",
            "attempt_number": 1
        }

        status = await self.service.get_webhook_delivery_status(delivery_id)

        assert status is not None
        assert status["delivery_id"] == delivery_id
        self.mock_delivery_service.get_delivery_status.assert_called_once_with(delivery_id)

    # Retry Tests (1 test)

    @pytest.mark.asyncio
    async def test_retry_failed_webhook(self):
        """Test retrying failed webhook delivery."""
        delivery_id = str(uuid4())
        admin_user_id = "admin_123"

        self.mock_delivery_service.retry_failed_delivery.return_value = True

        success = await self.service.retry_failed_webhook(
            delivery_id=delivery_id,
            admin_user_id=admin_user_id,
            notes="Manual retry"
        )

        assert success is True
        self.mock_delivery_service.retry_failed_delivery.assert_called_once()
        self.mock_audit_service.log_event.assert_called_once()

    # Health Check Test (1 test)

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Test health check returns service status."""
        self.mock_delivery_service.check_health.return_value = {
            "status": "healthy",
            "pending_deliveries": 10
        }

        health = await self.service.check_health()

        assert health["status"] == "healthy"
        assert health["service"] == "ReliableWebhookNotificationService"
        assert "webhook_delivery_service" in health


class TestWebhookNotificationMixin:
    """Test webhook notification mixin functionality."""

    def test_mixin_initialization(self):
        """Test mixin initializes correctly."""
        class TestClass(WebhookNotificationMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()
        assert instance._reliable_webhook_service is None

    @pytest.mark.asyncio
    async def test_mixin_send_reliable_webhook_with_service(self):
        """Test mixin sends webhook when service is set."""
        class TestClass(WebhookNotificationMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()
        mock_service = AsyncMock()
        mock_service.send_reliable_webhook.return_value = str(uuid4())

        instance.set_reliable_webhook_service(mock_service)

        delivery_id = await instance.send_reliable_webhook(
            endpoint_id=str(uuid4()),
            event_type=WebhookEventType.DOCUMENT_PROCESSED,
            payload={"test": "data"},
            tenant_id=str(uuid4())
        )

        assert delivery_id is not None
        mock_service.send_reliable_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixin_send_reliable_webhook_without_service(self):
        """Test mixin returns None when service not set."""
        class TestClass(WebhookNotificationMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        delivery_id = await instance.send_reliable_webhook(
            endpoint_id=str(uuid4()),
            event_type=WebhookEventType.DOCUMENT_PROCESSED,
            payload={"test": "data"},
            tenant_id=str(uuid4())
        )

        assert delivery_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])