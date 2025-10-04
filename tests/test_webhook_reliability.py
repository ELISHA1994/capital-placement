"""
Tests for webhook reliability features.

This module tests the webhook delivery reliability system including
retry mechanisms, circuit breakers, dead letter queues, and monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.services.webhook.signature_service import WebhookSignatureService
from app.services.webhook.circuit_breaker_service import WebhookCircuitBreakerService
from app.services.webhook.delivery_service import WebhookDeliveryService
from app.services.webhook.dead_letter_service import WebhookDeadLetterService
from app.services.webhook.stats_service import WebhookStatsService
from app.services.webhook.reliable_notification_adapter import ReliableWebhookNotificationService
from app.models.webhook_models import (
    WebhookEventType,
    WebhookDeliveryStatus,
    WebhookFailureReason,
    CircuitBreakerState,
    RetryPolicy
)
from app.domain.interfaces import WebhookDeliveryResult


class TestWebhookSignatureService:
    """Test webhook signature generation and verification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.signature_service = WebhookSignatureService()
        self.test_payload = '{"event": "test", "data": "example"}'
        self.test_secret = "test_secret_key_123"
    
    def test_generate_signature(self):
        """Test signature generation."""
        signature = self.signature_service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )
        
        assert signature.startswith("sha256=")
        assert len(signature) > 10
    
    def test_verify_signature_success(self):
        """Test successful signature verification."""
        signature = self.signature_service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )
        
        is_valid = self.signature_service.verify_signature(
            self.test_payload, signature, self.test_secret, "sha256"
        )
        
        assert is_valid is True
    
    def test_verify_signature_failure(self):
        """Test failed signature verification."""
        signature = self.signature_service.generate_signature(
            self.test_payload, self.test_secret, "sha256"
        )
        
        # Use wrong secret
        is_valid = self.signature_service.verify_signature(
            self.test_payload, signature, "wrong_secret", "sha256"
        )
        
        assert is_valid is False
    
    def test_generate_timestamp_signature(self):
        """Test timestamped signature generation."""
        timestamp = int(datetime.utcnow().timestamp())
        
        signature = self.signature_service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )
        
        assert signature.startswith(f"t={timestamp},v1=")
    
    def test_verify_timestamp_signature_success(self):
        """Test successful timestamped signature verification."""
        timestamp = int(datetime.utcnow().timestamp())
        
        signature = self.signature_service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )
        
        is_valid = self.signature_service.verify_timestamp_signature(
            self.test_payload, signature, self.test_secret, tolerance_seconds=300
        )
        
        assert is_valid is True
    
    def test_verify_timestamp_signature_expired(self):
        """Test timestamped signature verification with expired timestamp."""
        # Use timestamp from 10 minutes ago
        timestamp = int((datetime.utcnow() - timedelta(minutes=10)).timestamp())
        
        signature = self.signature_service.generate_timestamp_signature(
            self.test_payload, timestamp, self.test_secret, "sha256"
        )
        
        is_valid = self.signature_service.verify_timestamp_signature(
            self.test_payload, signature, self.test_secret, tolerance_seconds=300  # 5 minutes
        )
        
        assert is_valid is False
    
    def test_generate_test_signature(self):
        """Test test signature generation."""
        payload, secret, signature = self.signature_service.generate_test_signature()
        
        assert payload is not None
        assert secret is not None
        assert signature is not None
        assert signature.startswith("sha256=")
        
        # Verify the test signature
        is_valid = self.signature_service.verify_signature(payload, signature, secret)
        assert is_valid is True


class TestWebhookCircuitBreakerService:
    """Test webhook circuit breaker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_database = AsyncMock()
        self.circuit_breaker = WebhookCircuitBreakerService(self.mock_database)
        self.endpoint_id = str(uuid4())
    
    @pytest.mark.asyncio
    async def test_should_allow_request_closed_circuit(self):
        """Test request allowed when circuit is closed."""
        # Mock endpoint data with closed circuit
        self.mock_database.get_item.return_value = {
            "circuit_state": "closed",
            "failure_count": 0,
            "retry_policy": {"failure_threshold": 5}
        }
        
        should_allow = await self.circuit_breaker.should_allow_request(self.endpoint_id)
        assert should_allow is True
    
    @pytest.mark.asyncio
    async def test_should_deny_request_open_circuit(self):
        """Test request denied when circuit is open."""
        # Mock endpoint data with open circuit
        self.mock_database.get_item.return_value = {
            "circuit_state": "open",
            "failure_count": 10,
            "circuit_opened_at": datetime.utcnow(),
            "retry_policy": {"recovery_timeout_seconds": 60}
        }
        
        should_allow = await self.circuit_breaker.should_allow_request(self.endpoint_id)
        assert should_allow is False
    
    @pytest.mark.asyncio
    async def test_record_success_resets_failure_count(self):
        """Test recording success resets failure count."""
        # Mock current state
        self.circuit_breaker._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.CLOSED,
            "failure_count": 3,
            "total_calls": 10,
            "successful_calls": 7
        }
        
        await self.circuit_breaker.record_success(self.endpoint_id, 150)
        
        # Check failure count was reset
        state = self.circuit_breaker._circuit_states[self.endpoint_id]
        assert state["failure_count"] == 0
        assert state["successful_calls"] == 8
        assert state["total_calls"] == 11
    
    @pytest.mark.asyncio
    async def test_record_failure_opens_circuit(self):
        """Test recording failures opens circuit when threshold exceeded."""
        # Mock current state near threshold
        self.circuit_breaker._circuit_states[self.endpoint_id] = {
            "state": CircuitBreakerState.CLOSED,
            "failure_count": 4,
            "failure_threshold": 5,
            "total_calls": 10,
            "failed_calls": 4
        }
        
        await self.circuit_breaker.record_failure(self.endpoint_id, "timeout")
        
        # Check circuit was opened
        state = self.circuit_breaker._circuit_states[self.endpoint_id]
        assert state["failure_count"] == 5
        assert state["state"] == CircuitBreakerState.OPEN


class TestWebhookDeliveryService:
    """Test webhook delivery service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_database = AsyncMock()
        self.mock_circuit_breaker = AsyncMock()
        self.mock_signature_service = Mock()
        
        self.delivery_service = WebhookDeliveryService(
            database=self.mock_database,
            circuit_breaker=self.mock_circuit_breaker,
            signature_service=self.mock_signature_service
        )
        
        self.endpoint_id = str(uuid4())
        self.tenant_id = str(uuid4())
        self.test_payload = {"event": "test", "data": "example"}
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_queues_delivery(self):
        """Test webhook delivery queuing."""
        # Mock endpoint exists and is enabled
        self.mock_database.get_item.return_value = {
            "id": self.endpoint_id,
            "url": "https://example.com/webhook",
            "enabled": True,
            "retry_policy": {"max_attempts": 5}
        }
        
        # Mock successful database creation
        self.mock_database.create_item.return_value = None
        
        delivery_id = await self.delivery_service.deliver_webhook(
            endpoint_id=self.endpoint_id,
            event_type="document.processed",
            payload=self.test_payload,
            tenant_id=self.tenant_id
        )
        
        assert delivery_id is not None
        assert len(delivery_id) > 0
        
        # Verify database calls
        self.mock_database.get_item.assert_called_once()
        self.mock_database.create_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_immediate_success(self):
        """Test immediate webhook delivery success."""
        # Mock signature generation
        self.mock_signature_service.generate_signature.return_value = "sha256=abc123"
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.reason_phrase = "OK"
            mock_response.headers = {"content-type": "application/json"}
            mock_response.aread.return_value = b'{"status": "received"}'
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await self.delivery_service.deliver_webhook_immediate(
                url="https://example.com/webhook",
                payload=self.test_payload,
                secret="test_secret"
            )
            
            assert result.success is True
            assert result.status_code == 200
            assert result.response_time_ms is not None
            assert result.signature_verified is True
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_immediate_failure(self):
        """Test immediate webhook delivery failure."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock HTTP error response
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.reason_phrase = "Internal Server Error"
            mock_response.headers = {}
            mock_response.aread.return_value = b'{"error": "server error"}'
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await self.delivery_service.deliver_webhook_immediate(
                url="https://example.com/webhook",
                payload=self.test_payload
            )
            
            assert result.success is False
            assert result.status_code == 500
            assert result.failure_reason == WebhookFailureReason.HTTP_ERROR.value
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_calculate_retry_schedule(self):
        """Test retry schedule calculation."""
        delivery_id = str(uuid4())
        
        # Mock delivery record
        self.mock_database.get_item.return_value = {
            "id": delivery_id,
            "endpoint_id": self.endpoint_id,
            "attempt_number": 2,
            "tenant_id": self.tenant_id
        }
        
        # Mock endpoint config
        self.delivery_service._get_endpoint_config = AsyncMock(return_value={
            "retry_policy": {
                "max_attempts": 5,
                "base_delay_seconds": 1.0,
                "max_delay_seconds": 300.0,
                "backoff_multiplier": 2.0,
                "jitter_enabled": True,
                "jitter_max_seconds": 5.0
            }
        })
        
        schedule = await self.delivery_service.calculate_retry_schedule(
            delivery_id, WebhookFailureReason.TIMEOUT.value
        )
        
        assert schedule.should_retry is True
        assert schedule.attempt_number == 3
        assert schedule.delay_seconds is not None
        assert schedule.delay_seconds >= 2.0  # Base delay * multiplier^(attempt-1)
        assert schedule.next_attempt_at is not None


class TestWebhookDeadLetterService:
    """Test webhook dead letter queue functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_database = AsyncMock()
        self.dead_letter_service = WebhookDeadLetterService(self.mock_database)
        self.delivery_id = str(uuid4())
        self.tenant_id = str(uuid4())
    
    @pytest.mark.asyncio
    async def test_move_to_dead_letter(self):
        """Test moving delivery to dead letter queue."""
        # Mock original delivery record
        self.mock_database.get_item.return_value = {
            "id": self.delivery_id,
            "tenant_id": self.tenant_id,
            "endpoint_id": str(uuid4()),
            "event_type": "document.processed",
            "event_id": "event_123",
            "payload": {"test": "data"},
            "attempt_number": 5,
            "scheduled_at": datetime.utcnow()
        }
        
        # Mock successful database operations
        self.mock_database.create_item.return_value = None
        self.mock_database.update_item.return_value = None
        
        dead_letter_id = await self.dead_letter_service.move_to_dead_letter(
            delivery_id=self.delivery_id,
            final_failure_reason=WebhookFailureReason.MAX_RETRIES_EXCEEDED.value,
            final_error_message="Maximum retry attempts exceeded"
        )
        
        assert dead_letter_id is not None
        assert len(dead_letter_id) > 0
        
        # Verify database calls
        self.mock_database.get_item.assert_called_once()
        self.mock_database.create_item.assert_called_once()
        self.mock_database.update_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_dead_letter(self):
        """Test retrying dead letter delivery."""
        dead_letter_id = str(uuid4())
        admin_user_id = "admin_123"
        
        # Mock dead letter record
        self.mock_database.get_item.return_value = {
            "id": dead_letter_id,
            "tenant_id": self.tenant_id,
            "endpoint_id": str(uuid4()),
            "event_type": "document.processed",
            "event_id": "event_123",
            "payload": {"test": "data"},
            "can_retry": True,
            "retry_count": 0
        }
        
        # Mock successful database operations
        self.mock_database.create_item.return_value = None
        self.mock_database.update_item.return_value = None
        
        new_delivery_id = await self.dead_letter_service.retry_dead_letter(
            dead_letter_id=dead_letter_id,
            admin_user_id=admin_user_id,
            notes="Manual retry by admin"
        )
        
        assert new_delivery_id is not None
        assert len(new_delivery_id) > 0
        
        # Verify database calls
        self.mock_database.get_item.assert_called_once()
        self.mock_database.create_item.assert_called_once()
        self.mock_database.update_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resolve_dead_letter(self):
        """Test resolving dead letter."""
        dead_letter_id = str(uuid4())
        admin_user_id = "admin_123"
        
        # Mock dead letter record exists
        self.mock_database.get_item.return_value = {
            "id": dead_letter_id,
            "tenant_id": self.tenant_id
        }
        
        # Mock successful update
        self.mock_database.update_item.return_value = None
        
        success = await self.dead_letter_service.resolve_dead_letter(
            dead_letter_id=dead_letter_id,
            resolution_action="fixed_endpoint_url",
            admin_user_id=admin_user_id,
            notes="Fixed endpoint URL configuration"
        )
        
        assert success is True
        
        # Verify database calls
        self.mock_database.get_item.assert_called_once()
        self.mock_database.update_item.assert_called_once()


class TestReliableWebhookNotificationService:
    """Test reliable webhook notification service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_delivery_service = AsyncMock()
        self.mock_audit_service = AsyncMock()
        
        self.notification_service = ReliableWebhookNotificationService(
            webhook_delivery_service=self.mock_delivery_service,
            audit_service=self.mock_audit_service
        )
        
        self.test_url = "https://example.com/webhook"
        self.test_payload = {"event": "test", "data": "example"}
    
    @pytest.mark.asyncio
    async def test_send_webhook_success(self):
        """Test successful webhook sending."""
        # Mock successful delivery
        self.mock_delivery_service.deliver_webhook_immediate.return_value = WebhookDeliveryResult(
            success=True,
            status_code=200,
            response_time_ms=150,
            signature_verified=True
        )
        
        result = await self.notification_service.send_webhook(
            url=self.test_url,
            payload=self.test_payload,
            secret="test_secret"
        )
        
        assert result is True
        
        # Verify delivery service was called
        self.mock_delivery_service.deliver_webhook_immediate.assert_called_once()
        
        # Verify audit log was created
        self.mock_audit_service.log_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_webhook_failure(self):
        """Test webhook sending failure."""
        # Mock failed delivery
        self.mock_delivery_service.deliver_webhook_immediate.return_value = WebhookDeliveryResult(
            success=False,
            status_code=500,
            error_message="Internal Server Error",
            failure_reason=WebhookFailureReason.HTTP_ERROR.value
        )
        
        result = await self.notification_service.send_webhook(
            url=self.test_url,
            payload=self.test_payload
        )
        
        assert result is False
        
        # Verify audit log for failure was created
        self.mock_audit_service.log_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_reliable_webhook(self):
        """Test reliable webhook delivery with queuing."""
        delivery_id = str(uuid4())
        endpoint_id = str(uuid4())
        tenant_id = str(uuid4())
        
        # Mock successful queuing
        self.mock_delivery_service.deliver_webhook.return_value = delivery_id
        
        result = await self.notification_service.send_reliable_webhook(
            endpoint_id=endpoint_id,
            event_type=WebhookEventType.DOCUMENT_PROCESSED,
            payload=self.test_payload,
            tenant_id=tenant_id
        )
        
        assert result == delivery_id
        
        # Verify delivery service was called with correct parameters
        self.mock_delivery_service.deliver_webhook.assert_called_once_with(
            endpoint_id=endpoint_id,
            event_type=WebhookEventType.DOCUMENT_PROCESSED.value,
            payload=self.test_payload,
            tenant_id=tenant_id,
            event_id=None,
            correlation_id=None,
            priority=0
        )
        
        # Verify audit log was created
        self.mock_audit_service.log_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_failed_webhook(self):
        """Test retrying failed webhook."""
        delivery_id = str(uuid4())
        admin_user_id = "admin_123"
        
        # Mock successful retry scheduling
        self.mock_delivery_service.retry_failed_delivery.return_value = True
        
        result = await self.notification_service.retry_failed_webhook(
            delivery_id=delivery_id,
            admin_user_id=admin_user_id,
            notes="Manual retry by admin"
        )
        
        assert result is True
        
        # Verify delivery service was called
        self.mock_delivery_service.retry_failed_delivery.assert_called_once_with(
            delivery_id=delivery_id,
            admin_user_id=admin_user_id,
            notes="Manual retry by admin"
        )
        
        # Verify audit log was created
        self.mock_audit_service.log_event.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])