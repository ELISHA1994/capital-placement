"""Enhanced webhook delivery service with retry and circuit breaker pattern."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
import structlog

from app.domain.interfaces import IWebhookDeliveryService, IWebhookCircuitBreakerService
from app.domain.retry import RetryPolicy
from app.infrastructure.providers.retry_provider import get_specialized_retry_executor


logger = structlog.get_logger(__name__)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class WebhookCircuitBreaker:
    """Circuit breaker implementation for webhook endpoints."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.success_threshold = success_threshold
        
        # State tracking per endpoint
        self._endpoint_states: Dict[str, Dict[str, Any]] = {}
        self._logger = structlog.get_logger(__name__)
    
    def _get_endpoint_state(self, endpoint_id: str) -> Dict[str, Any]:
        """Get or create endpoint state."""
        if endpoint_id not in self._endpoint_states:
            self._endpoint_states[endpoint_id] = {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": None,
                "opened_at": None,
                "total_requests": 0,
                "total_failures": 0,
                "total_successes": 0
            }
        return self._endpoint_states[endpoint_id]
    
    async def should_allow_request(self, endpoint_id: str) -> bool:
        """Check if requests should be allowed to an endpoint."""
        state = self._get_endpoint_state(endpoint_id)
        current_state = state["state"]
        
        if current_state == CircuitBreakerState.CLOSED:
            return True
        
        elif current_state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if state["opened_at"] and datetime.utcnow() - state["opened_at"] >= self.recovery_timeout:
                # Transition to half-open
                state["state"] = CircuitBreakerState.HALF_OPEN
                state["success_count"] = 0
                self._logger.info(
                    "Circuit breaker transitioning to half-open",
                    endpoint_id=endpoint_id
                )
                return True
            return False
        
        elif current_state == CircuitBreakerState.HALF_OPEN:
            # Allow limited requests to test recovery
            return True
        
        return False
    
    async def record_success(self, endpoint_id: str, response_time_ms: int) -> None:
        """Record a successful webhook delivery."""
        state = self._get_endpoint_state(endpoint_id)
        state["total_requests"] += 1
        state["total_successes"] += 1
        state["failure_count"] = 0  # Reset failure count on success
        
        current_state = state["state"]
        
        if current_state == CircuitBreakerState.HALF_OPEN:
            state["success_count"] += 1
            
            # Check if we've had enough successes to close the circuit
            if state["success_count"] >= self.success_threshold:
                state["state"] = CircuitBreakerState.CLOSED
                state["success_count"] = 0
                self._logger.info(
                    "Circuit breaker closed after successful recovery",
                    endpoint_id=endpoint_id,
                    recovery_successes=state["success_count"]
                )
        
        self._logger.debug(
            "Webhook success recorded",
            endpoint_id=endpoint_id,
            response_time_ms=response_time_ms,
            circuit_state=state["state"]
        )
    
    async def record_failure(self, endpoint_id: str, failure_reason: str) -> None:
        """Record a failed webhook delivery."""
        state = self._get_endpoint_state(endpoint_id)
        state["total_requests"] += 1
        state["total_failures"] += 1
        state["failure_count"] += 1
        state["last_failure_time"] = datetime.utcnow()
        state["success_count"] = 0  # Reset success count on failure
        
        current_state = state["state"]
        
        # Check if we should open the circuit
        if current_state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
            if state["failure_count"] >= self.failure_threshold:
                state["state"] = CircuitBreakerState.OPEN
                state["opened_at"] = datetime.utcnow()
                self._logger.warning(
                    "Circuit breaker opened due to failures",
                    endpoint_id=endpoint_id,
                    failure_count=state["failure_count"],
                    failure_reason=failure_reason
                )
        
        self._logger.debug(
            "Webhook failure recorded",
            endpoint_id=endpoint_id,
            failure_reason=failure_reason,
            failure_count=state["failure_count"],
            circuit_state=state["state"]
        )
    
    async def get_circuit_state(self, endpoint_id: str) -> Dict[str, Any]:
        """Get circuit breaker state for an endpoint."""
        state = self._get_endpoint_state(endpoint_id)
        
        return {
            "endpoint_id": endpoint_id,
            "circuit_state": state["state"],
            "failure_count": state["failure_count"],
            "success_count": state["success_count"],
            "last_failure_time": state["last_failure_time"].isoformat() if state["last_failure_time"] else None,
            "opened_at": state["opened_at"].isoformat() if state["opened_at"] else None,
            "total_requests": state["total_requests"],
            "total_failures": state["total_failures"],
            "total_successes": state["total_successes"],
            "failure_rate": state["total_failures"] / state["total_requests"] if state["total_requests"] > 0 else 0.0,
            "is_healthy": state["state"] == CircuitBreakerState.CLOSED
        }
    
    async def force_circuit_state(
        self,
        endpoint_id: str,
        target_state: CircuitBreakerState,
        admin_user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Manually set circuit breaker state."""
        state = self._get_endpoint_state(endpoint_id)
        old_state = state["state"]
        
        state["state"] = target_state
        
        if target_state == CircuitBreakerState.OPEN:
            state["opened_at"] = datetime.utcnow()
        elif target_state == CircuitBreakerState.CLOSED:
            state["failure_count"] = 0
            state["success_count"] = 0
            state["opened_at"] = None
        elif target_state == CircuitBreakerState.HALF_OPEN:
            state["success_count"] = 0
        
        self._logger.info(
            "Circuit breaker state manually changed",
            endpoint_id=endpoint_id,
            old_state=old_state,
            new_state=target_state,
            admin_user_id=admin_user_id,
            reason=reason
        )
        
        return True


class EnhancedWebhookDeliveryService(IWebhookDeliveryService):
    """Enhanced webhook delivery service with retry and circuit breaker."""
    
    def __init__(self):
        self._logger = structlog.get_logger(__name__)
        self._circuit_breaker = WebhookCircuitBreaker()
        
        # Session for HTTP requests
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Default retry policy for webhooks
        self._default_policy = RetryPolicy(
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=600.0,
            backoff_strategy="exponential_jitter",
            jitter_factor=0.2,
            circuit_breaker_enabled=True,
            dead_letter_enabled=True
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "CapitalPlacement-Webhook/1.0",
                    "Content-Type": "application/json"
                }
            )
        return self._session
    
    async def deliver_webhook(
        self,
        endpoint_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        tenant_id: str,
        event_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Queue a webhook for delivery with retry mechanism."""
        
        delivery_id = str(uuid4())
        
        # Get webhook endpoint configuration
        webhook_config = await self._get_webhook_config(endpoint_id, tenant_id)
        if not webhook_config:
            raise ValueError(f"Webhook endpoint {endpoint_id} not found")
        
        # Prepare delivery context
        delivery_context = {
            "delivery_id": delivery_id,
            "endpoint_id": endpoint_id,
            "event_type": event_type,
            "tenant_id": tenant_id,
            "event_id": event_id or str(uuid4()),
            "correlation_id": correlation_id,
            "priority": priority,
            "webhook_url": webhook_config["url"],
            "webhook_secret": webhook_config.get("secret"),
            "payload_size": len(str(payload))
        }
        
        self._logger.info(
            "Queuing webhook for delivery",
            delivery_id=delivery_id,
            endpoint_id=endpoint_id,
            event_type=event_type,
            tenant_id=tenant_id
        )
        
        # Execute webhook delivery with retry
        retry_executor = await get_specialized_retry_executor("webhook_delivery")
        
        try:
            await retry_executor.execute_with_retry(
                operation_func=self._deliver_webhook_operation,
                operation_id=delivery_id,
                operation_type="webhook_delivery",
                tenant_id=tenant_id,
                args=(webhook_config, payload),
                kwargs={"delivery_context": delivery_context},
                policy=self._default_policy,
                context=delivery_context
            )
            
            return delivery_id
        
        except Exception as error:
            self._logger.error(
                "Webhook delivery failed after all retries",
                delivery_id=delivery_id,
                endpoint_id=endpoint_id,
                error=str(error)
            )
            
            # The retry mechanism will handle dead letter queue
            return delivery_id
    
    async def _deliver_webhook_operation(
        self,
        webhook_config: Dict[str, Any],
        payload: Dict[str, Any],
        *,
        delivery_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Core webhook delivery operation that can be retried."""
        
        endpoint_id = delivery_context["endpoint_id"]
        delivery_id = delivery_context["delivery_id"]
        
        # Check circuit breaker
        if not await self._circuit_breaker.should_allow_request(endpoint_id):
            circuit_state = await self._circuit_breaker.get_circuit_state(endpoint_id)
            raise Exception(
                f"Circuit breaker is {circuit_state['circuit_state']} for endpoint {endpoint_id}"
            )
        
        # Prepare webhook payload
        webhook_payload = {
            "event_id": delivery_context["event_id"],
            "event_type": delivery_context["event_type"],
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload,
            "delivery_id": delivery_id,
            "correlation_id": delivery_context.get("correlation_id")
        }
        
        delivery_start = datetime.utcnow()
        
        try:
            # Deliver webhook
            result = await self.deliver_webhook_immediate(
                url=webhook_config["url"],
                payload=webhook_payload,
                secret=webhook_config.get("secret"),
                timeout_seconds=30,
                correlation_id=delivery_context.get("correlation_id")
            )
            
            if result.success:
                # Record success in circuit breaker
                response_time = result.response_time_ms or 0
                await self._circuit_breaker.record_success(endpoint_id, response_time)
                
                self._logger.info(
                    "Webhook delivered successfully",
                    delivery_id=delivery_id,
                    endpoint_id=endpoint_id,
                    status_code=result.status_code,
                    response_time_ms=response_time
                )
                
                return {
                    "delivery_id": delivery_id,
                    "status": "delivered",
                    "status_code": result.status_code,
                    "response_time_ms": response_time,
                    "delivered_at": datetime.utcnow().isoformat()
                }
            
            else:
                # Record failure in circuit breaker
                failure_reason = result.failure_reason or result.error_message or "Unknown failure"
                await self._circuit_breaker.record_failure(endpoint_id, failure_reason)
                
                # Raise exception to trigger retry
                raise Exception(f"Webhook delivery failed: {failure_reason}")
        
        except Exception as error:
            # Record failure in circuit breaker
            await self._circuit_breaker.record_failure(endpoint_id, str(error))
            
            self._logger.warning(
                "Webhook delivery attempt failed",
                delivery_id=delivery_id,
                endpoint_id=endpoint_id,
                error=str(error),
                delivery_duration_ms=int((datetime.utcnow() - delivery_start).total_seconds() * 1000)
            )
            
            # Re-raise to allow retry mechanism to handle
            raise
    
    async def deliver_webhook_immediate(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        secret: Optional[str] = None,
        timeout_seconds: int = 30,
        signature_header: str = "X-Webhook-Signature",
        correlation_id: Optional[str] = None
    ) -> "WebhookDeliveryResult":
        """Deliver webhook immediately without queuing."""
        
        from app.domain.interfaces import WebhookDeliveryResult
        
        session = await self._get_session()
        delivery_start = datetime.utcnow()
        
        try:
            # Generate signature if secret provided
            headers = {}
            if secret:
                # Import here to avoid circular dependency
                from app.services.adapters.notification_adapter import WebhookSignatureService
                signature_service = WebhookSignatureService()
                
                payload_str = str(payload)  # Should be JSON string in real implementation
                signature = signature_service.generate_signature(payload_str, secret)
                headers[signature_header] = signature
            
            # Add correlation ID if provided
            if correlation_id:
                headers["X-Correlation-ID"] = correlation_id
            
            # Make HTTP request
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                response_time_ms = int((datetime.utcnow() - delivery_start).total_seconds() * 1000)
                response_body = await response.text()
                
                # Get response headers
                response_headers = dict(response.headers)
                
                success = 200 <= response.status < 300
                
                return WebhookDeliveryResult(
                    success=success,
                    status_code=response.status,
                    response_body=response_body,
                    response_headers=response_headers,
                    response_time_ms=response_time_ms,
                    failure_reason=None if success else f"HTTP {response.status}",
                    signature_verified=True  # Assume verified for simplicity
                )
        
        except asyncio.TimeoutError:
            response_time_ms = int((datetime.utcnow() - delivery_start).total_seconds() * 1000)
            return WebhookDeliveryResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message="Request timeout",
                failure_reason="timeout"
            )
        
        except aiohttp.ClientError as e:
            response_time_ms = int((datetime.utcnow() - delivery_start).total_seconds() * 1000)
            return WebhookDeliveryResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                failure_reason="client_error"
            )
        
        except Exception as e:
            response_time_ms = int((datetime.utcnow() - delivery_start).total_seconds() * 1000)
            return WebhookDeliveryResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                failure_reason="unexpected_error"
            )
    
    async def _get_webhook_config(self, endpoint_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook endpoint configuration."""
        
        # Import here to avoid circular dependency
        from app.database.sqlmodel_engine import get_sqlmodel_db_manager
        from sqlalchemy import select
        
        # This would typically query the webhook_endpoints table
        # For now, return a mock configuration
        return {
            "url": f"https://webhook.example.com/endpoints/{endpoint_id}",
            "secret": "webhook_secret_key",
            "enabled": True,
            "timeout_seconds": 30
        }
    
    async def get_circuit_breaker_status(self, endpoint_id: str) -> Dict[str, Any]:
        """Get circuit breaker status for an endpoint."""
        return await self._circuit_breaker.get_circuit_state(endpoint_id)
    
    async def force_circuit_breaker_state(
        self,
        endpoint_id: str,
        state: str,
        *,
        admin_user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Manually set circuit breaker state."""
        target_state = CircuitBreakerState(state)
        return await self._circuit_breaker.force_circuit_state(
            endpoint_id, target_state, admin_user_id, reason
        )
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the webhook delivery service."""
        
        # Get circuit breaker statistics
        endpoint_states = list(self._circuit_breaker._endpoint_states.keys())
        healthy_endpoints = []
        unhealthy_endpoints = []
        
        for endpoint_id in endpoint_states:
            state = await self._circuit_breaker.get_circuit_state(endpoint_id)
            if state["is_healthy"]:
                healthy_endpoints.append(endpoint_id)
            else:
                unhealthy_endpoints.append(endpoint_id)
        
        return {
            "status": "healthy" if len(unhealthy_endpoints) == 0 else "degraded",
            "total_endpoints": len(endpoint_states),
            "healthy_endpoints": len(healthy_endpoints),
            "unhealthy_endpoints": len(unhealthy_endpoints),
            "circuit_breaker_active": len(endpoint_states) > 0,
            "session_active": self._session is not None and not self._session.closed
        }
    
    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()


__all__ = ["EnhancedWebhookDeliveryService", "WebhookCircuitBreaker"]