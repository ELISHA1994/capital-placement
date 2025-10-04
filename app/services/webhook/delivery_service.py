"""
Webhook delivery service with retry mechanisms and reliability features.

This service provides robust webhook delivery with exponential backoff retry,
circuit breaker integration, signature generation, and comprehensive logging.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
import structlog

from app.domain.interfaces import (
    IWebhookDeliveryService, 
    IWebhookCircuitBreakerService,
    IWebhookSignatureService,
    IDatabase,
    WebhookDeliveryResult,
    WebhookRetrySchedule
)
from app.models.webhook_models import (
    WebhookDeliveryStatus,
    WebhookFailureReason,
    RetryPolicy
)

logger = structlog.get_logger(__name__)


class WebhookDeliveryService(IWebhookDeliveryService):
    """Service for reliable webhook delivery with retry mechanisms."""

    def __init__(
        self,
        database: IDatabase,
        circuit_breaker: IWebhookCircuitBreakerService,
        signature_service: IWebhookSignatureService
    ):
        """
        Initialize webhook delivery service.
        
        Args:
            database: Database interface for persistence
            circuit_breaker: Circuit breaker service
            signature_service: Signature generation service
        """
        self.database = database
        self.circuit_breaker = circuit_breaker
        self.signature_service = signature_service
        self._delivery_semaphore = asyncio.Semaphore(10)  # Limit concurrent deliveries
        
    async def check_health(self) -> Dict[str, Any]:
        """Return webhook delivery service health."""
        try:
            # Check pending deliveries count
            pending_count = await self._get_pending_deliveries_count()
            
            return {
                "status": "healthy",
                "service": "WebhookDeliveryService",
                "pending_deliveries": pending_count,
                "max_concurrent_deliveries": 10,
                "available_slots": self._delivery_semaphore._value
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "WebhookDeliveryService",
                "error": str(e)
            }
    
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
        """
        Queue a webhook for delivery with retry mechanism.
        
        Args:
            endpoint_id: Webhook endpoint identifier
            event_type: Type of event triggering the webhook
            payload: Webhook payload data
            tenant_id: Tenant identifier
            event_id: Unique event identifier
            correlation_id: Correlation ID for tracking
            priority: Delivery priority (higher = more urgent)
            
        Returns:
            Delivery ID for tracking
        """
        delivery_id = str(uuid4())
        
        try:
            # Get endpoint configuration
            endpoint = await self._get_endpoint_config(endpoint_id)
            if not endpoint:
                raise ValueError(f"Webhook endpoint not found: {endpoint_id}")
            
            if not endpoint.get("enabled", True):
                raise ValueError(f"Webhook endpoint is disabled: {endpoint_id}")
            
            # Create delivery record
            delivery_record = {
                "id": delivery_id,
                "tenant_id": tenant_id,
                "endpoint_id": endpoint_id,
                "event_type": event_type,
                "event_id": event_id or str(uuid4()),
                "payload": payload,
                "status": WebhookDeliveryStatus.PENDING.value,
                "attempt_number": 1,
                "max_attempts": endpoint.get("retry_policy", {}).get("max_attempts", 5),
                "scheduled_at": datetime.utcnow(),
                "correlation_id": correlation_id,
                "priority": priority,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Store delivery record
            await self.database.create_item("webhook_deliveries", delivery_record)
            
            # Schedule immediate delivery
            asyncio.create_task(self._process_single_delivery(delivery_id))
            
            logger.info(
                "Webhook delivery queued",
                delivery_id=delivery_id,
                endpoint_id=endpoint_id,
                event_type=event_type,
                tenant_id=tenant_id,
                priority=priority
            )
            
            return delivery_id
            
        except Exception as e:
            logger.error(
                "Failed to queue webhook delivery",
                endpoint_id=endpoint_id,
                event_type=event_type,
                tenant_id=tenant_id,
                error=str(e)
            )
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
    ) -> WebhookDeliveryResult:
        """
        Deliver webhook immediately without queuing.
        
        Args:
            url: Webhook URL
            payload: Webhook payload
            secret: Secret for signature generation
            timeout_seconds: Request timeout
            signature_header: Header name for signature
            correlation_id: Correlation ID for tracking
            
        Returns:
            WebhookDeliveryResult with delivery outcome
        """
        start_time = time.time()
        
        try:
            # Prepare payload
            payload_str = json.dumps(payload, separators=(',', ':'))
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "CapitalPlacement-Webhook/1.0",
                "X-Webhook-Delivery": str(uuid4()),
                "X-Webhook-Timestamp": str(int(time.time()))
            }
            
            if correlation_id:
                headers["X-Correlation-ID"] = correlation_id
            
            # Generate signature if secret provided
            signature_verified = None
            if secret:
                signature = self.signature_service.generate_signature(
                    payload_str, secret, "sha256"
                )
                headers[signature_header] = signature
                signature_verified = True
            
            # Make HTTP request
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    url,
                    content=payload_str,
                    headers=headers
                )
                
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Read response body (limit size to prevent memory issues)
                response_body = await response.aread()
                if len(response_body) > 10000:  # Limit to 10KB
                    response_body = response_body[:10000]
                
                response_body_str = response_body.decode('utf-8', errors='ignore')
                
                # Check if successful
                if 200 <= response.status_code < 300:
                    logger.info(
                        "Webhook delivered successfully",
                        url=url,
                        status_code=response.status_code,
                        response_time_ms=response_time_ms,
                        correlation_id=correlation_id
                    )
                    
                    return WebhookDeliveryResult(
                        success=True,
                        status_code=response.status_code,
                        response_body=response_body_str,
                        response_headers=dict(response.headers),
                        response_time_ms=response_time_ms,
                        signature_verified=signature_verified
                    )
                else:
                    logger.warning(
                        "Webhook delivery failed with HTTP error",
                        url=url,
                        status_code=response.status_code,
                        response_body=response_body_str[:500],
                        response_time_ms=response_time_ms,
                        correlation_id=correlation_id
                    )
                    
                    return WebhookDeliveryResult(
                        success=False,
                        status_code=response.status_code,
                        response_body=response_body_str,
                        response_headers=dict(response.headers),
                        response_time_ms=response_time_ms,
                        error_message=f"HTTP {response.status_code}: {response.reason_phrase}",
                        failure_reason=WebhookFailureReason.HTTP_ERROR.value,
                        signature_verified=signature_verified
                    )
        
        except httpx.TimeoutException:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                "Webhook delivery timeout",
                url=url,
                timeout_seconds=timeout_seconds,
                response_time_ms=response_time_ms,
                correlation_id=correlation_id
            )
            
            return WebhookDeliveryResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message=f"Request timeout after {timeout_seconds}s",
                failure_reason=WebhookFailureReason.TIMEOUT.value,
                signature_verified=signature_verified
            )
        
        except httpx.ConnectError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.warning(
                "Webhook delivery connection error",
                url=url,
                error=str(e),
                response_time_ms=response_time_ms,
                correlation_id=correlation_id
            )
            
            return WebhookDeliveryResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message=f"Connection error: {str(e)}",
                failure_reason=WebhookFailureReason.CONNECTION_ERROR.value,
                signature_verified=signature_verified
            )
        
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Webhook delivery unexpected error",
                url=url,
                error=str(e),
                response_time_ms=response_time_ms,
                correlation_id=correlation_id
            )
            
            return WebhookDeliveryResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message=f"Unexpected error: {str(e)}",
                failure_reason=WebhookFailureReason.CONNECTION_ERROR.value,
                signature_verified=signature_verified
            )
    
    async def retry_failed_delivery(
        self,
        delivery_id: str,
        *,
        override_max_attempts: bool = False,
        new_max_attempts: Optional[int] = None,
        admin_user_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Manually retry a failed webhook delivery.
        
        Args:
            delivery_id: Delivery ID to retry
            override_max_attempts: Override the max attempts limit
            new_max_attempts: New max attempts if overriding
            admin_user_id: Admin user performing the retry
            notes: Notes about the retry
            
        Returns:
            True if retry was scheduled successfully
        """
        try:
            # Get delivery record
            delivery = await self.database.get_item(
                "webhook_deliveries",
                delivery_id,
                delivery_id
            )
            
            if not delivery:
                logger.warning("Delivery not found for retry", delivery_id=delivery_id)
                return False
            
            # Check if retry is appropriate
            current_status = delivery.get("status")
            if current_status not in [
                WebhookDeliveryStatus.FAILED.value,
                WebhookDeliveryStatus.DEAD_LETTER.value
            ]:
                logger.warning(
                    "Cannot retry delivery in current status",
                    delivery_id=delivery_id,
                    current_status=current_status
                )
                return False
            
            # Update delivery record for retry
            updates = {
                "status": WebhookDeliveryStatus.PENDING.value,
                "manual_retry": True,
                "retry_admin_user_id": admin_user_id,
                "retry_notes": notes,
                "retry_scheduled_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Override max attempts if requested
            if override_max_attempts and new_max_attempts:
                updates["max_attempts"] = new_max_attempts
                updates["max_attempts_overridden"] = True
            
            await self.database.update_item("webhook_deliveries", delivery_id, updates)
            
            # Schedule delivery
            asyncio.create_task(self._process_single_delivery(delivery_id))
            
            logger.info(
                "Manual retry scheduled for webhook delivery",
                delivery_id=delivery_id,
                admin_user_id=admin_user_id,
                override_max_attempts=override_max_attempts,
                new_max_attempts=new_max_attempts,
                notes=notes
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to schedule retry for webhook delivery",
                delivery_id=delivery_id,
                admin_user_id=admin_user_id,
                error=str(e)
            )
            return False
    
    async def cancel_delivery(
        self,
        delivery_id: str,
        *,
        reason: str,
        admin_user_id: Optional[str] = None
    ) -> bool:
        """
        Cancel a pending webhook delivery.
        
        Args:
            delivery_id: Delivery ID to cancel
            reason: Cancellation reason
            admin_user_id: Admin user performing the cancellation
            
        Returns:
            True if cancellation was successful
        """
        try:
            # Get delivery record
            delivery = await self.database.get_item(
                "webhook_deliveries",
                delivery_id,
                delivery_id
            )
            
            if not delivery:
                logger.warning("Delivery not found for cancellation", delivery_id=delivery_id)
                return False
            
            # Check if cancellation is appropriate
            current_status = delivery.get("status")
            if current_status not in [
                WebhookDeliveryStatus.PENDING.value,
                WebhookDeliveryStatus.RETRYING.value
            ]:
                logger.warning(
                    "Cannot cancel delivery in current status",
                    delivery_id=delivery_id,
                    current_status=current_status
                )
                return False
            
            # Update delivery record
            updates = {
                "status": WebhookDeliveryStatus.CANCELLED.value,
                "cancelled_at": datetime.utcnow(),
                "cancelled_by": admin_user_id,
                "cancellation_reason": reason,
                "failure_reason": WebhookFailureReason.CANCELLED_BY_ADMIN.value,
                "error_message": f"Cancelled by admin: {reason}",
                "updated_at": datetime.utcnow()
            }
            
            await self.database.update_item("webhook_deliveries", delivery_id, updates)
            
            logger.info(
                "Webhook delivery cancelled",
                delivery_id=delivery_id,
                admin_user_id=admin_user_id,
                reason=reason
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cancel webhook delivery",
                delivery_id=delivery_id,
                admin_user_id=admin_user_id,
                error=str(e)
            )
            return False
    
    async def get_delivery_status(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a webhook delivery.
        
        Args:
            delivery_id: Delivery ID to check
            
        Returns:
            Delivery status details or None if not found
        """
        try:
            delivery = await self.database.get_item(
                "webhook_deliveries",
                delivery_id,
                delivery_id
            )
            
            if not delivery:
                return None
            
            # Format response
            return {
                "delivery_id": delivery_id,
                "status": delivery.get("status"),
                "attempt_number": delivery.get("attempt_number", 1),
                "max_attempts": delivery.get("max_attempts", 5),
                "event_type": delivery.get("event_type"),
                "endpoint_id": delivery.get("endpoint_id"),
                "tenant_id": delivery.get("tenant_id"),
                "scheduled_at": delivery.get("scheduled_at"),
                "first_attempted_at": delivery.get("first_attempted_at"),
                "last_attempted_at": delivery.get("last_attempted_at"),
                "delivered_at": delivery.get("delivered_at"),
                "next_retry_at": delivery.get("next_retry_at"),
                "http_status_code": delivery.get("http_status_code"),
                "response_time_ms": delivery.get("response_time_ms"),
                "failure_reason": delivery.get("failure_reason"),
                "error_message": delivery.get("error_message"),
                "correlation_id": delivery.get("correlation_id"),
                "created_at": delivery.get("created_at"),
                "updated_at": delivery.get("updated_at")
            }
            
        except Exception as e:
            logger.error(
                "Failed to get delivery status",
                delivery_id=delivery_id,
                error=str(e)
            )
            return None
    
    async def process_delivery_queue(
        self,
        *,
        max_deliveries: int = 50,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Process pending webhook deliveries from queue.
        
        Args:
            max_deliveries: Maximum deliveries to process
            timeout_seconds: Processing timeout
            
        Returns:
            Processing summary with statistics
        """
        start_time = time.time()
        processed = 0
        successful = 0
        failed = 0
        
        try:
            # Get pending deliveries
            pending_deliveries = await self._get_pending_deliveries(limit=max_deliveries)
            
            if not pending_deliveries:
                return {
                    "processed": 0,
                    "successful": 0,
                    "failed": 0,
                    "duration_seconds": 0,
                    "message": "No pending deliveries"
                }
            
            # Process deliveries concurrently
            tasks = []
            for delivery in pending_deliveries:
                if time.time() - start_time > timeout_seconds:
                    break
                
                task = asyncio.create_task(
                    self._process_single_delivery(delivery["id"])
                )
                tasks.append(task)
            
            # Wait for completion
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            for result in results:
                processed += 1
                if isinstance(result, Exception):
                    failed += 1
                elif result:
                    successful += 1
                else:
                    failed += 1
            
            duration = time.time() - start_time
            
            logger.info(
                "Webhook delivery queue processing completed",
                processed=processed,
                successful=successful,
                failed=failed,
                duration_seconds=duration
            )
            
            return {
                "processed": processed,
                "successful": successful,
                "failed": failed,
                "duration_seconds": duration,
                "available_deliveries": len(pending_deliveries),
                "timeout_reached": duration >= timeout_seconds
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Error processing webhook delivery queue",
                processed=processed,
                duration_seconds=duration,
                error=str(e)
            )
            
            return {
                "processed": processed,
                "successful": successful,
                "failed": failed,
                "duration_seconds": duration,
                "error": str(e)
            }
    
    async def calculate_retry_schedule(
        self,
        delivery_id: str,
        failure_reason: str
    ) -> WebhookRetrySchedule:
        """
        Calculate when to retry a failed delivery.
        
        Args:
            delivery_id: Delivery ID that failed
            failure_reason: Reason for failure
            
        Returns:
            WebhookRetrySchedule with retry timing
        """
        try:
            # Get delivery record
            delivery = await self.database.get_item(
                "webhook_deliveries",
                delivery_id,
                delivery_id
            )
            
            if not delivery:
                return WebhookRetrySchedule(
                    should_retry=False,
                    reason="Delivery record not found"
                )
            
            # Get endpoint configuration
            endpoint = await self._get_endpoint_config(delivery["endpoint_id"])
            if not endpoint:
                return WebhookRetrySchedule(
                    should_retry=False,
                    reason="Endpoint not found"
                )
            
            # Get retry policy
            retry_policy = endpoint.get("retry_policy", {})
            max_attempts = retry_policy.get("max_attempts", 5)
            base_delay = retry_policy.get("base_delay_seconds", 1.0)
            max_delay = retry_policy.get("max_delay_seconds", 300.0)
            backoff_multiplier = retry_policy.get("backoff_multiplier", 2.0)
            jitter_enabled = retry_policy.get("jitter_enabled", True)
            jitter_max = retry_policy.get("jitter_max_seconds", 5.0)
            
            # Check retry eligibility
            current_attempt = delivery.get("attempt_number", 1)
            
            # Check if we've exceeded max attempts
            if current_attempt >= max_attempts:
                return WebhookRetrySchedule(
                    should_retry=False,
                    attempt_number=current_attempt,
                    reason=f"Maximum attempts ({max_attempts}) exceeded"
                )
            
            # Check if failure reason is retryable
            non_retryable_reasons = {
                WebhookFailureReason.SIGNATURE_VERIFICATION_FAILED.value,
                WebhookFailureReason.INVALID_URL.value,
                WebhookFailureReason.CANCELLED_BY_ADMIN.value
            }
            
            if failure_reason in non_retryable_reasons:
                return WebhookRetrySchedule(
                    should_retry=False,
                    attempt_number=current_attempt,
                    reason=f"Non-retryable failure: {failure_reason}"
                )
            
            # Calculate delay with exponential backoff
            next_attempt = current_attempt + 1
            delay = min(base_delay * (backoff_multiplier ** (current_attempt - 1)), max_delay)
            
            # Add jitter if enabled
            if jitter_enabled:
                jitter = random.uniform(0, jitter_max)
                delay += jitter
            
            # Calculate next attempt time
            next_attempt_at = datetime.utcnow() + timedelta(seconds=delay)
            
            return WebhookRetrySchedule(
                should_retry=True,
                next_attempt_at=next_attempt_at,
                delay_seconds=delay,
                attempt_number=next_attempt,
                reason=f"Retry {next_attempt}/{max_attempts} scheduled"
            )
            
        except Exception as e:
            logger.error(
                "Error calculating retry schedule",
                delivery_id=delivery_id,
                failure_reason=failure_reason,
                error=str(e)
            )
            
            return WebhookRetrySchedule(
                should_retry=False,
                reason=f"Error calculating schedule: {str(e)}"
            )
    
    async def _process_single_delivery(self, delivery_id: str) -> bool:
        """Process a single webhook delivery."""
        async with self._delivery_semaphore:
            try:
                # Get delivery record
                delivery = await self.database.get_item(
                    "webhook_deliveries",
                    delivery_id,
                    delivery_id
                )
                
                if not delivery:
                    logger.warning("Delivery record not found", delivery_id=delivery_id)
                    return False
                
                # Skip if already processed
                if delivery.get("status") not in [
                    WebhookDeliveryStatus.PENDING.value,
                    WebhookDeliveryStatus.RETRYING.value
                ]:
                    return True
                
                # Get endpoint configuration
                endpoint = await self._get_endpoint_config(delivery["endpoint_id"])
                if not endpoint:
                    await self._mark_delivery_failed(
                        delivery_id,
                        WebhookFailureReason.INVALID_URL,
                        "Endpoint configuration not found"
                    )
                    return False
                
                # Check circuit breaker
                if not await self.circuit_breaker.should_allow_request(endpoint["id"]):
                    # Schedule retry if circuit is open
                    retry_schedule = await self.calculate_retry_schedule(
                        delivery_id,
                        WebhookFailureReason.CIRCUIT_BREAKER_OPEN.value
                    )
                    
                    if retry_schedule.should_retry:
                        await self._schedule_retry(delivery_id, retry_schedule)
                    else:
                        await self._mark_delivery_failed(
                            delivery_id,
                            WebhookFailureReason.CIRCUIT_BREAKER_OPEN,
                            "Circuit breaker open, max retries exceeded"
                        )
                    
                    return False
                
                # Mark as sending
                await self._update_delivery_status(
                    delivery_id,
                    WebhookDeliveryStatus.SENDING,
                    {"first_attempted_at": datetime.utcnow()}
                )
                
                # Perform delivery
                result = await self.deliver_webhook_immediate(
                    url=endpoint["url"],
                    payload=delivery["payload"],
                    secret=endpoint.get("secret"),
                    timeout_seconds=endpoint.get("timeout_seconds", 30),
                    signature_header=endpoint.get("signature_header", "X-Webhook-Signature"),
                    correlation_id=delivery.get("correlation_id")
                )
                
                # Record result
                if result.success:
                    await self._mark_delivery_successful(delivery_id, result)
                    await self.circuit_breaker.record_success(
                        endpoint["id"],
                        result.response_time_ms or 0
                    )
                    return True
                else:
                    await self._handle_delivery_failure(delivery_id, result)
                    await self.circuit_breaker.record_failure(
                        endpoint["id"],
                        result.failure_reason or "unknown"
                    )
                    return False
                
            except Exception as e:
                logger.error(
                    "Error processing webhook delivery",
                    delivery_id=delivery_id,
                    error=str(e)
                )
                
                await self._mark_delivery_failed(
                    delivery_id,
                    WebhookFailureReason.CONNECTION_ERROR,
                    f"Processing error: {str(e)}"
                )
                return False
    
    async def _handle_delivery_failure(
        self,
        delivery_id: str,
        result: WebhookDeliveryResult
    ) -> None:
        """Handle delivery failure and schedule retry if appropriate."""
        try:
            # Update delivery record with failure details
            await self._update_delivery_with_result(delivery_id, result, success=False)
            
            # Calculate retry schedule
            failure_reason = result.failure_reason or WebhookFailureReason.CONNECTION_ERROR.value
            retry_schedule = await self.calculate_retry_schedule(delivery_id, failure_reason)
            
            if retry_schedule.should_retry:
                await self._schedule_retry(delivery_id, retry_schedule)
            else:
                await self._mark_delivery_failed(
                    delivery_id,
                    WebhookFailureReason(failure_reason),
                    result.error_message or "Delivery failed"
                )
            
        except Exception as e:
            logger.error(
                "Error handling delivery failure",
                delivery_id=delivery_id,
                error=str(e)
            )
    
    async def _mark_delivery_successful(
        self,
        delivery_id: str,
        result: WebhookDeliveryResult
    ) -> None:
        """Mark delivery as successful."""
        await self._update_delivery_with_result(delivery_id, result, success=True)
        
        updates = {
            "status": WebhookDeliveryStatus.DELIVERED.value,
            "delivered_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self.database.update_item("webhook_deliveries", delivery_id, updates)
        
        logger.info("Webhook delivery successful", delivery_id=delivery_id)
    
    async def _mark_delivery_failed(
        self,
        delivery_id: str,
        failure_reason: WebhookFailureReason,
        error_message: str
    ) -> None:
        """Mark delivery as permanently failed."""
        updates = {
            "status": WebhookDeliveryStatus.FAILED.value,
            "failure_reason": failure_reason.value,
            "error_message": error_message,
            "last_attempted_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self.database.update_item("webhook_deliveries", delivery_id, updates)
        
        logger.warning(
            "Webhook delivery failed permanently",
            delivery_id=delivery_id,
            failure_reason=failure_reason.value,
            error_message=error_message
        )
    
    async def _schedule_retry(
        self,
        delivery_id: str,
        retry_schedule: WebhookRetrySchedule
    ) -> None:
        """Schedule delivery retry."""
        updates = {
            "status": WebhookDeliveryStatus.RETRYING.value,
            "attempt_number": retry_schedule.attempt_number,
            "next_retry_at": retry_schedule.next_attempt_at,
            "last_attempted_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await self.database.update_item("webhook_deliveries", delivery_id, updates)
        
        # Schedule the retry
        delay = retry_schedule.delay_seconds or 0
        asyncio.create_task(self._delayed_retry(delivery_id, delay))
        
        logger.info(
            "Webhook delivery retry scheduled",
            delivery_id=delivery_id,
            attempt_number=retry_schedule.attempt_number,
            delay_seconds=delay,
            next_attempt_at=retry_schedule.next_attempt_at
        )
    
    async def _delayed_retry(self, delivery_id: str, delay_seconds: float) -> None:
        """Perform delayed retry of webhook delivery."""
        try:
            await asyncio.sleep(delay_seconds)
            await self._process_single_delivery(delivery_id)
        except Exception as e:
            logger.error(
                "Error in delayed retry",
                delivery_id=delivery_id,
                delay_seconds=delay_seconds,
                error=str(e)
            )
    
    async def _update_delivery_status(
        self,
        delivery_id: str,
        status: WebhookDeliveryStatus,
        additional_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update delivery status."""
        updates = {
            "status": status.value,
            "updated_at": datetime.utcnow()
        }
        
        if additional_updates:
            updates.update(additional_updates)
        
        await self.database.update_item("webhook_deliveries", delivery_id, updates)
    
    async def _update_delivery_with_result(
        self,
        delivery_id: str,
        result: WebhookDeliveryResult,
        success: bool
    ) -> None:
        """Update delivery record with HTTP result details."""
        updates = {
            "http_status_code": result.status_code,
            "response_body": result.response_body,
            "response_headers": result.response_headers,
            "response_time_ms": result.response_time_ms,
            "signature_verified": result.signature_verified,
            "last_attempted_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if not success:
            updates["error_message"] = result.error_message
            updates["failure_reason"] = result.failure_reason
        
        await self.database.update_item("webhook_deliveries", delivery_id, updates)
    
    async def _get_endpoint_config(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook endpoint configuration."""
        try:
            return await self.database.get_item("webhook_endpoints", endpoint_id, endpoint_id)
        except Exception as e:
            logger.error(
                "Error getting endpoint configuration",
                endpoint_id=endpoint_id,
                error=str(e)
            )
            return None
    
    async def _get_pending_deliveries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending webhook deliveries."""
        try:
            # Query for pending or retrying deliveries that are ready
            current_time = datetime.utcnow()
            
            query = """
                SELECT * FROM webhook_deliveries 
                WHERE status IN ('pending', 'retrying') 
                AND (next_retry_at IS NULL OR next_retry_at <= %s)
                ORDER BY priority DESC, scheduled_at ASC 
                LIMIT %s
            """
            
            return await self.database.query_items(
                "webhook_deliveries",
                query,
                [{"name": "@current_time", "value": current_time}, {"name": "@limit", "value": limit}]
            )
            
        except Exception as e:
            logger.error("Error getting pending deliveries", error=str(e))
            return []
    
    async def _get_pending_deliveries_count(self) -> int:
        """Get count of pending deliveries."""
        try:
            query = """
                SELECT COUNT(*) as count FROM webhook_deliveries 
                WHERE status IN ('pending', 'retrying')
            """
            
            result = await self.database.query_items("webhook_deliveries", query)
            return result[0]["count"] if result else 0
            
        except Exception as e:
            logger.warning("Error getting pending deliveries count", error=str(e))
            return 0


__all__ = ["WebhookDeliveryService"]