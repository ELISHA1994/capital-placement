"""
Reliable notification adapter with webhook delivery reliability.

This adapter replaces the simple webhook sending with a comprehensive
reliable delivery system including retry mechanisms, dead letter queues,
and monitoring.
"""

import json
from typing import Any, Dict, Optional
from uuid import uuid4

import structlog

from app.domain.interfaces import (
    INotificationService,
    IWebhookDeliveryService,
    IAuditService
)
from app.api.schemas.webhook_schemas import WebhookEventType

logger = structlog.get_logger(__name__)


class ReliableWebhookNotificationService(INotificationService):
    """Notification service with reliable webhook delivery."""

    def __init__(
        self,
        webhook_delivery_service: IWebhookDeliveryService,
        audit_service: Optional[IAuditService] = None
    ):
        """
        Initialize reliable notification service.
        
        Args:
            webhook_delivery_service: Webhook delivery service
            audit_service: Audit service for logging (optional)
        """
        self.webhook_delivery_service = webhook_delivery_service
        self.audit_service = audit_service
        self._sent_email_count = 0
        self._sent_webhooks = 0
        self._sent_push = 0
        
    async def check_health(self) -> Dict[str, Any]:
        """Return notification service health."""
        try:
            webhook_health = await self.webhook_delivery_service.check_health()
            
            return {
                "status": "healthy",
                "service": "ReliableWebhookNotificationService",
                "emails_sent": self._sent_email_count,
                "webhooks_sent": self._sent_webhooks,
                "push_sent": self._sent_push,
                "webhook_delivery_service": webhook_health
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "ReliableWebhookNotificationService",
                "error": str(e)
            }
    
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        is_html: bool = False
    ) -> bool:
        """
        Send email notification.
        
        Note: This is a placeholder implementation.
        In production, integrate with actual email service.
        """
        try:
            self._sent_email_count += 1
            
            logger.info(
                "Email notification sent (simulated)",
                recipient=to,
                subject=subject,
                is_html=is_html,
                body_length=len(body)
            )
            
            # Log to audit service if available
            if self.audit_service:
                try:
                    await self.audit_service.log_event(
                        event_type="notification.email.sent",
                        tenant_id="system",  # TODO: Get from context
                        action="send_email",
                        resource_type="email_notification",
                        details={
                            "recipient": to,
                            "subject": subject,
                            "is_html": is_html,
                            "body_length": len(body)
                        }
                    )
                except Exception as audit_error:
                    logger.warning("Failed to log email audit event", error=str(audit_error))
            
            return True
            
        except Exception as err:
            logger.error("Failed to send email", recipient=to, error=str(err))
            return False
    
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None
    ) -> bool:
        """
        Send webhook notification using reliable delivery service.
        
        Args:
            url: Webhook URL
            payload: Webhook payload
            secret: Secret for signature generation
            
        Returns:
            True if webhook was queued successfully
        """
        try:
            # Use immediate delivery for backward compatibility
            # In a full implementation, you might want to queue this
            correlation_id = str(uuid4())
            
            result = await self.webhook_delivery_service.deliver_webhook_immediate(
                url=url,
                payload=payload,
                secret=secret,
                correlation_id=correlation_id
            )
            
            self._sent_webhooks += 1
            
            if result.success:
                logger.info(
                    "Webhook notification delivered successfully",
                    url=url,
                    status_code=result.status_code,
                    response_time_ms=result.response_time_ms,
                    correlation_id=correlation_id
                )
                
                # Log successful delivery to audit service
                if self.audit_service:
                    try:
                        await self.audit_service.log_event(
                            event_type="notification.webhook.delivered",
                            tenant_id="system",  # TODO: Get from context
                            action="send_webhook",
                            resource_type="webhook_notification",
                            details={
                                "url": url,
                                "status_code": result.status_code,
                                "response_time_ms": result.response_time_ms,
                                "payload_size": len(json.dumps(payload)),
                                "correlation_id": correlation_id,
                                "signature_verified": result.signature_verified
                            }
                        )
                    except Exception as audit_error:
                        logger.warning("Failed to log webhook audit event", error=str(audit_error))
                
                return True
            else:
                logger.warning(
                    "Webhook notification delivery failed",
                    url=url,
                    status_code=result.status_code,
                    error_message=result.error_message,
                    failure_reason=result.failure_reason,
                    correlation_id=correlation_id
                )
                
                # Log failed delivery to audit service
                if self.audit_service:
                    try:
                        await self.audit_service.log_event(
                            event_type="notification.webhook.failed",
                            tenant_id="system",  # TODO: Get from context
                            action="send_webhook",
                            resource_type="webhook_notification",
                            details={
                                "url": url,
                                "status_code": result.status_code,
                                "error_message": result.error_message,
                                "failure_reason": result.failure_reason,
                                "payload_size": len(json.dumps(payload)),
                                "correlation_id": correlation_id
                            },
                            error_code=str(result.status_code) if result.status_code else None,
                            error_message=result.error_message
                        )
                    except Exception as audit_error:
                        logger.warning("Failed to log webhook failure audit event", error=str(audit_error))
                
                return False
                
        except Exception as err:
            logger.error("Failed to send webhook", url=url, error=str(err))
            
            # Log exception to audit service
            if self.audit_service:
                try:
                    await self.audit_service.log_event(
                        event_type="notification.webhook.error",
                        tenant_id="system",  # TODO: Get from context
                        action="send_webhook",
                        resource_type="webhook_notification",
                        details={
                            "url": url,
                            "payload_size": len(json.dumps(payload)),
                            "error": str(err)
                        },
                        error_message=str(err)
                    )
                except Exception as audit_error:
                    logger.warning("Failed to log webhook error audit event", error=str(audit_error))
            
            return False
    
    async def send_push_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send push notification.
        
        Note: This is a placeholder implementation.
        In production, integrate with actual push notification service.
        """
        try:
            self._sent_push += 1
            
            logger.info(
                "Push notification sent (simulated)",
                user_id=user_id,
                title=title,
                message_length=len(message),
                has_data=bool(data)
            )
            
            # Log to audit service if available
            if self.audit_service:
                try:
                    await self.audit_service.log_event(
                        event_type="notification.push.sent",
                        tenant_id="system",  # TODO: Get from context
                        action="send_push_notification",
                        resource_type="push_notification",
                        user_id=user_id,
                        details={
                            "title": title,
                            "message_length": len(message),
                            "has_data": bool(data)
                        }
                    )
                except Exception as audit_error:
                    logger.warning("Failed to log push notification audit event", error=str(audit_error))
            
            return True
            
        except Exception as err:
            logger.error("Failed to send push notification", user_id=user_id, error=str(err))
            return False
    
    async def send_reliable_webhook(
        self,
        endpoint_id: str,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        *,
        tenant_id: str,
        event_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """
        Send webhook using the reliable delivery system with queuing and retry.
        
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
        try:
            delivery_id = await self.webhook_delivery_service.deliver_webhook(
                endpoint_id=endpoint_id,
                event_type=event_type.value,
                payload=payload,
                tenant_id=tenant_id,
                event_id=event_id,
                correlation_id=correlation_id,
                priority=priority
            )
            
            self._sent_webhooks += 1
            
            logger.info(
                "Reliable webhook queued for delivery",
                delivery_id=delivery_id,
                endpoint_id=endpoint_id,
                event_type=event_type.value,
                tenant_id=tenant_id,
                priority=priority
            )
            
            # Log to audit service
            if self.audit_service:
                try:
                    await self.audit_service.log_event(
                        event_type="notification.webhook.queued",
                        tenant_id=tenant_id,
                        action="queue_webhook",
                        resource_type="webhook_notification",
                        details={
                            "delivery_id": delivery_id,
                            "endpoint_id": endpoint_id,
                            "event_type": event_type.value,
                            "payload_size": len(json.dumps(payload)),
                            "priority": priority,
                            "correlation_id": correlation_id
                        }
                    )
                except Exception as audit_error:
                    logger.warning("Failed to log webhook queuing audit event", error=str(audit_error))
            
            return delivery_id
            
        except Exception as err:
            logger.error(
                "Failed to queue reliable webhook",
                endpoint_id=endpoint_id,
                event_type=event_type.value,
                tenant_id=tenant_id,
                error=str(err)
            )
            raise
    
    async def get_webhook_delivery_status(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a webhook delivery.
        
        Args:
            delivery_id: Delivery ID to check
            
        Returns:
            Delivery status details or None if not found
        """
        try:
            return await self.webhook_delivery_service.get_delivery_status(delivery_id)
        except Exception as e:
            logger.error("Failed to get webhook delivery status", delivery_id=delivery_id, error=str(e))
            return None
    
    async def retry_failed_webhook(
        self,
        delivery_id: str,
        *,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Manually retry a failed webhook delivery.
        
        Args:
            delivery_id: Delivery ID to retry
            admin_user_id: Admin user performing the retry
            notes: Notes about the retry
            
        Returns:
            True if retry was scheduled successfully
        """
        try:
            success = await self.webhook_delivery_service.retry_failed_delivery(
                delivery_id=delivery_id,
                admin_user_id=admin_user_id,
                notes=notes
            )
            
            if success:
                logger.info(
                    "Webhook delivery retry scheduled",
                    delivery_id=delivery_id,
                    admin_user_id=admin_user_id,
                    notes=notes
                )
                
                # Log to audit service
                if self.audit_service:
                    try:
                        await self.audit_service.log_event(
                            event_type="notification.webhook.retry_scheduled",
                            tenant_id="system",  # TODO: Get from context
                            action="retry_webhook",
                            resource_type="webhook_notification",
                            user_id=admin_user_id,
                            details={
                                "delivery_id": delivery_id,
                                "notes": notes
                            }
                        )
                    except Exception as audit_error:
                        logger.warning("Failed to log webhook retry audit event", error=str(audit_error))
            
            return success
            
        except Exception as e:
            logger.error(
                "Failed to retry webhook delivery",
                delivery_id=delivery_id,
                admin_user_id=admin_user_id,
                error=str(e)
            )
            return False


class WebhookNotificationMixin:
    """Mixin to add reliable webhook functionality to existing notification services."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reliable_webhook_service: Optional[ReliableWebhookNotificationService] = None
    
    def set_reliable_webhook_service(self, service: ReliableWebhookNotificationService) -> None:
        """Set the reliable webhook service."""
        self._reliable_webhook_service = service
    
    async def send_reliable_webhook(
        self,
        endpoint_id: str,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        *,
        tenant_id: str,
        event_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 0
    ) -> Optional[str]:
        """Send webhook using reliable delivery if available."""
        if self._reliable_webhook_service:
            return await self._reliable_webhook_service.send_reliable_webhook(
                endpoint_id=endpoint_id,
                event_type=event_type,
                payload=payload,
                tenant_id=tenant_id,
                event_id=event_id,
                correlation_id=correlation_id,
                priority=priority
            )
        else:
            logger.warning("Reliable webhook service not available, falling back to basic delivery")
            return None


__all__ = [
    "ReliableWebhookNotificationService",
    "WebhookNotificationMixin"
]