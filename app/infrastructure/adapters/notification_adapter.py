"""Notification adapter implementations for local and production environments."""

from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional
import structlog
import httpx

from app.domain.interfaces import INotificationService

logger = structlog.get_logger(__name__)


class LocalNotificationService(INotificationService):
    """Simple notification service that logs messages locally for development."""

    def __init__(self):
        self._sent_email_count = 0
        self._sent_webhooks = 0
        self._sent_push = 0

    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "status": "healthy",
            "service": "LocalNotificationService",
            "emails_sent": self._sent_email_count,
            "webhooks_sent": self._sent_webhooks,
            "push_sent": self._sent_push,
        }

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        is_html: bool = False
    ) -> bool:
        """Send email notification (logs locally)."""
        try:
            self._sent_email_count += 1
            logger.info(
                "Email notification dispatched (local)",
                recipient=to,
                subject=subject,
                is_html=is_html,
                body_length=len(body),
                body_content=body,
            )
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
        """Send webhook notification (logs locally)."""
        try:
            self._sent_webhooks += 1
            logger.info("Webhook notification dispatched (local)", url=url)
            logger.debug("Webhook payload", url=url, payload=payload, secret_provided=bool(secret))
            return True
        except Exception as err:
            logger.error("Failed to send webhook", url=url, error=str(err))
            return False

    async def send_push_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send push notification (logs locally)."""
        try:
            self._sent_push += 1
            logger.info("Push notification dispatched (local)", user_id=user_id, title=title)
            logger.debug("Push payload", user_id=user_id, title=title, message=message, data=data)
            return True
        except Exception as err:
            logger.error("Failed to send push notification", user_id=user_id, error=str(err))
            return False


class NotificationAdapter:
    """Production notification adapter with HTTP webhook support."""

    def __init__(self):
        self.timeout = 30.0

    async def send_webhook(self, url: str, payload: Dict[str, Any]) -> None:
        """Send webhook notification."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            logger.debug(
                "Webhook notification sent successfully",
                url=url,
                payload_size=len(str(payload))
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "Webhook HTTP error",
                url=url,
                status_code=e.response.status_code,
                error=str(e)
            )
            # Don't re-raise to avoid breaking business flow
            
        except httpx.RequestError as e:
            logger.error(
                "Webhook request error",
                url=url,
                error=str(e)
            )
            # Don't re-raise to avoid breaking business flow
            
        except Exception as e:
            logger.error(
                "Webhook unknown error",
                url=url,
                error=str(e)
            )
            # Don't re-raise to avoid breaking business flow

    async def send_email(self, to: str, subject: str, body: str, **kwargs) -> None:
        """Send email notification (placeholder implementation)."""
        try:
            # This is a placeholder implementation
            # In a real system, this would integrate with an email service like:
            # - SendGrid
            # - AWS SES
            # - Mailgun
            # - SMTP server
            
            logger.info(
                "Email notification (simulated)",
                to=to,
                subject=subject,
                body_length=len(body)
            )
            
            # Simulate email sending delay
            await asyncio.sleep(0.1)
            
            logger.debug(
                "Email notification sent successfully",
                to=to,
                subject=subject
            )
            
        except Exception as e:
            logger.error(
                "Failed to send email",
                to=to,
                subject=subject,
                error=str(e)
            )
            # Don't re-raise to avoid breaking business flow

    async def send_sms(self, to: str, message: str, **kwargs) -> None:
        """Send SMS notification (placeholder implementation)."""
        try:
            # This is a placeholder implementation
            # In a real system, this would integrate with an SMS service like:
            # - Twilio
            # - AWS SNS
            # - MessageBird
            
            logger.info(
                "SMS notification (simulated)",
                to=to,
                message_length=len(message)
            )
            
            # Simulate SMS sending delay
            await asyncio.sleep(0.1)
            
            logger.debug(
                "SMS notification sent successfully",
                to=to,
                message_length=len(message)
            )
            
        except Exception as e:
            logger.error(
                "Failed to send SMS",
                to=to,
                error=str(e)
            )
            # Don't re-raise to avoid breaking business flow

    async def send_push_notification(self, device_token: str, title: str, body: str, **kwargs) -> None:
        """Send push notification (placeholder implementation)."""
        try:
            # This is a placeholder implementation
            # In a real system, this would integrate with push services like:
            # - Firebase Cloud Messaging (FCM)
            # - Apple Push Notification Service (APNs)
            # - OneSignal
            
            logger.info(
                "Push notification (simulated)",
                device_token=device_token[:10] + "...",  # Mask token for privacy
                title=title,
                body_length=len(body)
            )
            
            # Simulate push notification delay
            await asyncio.sleep(0.1)
            
            logger.debug(
                "Push notification sent successfully",
                device_token=device_token[:10] + "...",
                title=title
            )
            
        except Exception as e:
            logger.error(
                "Failed to send push notification",
                device_token=device_token[:10] + "...",
                title=title,
                error=str(e)
            )
            # Don't re-raise to avoid breaking business flow


__all__ = ["LocalNotificationService", "NotificationAdapter"]