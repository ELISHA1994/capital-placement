"""Local notification service for development environments."""

import structlog
from typing import Any, Dict, Optional

from app.domain.interfaces import INotificationService


logger = structlog.get_logger(__name__)


class LocalNotificationService(INotificationService):
    """Simple notification service that logs messages locally."""

    def __init__(self):
        self._sent_email_count = 0
        self._sent_webhooks = 0
        self._sent_push = 0

    async def check_health(self) -> Dict[str, Any]:
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
        try:
            self._sent_email_count += 1
            logger.info(
                "Password reset email dispatched",
                recipient=to,
                subject=subject,
                is_html=is_html,
                body_length=len(body),
                body_content=body,
            )
            return True
        except Exception as err:  # pragma: no cover - defensive logging
            logger.error("Failed to send email", recipient=to, error=str(err))
            return False

    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None
    ) -> bool:
        try:
            self._sent_webhooks += 1
            logger.info("Webhook notification dispatched", url=url)
            logger.debug("Webhook payload", url=url, payload=payload, secret_provided=bool(secret))
            return True
        except Exception as err:  # pragma: no cover - defensive logging
            logger.error("Failed to send webhook", url=url, error=str(err))
            return False

    async def send_push_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        try:
            self._sent_push += 1
            logger.info("Push notification dispatched", user_id=user_id, title=title)
            logger.debug("Push payload", user_id=user_id, title=title, message=message, data=data)
            return True
        except Exception as err:  # pragma: no cover - defensive logging
            logger.error("Failed to send push notification", user_id=user_id, error=str(err))
            return False
