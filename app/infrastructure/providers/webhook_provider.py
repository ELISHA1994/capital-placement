"""
Webhook service provider for dependency injection.

This module provides singleton instances of webhook-related services
following the established provider pattern in the application.
"""

from typing import Optional

from app.domain.interfaces import IDatabase
from app.services.webhook.signature_service import WebhookSignatureService
from app.services.webhook.circuit_breaker_service import WebhookCircuitBreakerService
from app.services.webhook.delivery_service import WebhookDeliveryService
from app.services.webhook.dead_letter_service import WebhookDeadLetterService
from app.services.webhook.stats_service import WebhookStatsService
from app.services.webhook.reliable_notification_adapter import ReliableWebhookNotificationService
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.audit_provider import get_audit_service

# Singleton instances
_webhook_signature_service: Optional[WebhookSignatureService] = None
_webhook_circuit_breaker_service: Optional[WebhookCircuitBreakerService] = None
_webhook_delivery_service: Optional[WebhookDeliveryService] = None
_webhook_dead_letter_service: Optional[WebhookDeadLetterService] = None
_webhook_stats_service: Optional[WebhookStatsService] = None
_reliable_webhook_notification_service: Optional[ReliableWebhookNotificationService] = None


async def get_webhook_signature_service() -> WebhookSignatureService:
    """Get webhook signature service singleton."""
    global _webhook_signature_service
    
    if _webhook_signature_service is None:
        _webhook_signature_service = WebhookSignatureService()
    
    return _webhook_signature_service


async def get_webhook_circuit_breaker_service() -> WebhookCircuitBreakerService:
    """Get webhook circuit breaker service singleton."""
    global _webhook_circuit_breaker_service
    
    if _webhook_circuit_breaker_service is None:
        database = await get_postgres_adapter()
        _webhook_circuit_breaker_service = WebhookCircuitBreakerService(database)
    
    return _webhook_circuit_breaker_service


async def get_webhook_delivery_service() -> WebhookDeliveryService:
    """Get webhook delivery service singleton."""
    global _webhook_delivery_service
    
    if _webhook_delivery_service is None:
        database = await get_postgres_adapter()
        circuit_breaker = await get_webhook_circuit_breaker_service()
        signature_service = await get_webhook_signature_service()
        
        _webhook_delivery_service = WebhookDeliveryService(
            database=database,
            circuit_breaker=circuit_breaker,
            signature_service=signature_service
        )
    
    return _webhook_delivery_service


async def get_webhook_dead_letter_service() -> WebhookDeadLetterService:
    """Get webhook dead letter service singleton."""
    global _webhook_dead_letter_service
    
    if _webhook_dead_letter_service is None:
        database = await get_postgres_adapter()
        _webhook_dead_letter_service = WebhookDeadLetterService(database)
    
    return _webhook_dead_letter_service


async def get_webhook_stats_service() -> WebhookStatsService:
    """Get webhook stats service singleton."""
    global _webhook_stats_service
    
    if _webhook_stats_service is None:
        database = await get_postgres_adapter()
        _webhook_stats_service = WebhookStatsService(database)
    
    return _webhook_stats_service


async def get_reliable_webhook_notification_service() -> ReliableWebhookNotificationService:
    """Get reliable webhook notification service singleton."""
    global _reliable_webhook_notification_service
    
    if _reliable_webhook_notification_service is None:
        delivery_service = await get_webhook_delivery_service()
        audit_service = await get_audit_service()
        
        _reliable_webhook_notification_service = ReliableWebhookNotificationService(
            webhook_delivery_service=delivery_service,
            audit_service=audit_service
        )
    
    return _reliable_webhook_notification_service


def reset_webhook_services() -> None:
    """Reset all webhook service singletons (for testing)."""
    global _webhook_signature_service
    global _webhook_circuit_breaker_service
    global _webhook_delivery_service
    global _webhook_dead_letter_service
    global _webhook_stats_service
    global _reliable_webhook_notification_service
    
    _webhook_signature_service = None
    _webhook_circuit_breaker_service = None
    _webhook_delivery_service = None
    _webhook_dead_letter_service = None
    _webhook_stats_service = None
    _reliable_webhook_notification_service = None


__all__ = [
    "get_webhook_signature_service",
    "get_webhook_circuit_breaker_service", 
    "get_webhook_delivery_service",
    "get_webhook_dead_letter_service",
    "get_webhook_stats_service",
    "get_reliable_webhook_notification_service",
    "reset_webhook_services"
]