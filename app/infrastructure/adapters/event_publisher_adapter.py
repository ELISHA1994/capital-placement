"""Event publisher adapter for domain events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from app.domain.events.base import DomainEvent, IDomainEventPublisher
from app.domain.interfaces import IEventPublisher

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


class EventPublisherAdapter(IEventPublisher, IDomainEventPublisher):
    """Adapter implementation of IEventPublisher interface.

    This is a simple in-memory event publisher that can be extended
    to integrate with message queues, event stores, or other event systems.
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable[[Any], Any] | Any]] = {}
        self._published_events: list[Any] = []

    async def publish(self, event: DomainEvent) -> bool:
        """Publish DomainEvent using event type as topic."""
        try:
            topic = event.event_type
            event_payload = event.to_dict()
            event_payload["occurred_at"] = event.occurred_at.isoformat()
            return await self.publish_event(topic, event_payload)
        except Exception as exc:
            logger.error(
                "Failed to publish domain event",
                event_type=getattr(event, "event_type", type(event).__name__),
                error=str(exc),
            )
            return False

    async def publish_batch(self, events: list[DomainEvent]) -> bool:
        """Publish a batch of DomainEvents."""
        if not events:
            return True

        results: list[bool] = []
        for event in events:
            results.append(await self.publish(event))
        return all(results)

    async def publish_and_wait(self, event: DomainEvent, timeout_seconds: int = 30) -> bool:
        """Publish event and assume immediate completion."""
        # For the in-memory adapter we publish synchronously; timeout hint ignored.
        return await self.publish(event)

    async def publish_event(self, topic: str, event_data: dict[str, Any]) -> bool:
        """Publish a single event to a topic."""
        try:
            logger.debug(
                "Publishing event to topic",
                topic=topic,
                event_id=event_data.get('id', 'unknown')
            )

            # Store event for debugging/auditing
            self._published_events.append({'topic': topic, 'data': event_data})

            # Notify registered handlers for this topic
            handlers = self._handlers.get(topic, [])
            for handler in handlers:
                try:
                    if callable(handler):
                        await handler(event_data) if hasattr(handler, '__await__') else handler(event_data)
                except Exception as e:
                    logger.error(
                        "Event handler failed",
                        topic=topic,
                        handler=handler.__name__ if hasattr(handler, '__name__') else str(handler),
                        error=str(e)
                    )

            logger.debug(
                "Event published successfully to topic",
                topic=topic,
                handlers_notified=len(handlers)
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to publish event to topic",
                topic=topic,
                error=str(e)
            )
            return False

    async def publish_events(self, topic: str, events: list[dict[str, Any]]) -> bool:
        """Publish multiple events to a topic."""
        try:
            logger.debug(
                "Publishing multiple events to topic",
                topic=topic,
                event_count=len(events)
            )

            success = True
            for event_data in events:
                result = await self.publish_event(topic, event_data)
                if not result:
                    success = False

            logger.debug(
                "Multiple events published to topic",
                topic=topic,
                event_count=len(events),
                success=success
            )
            return success

        except Exception as e:
            logger.error(
                "Failed to publish multiple events to topic",
                topic=topic,
                error=str(e)
            )
            return False

    async def check_health(self) -> dict[str, Any]:
        """Health check for event publisher."""
        return {
            "status": "healthy",
            "handlers_registered": len(self._handlers),
            "events_published": len(self._published_events)
        }

    async def health_check(self) -> dict[str, Any]:
        """Backward-compatible alias for legacy callers."""
        return await self.check_health()

    def register_handler(self, event_type: str, handler: Callable[[Any], Any] | Any) -> None:
        """Register event handler for specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

        logger.debug(
            "Event handler registered",
            event_type=event_type,
            handler=handler.__name__ if hasattr(handler, '__name__') else str(handler)
        )

    def unregister_handler(self, event_type: str, handler: Callable[[Any], Any] | Any) -> None:
        """Unregister event handler."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.debug(
                    "Event handler unregistered",
                    event_type=event_type,
                    handler=handler.__name__ if hasattr(handler, '__name__') else str(handler)
                )
            except ValueError:
                logger.warning(
                    "Handler not found for unregistration",
                    event_type=event_type,
                    handler=handler.__name__ if hasattr(handler, '__name__') else str(handler)
                )

    def get_published_events(self) -> list[Any]:
        """Get list of published events (for testing/debugging)."""
        return self._published_events.copy()

    def clear_events(self) -> None:
        """Clear published events list (for testing)."""
        self._published_events.clear()

    def get_handlers(self, event_type: str) -> list[Callable[[Any], Any] | Any]:
        """Get registered handlers for event type."""
        return self._handlers.get(event_type, []).copy()


# Global singleton instance
_event_publisher: EventPublisherAdapter | None = None


def get_event_publisher() -> EventPublisherAdapter:
    """Get singleton event publisher instance."""
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = EventPublisherAdapter()
    return _event_publisher


__all__ = ["EventPublisherAdapter", "get_event_publisher"]
