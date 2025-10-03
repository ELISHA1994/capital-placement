"""Event publisher adapter for domain events."""

from __future__ import annotations

import structlog
from typing import Any, List, Dict

from app.domain.interfaces import IEventPublisher

logger = structlog.get_logger(__name__)


class EventPublisherAdapter(IEventPublisher):
    """Adapter implementation of IEventPublisher interface.
    
    This is a simple in-memory event publisher that can be extended
    to integrate with message queues, event stores, or other event systems.
    """

    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}
        self._published_events: List[Any] = []

    async def publish(self, event: Any) -> None:
        """Publish domain event."""
        try:
            event_type = type(event).__name__
            
            logger.debug(
                "Publishing domain event",
                event_type=event_type,
                event_id=getattr(event, 'id', 'unknown')
            )
            
            # Store event for debugging/auditing
            self._published_events.append(event)
            
            # Notify registered handlers
            handlers = self._handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if callable(handler):
                        await handler(event) if hasattr(handler, '__await__') else handler(event)
                except Exception as e:
                    logger.error(
                        "Event handler failed",
                        event_type=event_type,
                        handler=handler.__name__ if hasattr(handler, '__name__') else str(handler),
                        error=str(e)
                    )
            
            logger.debug(
                "Domain event published successfully",
                event_type=event_type,
                handlers_notified=len(handlers)
            )
            
        except Exception as e:
            logger.error(
                "Failed to publish domain event",
                event=str(event),
                error=str(e)
            )
            # Don't re-raise to avoid breaking business operations

    def register_handler(self, event_type: str, handler: callable) -> None:
        """Register event handler for specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        
        logger.debug(
            "Event handler registered",
            event_type=event_type,
            handler=handler.__name__ if hasattr(handler, '__name__') else str(handler)
        )

    def unregister_handler(self, event_type: str, handler: callable) -> None:
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

    def get_published_events(self) -> List[Any]:
        """Get list of published events (for testing/debugging)."""
        return self._published_events.copy()

    def clear_events(self) -> None:
        """Clear published events list (for testing)."""
        self._published_events.clear()

    def get_handlers(self, event_type: str) -> List[callable]:
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