"""Base domain event infrastructure."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from app.domain.value_objects import TenantId


@dataclass(kw_only=True)
class DomainEvent:
    """Base class for all domain events."""

    tenant_id: TenantId
    event_id: UUID = field(default_factory=uuid4)
    event_type: str = field(init=False)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None

    def __post_init__(self) -> None:
        """Set event type from class name if not already set."""
        if not hasattr(self, 'event_type') or not self.event_type:
            self.event_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "tenant_id": str(self.tenant_id),
            "occurred_at": self.occurred_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """Create event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            tenant_id=TenantId(data["tenant_id"]),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            version=data.get("version", 1),
            metadata=data.get("metadata", {}),
            correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") else None,
            causation_id=UUID(data["causation_id"]) if data.get("causation_id") else None,
        )


class IDomainEventPublisher(ABC):
    """Interface for publishing domain events."""

    @abstractmethod
    async def publish(self, event: DomainEvent) -> bool:
        """Publish a single domain event."""
        pass

    @abstractmethod
    async def publish_batch(self, events: list[DomainEvent]) -> bool:
        """Publish multiple domain events as a batch."""
        pass

    @abstractmethod
    async def publish_and_wait(self, event: DomainEvent, timeout_seconds: int = 30) -> bool:
        """Publish event and wait for acknowledgment."""
        pass


__all__ = ["DomainEvent", "IDomainEventPublisher"]
