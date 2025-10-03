"""Domain events package."""

from .base import DomainEvent, IDomainEventPublisher
from .profile_events import (
    ProfileCreatedEvent,
    ProfileUpdatedEvent,
    ProfileDeletedEvent,
    ProfileViewedEvent,
    ProfileSearchedEvent,
    ProfileProcessingCompletedEvent,
    ProfileProcessingFailedEvent
)
from .user_events import (
    UserCreatedEvent,
    UserLoggedInEvent,
    UserPasswordChangedEvent,
    UserRoleChangedEvent,
    UserDeletedEvent
)
from .tenant_events import (
    TenantCreatedEvent,
    TenantSubscriptionChangedEvent,
    TenantLimitExceededEvent,
    TenantDeletedEvent
)

__all__ = [
    # Base
    "DomainEvent",
    "IDomainEventPublisher",
    # Profile events
    "ProfileCreatedEvent",
    "ProfileUpdatedEvent",
    "ProfileDeletedEvent",
    "ProfileViewedEvent",
    "ProfileSearchedEvent",
    "ProfileProcessingCompletedEvent",
    "ProfileProcessingFailedEvent",
    # User events
    "UserCreatedEvent",
    "UserLoggedInEvent",
    "UserPasswordChangedEvent",
    "UserRoleChangedEvent",
    "UserDeletedEvent",
    # Tenant events
    "TenantCreatedEvent",
    "TenantSubscriptionChangedEvent",
    "TenantLimitExceededEvent",
    "TenantDeletedEvent",
]