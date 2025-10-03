"""Tenant-related domain events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import DomainEvent
from app.domain.value_objects import TenantId, UserId
from app.domain.entities.tenant import SubscriptionTier, TenantStatus


@dataclass
class TenantCreatedEvent(DomainEvent):
    """Event fired when a new tenant is created."""
    
    name: str
    tenant_type: str
    subscription_tier: SubscriptionTier
    created_by_user_id: Optional[UserId] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "name": self.name,
            "tenant_type": self.tenant_type,
            "subscription_tier": self.subscription_tier.value,
            "created_by_user_id": str(self.created_by_user_id) if self.created_by_user_id else None,
        })
        return base


@dataclass
class TenantSubscriptionChangedEvent(DomainEvent):
    """Event fired when a tenant's subscription changes."""
    
    previous_tier: SubscriptionTier
    new_tier: SubscriptionTier
    changed_by_user_id: Optional[UserId] = None
    billing_cycle_months: int = 1
    amount_paid: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "previous_tier": self.previous_tier.value,
            "new_tier": self.new_tier.value,
            "changed_by_user_id": str(self.changed_by_user_id) if self.changed_by_user_id else None,
            "billing_cycle_months": self.billing_cycle_months,
            "amount_paid": self.amount_paid,
        })
        return base


@dataclass
class TenantLimitExceededEvent(DomainEvent):
    """Event fired when a tenant exceeds usage limits."""
    
    limit_type: str  # "users", "profiles", "storage", "searches", "api_calls"
    current_usage: int
    limit_value: int
    exceeded_by: int
    action_taken: str  # "blocked", "warned", "auto_upgrade", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "limit_type": self.limit_type,
            "current_usage": self.current_usage,
            "limit_value": self.limit_value,
            "exceeded_by": self.exceeded_by,
            "action_taken": self.action_taken,
        })
        return base


@dataclass
class TenantDeletedEvent(DomainEvent):
    """Event fired when a tenant is deleted."""
    
    name: str
    deleted_by_user_id: Optional[UserId] = None
    deletion_reason: Optional[str] = None
    soft_delete: bool = True
    data_retention_days: int = 90
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "name": self.name,
            "deleted_by_user_id": str(self.deleted_by_user_id) if self.deleted_by_user_id else None,
            "deletion_reason": self.deletion_reason,
            "soft_delete": self.soft_delete,
            "data_retention_days": self.data_retention_days,
        })
        return base


@dataclass
class TenantStatusChangedEvent(DomainEvent):
    """Event fired when a tenant's status changes."""
    
    previous_status: TenantStatus
    new_status: TenantStatus
    changed_by_user_id: Optional[UserId] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "previous_status": self.previous_status.value,
            "new_status": self.new_status.value,
            "changed_by_user_id": str(self.changed_by_user_id) if self.changed_by_user_id else None,
            "reason": self.reason,
        })
        return base


@dataclass
class TenantSubscriptionExpiredEvent(DomainEvent):
    """Event fired when a tenant's subscription expires."""
    
    subscription_tier: SubscriptionTier
    expired_at: str  # ISO datetime string
    grace_period_days: int = 7
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "subscription_tier": self.subscription_tier.value,
            "expired_at": self.expired_at,
            "grace_period_days": self.grace_period_days,
        })
        return base


@dataclass
class TenantUsageThresholdReachedEvent(DomainEvent):
    """Event fired when a tenant reaches usage thresholds."""
    
    usage_type: str  # "users", "profiles", "storage", etc.
    threshold_percentage: float  # 80.0 for 80% threshold
    current_usage: int
    limit_value: int
    warning_level: str  # "low", "medium", "high", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "usage_type": self.usage_type,
            "threshold_percentage": self.threshold_percentage,
            "current_usage": self.current_usage,
            "limit_value": self.limit_value,
            "warning_level": self.warning_level,
        })
        return base


__all__ = [
    "TenantCreatedEvent",
    "TenantSubscriptionChangedEvent",
    "TenantLimitExceededEvent",
    "TenantDeletedEvent",
    "TenantStatusChangedEvent",
    "TenantSubscriptionExpiredEvent",
    "TenantUsageThresholdReachedEvent",
]