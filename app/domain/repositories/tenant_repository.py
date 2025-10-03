"""Domain repository interface for Tenant aggregates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.entities.tenant import Tenant, TenantStatus, TenantType, SubscriptionTier
from app.domain.value_objects import TenantId


class ITenantRepository(ABC):
    """Repository interface for Tenant aggregate."""

    @abstractmethod
    async def save(self, tenant: Tenant) -> Tenant:
        """Save a tenant to persistent storage."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, tenant_id: TenantId) -> Optional[Tenant]:
        """Get a tenant by ID."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Tenant]:
        """Get a tenant by name."""
        raise NotImplementedError

    @abstractmethod
    async def list_all(
        self,
        status: Optional[TenantStatus] = None,
        tenant_type: Optional[TenantType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """List all tenants with optional filtering."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, tenant_id: TenantId) -> bool:
        """Delete a tenant (hard delete)."""
        raise NotImplementedError

    @abstractmethod
    async def count_all(self, status: Optional[TenantStatus] = None) -> int:
        """Count all tenants."""
        raise NotImplementedError

    @abstractmethod
    async def get_system_tenant(self) -> Optional[Tenant]:
        """Get the system tenant."""
        raise NotImplementedError

    @abstractmethod
    async def list_by_subscription_tier(
        self,
        tier: SubscriptionTier,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """List tenants by subscription tier."""
        raise NotImplementedError

    @abstractmethod
    async def list_expiring_subscriptions(
        self,
        days_until_expiry: int = 7
    ) -> List[Tenant]:
        """List tenants with subscriptions expiring soon."""
        raise NotImplementedError

    @abstractmethod
    async def list_over_limits(self) -> List[Tenant]:
        """List tenants that have exceeded their usage limits."""
        raise NotImplementedError

    @abstractmethod
    async def update_usage_counters(
        self,
        tenant_id: TenantId,
        user_count_delta: int = 0,
        profile_count_delta: int = 0,
        storage_delta_gb: float = 0.0,
        searches_delta: int = 0,
        api_calls_delta: int = 0
    ) -> bool:
        """Update tenant usage counters atomically."""
        raise NotImplementedError


__all__ = ["ITenantRepository"]