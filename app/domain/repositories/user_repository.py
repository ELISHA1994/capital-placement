"""Domain repository interface for User aggregates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.entities.user import User, UserRole, UserStatus
from app.domain.value_objects import UserId, TenantId, EmailAddress


class IUserRepository(ABC):
    """Repository interface for User aggregate."""

    @abstractmethod
    async def save(self, user: User) -> User:
        """Save a user to persistent storage."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, user_id: UserId, tenant_id: TenantId) -> Optional[User]:
        """Get a user by ID within tenant scope."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_email(self, email: EmailAddress, tenant_id: TenantId) -> Optional[User]:
        """Get a user by email within tenant scope."""
        raise NotImplementedError

    @abstractmethod
    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        role: Optional[UserRole] = None,
        status: Optional[UserStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[User]:
        """List users for a tenant with optional filtering."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, user_id: UserId, tenant_id: TenantId) -> bool:
        """Delete a user (hard delete)."""
        raise NotImplementedError

    @abstractmethod
    async def count_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[UserStatus] = None
    ) -> int:
        """Count users for a tenant."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_password_reset_token(self, token: str) -> Optional[User]:
        """Get user by password reset token."""
        raise NotImplementedError

    @abstractmethod
    async def get_by_email_verification_token(self, token: str) -> Optional[User]:
        """Get user by email verification token."""
        raise NotImplementedError

    @abstractmethod
    async def list_inactive_users(
        self,
        tenant_id: TenantId,
        days_inactive: int = 90
    ) -> List[User]:
        """List users who have been inactive for specified days."""
        raise NotImplementedError

    @abstractmethod
    async def get_admins_by_tenant(self, tenant_id: TenantId) -> List[User]:
        """Get all admin users for a tenant."""
        raise NotImplementedError


__all__ = ["IUserRepository"]