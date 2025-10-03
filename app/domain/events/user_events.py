"""User-related domain events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import DomainEvent
from app.domain.value_objects import UserId, TenantId
from app.domain.entities.user import UserRole


@dataclass
class UserCreatedEvent(DomainEvent):
    """Event fired when a new user is created."""
    
    user_id: UserId
    email: str
    full_name: str
    role: UserRole
    created_by_user_id: Optional[UserId] = None
    invitation_sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value,
            "created_by_user_id": str(self.created_by_user_id) if self.created_by_user_id else None,
            "invitation_sent": self.invitation_sent,
        })
        return base


@dataclass
class UserLoggedInEvent(DomainEvent):
    """Event fired when a user logs in."""
    
    user_id: UserId
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    login_method: str = "password"  # "password", "sso", "token", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "login_method": self.login_method,
        })
        return base


@dataclass
class UserPasswordChangedEvent(DomainEvent):
    """Event fired when a user changes their password."""
    
    user_id: UserId
    changed_by_user_id: UserId  # May be different for admin password resets
    reset_token_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "changed_by_user_id": str(self.changed_by_user_id),
            "reset_token_used": self.reset_token_used,
        })
        return base


@dataclass
class UserRoleChangedEvent(DomainEvent):
    """Event fired when a user's role is changed."""
    
    user_id: UserId
    previous_role: UserRole
    new_role: UserRole
    changed_by_user_id: UserId
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "previous_role": self.previous_role.value,
            "new_role": self.new_role.value,
            "changed_by_user_id": str(self.changed_by_user_id),
            "reason": self.reason,
        })
        return base


@dataclass
class UserDeletedEvent(DomainEvent):
    """Event fired when a user is deleted."""
    
    user_id: UserId
    deleted_by_user_id: UserId
    deletion_reason: Optional[str] = None
    soft_delete: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "deleted_by_user_id": str(self.deleted_by_user_id),
            "deletion_reason": self.deletion_reason,
            "soft_delete": self.soft_delete,
        })
        return base


@dataclass
class UserEmailVerifiedEvent(DomainEvent):
    """Event fired when a user verifies their email."""
    
    user_id: UserId
    email: str
    verification_token_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "email": self.email,
            "verification_token_used": self.verification_token_used,
        })
        return base


@dataclass
class UserLoginFailedEvent(DomainEvent):
    """Event fired when a user login fails."""
    
    email: str
    failure_reason: str  # "invalid_password", "account_locked", "user_not_found", etc.
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "email": self.email,
            "failure_reason": self.failure_reason,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        })
        return base


@dataclass
class UserAccountLockedEvent(DomainEvent):
    """Event fired when a user account is locked due to failed attempts."""
    
    user_id: UserId
    failed_attempts_count: int
    lockout_duration_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "user_id": str(self.user_id),
            "failed_attempts_count": self.failed_attempts_count,
            "lockout_duration_minutes": self.lockout_duration_minutes,
        })
        return base


__all__ = [
    "UserCreatedEvent",
    "UserLoggedInEvent",
    "UserPasswordChangedEvent",
    "UserRoleChangedEvent",
    "UserDeletedEvent",
    "UserEmailVerifiedEvent",
    "UserLoginFailedEvent",
    "UserAccountLockedEvent",
]