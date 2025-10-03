"""Pure domain representation of user aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.domain.value_objects import UserId, TenantId, EmailAddress


class UserRole(str, Enum):
    """User role types in the system."""

    ADMIN = "admin"
    RECRUITER = "recruiter"
    VIEWER = "viewer"
    SUPER_ADMIN = "super_admin"


class UserStatus(str, Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    DELETED = "deleted"


@dataclass
class UserPreferences:
    """User preferences and settings."""

    language: str = "en"
    timezone: str = "UTC"
    email_notifications: bool = True
    push_notifications: bool = True
    weekly_digest: bool = True
    marketing_emails: bool = False
    theme: str = "light"
    items_per_page: int = 20

    def update_notification_settings(
        self, 
        email: Optional[bool] = None,
        push: Optional[bool] = None,
        digest: Optional[bool] = None,
        marketing: Optional[bool] = None
    ) -> None:
        """Update notification preferences."""
        if email is not None:
            self.email_notifications = email
        if push is not None:
            self.push_notifications = push
        if digest is not None:
            self.weekly_digest = digest
        if marketing is not None:
            self.marketing_emails = marketing


@dataclass
class UserActivity:
    """User activity tracking."""

    last_login_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    login_count: int = 0
    session_count: int = 0
    failed_login_attempts: int = 0
    last_failed_login_at: Optional[datetime] = None

    def record_successful_login(self) -> None:
        """Record a successful login."""
        now = datetime.utcnow()
        self.last_login_at = now
        self.last_active_at = now
        self.login_count += 1
        self.session_count += 1
        self.failed_login_attempts = 0  # Reset failed attempts

    def record_failed_login(self) -> None:
        """Record a failed login attempt."""
        self.failed_login_attempts += 1
        self.last_failed_login_at = datetime.utcnow()

    def record_activity(self) -> None:
        """Record user activity."""
        self.last_active_at = datetime.utcnow()

    def is_account_locked(self, max_attempts: int = 5) -> bool:
        """Check if account should be locked due to failed attempts."""
        return self.failed_login_attempts >= max_attempts


@dataclass
class UserSecurity:
    """User security settings and information."""

    password_hash: str
    password_salt: Optional[str] = None
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None
    email_verification_token: Optional[str] = None
    email_verified: bool = False
    email_verified_at: Optional[datetime] = None
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    recovery_codes: List[str] = field(default_factory=list)

    def verify_email(self) -> None:
        """Mark email as verified."""
        self.email_verified = True
        self.email_verified_at = datetime.utcnow()
        self.email_verification_token = None

    def set_password_reset_token(self, token: str, expires_at: datetime) -> None:
        """Set password reset token with expiration."""
        self.password_reset_token = token
        self.password_reset_expires = expires_at

    def clear_password_reset_token(self) -> None:
        """Clear password reset token."""
        self.password_reset_token = None
        self.password_reset_expires = None

    def is_password_reset_valid(self) -> bool:
        """Check if password reset token is still valid."""
        if not self.password_reset_token or not self.password_reset_expires:
            return False
        return datetime.utcnow() < self.password_reset_expires

    def enable_two_factor(self, secret: str, recovery_codes: List[str]) -> None:
        """Enable two-factor authentication."""
        self.two_factor_enabled = True
        self.two_factor_secret = secret
        self.recovery_codes = recovery_codes

    def disable_two_factor(self) -> None:
        """Disable two-factor authentication."""
        self.two_factor_enabled = False
        self.two_factor_secret = None
        self.recovery_codes = []


@dataclass
class User:
    """Aggregate root representing a system user."""

    id: UserId
    tenant_id: TenantId
    email: EmailAddress
    full_name: str
    role: UserRole
    status: UserStatus
    security: UserSecurity
    preferences: UserPreferences = field(default_factory=UserPreferences)
    activity: UserActivity = field(default_factory=UserActivity)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def authenticate(self, provided_password: str, password_verifier) -> bool:
        """Authenticate user with provided password."""
        if self.status != UserStatus.ACTIVE:
            return False
        
        if self.activity.is_account_locked():
            return False
        
        is_valid = password_verifier(provided_password, self.security.password_hash)
        
        if is_valid:
            self.activity.record_successful_login()
            self.updated_at = datetime.utcnow()
        else:
            self.activity.record_failed_login()
        
        return is_valid

    def change_password(self, new_password_hash: str) -> None:
        """Change user password."""
        self.security.password_hash = new_password_hash
        self.security.clear_password_reset_token()
        self.updated_at = datetime.utcnow()

    def update_profile(self, full_name: Optional[str] = None) -> None:
        """Update user profile information."""
        if full_name is not None:
            self.full_name = full_name
        
        self.updated_at = datetime.utcnow()

    def change_role(self, new_role: UserRole, changed_by: UserId) -> None:
        """Change user role (domain business rule: only admins can change roles)."""
        self.role = new_role
        self.metadata["role_changed_by"] = str(changed_by)
        self.metadata["role_changed_at"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()

    def activate(self) -> None:
        """Activate user account."""
        if self.status == UserStatus.DELETED:
            raise ValueError("Cannot activate deleted user")
        
        self.status = UserStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Deactivate user account."""
        self.status = UserStatus.INACTIVE
        self.updated_at = datetime.utcnow()

    def suspend(self, reason: Optional[str] = None) -> None:
        """Suspend user account."""
        self.status = UserStatus.SUSPENDED
        if reason:
            self.metadata["suspension_reason"] = reason
        self.metadata["suspended_at"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()

    def mark_deleted(self) -> None:
        """Soft delete user account."""
        self.status = UserStatus.DELETED
        self.updated_at = datetime.utcnow()

    def verify_email(self) -> None:
        """Verify user email address."""
        self.security.verify_email()
        if self.status == UserStatus.PENDING_VERIFICATION:
            self.status = UserStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def record_activity(self) -> None:
        """Record user activity."""
        self.activity.record_activity()

    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]

    def can_manage_users(self) -> bool:
        """Check if user can manage other users."""
        return self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]

    def can_view_analytics(self) -> bool:
        """Check if user can view analytics."""
        return self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN, UserRole.RECRUITER]

    def can_search_profiles(self) -> bool:
        """Check if user can search profiles."""
        return self.status == UserStatus.ACTIVE and self.role != UserRole.VIEWER

    def can_upload_profiles(self) -> bool:
        """Check if user can upload profiles."""
        return self.status == UserStatus.ACTIVE and self.role in [UserRole.ADMIN, UserRole.RECRUITER]

    def get_display_name(self) -> str:
        """Get user display name."""
        return self.full_name or str(self.email)

    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE

    def days_since_last_login(self) -> Optional[int]:
        """Calculate days since last login."""
        if not self.activity.last_login_at:
            return None
        
        delta = datetime.utcnow() - self.activity.last_login_at
        return delta.days


__all__ = [
    "User",
    "UserRole",
    "UserStatus",
    "UserPreferences",
    "UserActivity",
    "UserSecurity",
]