"""
Authentication and authorization SQLModel models.

This module provides SQLModel-based authentication models with database persistence
while preserving all original Pydantic functionality and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB
from sqlalchemy.sql import func
from sqlmodel import Field, SQLModel, Relationship
from pydantic import field_validator

from .base import AuditableModel, BaseModel, TenantModel


# Validation and data models (non-table models)
class UserRole(BaseModel):
    """User role definition for validation and API responses."""
    name: str = Field(..., description="Role name")
    permissions: List[str] = Field(..., description="List of permissions")
    description: Optional[str] = Field(None, description="Role description")


class Permission(BaseModel):
    """Permission definition for validation and API responses."""
    name: str = Field(..., description="Permission name")
    resource: str = Field(..., description="Resource type")
    action: str = Field(..., description="Action type")
    description: Optional[str] = Field(None, description="Permission description")


# Database table models
class UserTable(TenantModel, table=True):
    """
    User account table model with SQLModel.
    
    Matches the actual database schema for the users table.
    """
    __tablename__ = "users"
    
    # Core user fields
    email: str = Field(
        sa_column=Column(String(255), nullable=False, unique=True, index=True),
        description="Email address"
    )
    hashed_password: str = Field(
        sa_column=Column(String(255), nullable=False),
        description="Hashed password"
    )
    first_name: str = Field(
        sa_column=Column(String(255), nullable=False),
        description="First name"
    )
    last_name: str = Field(
        sa_column=Column(String(255), nullable=False),
        description="Last name"
    )
    full_name: str = Field(
        sa_column=Column(String(255), nullable=False),
        description="Full name"
    )
    
    # Status flags
    is_active: bool = Field(
        default=True,
        sa_column=Column(Boolean, default=True, nullable=False, index=True),
        description="Account is active"
    )
    is_verified: bool = Field(
        default=False,
        sa_column=Column(Boolean, default=False, nullable=False),
        description="Email is verified"
    )
    is_superuser: bool = Field(
        default=False,
        sa_column=Column(Boolean, default=False, nullable=False),
        description="Superuser status"
    )
    
    # Roles and permissions (JSONB arrays)
    roles: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, default=list),
        description="Assigned roles array"
    )
    permissions: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, default=list),
        description="Direct permissions array"
    )
    
    # Activity tracking
    last_login_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
        description="Last login timestamp"
    )
    failed_login_attempts: int = Field(
        default=0,
        sa_column=Column(Integer, default=0, nullable=False),
        description="Failed login attempts counter"
    )
    locked_until: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
        description="Account locked until timestamp"
    )
    
    # Profile settings and AI preferences (JSONB)
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="User settings and preferences"
    )
    ai_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default={}),
        description="AI-related preferences"
    )
    
    # User methods (preserve original functionality)
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def add_role(self, role: str) -> None:
        """Add a role if not already present."""
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: str) -> None:
        """Remove a role if present."""
        if role in self.roles:
            self.roles.remove(role)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def add_permission(self, permission: str) -> None:
        """Add a permission if not already present."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str) -> None:
        """Remove a permission if present."""
        if permission in self.permissions:
            self.permissions.remove(permission)
    
    def update_login(self) -> None:
        """Update login tracking information."""
        self.last_login_at = datetime.utcnow()
        self.failed_login_attempts = 0  # Reset on successful login
        self.locked_until = None
        self.updated_at = datetime.utcnow()
    
    def increment_failed_login(self) -> None:
        """Increment failed login attempts."""
        self.failed_login_attempts += 1
        self.updated_at = datetime.utcnow()
    
    def lock_account(self, until: datetime) -> None:
        """Lock account until specified time."""
        self.locked_until = until
        self.updated_at = datetime.utcnow()
    
    def unlock_account(self) -> None:
        """Unlock account and reset failed attempts."""
        self.locked_until = None
        self.failed_login_attempts = 0
        self.updated_at = datetime.utcnow()
    
    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until


class UserSessionTable(AuditableModel, table=True):
    """
    User session management table.
    
    Tracks active user sessions for security and session management.
    """
    __tablename__ = "user_sessions"
    
    user_id: UUID = Field(
        sa_column=Column(
            PostgreSQLUUID(as_uuid=True), 
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, 
            index=True
        ),
        description="User ID with CASCADE DELETE constraint"
    )
    
    session_token: str = Field(
        sa_column=Column(String(255), nullable=False, unique=True, index=True),
        description="Session token hash"
    )
    
    ip_address: str = Field(
        sa_column=Column(String(45), nullable=False),  # IPv6 support
        description="Client IP address"
    )
    
    user_agent: str = Field(
        sa_column=Column(Text, nullable=False),
        description="Client user agent"
    )
    
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
        description="Session expiration time"
    )
    
    last_activity: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True), 
            nullable=False, 
            default=func.now(),
            onupdate=func.now()
        ),
        description="Last activity timestamp"
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def extend_session(self, additional_seconds: int = 3600) -> None:
        """Extend session expiration time."""
        self.expires_at = datetime.utcnow() + datetime.timedelta(seconds=additional_seconds)
        self.last_activity = datetime.utcnow()
        self.update_timestamp()


class APIKeyTable(AuditableModel, table=True):
    """
    API Key management table.
    
    Manages API keys for programmatic access to the system.
    """
    __tablename__ = "api_keys"
    
    key_hash: str = Field(
        sa_column=Column(String(255), nullable=False, unique=True, index=True),
        description="Hashed API key"
    )
    
    name: str = Field(
        sa_column=Column(String(255), nullable=False),
        description="Key name/description"
    )
    
    permissions: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, default=list),
        description="API key permissions"
    )
    
    # Usage tracking
    usage_count: int = Field(
        default=0,
        sa_column=Column(Integer, default=0, nullable=False),
        description="Number of uses"
    )
    
    last_used_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
        description="Last used timestamp"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=1000,
        sa_column=Column(Integer, default=1000, nullable=False),
        description="Requests per hour limit"
    )
    
    # Expiration
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True, index=True),
        description="Expiration timestamp"
    )
    
    def record_usage(self) -> None:
        """Record API key usage."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
        self.update_timestamp()
    
    @property
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission."""
        return permission in self.permissions


# Request/Response models (preserve original API contracts)
class UserCreate(BaseModel):
    """User creation request model."""
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: str = Field(..., description="Full name")
    username: Optional[str] = Field(None, description="Username (optional)")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Basic email validation."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v.lower()


class UserUpdate(BaseModel):
    """User update request model."""
    full_name: Optional[str] = None
    username: Optional[str] = None
    roles: Optional[List[str]] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


class UserLogin(BaseModel):
    """User login request model."""
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant authentication")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        return v.lower()


class TokenData(BaseModel):
    """JWT token payload model."""
    sub: str = Field(..., description="User ID")  # Subject
    email: str = Field(..., description="Email address")
    tenant_id: str = Field(..., description="Tenant ID")
    roles: List[str] = Field(..., description="User roles")
    permissions: List[str] = Field(..., description="User permissions")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    token_type: str = Field("access", description="Token type")


class TokenResponse(BaseModel):
    """Authentication token response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str = Field(..., description="Refresh token")


class CurrentUser(BaseModel):
    """Current authenticated user context model."""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="Email address")
    full_name: str = Field(..., description="Full name")
    tenant_id: str = Field(..., description="Tenant ID")
    roles: List[str] = Field(..., description="User roles")
    permissions: List[str] = Field(..., description="User permissions")
    is_active: bool = Field(..., description="Account is active")
    is_superuser: bool = Field(False, description="Superuser status")
    
    @classmethod
    def from_user_table(cls, user: UserTable) -> "CurrentUser":
        """Create CurrentUser from UserTable instance."""
        return cls(
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            tenant_id=str(user.tenant_id),
            roles=user.roles,
            permissions=user.permissions,
            is_active=user.is_active,
            is_superuser=user.is_superuser
        )


class APIKeyCreate(BaseModel):
    """API Key creation request model."""
    name: str = Field(..., description="Key name/description")
    permissions: List[str] = Field(..., description="Key permissions")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")
    rate_limit_requests: int = Field(1000, ge=1, description="Requests per hour limit")


class APIKeyResponse(BaseModel):
    """API Key creation response model."""
    api_key: str = Field(..., description="The actual API key (shown only once)")
    key_id: str = Field(..., description="Key identifier")
    name: str = Field(..., description="Key name")
    permissions: List[str] = Field(..., description="Key permissions")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    created_at: str = Field(..., description="Creation timestamp")


class APIKeyInfo(BaseModel):
    """API Key information model (without the key itself)."""
    id: str
    name: str
    permissions: List[str]
    is_active: bool
    usage_count: int
    last_used_at: Optional[str]
    created_at: str
    expires_at: Optional[str]
    rate_limit_requests: int
    
    @classmethod
    def from_api_key_table(cls, api_key: APIKeyTable) -> "APIKeyInfo":
        """Create APIKeyInfo from APIKeyTable instance."""
        return cls(
            id=str(api_key.id),
            name=api_key.name,
            permissions=api_key.permissions,
            is_active=not api_key.is_deleted,
            usage_count=api_key.usage_count,
            last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            created_at=api_key.created_at.isoformat(),
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
            rate_limit_requests=api_key.rate_limit_requests
        )


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: str = Field(..., description="Email address")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        return v.lower()


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class TenantContext(BaseModel):
    """Tenant context for requests."""
    tenant_id: str = Field(..., description="Tenant ID")
    tenant_type: str = Field(..., description="Tenant type")
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(True, description="Tenant is active")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split(".")
        value = self.configuration
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


class AuthenticationResult(BaseModel):
    """Authentication result model."""
    success: bool = Field(..., description="Authentication successful")
    user: Optional[CurrentUser] = Field(None, description="User information")
    tokens: Optional[TokenResponse] = Field(None, description="Authentication tokens")
    error: Optional[str] = Field(None, description="Error message")


class AuthorizationResult(BaseModel):
    """Authorization check result model."""
    allowed: bool = Field(..., description="Access is allowed")
    reason: Optional[str] = Field(None, description="Reason if denied")
    required_permissions: Optional[List[str]] = Field(None, description="Required permissions")


class SessionInfo(BaseModel):
    """User session information model."""
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    tenant_id: str = Field(..., description="Tenant ID")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str = Field(..., description="Client user agent")
    created_at: str = Field(..., description="Session start time")
    last_activity: str = Field(..., description="Last activity time")
    expires_at: str = Field(..., description="Session expiration time")
    
    @classmethod
    def from_session_table(cls, session: UserSessionTable, user: UserTable) -> "SessionInfo":
        """Create SessionInfo from UserSessionTable instance."""
        return cls(
            session_id=str(session.id),
            user_id=str(session.user_id),
            tenant_id=str(user.tenant_id),
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            created_at=session.created_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            expires_at=session.expires_at.isoformat()
        )


class AuditLog(BaseModel):
    """Security audit log entry model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="Tenant ID")
    user_id: Optional[str] = Field(None, description="User ID")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: str = Field(..., description="Client user agent")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Risk assessment
    risk_level: str = Field("low", description="Risk level: low, medium, high")
    suspicious: bool = Field(False, description="Flagged as suspicious activity")


# Backward compatibility aliases
User = UserTable  # For backward compatibility during migration