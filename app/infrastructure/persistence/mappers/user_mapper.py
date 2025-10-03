"""Mapper between User domain entities and UserTable persistence models."""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from app.domain.entities.user import (
    User,
    UserRole,
    UserStatus,
    UserPreferences,
    UserActivity,
    UserSecurity
)
from app.domain.value_objects import (
    UserId,
    TenantId,
    EmailAddress
)
from app.models.auth import UserTable  # SQLModel persistence model


class UserMapper:
    """Maps between User domain entities and UserTable persistence models."""

    @staticmethod
    def to_domain(user_table: UserTable) -> User:
        """Convert UserTable (persistence) to User (domain)."""
        # Map user preferences
        preferences_dict = user_table.settings.get('preferences', {}) if user_table.settings else {}
        preferences = UserMapper._map_preferences_to_domain(preferences_dict)
        
        # Map user activity
        activity = UserMapper._map_activity_to_domain(
            user_table.last_login_at,
            user_table.failed_login_attempts,
            user_table.locked_until
        )
        
        # Map user security
        security = UserMapper._map_security_to_domain(
            user_table.hashed_password,
            user_table.is_verified,
            user_table.settings
        )
        
        # Map role from roles array (take first role or default to VIEWER)
        user_role = UserRole.VIEWER
        if user_table.roles and len(user_table.roles) > 0:
            role_str = user_table.roles[0].upper()
            try:
                user_role = UserRole(role_str)
            except ValueError:
                user_role = UserRole.VIEWER
        
        # Map status
        user_status = UserStatus.ACTIVE if user_table.is_active else UserStatus.INACTIVE
        if hasattr(user_table, 'is_deleted') and user_table.is_deleted:
            user_status = UserStatus.DELETED
        elif user_table.is_locked:
            user_status = UserStatus.SUSPENDED
        elif not user_table.is_verified:
            user_status = UserStatus.PENDING_VERIFICATION
        
        # Create domain entity
        return User(
            id=UserId(user_table.id),
            tenant_id=TenantId(user_table.tenant_id),
            email=EmailAddress(user_table.email),
            full_name=user_table.full_name,
            role=user_role,
            status=user_status,
            security=security,
            preferences=preferences,
            activity=activity,
            metadata=user_table.settings.get('metadata', {}) if user_table.settings else {},
            created_at=user_table.created_at,
            updated_at=user_table.updated_at
        )

    @staticmethod
    def to_persistence(user: User) -> UserTable:
        """Convert User (domain) to UserTable (persistence)."""
        # Map preferences to settings
        settings_dict = UserMapper._map_preferences_to_persistence(user.preferences)
        settings_dict['metadata'] = user.metadata
        
        # Map security to persistence fields
        security_dict = UserMapper._map_security_to_persistence(user.security)
        settings_dict.update(security_dict)
        
        # Split full name into first and last name (simple split)
        name_parts = user.full_name.split(' ', 1)
        first_name = name_parts[0] if name_parts else ''
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        # Create persistence model
        user_table = UserTable(
            id=user.id.value,
            tenant_id=user.tenant_id.value,
            email=str(user.email),
            hashed_password=user.security.password_hash,
            first_name=first_name,
            last_name=last_name,
            full_name=user.full_name,
            is_active=user.status == UserStatus.ACTIVE,
            is_verified=user.security.email_verified,
            is_superuser=user.role == UserRole.SUPER_ADMIN,
            roles=[user.role.value],
            permissions=[],  # Could be expanded to map permissions
            last_login_at=user.activity.last_login_at,
            failed_login_attempts=user.activity.failed_login_attempts,
            locked_until=user.activity.last_failed_login_at if user.activity.is_account_locked() else None,
            settings=settings_dict,
            ai_preferences={},  # Could be expanded
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
        return user_table

    @staticmethod
    def _map_preferences_to_domain(preferences_dict: Dict[str, Any]) -> UserPreferences:
        """Map settings preferences to domain UserPreferences."""
        return UserPreferences(
            language=preferences_dict.get('language', 'en'),
            timezone=preferences_dict.get('timezone', 'UTC'),
            email_notifications=preferences_dict.get('email_notifications', True),
            push_notifications=preferences_dict.get('push_notifications', True),
            weekly_digest=preferences_dict.get('weekly_digest', True),
            marketing_emails=preferences_dict.get('marketing_emails', False),
            theme=preferences_dict.get('theme', 'light'),
            items_per_page=preferences_dict.get('items_per_page', 20)
        )

    @staticmethod
    def _map_preferences_to_persistence(preferences: UserPreferences) -> Dict[str, Any]:
        """Map domain UserPreferences to settings dict."""
        return {
            'preferences': {
                'language': preferences.language,
                'timezone': preferences.timezone,
                'email_notifications': preferences.email_notifications,
                'push_notifications': preferences.push_notifications,
                'weekly_digest': preferences.weekly_digest,
                'marketing_emails': preferences.marketing_emails,
                'theme': preferences.theme,
                'items_per_page': preferences.items_per_page
            }
        }

    @staticmethod
    def _map_activity_to_domain(
        last_login_at: Optional[datetime],
        failed_login_attempts: int,
        locked_until: Optional[datetime]
    ) -> UserActivity:
        """Map activity fields to domain UserActivity."""
        return UserActivity(
            last_login_at=last_login_at,
            last_active_at=last_login_at,  # Use last_login as approximation
            login_count=0,  # Not tracked in current schema
            session_count=0,  # Not tracked in current schema
            failed_login_attempts=failed_login_attempts,
            last_failed_login_at=locked_until  # Approximation
        )

    @staticmethod
    def _map_security_to_domain(
        hashed_password: str,
        is_verified: bool,
        settings: Dict[str, Any]
    ) -> UserSecurity:
        """Map security fields to domain UserSecurity."""
        security_settings = settings.get('security', {}) if settings else {}
        
        return UserSecurity(
            password_hash=hashed_password,
            password_salt=security_settings.get('password_salt'),
            password_reset_token=security_settings.get('password_reset_token'),
            password_reset_expires=security_settings.get('password_reset_expires'),
            email_verification_token=security_settings.get('email_verification_token'),
            email_verified=is_verified,
            email_verified_at=security_settings.get('email_verified_at'),
            two_factor_enabled=security_settings.get('two_factor_enabled', False),
            two_factor_secret=security_settings.get('two_factor_secret'),
            recovery_codes=security_settings.get('recovery_codes', [])
        )

    @staticmethod
    def _map_security_to_persistence(security: UserSecurity) -> Dict[str, Any]:
        """Map domain UserSecurity to settings dict."""
        return {
            'security': {
                'password_salt': security.password_salt,
                'password_reset_token': security.password_reset_token,
                'password_reset_expires': security.password_reset_expires.isoformat() if security.password_reset_expires else None,
                'email_verification_token': security.email_verification_token,
                'email_verified_at': security.email_verified_at.isoformat() if security.email_verified_at else None,
                'two_factor_enabled': security.two_factor_enabled,
                'two_factor_secret': security.two_factor_secret,
                'recovery_codes': security.recovery_codes
            }
        }

    @staticmethod
    def update_persistence_from_domain(user_table: UserTable, user: User) -> UserTable:
        """Update existing UserTable with data from User domain entity."""
        # Update basic fields
        user_table.email = str(user.email)
        user_table.full_name = user.full_name
        
        # Split full name
        name_parts = user.full_name.split(' ', 1)
        user_table.first_name = name_parts[0] if name_parts else ''
        user_table.last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        # Update status fields
        user_table.is_active = user.status == UserStatus.ACTIVE
        user_table.is_verified = user.security.email_verified
        user_table.is_superuser = user.role == UserRole.SUPER_ADMIN
        
        # Update roles
        user_table.roles = [user.role.value]
        
        # Update activity fields
        user_table.last_login_at = user.activity.last_login_at
        user_table.failed_login_attempts = user.activity.failed_login_attempts
        user_table.locked_until = user.activity.last_failed_login_at if user.activity.is_account_locked() else None
        
        # Update security
        user_table.hashed_password = user.security.password_hash
        
        # Update settings
        settings_dict = UserMapper._map_preferences_to_persistence(user.preferences)
        settings_dict['metadata'] = user.metadata
        security_dict = UserMapper._map_security_to_persistence(user.security)
        settings_dict.update(security_dict)
        user_table.settings = settings_dict
        
        # Update timestamps
        user_table.updated_at = user.updated_at
        
        return user_table


__all__ = ["UserMapper"]