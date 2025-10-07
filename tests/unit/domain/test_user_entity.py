"""
Pure domain tests for User entity.

These tests validate the business logic of the User aggregate root and its value objects
WITHOUT any infrastructure dependencies (no database, no mappers, no infrastructure).

Test Coverage:
- User entity creation and initialization
- UserRole enum handling
- UserStatus state transitions (ACTIVE, INACTIVE, SUSPENDED, PENDING_VERIFICATION, DELETED)
- UserPreferences business logic
- UserActivity tracking and login management
- UserSecurity operations (2FA, email verification, password reset)
- Authentication business rules
- Profile management methods
- Role management and metadata
- Permission methods
- Display name and full name handling
- Days since last login calculations
- Validation rules
- Edge cases and boundary conditions
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from hypothesis import given, strategies as st

from app.domain.entities.user import (
    User,
    UserRole,
    UserStatus,
    UserPreferences,
    UserActivity,
    UserSecurity,
)
from app.domain.value_objects import UserId, TenantId, EmailAddress


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def valid_email() -> EmailAddress:
    """Create valid email address."""
    return EmailAddress("john.doe@example.com")


@pytest.fixture
def valid_tenant_id() -> TenantId:
    """Create valid tenant ID."""
    return TenantId(uuid4())


@pytest.fixture
def valid_user_id() -> UserId:
    """Create valid user ID."""
    return UserId(uuid4())


@pytest.fixture
def valid_security() -> UserSecurity:
    """Create valid user security object."""
    return UserSecurity(
        password_hash="$2b$12$hashedpassword",
        email_verified=False,
    )


@pytest.fixture
def minimal_user(
    valid_user_id: UserId,
    valid_tenant_id: TenantId,
    valid_email: EmailAddress,
    valid_security: UserSecurity,
) -> User:
    """Create user with minimal required fields."""
    return User(
        id=valid_user_id,
        tenant_id=valid_tenant_id,
        email=valid_email,
        full_name="John Doe",
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
        security=valid_security,
    )


@pytest.fixture
def admin_user(
    valid_tenant_id: TenantId,
    valid_email: EmailAddress,
) -> User:
    """Create admin user."""
    return User(
        id=UserId(uuid4()),
        tenant_id=valid_tenant_id,
        email=valid_email,
        full_name="Admin User",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        security=UserSecurity(password_hash="$2b$12$hashedpassword"),
    )


@pytest.fixture
def super_admin_user(
    valid_tenant_id: TenantId,
    valid_email: EmailAddress,
) -> User:
    """Create super admin user."""
    return User(
        id=UserId(uuid4()),
        tenant_id=valid_tenant_id,
        email=valid_email,
        full_name="Super Admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
        security=UserSecurity(password_hash="$2b$12$hashedpassword"),
    )


# ============================================================================
# 1. Entity Creation & Initialization Tests (6 tests)
# ============================================================================


class TestUserCreation:
    """Test user entity creation and initialization."""

    def test_create_user_with_required_fields(
        self,
        valid_user_id: UserId,
        valid_tenant_id: TenantId,
        valid_email: EmailAddress,
        valid_security: UserSecurity,
    ):
        """Should create user with only required fields."""
        # Act
        user = User(
            id=valid_user_id,
            tenant_id=valid_tenant_id,
            email=valid_email,
            full_name="Jane Smith",
            role=UserRole.RECRUITER,
            status=UserStatus.ACTIVE,
            security=valid_security,
        )

        # Assert
        assert user.id == valid_user_id
        assert user.tenant_id == valid_tenant_id
        assert user.email == valid_email
        assert user.full_name == "Jane Smith"
        assert user.role == UserRole.RECRUITER
        assert user.status == UserStatus.ACTIVE
        assert user.security == valid_security
        assert user.created_at is not None
        assert user.updated_at is not None
        assert isinstance(user.created_at, datetime)

    def test_user_default_values_on_creation(self, minimal_user: User):
        """Should initialize with correct default values."""
        # Assert
        assert isinstance(minimal_user.preferences, UserPreferences)
        assert isinstance(minimal_user.activity, UserActivity)
        assert minimal_user.metadata == {}
        assert minimal_user.preferences.language == "en"
        assert minimal_user.preferences.timezone == "UTC"
        assert minimal_user.activity.login_count == 0
        assert minimal_user.activity.last_login_at is None

    def test_create_user_with_preferences(
        self,
        valid_user_id: UserId,
        valid_tenant_id: TenantId,
        valid_email: EmailAddress,
        valid_security: UserSecurity,
    ):
        """Should create user with custom preferences."""
        # Arrange
        custom_prefs = UserPreferences(
            language="es",
            timezone="America/New_York",
            theme="dark",
            items_per_page=50,
        )

        # Act
        user = User(
            id=valid_user_id,
            tenant_id=valid_tenant_id,
            email=valid_email,
            full_name="Test User",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            security=valid_security,
            preferences=custom_prefs,
        )

        # Assert
        assert user.preferences.language == "es"
        assert user.preferences.timezone == "America/New_York"
        assert user.preferences.theme == "dark"
        assert user.preferences.items_per_page == 50

    def test_create_user_with_activity(
        self,
        valid_user_id: UserId,
        valid_tenant_id: TenantId,
        valid_email: EmailAddress,
        valid_security: UserSecurity,
    ):
        """Should create user with existing activity data."""
        # Arrange
        activity = UserActivity(
            login_count=5,
            session_count=10,
            last_login_at=datetime.utcnow(),
        )

        # Act
        user = User(
            id=valid_user_id,
            tenant_id=valid_tenant_id,
            email=valid_email,
            full_name="Test User",
            role=UserRole.RECRUITER,
            status=UserStatus.ACTIVE,
            security=valid_security,
            activity=activity,
        )

        # Assert
        assert user.activity.login_count == 5
        assert user.activity.session_count == 10
        assert user.activity.last_login_at is not None

    def test_create_user_with_metadata(
        self,
        valid_user_id: UserId,
        valid_tenant_id: TenantId,
        valid_email: EmailAddress,
        valid_security: UserSecurity,
    ):
        """Should create user with custom metadata."""
        # Arrange
        metadata = {"source": "invitation", "department": "engineering"}

        # Act
        user = User(
            id=valid_user_id,
            tenant_id=valid_tenant_id,
            email=valid_email,
            full_name="Test User",
            role=UserRole.VIEWER,
            status=UserStatus.ACTIVE,
            security=valid_security,
            metadata=metadata,
        )

        # Assert
        assert user.metadata["source"] == "invitation"
        assert user.metadata["department"] == "engineering"

    def test_create_user_all_roles_are_valid(
        self,
        valid_tenant_id: TenantId,
        valid_email: EmailAddress,
        valid_security: UserSecurity,
    ):
        """Should accept all valid UserRole enum values."""
        # Arrange
        roles = [UserRole.ADMIN, UserRole.RECRUITER, UserRole.VIEWER, UserRole.SUPER_ADMIN]

        # Act & Assert
        for role in roles:
            user = User(
                id=UserId(uuid4()),
                tenant_id=valid_tenant_id,
                email=valid_email,
                full_name="Test User",
                role=role,
                status=UserStatus.ACTIVE,
                security=valid_security,
            )
            assert user.role == role


# ============================================================================
# 2. UserRole Enum Tests (2 tests)
# ============================================================================


class TestUserRoleEnum:
    """Test UserRole enum values and behavior."""

    def test_all_user_role_values_defined(self):
        """Should verify all UserRole enum values are defined."""
        # Assert
        assert UserRole.ADMIN == "admin"
        assert UserRole.RECRUITER == "recruiter"
        assert UserRole.VIEWER == "viewer"
        assert UserRole.SUPER_ADMIN == "super_admin"

    def test_user_role_is_string_enum(self):
        """Should verify UserRole is string-based enum."""
        # Assert
        assert isinstance(UserRole.ADMIN.value, str)
        assert isinstance(UserRole.RECRUITER.value, str)
        assert isinstance(UserRole.VIEWER.value, str)
        assert isinstance(UserRole.SUPER_ADMIN.value, str)


# ============================================================================
# 3. UserStatus State Transition Tests (10 tests)
# ============================================================================


class TestUserStatusTransitions:
    """Test user status state transition methods."""

    def test_activate_sets_status_to_active(self, minimal_user: User):
        """Should change status to ACTIVE and update timestamp."""
        # Arrange
        minimal_user.status = UserStatus.INACTIVE
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.activate()

        # Assert
        assert minimal_user.status == UserStatus.ACTIVE
        assert minimal_user.updated_at > original_updated_at

    def test_activate_raises_error_for_deleted_user(self, minimal_user: User):
        """Should raise ValueError when trying to activate DELETED user."""
        # Arrange
        minimal_user.status = UserStatus.DELETED

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot activate deleted user"):
            minimal_user.activate()

    def test_deactivate_sets_status_to_inactive(self, minimal_user: User):
        """Should change status to INACTIVE and update timestamp."""
        # Arrange
        minimal_user.status = UserStatus.ACTIVE
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.deactivate()

        # Assert
        assert minimal_user.status == UserStatus.INACTIVE
        assert minimal_user.updated_at > original_updated_at

    def test_suspend_sets_status_to_suspended(self, minimal_user: User):
        """Should change status to SUSPENDED and update timestamp."""
        # Arrange
        minimal_user.status = UserStatus.ACTIVE
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.suspend()

        # Assert
        assert minimal_user.status == UserStatus.SUSPENDED
        assert minimal_user.updated_at > original_updated_at

    def test_suspend_with_reason_stores_metadata(self, minimal_user: User):
        """Should store suspension reason in metadata."""
        # Act
        minimal_user.suspend(reason="Multiple failed login attempts")

        # Assert
        assert minimal_user.status == UserStatus.SUSPENDED
        assert minimal_user.metadata["suspension_reason"] == "Multiple failed login attempts"
        assert "suspended_at" in minimal_user.metadata

    def test_suspend_without_reason_works(self, minimal_user: User):
        """Should suspend without reason gracefully."""
        # Act
        minimal_user.suspend()

        # Assert
        assert minimal_user.status == UserStatus.SUSPENDED
        assert "suspension_reason" not in minimal_user.metadata
        assert "suspended_at" in minimal_user.metadata

    def test_mark_deleted_sets_status_to_deleted(self, minimal_user: User):
        """Should soft-delete user by changing status to DELETED."""
        # Arrange
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.mark_deleted()

        # Assert
        assert minimal_user.status == UserStatus.DELETED
        assert minimal_user.updated_at > original_updated_at

    def test_all_user_status_enum_values_exist(self):
        """Should verify all UserStatus enum values are defined."""
        # Assert
        assert UserStatus.ACTIVE == "active"
        assert UserStatus.INACTIVE == "inactive"
        assert UserStatus.SUSPENDED == "suspended"
        assert UserStatus.PENDING_VERIFICATION == "pending_verification"
        assert UserStatus.DELETED == "deleted"

    def test_status_transitions_preserve_other_fields(self, minimal_user: User):
        """Should not modify other fields during status transitions."""
        # Arrange
        original_id = minimal_user.id
        original_email = minimal_user.email
        original_role = minimal_user.role

        # Act
        minimal_user.suspend()

        # Assert - only status and updated_at should change
        assert minimal_user.id == original_id
        assert minimal_user.email == original_email
        assert minimal_user.role == original_role

    def test_is_active_returns_correct_value(self, minimal_user: User):
        """Should return True only when status is ACTIVE."""
        # Test ACTIVE
        minimal_user.status = UserStatus.ACTIVE
        assert minimal_user.is_active() is True

        # Test other statuses
        minimal_user.status = UserStatus.INACTIVE
        assert minimal_user.is_active() is False

        minimal_user.status = UserStatus.SUSPENDED
        assert minimal_user.is_active() is False

        minimal_user.status = UserStatus.DELETED
        assert minimal_user.is_active() is False


# ============================================================================
# 4. UserPreferences Business Logic Tests (8 tests)
# ============================================================================


class TestUserPreferences:
    """Test UserPreferences value object and update methods."""

    def test_preferences_default_values(self):
        """Should initialize with correct default values."""
        # Act
        prefs = UserPreferences()

        # Assert
        assert prefs.language == "en"
        assert prefs.timezone == "UTC"
        assert prefs.email_notifications is True
        assert prefs.push_notifications is True
        assert prefs.weekly_digest is True
        assert prefs.marketing_emails is False
        assert prefs.theme == "light"
        assert prefs.items_per_page == 20

    def test_update_notification_settings_email_only(self, minimal_user: User):
        """Should update only email notification setting."""
        # Act
        minimal_user.preferences.update_notification_settings(email=False)

        # Assert
        assert minimal_user.preferences.email_notifications is False
        assert minimal_user.preferences.push_notifications is True
        assert minimal_user.preferences.weekly_digest is True
        assert minimal_user.preferences.marketing_emails is False

    def test_update_notification_settings_push_only(self, minimal_user: User):
        """Should update only push notification setting."""
        # Act
        minimal_user.preferences.update_notification_settings(push=False)

        # Assert
        assert minimal_user.preferences.email_notifications is True
        assert minimal_user.preferences.push_notifications is False
        assert minimal_user.preferences.weekly_digest is True

    def test_update_notification_settings_digest_only(self, minimal_user: User):
        """Should update only weekly digest setting."""
        # Act
        minimal_user.preferences.update_notification_settings(digest=False)

        # Assert
        assert minimal_user.preferences.email_notifications is True
        assert minimal_user.preferences.push_notifications is True
        assert minimal_user.preferences.weekly_digest is False

    def test_update_notification_settings_marketing_only(self, minimal_user: User):
        """Should update only marketing emails setting."""
        # Act
        minimal_user.preferences.update_notification_settings(marketing=True)

        # Assert
        assert minimal_user.preferences.email_notifications is True
        assert minimal_user.preferences.marketing_emails is True

    def test_update_notification_settings_multiple_values(self, minimal_user: User):
        """Should update multiple notification settings at once."""
        # Act
        minimal_user.preferences.update_notification_settings(
            email=False,
            push=False,
            digest=False,
            marketing=True,
        )

        # Assert
        assert minimal_user.preferences.email_notifications is False
        assert minimal_user.preferences.push_notifications is False
        assert minimal_user.preferences.weekly_digest is False
        assert minimal_user.preferences.marketing_emails is True

    def test_update_notification_settings_with_none_values(self, minimal_user: User):
        """Should skip None values and keep existing settings."""
        # Arrange - set initial values
        minimal_user.preferences.email_notifications = False

        # Act - pass None for email
        minimal_user.preferences.update_notification_settings(
            email=None,
            push=False,
        )

        # Assert - email should remain unchanged
        assert minimal_user.preferences.email_notifications is False
        assert minimal_user.preferences.push_notifications is False

    def test_preferences_custom_values(self):
        """Should accept and store custom preference values."""
        # Act
        prefs = UserPreferences(
            language="fr",
            timezone="Europe/Paris",
            theme="dark",
            items_per_page=100,
        )

        # Assert
        assert prefs.language == "fr"
        assert prefs.timezone == "Europe/Paris"
        assert prefs.theme == "dark"
        assert prefs.items_per_page == 100


# ============================================================================
# 5. UserActivity Tracking Tests (10 tests)
# ============================================================================


class TestUserActivityTracking:
    """Test UserActivity tracking and login management."""

    def test_record_successful_login_increments_counters(self, minimal_user: User):
        """Should increment login_count and session_count."""
        # Arrange
        original_login_count = minimal_user.activity.login_count
        original_session_count = minimal_user.activity.session_count

        # Act
        minimal_user.activity.record_successful_login()

        # Assert
        assert minimal_user.activity.login_count == original_login_count + 1
        assert minimal_user.activity.session_count == original_session_count + 1

    def test_record_successful_login_sets_timestamps(self, minimal_user: User):
        """Should set last_login_at and last_active_at timestamps."""
        # Act
        minimal_user.activity.record_successful_login()

        # Assert
        assert minimal_user.activity.last_login_at is not None
        assert minimal_user.activity.last_active_at is not None
        assert isinstance(minimal_user.activity.last_login_at, datetime)

    def test_record_successful_login_resets_failed_attempts(self, minimal_user: User):
        """Should reset failed_login_attempts to zero."""
        # Arrange
        minimal_user.activity.failed_login_attempts = 3

        # Act
        minimal_user.activity.record_successful_login()

        # Assert
        assert minimal_user.activity.failed_login_attempts == 0

    def test_record_failed_login_increments_attempts(self, minimal_user: User):
        """Should increment failed_login_attempts counter."""
        # Arrange
        original_attempts = minimal_user.activity.failed_login_attempts

        # Act
        minimal_user.activity.record_failed_login()

        # Assert
        assert minimal_user.activity.failed_login_attempts == original_attempts + 1

    def test_record_failed_login_sets_timestamp(self, minimal_user: User):
        """Should set last_failed_login_at timestamp."""
        # Act
        minimal_user.activity.record_failed_login()

        # Assert
        assert minimal_user.activity.last_failed_login_at is not None
        assert isinstance(minimal_user.activity.last_failed_login_at, datetime)

    def test_is_account_locked_default_threshold(self, minimal_user: User):
        """Should lock account after 5 failed attempts (default)."""
        # Arrange
        minimal_user.activity.failed_login_attempts = 4
        assert minimal_user.activity.is_account_locked() is False

        # Act - 5th failed attempt
        minimal_user.activity.record_failed_login()

        # Assert
        assert minimal_user.activity.is_account_locked() is True

    def test_is_account_locked_custom_threshold(self, minimal_user: User):
        """Should lock account at custom threshold."""
        # Arrange
        minimal_user.activity.failed_login_attempts = 2

        # Assert
        assert minimal_user.activity.is_account_locked(max_attempts=3) is False
        assert minimal_user.activity.is_account_locked(max_attempts=2) is True

    def test_record_activity_updates_last_active_at(self, minimal_user: User):
        """Should update last_active_at timestamp."""
        # Arrange
        original_last_active = minimal_user.activity.last_active_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.activity.record_activity()

        # Assert
        assert minimal_user.activity.last_active_at is not None
        if original_last_active:
            assert minimal_user.activity.last_active_at > original_last_active

    def test_multiple_failed_login_attempts(self, minimal_user: User):
        """Should track multiple consecutive failed login attempts."""
        # Act
        minimal_user.activity.record_failed_login()
        minimal_user.activity.record_failed_login()
        minimal_user.activity.record_failed_login()

        # Assert
        assert minimal_user.activity.failed_login_attempts == 3

    def test_record_activity_via_user_method(self, minimal_user: User):
        """Should record activity through user.record_activity method."""
        # Act
        minimal_user.record_activity()

        # Assert
        assert minimal_user.activity.last_active_at is not None


# ============================================================================
# 6. UserSecurity Operations Tests (12 tests)
# ============================================================================


class TestUserSecurityOperations:
    """Test UserSecurity operations including 2FA, email verification, and password reset."""

    def test_security_default_values(self):
        """Should initialize with correct default security values."""
        # Act
        security = UserSecurity(password_hash="hashed_password")

        # Assert
        assert security.password_hash == "hashed_password"
        assert security.password_salt is None
        assert security.password_reset_token is None
        assert security.password_reset_expires is None
        assert security.email_verification_token is None
        assert security.email_verified is False
        assert security.email_verified_at is None
        assert security.two_factor_enabled is False
        assert security.two_factor_secret is None
        assert security.recovery_codes == []

    def test_verify_email_marks_verified(self, minimal_user: User):
        """Should mark email as verified and set timestamp."""
        # Act
        minimal_user.security.verify_email()

        # Assert
        assert minimal_user.security.email_verified is True
        assert minimal_user.security.email_verified_at is not None
        assert minimal_user.security.email_verification_token is None

    def test_verify_email_via_user_method(self, minimal_user: User):
        """Should verify email through user.verify_email method."""
        # Arrange
        minimal_user.status = UserStatus.PENDING_VERIFICATION

        # Act
        minimal_user.verify_email()

        # Assert
        assert minimal_user.security.email_verified is True
        assert minimal_user.status == UserStatus.ACTIVE

    def test_verify_email_does_not_change_active_status(self, minimal_user: User):
        """Should not change status if already ACTIVE."""
        # Arrange
        minimal_user.status = UserStatus.ACTIVE

        # Act
        minimal_user.verify_email()

        # Assert
        assert minimal_user.status == UserStatus.ACTIVE

    def test_set_password_reset_token(self, minimal_user: User):
        """Should set password reset token with expiration."""
        # Arrange
        token = "reset_token_12345"
        expires_at = datetime.utcnow() + timedelta(hours=1)

        # Act
        minimal_user.security.set_password_reset_token(token, expires_at)

        # Assert
        assert minimal_user.security.password_reset_token == token
        assert minimal_user.security.password_reset_expires == expires_at

    def test_clear_password_reset_token(self, minimal_user: User):
        """Should clear password reset token and expiration."""
        # Arrange
        minimal_user.security.password_reset_token = "token"
        minimal_user.security.password_reset_expires = datetime.utcnow()

        # Act
        minimal_user.security.clear_password_reset_token()

        # Assert
        assert minimal_user.security.password_reset_token is None
        assert minimal_user.security.password_reset_expires is None

    def test_is_password_reset_valid_returns_true_when_valid(self, minimal_user: User):
        """Should return True when token exists and not expired."""
        # Arrange
        minimal_user.security.password_reset_token = "token"
        minimal_user.security.password_reset_expires = datetime.utcnow() + timedelta(hours=1)

        # Act
        result = minimal_user.security.is_password_reset_valid()

        # Assert
        assert result is True

    def test_is_password_reset_valid_returns_false_when_expired(self, minimal_user: User):
        """Should return False when token is expired."""
        # Arrange
        minimal_user.security.password_reset_token = "token"
        minimal_user.security.password_reset_expires = datetime.utcnow() - timedelta(hours=1)

        # Act
        result = minimal_user.security.is_password_reset_valid()

        # Assert
        assert result is False

    def test_is_password_reset_valid_returns_false_when_no_token(self, minimal_user: User):
        """Should return False when no token exists."""
        # Act
        result = minimal_user.security.is_password_reset_valid()

        # Assert
        assert result is False

    def test_enable_two_factor(self, minimal_user: User):
        """Should enable two-factor authentication with secret and recovery codes."""
        # Arrange
        secret = "JBSWY3DPEHPK3PXP"
        recovery_codes = ["code1", "code2", "code3"]

        # Act
        minimal_user.security.enable_two_factor(secret, recovery_codes)

        # Assert
        assert minimal_user.security.two_factor_enabled is True
        assert minimal_user.security.two_factor_secret == secret
        assert minimal_user.security.recovery_codes == recovery_codes

    def test_disable_two_factor(self, minimal_user: User):
        """Should disable two-factor authentication and clear data."""
        # Arrange
        minimal_user.security.enable_two_factor("secret", ["code1", "code2"])

        # Act
        minimal_user.security.disable_two_factor()

        # Assert
        assert minimal_user.security.two_factor_enabled is False
        assert minimal_user.security.two_factor_secret is None
        assert minimal_user.security.recovery_codes == []

    def test_change_password_updates_hash_and_clears_reset(self, minimal_user: User):
        """Should update password hash and clear reset token."""
        # Arrange
        minimal_user.security.password_reset_token = "token"
        minimal_user.security.password_reset_expires = datetime.utcnow()
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.change_password("$2b$12$newhashedpassword")

        # Assert
        assert minimal_user.security.password_hash == "$2b$12$newhashedpassword"
        assert minimal_user.security.password_reset_token is None
        assert minimal_user.security.password_reset_expires is None
        assert minimal_user.updated_at > original_updated_at


# ============================================================================
# 7. Authentication Business Logic Tests (8 tests)
# ============================================================================


class TestUserAuthentication:
    """Test authentication business logic."""

    def test_authenticate_success_with_correct_password(self, minimal_user: User):
        """Should return True and record login for correct password."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return provided == "correct_password"

        minimal_user.status = UserStatus.ACTIVE
        original_login_count = minimal_user.activity.login_count

        # Act
        result = minimal_user.authenticate("correct_password", password_verifier)

        # Assert
        assert result is True
        assert minimal_user.activity.login_count == original_login_count + 1
        assert minimal_user.activity.failed_login_attempts == 0

    def test_authenticate_fails_with_incorrect_password(self, minimal_user: User):
        """Should return False and record failed attempt for incorrect password."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return provided == "correct_password"

        minimal_user.status = UserStatus.ACTIVE
        original_attempts = minimal_user.activity.failed_login_attempts

        # Act
        result = minimal_user.authenticate("wrong_password", password_verifier)

        # Assert
        assert result is False
        assert minimal_user.activity.failed_login_attempts == original_attempts + 1

    def test_authenticate_fails_when_user_inactive(self, minimal_user: User):
        """Should return False when user status is not ACTIVE."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return True

        minimal_user.status = UserStatus.INACTIVE

        # Act
        result = minimal_user.authenticate("password", password_verifier)

        # Assert
        assert result is False

    def test_authenticate_fails_when_user_suspended(self, minimal_user: User):
        """Should return False when user is suspended."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return True

        minimal_user.status = UserStatus.SUSPENDED

        # Act
        result = minimal_user.authenticate("password", password_verifier)

        # Assert
        assert result is False

    def test_authenticate_fails_when_account_locked(self, minimal_user: User):
        """Should return False when account is locked due to failed attempts."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return True

        minimal_user.status = UserStatus.ACTIVE
        minimal_user.activity.failed_login_attempts = 5

        # Act
        result = minimal_user.authenticate("password", password_verifier)

        # Assert
        assert result is False

    def test_authenticate_updates_timestamp_on_success(self, minimal_user: User):
        """Should update updated_at timestamp on successful authentication."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return True

        minimal_user.status = UserStatus.ACTIVE
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.authenticate("password", password_verifier)

        # Assert
        assert minimal_user.updated_at > original_updated_at

    def test_authenticate_resets_failed_attempts_on_success(self, minimal_user: User):
        """Should reset failed attempts counter on successful authentication."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return True

        minimal_user.status = UserStatus.ACTIVE
        minimal_user.activity.failed_login_attempts = 3

        # Act
        minimal_user.authenticate("password", password_verifier)

        # Assert
        assert minimal_user.activity.failed_login_attempts == 0

    def test_authenticate_does_not_update_timestamp_on_failure(self, minimal_user: User):
        """Should not update updated_at timestamp on failed authentication."""
        # Arrange
        def password_verifier(provided: str, stored_hash: str) -> bool:
            return False

        minimal_user.status = UserStatus.ACTIVE
        original_updated_at = minimal_user.updated_at

        # Act
        minimal_user.authenticate("wrong_password", password_verifier)

        # Assert
        # updated_at should not change on failed authentication
        assert minimal_user.updated_at == original_updated_at


# ============================================================================
# 8. Profile Management Tests (4 tests)
# ============================================================================


class TestUserProfileManagement:
    """Test user profile update methods."""

    def test_update_profile_changes_full_name(self, minimal_user: User):
        """Should update full_name."""
        # Arrange
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.update_profile(full_name="Jane Doe")

        # Assert
        assert minimal_user.full_name == "Jane Doe"
        assert minimal_user.updated_at > original_updated_at

    def test_update_profile_with_none_preserves_values(self, minimal_user: User):
        """Should preserve existing values when None is passed."""
        # Arrange
        original_name = minimal_user.full_name

        # Act
        minimal_user.update_profile(full_name=None)

        # Assert
        assert minimal_user.full_name == original_name

    def test_update_profile_updates_timestamp(self, minimal_user: User):
        """Should always update timestamp even with no changes."""
        # Arrange
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.update_profile()

        # Assert
        assert minimal_user.updated_at > original_updated_at

    def test_get_display_name_returns_full_name(self, minimal_user: User):
        """Should return full_name as display name."""
        # Arrange
        minimal_user.full_name = "John Smith"

        # Act
        display_name = minimal_user.get_display_name()

        # Assert
        assert display_name == "John Smith"


# ============================================================================
# 9. Role Management Tests (6 tests)
# ============================================================================


class TestUserRoleManagement:
    """Test role change and metadata tracking."""

    def test_change_role_updates_role(self, minimal_user: User):
        """Should update user role."""
        # Arrange
        changer_id = UserId(uuid4())

        # Act
        minimal_user.change_role(UserRole.ADMIN, changer_id)

        # Assert
        assert minimal_user.role == UserRole.ADMIN

    def test_change_role_stores_metadata(self, minimal_user: User):
        """Should store role change metadata."""
        # Arrange
        changer_id = UserId(uuid4())

        # Act
        minimal_user.change_role(UserRole.RECRUITER, changer_id)

        # Assert
        assert "role_changed_by" in minimal_user.metadata
        assert minimal_user.metadata["role_changed_by"] == str(changer_id)
        assert "role_changed_at" in minimal_user.metadata

    def test_change_role_updates_timestamp(self, minimal_user: User):
        """Should update updated_at timestamp."""
        # Arrange
        changer_id = UserId(uuid4())
        original_updated_at = minimal_user.updated_at

        # Act
        import time
        time.sleep(0.01)
        minimal_user.change_role(UserRole.ADMIN, changer_id)

        # Assert
        assert minimal_user.updated_at > original_updated_at

    def test_is_admin_returns_true_for_admin(self, admin_user: User):
        """Should return True for ADMIN role."""
        # Assert
        assert admin_user.is_admin() is True

    def test_is_admin_returns_true_for_super_admin(self, super_admin_user: User):
        """Should return True for SUPER_ADMIN role."""
        # Assert
        assert super_admin_user.is_admin() is True

    def test_is_admin_returns_false_for_non_admin(self, minimal_user: User):
        """Should return False for non-admin roles."""
        # Assert
        assert minimal_user.is_admin() is False


# ============================================================================
# 10. Permission Methods Tests (10 tests)
# ============================================================================


class TestUserPermissions:
    """Test permission checking methods."""

    def test_can_manage_users_admin(self, admin_user: User):
        """Should return True for ADMIN."""
        # Assert
        assert admin_user.can_manage_users() is True

    def test_can_manage_users_super_admin(self, super_admin_user: User):
        """Should return True for SUPER_ADMIN."""
        # Assert
        assert super_admin_user.can_manage_users() is True

    def test_can_manage_users_recruiter(self, minimal_user: User):
        """Should return False for RECRUITER."""
        # Arrange
        minimal_user.role = UserRole.RECRUITER

        # Assert
        assert minimal_user.can_manage_users() is False

    def test_can_view_analytics_admin(self, admin_user: User):
        """Should return True for ADMIN."""
        # Assert
        assert admin_user.can_view_analytics() is True

    def test_can_view_analytics_recruiter(self, minimal_user: User):
        """Should return True for RECRUITER."""
        # Arrange
        minimal_user.role = UserRole.RECRUITER

        # Assert
        assert minimal_user.can_view_analytics() is True

    def test_can_view_analytics_viewer(self, minimal_user: User):
        """Should return False for VIEWER."""
        # Arrange
        minimal_user.role = UserRole.VIEWER

        # Assert
        assert minimal_user.can_view_analytics() is False

    def test_can_search_profiles_active_admin(self, admin_user: User):
        """Should return True for active ADMIN."""
        # Arrange
        admin_user.status = UserStatus.ACTIVE

        # Assert
        assert admin_user.can_search_profiles() is True

    def test_can_search_profiles_inactive_admin(self, admin_user: User):
        """Should return False for inactive user regardless of role."""
        # Arrange
        admin_user.status = UserStatus.INACTIVE

        # Assert
        assert admin_user.can_search_profiles() is False

    def test_can_upload_profiles_recruiter(self, minimal_user: User):
        """Should return True for active RECRUITER."""
        # Arrange
        minimal_user.role = UserRole.RECRUITER
        minimal_user.status = UserStatus.ACTIVE

        # Assert
        assert minimal_user.can_upload_profiles() is True

    def test_can_upload_profiles_viewer(self, minimal_user: User):
        """Should return False for VIEWER."""
        # Arrange
        minimal_user.role = UserRole.VIEWER
        minimal_user.status = UserStatus.ACTIVE

        # Assert
        assert minimal_user.can_upload_profiles() is False


# ============================================================================
# 11. Days Since Last Login Tests (4 tests)
# ============================================================================


class TestDaysSinceLastLogin:
    """Test days_since_last_login calculation."""

    def test_days_since_last_login_returns_none_when_never_logged_in(self, minimal_user: User):
        """Should return None when user has never logged in."""
        # Assert
        assert minimal_user.days_since_last_login() is None

    def test_days_since_last_login_returns_zero_for_today(self, minimal_user: User):
        """Should return 0 for login today."""
        # Arrange
        minimal_user.activity.last_login_at = datetime.utcnow()

        # Act
        days = minimal_user.days_since_last_login()

        # Assert
        assert days == 0

    def test_days_since_last_login_calculates_correct_days(self, minimal_user: User):
        """Should calculate correct number of days."""
        # Arrange
        minimal_user.activity.last_login_at = datetime.utcnow() - timedelta(days=5)

        # Act
        days = minimal_user.days_since_last_login()

        # Assert
        assert days == 5

    def test_days_since_last_login_handles_large_intervals(self, minimal_user: User):
        """Should handle large time intervals correctly."""
        # Arrange
        minimal_user.activity.last_login_at = datetime.utcnow() - timedelta(days=365)

        # Act
        days = minimal_user.days_since_last_login()

        # Assert
        assert days == 365


# ============================================================================
# 12. Edge Cases and Validation Tests (6 tests)
# ============================================================================


class TestUserEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_verify_email_clears_verification_token(self, minimal_user: User):
        """Should clear email_verification_token after verification."""
        # Arrange
        minimal_user.security.email_verification_token = "token_12345"

        # Act
        minimal_user.security.verify_email()

        # Assert
        assert minimal_user.security.email_verification_token is None

    def test_multiple_status_transitions(self, minimal_user: User):
        """Should handle multiple status transitions correctly."""
        # Act & Assert
        minimal_user.suspend()
        assert minimal_user.status == UserStatus.SUSPENDED

        minimal_user.activate()
        assert minimal_user.status == UserStatus.ACTIVE

        minimal_user.deactivate()
        assert minimal_user.status == UserStatus.INACTIVE

        minimal_user.activate()
        assert minimal_user.status == UserStatus.ACTIVE

        minimal_user.mark_deleted()
        assert minimal_user.status == UserStatus.DELETED

    def test_get_display_name_falls_back_to_email(
        self,
        valid_user_id: UserId,
        valid_tenant_id: TenantId,
        valid_email: EmailAddress,
        valid_security: UserSecurity,
    ):
        """Should return email when full_name is empty."""
        # Arrange
        user = User(
            id=valid_user_id,
            tenant_id=valid_tenant_id,
            email=valid_email,
            full_name="",
            role=UserRole.VIEWER,
            status=UserStatus.ACTIVE,
            security=valid_security,
        )

        # Act
        display_name = user.get_display_name()

        # Assert
        assert display_name == str(valid_email)

    def test_account_locking_boundary_condition(self, minimal_user: User):
        """Should lock at exact threshold."""
        # Arrange
        minimal_user.activity.failed_login_attempts = 4

        # Assert - not locked yet
        assert minimal_user.activity.is_account_locked(max_attempts=5) is False

        # Act - one more failure
        minimal_user.activity.record_failed_login()

        # Assert - now locked
        assert minimal_user.activity.is_account_locked(max_attempts=5) is True

    def test_password_reset_expiration_boundary(self, minimal_user: User):
        """Should handle password reset expiration boundary correctly."""
        # Arrange - expires in 1 second
        minimal_user.security.password_reset_token = "token"
        minimal_user.security.password_reset_expires = datetime.utcnow() + timedelta(seconds=1)

        # Assert - valid now
        assert minimal_user.security.is_password_reset_valid() is True

        # Wait for expiration
        import time
        time.sleep(1.1)

        # Assert - expired now
        assert minimal_user.security.is_password_reset_valid() is False

    def test_metadata_can_be_updated(self, minimal_user: User):
        """Should allow metadata to be updated."""
        # Act
        minimal_user.metadata["custom_field"] = "custom_value"
        minimal_user.metadata["department"] = "engineering"

        # Assert
        assert minimal_user.metadata["custom_field"] == "custom_value"
        assert minimal_user.metadata["department"] == "engineering"


# ============================================================================
# 13. Property-Based Tests with Hypothesis (3 tests)
# ============================================================================


class TestUserPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        full_name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    def test_user_full_name_always_stored_correctly(self, full_name: str):
        """Property: Any valid full_name should be stored correctly."""
        # Arrange
        user = User(
            id=UserId(uuid4()),
            tenant_id=TenantId(uuid4()),
            email=EmailAddress("test@example.com"),
            full_name=full_name,
            role=UserRole.VIEWER,
            status=UserStatus.ACTIVE,
            security=UserSecurity(password_hash="hash"),
        )

        # Assert
        assert user.full_name == full_name

    @given(
        status=st.sampled_from([
            UserStatus.ACTIVE,
            UserStatus.INACTIVE,
            UserStatus.SUSPENDED,
        ])
    )
    def test_status_transitions_always_update_timestamp(self, status: UserStatus):
        """Property: Status transitions should always update updated_at."""
        # Arrange
        user = User(
            id=UserId(uuid4()),
            tenant_id=TenantId(uuid4()),
            email=EmailAddress("test@example.com"),
            full_name="Test User",
            role=UserRole.VIEWER,
            status=status,
            security=UserSecurity(password_hash="hash"),
        )
        original_updated_at = user.updated_at

        # Act
        import time
        time.sleep(0.01)
        user.deactivate()

        # Assert
        assert user.updated_at > original_updated_at

    @given(
        failed_attempts=st.integers(min_value=0, max_value=10),
        max_attempts=st.integers(min_value=1, max_value=10),
    )
    def test_account_locking_logic_is_consistent(self, failed_attempts: int, max_attempts: int):
        """Property: Account locking logic should be consistent."""
        # Arrange
        activity = UserActivity(failed_login_attempts=failed_attempts)

        # Act
        is_locked = activity.is_account_locked(max_attempts=max_attempts)

        # Assert
        assert is_locked == (failed_attempts >= max_attempts)