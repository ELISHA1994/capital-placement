"""
Comprehensive security tests for AuthenticationService

This test suite covers all critical security paths including:
- Authentication: login, token generation, validation
- Password management: hashing, strength validation, reset flows
- Session management: creation, tracking, termination
- Token management: generation, validation, refresh, revocation
- Security: rate limiting, blacklisting, audit logging
- Edge cases: expired tokens, invalid credentials, tenant isolation
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid import uuid4, UUID

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.infrastructure.auth.authentication_service import AuthenticationService
from app.infrastructure.persistence.models.auth_tables import (
    UserCreate, UserLogin, CurrentUser, TokenResponse, AuthenticationResult,
    RefreshTokenRequest, PasswordChangeRequest, PasswordResetRequest,
    PasswordResetConfirm, SessionInfo
)
from app.database.repositories.postgres import UserRepository, TenantRepository, UserSessionRepository
from app.domain.interfaces import ICacheService, INotificationService


@pytest.fixture
def mock_user_repository():
    """Mock user repository"""
    repo = Mock(spec=UserRepository)
    repo.create = AsyncMock()
    repo.get_by_email = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.update = AsyncMock()
    repo.update_last_login = AsyncMock()
    repo.find_by_criteria = AsyncMock()
    return repo


@pytest.fixture
def mock_tenant_repository():
    """Mock tenant repository"""
    repo = Mock(spec=TenantRepository)
    repo.get = AsyncMock()
    repo.get_active_tenants = AsyncMock()
    return repo


@pytest.fixture
def mock_cache_service():
    """Mock cache service"""
    cache = Mock(spec=ICacheService)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    cache.exists = AsyncMock(return_value=False)
    cache.delete_pattern = AsyncMock()
    return cache


@pytest.fixture
def mock_notification_service():
    """Mock notification service"""
    service = Mock(spec=INotificationService)
    service.send_email = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_session_repository():
    """Mock session repository"""
    repo = Mock(spec=UserSessionRepository)
    repo.create = AsyncMock()
    repo.list_by_user = AsyncMock(return_value=[])
    repo.find_by_id = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.delete_by_user = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def auth_service(
    mock_user_repository,
    mock_tenant_repository,
    mock_cache_service,
    mock_notification_service,
    mock_session_repository
):
    """Create authentication service with mocked dependencies"""
    return AuthenticationService(
        user_repository=mock_user_repository,
        tenant_repository=mock_tenant_repository,
        cache_manager=mock_cache_service,
        notification_service=mock_notification_service,
        session_repository=mock_session_repository
    )


@pytest.fixture
def sample_tenant():
    """Sample tenant data"""
    return {
        "id": str(uuid4()),
        "name": "test-tenant",
        "is_active": True,
        "is_suspended": False
    }


@pytest.fixture
def sample_user():
    """Sample user data"""
    return {
        "id": str(uuid4()),
        "email": "test@example.com",
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyWK4fJlI5d2",  # "password123"
        "full_name": "Test User",
        "first_name": "Test",
        "last_name": "User",
        "tenant_id": str(uuid4()),
        "roles": ["user"],
        "permissions": [],
        "is_active": True,
        "is_verified": True,
        "is_superuser": False
    }


# =============================================================================
# AUTHENTICATION TESTS
# =============================================================================

@pytest.mark.asyncio
class TestAuthentication:
    """Tests for user authentication"""

    async def test_authenticate_valid_credentials(
        self, auth_service, mock_user_repository, sample_user
    ):
        """Test successful authentication with valid credentials"""
        # Setup
        mock_user_repository.get_by_email.return_value = sample_user
        mock_user_repository.update_last_login.return_value = None

        credentials = UserLogin(
            email="test@example.com",
            password="password123",
            tenant_id=sample_user["tenant_id"]
        )

        # Execute
        with patch("app.utils.security.password_manager.verify_password", return_value=True):
            with patch("app.utils.security.token_manager.create_access_token", return_value="access_token"):
                with patch("app.utils.security.token_manager.create_refresh_token", return_value="refresh_token"):
                    result = await auth_service.authenticate(credentials)

        # Verify
        assert result.success is True
        assert result.user is not None
        assert result.user.email == "test@example.com"
        assert result.tokens is not None
        mock_user_repository.update_last_login.assert_called_once()

    async def test_authenticate_invalid_email(
        self, auth_service, mock_user_repository
    ):
        """Test authentication with non-existent email"""
        # Setup
        mock_user_repository.get_by_email.return_value = None

        credentials = UserLogin(
            email="nonexistent@example.com",
            password="password123",
            tenant_id=str(uuid4())
        )

        # Execute
        result = await auth_service.authenticate(credentials)

        # Verify
        assert result.success is False
        assert result.error == "Invalid credentials"
        assert result.user is None

    async def test_authenticate_invalid_password(
        self, auth_service, mock_user_repository, sample_user
    ):
        """Test authentication with wrong password"""
        # Setup
        mock_user_repository.get_by_email.return_value = sample_user

        credentials = UserLogin(
            email="test@example.com",
            password="wrongpassword",
            tenant_id=sample_user["tenant_id"]
        )

        # Execute
        with patch("app.utils.security.password_manager.verify_password", return_value=False):
            result = await auth_service.authenticate(credentials)

        # Verify
        assert result.success is False
        assert result.error == "Invalid credentials"

    async def test_authenticate_inactive_user(
        self, auth_service, mock_user_repository, sample_user
    ):
        """Test authentication with inactive user account"""
        # Setup
        sample_user["is_active"] = False
        mock_user_repository.get_by_email.return_value = sample_user

        credentials = UserLogin(
            email="test@example.com",
            password="password123",
            tenant_id=sample_user["tenant_id"]
        )

        # Execute
        result = await auth_service.authenticate(credentials)

        # Verify
        assert result.success is False
        assert result.error == "Account is not active"

    async def test_authenticate_rate_limited(
        self, auth_service, mock_cache_service
    ):
        """Test authentication rate limiting"""
        # Setup - simulate too many failed attempts
        mock_cache_service.get.return_value = 10  # Exceeds MAX_LOGIN_ATTEMPTS

        credentials = UserLogin(
            email="test@example.com",
            password="password123",
            tenant_id=str(uuid4())
        )

        # Execute
        result = await auth_service.authenticate(credentials)

        # Verify
        assert result.success is False
        assert "Too many login attempts" in result.error


# =============================================================================
# TOKEN MANAGEMENT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestTokenManagement:
    """Tests for JWT token generation, validation, and refresh"""

    async def test_verify_token_valid(
        self, auth_service, mock_cache_service, sample_user
    ):
        """Test token verification with valid token"""
        # Setup
        mock_cache_service.exists.return_value = False  # Not blacklisted
        mock_cache_service.get.return_value = sample_user  # Cached user data

        token_payload = {
            "sub": sample_user["id"],
            "email": sample_user["email"],
            "tenant_id": sample_user["tenant_id"],
            "roles": ["user"],
            "permissions": [],
            "token_type": "access",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp())
        }

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            with patch("app.utils.security.token_manager.extract_token_data") as mock_extract:
                mock_token_data = Mock(
                    sub=sample_user["id"],
                    email=sample_user["email"],
                    tenant_id=sample_user["tenant_id"],
                    roles=["user"],
                    permissions=[],
                    token_type="access"
                )
                mock_extract.return_value = mock_token_data

                result = await auth_service.verify_token("valid_token")

        # Verify
        assert result is not None
        assert result["user_id"] == sample_user["id"]
        assert result["email"] == sample_user["email"]

    async def test_verify_token_blacklisted(
        self, auth_service, mock_cache_service
    ):
        """Test token verification with blacklisted token"""
        # Setup
        mock_cache_service.exists.return_value = True  # Blacklisted

        token_payload = {"jti": "token_id", "exp": 9999999999}

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            result = await auth_service.verify_token("blacklisted_token")

        # Verify
        assert result is None

    async def test_verify_token_expired(
        self, auth_service
    ):
        """Test token verification with expired token"""
        # Setup - verify_token returns None for expired tokens

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=None):
            result = await auth_service.verify_token("expired_token")

        # Verify
        assert result is None

    async def test_refresh_tokens_valid(
        self, auth_service, mock_cache_service, sample_user
    ):
        """Test token refresh with valid refresh token"""
        # Setup
        mock_cache_service.exists.return_value = False  # Not blacklisted
        mock_cache_service.get.return_value = sample_user

        token_payload = {
            "sub": sample_user["id"],
            "tenant_id": sample_user["tenant_id"],
            "token_type": "refresh",
            "family": "token_family_123",
            "exp": int((datetime.utcnow() + timedelta(days=7)).timestamp())
        }

        request = RefreshTokenRequest(refresh_token="valid_refresh_token")

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            with patch("app.utils.security.token_manager.create_access_token", return_value="new_access_token"):
                with patch("app.utils.security.token_manager.create_refresh_token", return_value="new_refresh_token"):
                    result = await auth_service.refresh_tokens(request)

        # Verify
        assert result is not None
        assert result.access_token == "new_access_token"
        assert result.refresh_token == "new_refresh_token"
        # Old token should be blacklisted
        mock_cache_service.set.assert_called()

    async def test_refresh_tokens_invalid_type(
        self, auth_service
    ):
        """Test token refresh with access token (wrong type)"""
        # Setup
        token_payload = {
            "sub": str(uuid4()),
            "token_type": "access",  # Wrong type
            "exp": 9999999999
        }

        request = RefreshTokenRequest(refresh_token="access_token")

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            result = await auth_service.refresh_tokens(request)

        # Verify
        assert result is None

    async def test_revoke_token(
        self, auth_service, mock_cache_service
    ):
        """Test token revocation"""
        # Setup
        token_payload = {
            "jti": "token_id_123",
            "sub": str(uuid4()),
            "token_type": "refresh",
            "family": "token_family_123",
            "exp": 9999999999
        }

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            result = await auth_service.revoke_token("token_to_revoke")

        # Verify
        assert result is True
        # Token should be blacklisted
        mock_cache_service.set.assert_called()


# =============================================================================
# PASSWORD MANAGEMENT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestPasswordManagement:
    """Tests for password hashing, validation, change, and reset"""

    async def test_change_password_success(
        self, auth_service, mock_user_repository, sample_user
    ):
        """Test successful password change"""
        # Setup
        mock_user_repository.get_by_id.return_value = sample_user
        mock_user_repository.update.return_value = sample_user

        request = PasswordChangeRequest(
            current_password="password123",
            new_password="NewPassword123!"
        )

        # Execute
        with patch("app.utils.security.password_manager.verify_password", return_value=True):
            with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
                mock_validate.return_value = {"valid": True, "errors": []}
                with patch("app.utils.security.password_manager.hash_password", return_value="new_hash"):
                    result = await auth_service.change_password(sample_user["id"], request)

        # Verify
        assert result is True
        mock_user_repository.update.assert_called_once()

    async def test_change_password_wrong_current(
        self, auth_service, mock_user_repository, sample_user
    ):
        """Test password change with wrong current password"""
        # Setup
        mock_user_repository.get_by_id.return_value = sample_user

        request = PasswordChangeRequest(
            current_password="wrongpassword",
            new_password="NewPassword123!"
        )

        # Execute
        with patch("app.utils.security.password_manager.verify_password", return_value=False):
            result = await auth_service.change_password(sample_user["id"], request)

        # Verify
        assert result is False
        mock_user_repository.update.assert_not_called()

    async def test_change_password_weak_new_password(
        self, auth_service, mock_user_repository, sample_user
    ):
        """Test password change with weak new password"""
        # Setup
        mock_user_repository.get_by_id.return_value = sample_user

        request = PasswordChangeRequest(
            current_password="password123",
            new_password="weakpass"  # Changed to meet minimum length
        )

        # Execute
        with patch("app.utils.security.password_manager.verify_password", return_value=True):
            with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
                mock_validate.return_value = {
                    "valid": False,
                    "errors": ["Password too short", "Password too weak"]
                }

                with pytest.raises(ValueError, match="Password validation failed"):
                    await auth_service.change_password(sample_user["id"], request)

    async def test_request_password_reset_success(
        self, auth_service, mock_user_repository, mock_cache_service, sample_user
    ):
        """Test password reset request flow"""
        # Setup
        mock_user_repository.get_by_email.return_value = sample_user
        mock_cache_service.get.return_value = None  # No previous reset

        request = PasswordResetRequest(
            email="test@example.com",
            tenant_id=sample_user["tenant_id"],  # Already a string
            redirect_url="https://example.com/reset"
        )

        # Execute
        with patch("app.utils.security.security_validator.validate_email", return_value=True):
            result = await auth_service.request_password_reset(request)

        # Verify
        assert result is not None
        assert result["email"] == "test@example.com"
        assert "token" in result
        assert "reset_link" in result
        # Token should be cached
        mock_cache_service.set.assert_called()

    async def test_request_password_reset_invalid_email(
        self, auth_service
    ):
        """Test password reset with invalid email format"""
        # Setup
        request = PasswordResetRequest(
            email="invalid-email",
            tenant_id=str(uuid4())
        )

        # Execute
        with patch("app.utils.security.security_validator.validate_email", return_value=False):
            result = await auth_service.request_password_reset(request)

        # Verify
        assert result is None

    async def test_request_password_reset_throttled(
        self, auth_service, mock_cache_service
    ):
        """Test password reset request throttling"""
        # Setup
        mock_cache_service.exists.return_value = True  # Recently requested

        request = PasswordResetRequest(
            email="test@example.com",
            tenant_id=str(uuid4())
        )

        # Execute
        with patch("app.utils.security.security_validator.validate_email", return_value=True):
            result = await auth_service.request_password_reset(request)

        # Verify
        assert result is None

    async def test_confirm_password_reset_success(
        self, auth_service, mock_user_repository, mock_cache_service, sample_user
    ):
        """Test password reset confirmation"""
        # Setup
        reset_token = secrets.token_urlsafe(32)
        token_digest = hashlib.sha256(reset_token.encode("utf-8")).hexdigest()

        reset_payload = {
            "user_id": sample_user["id"],
            "tenant_id": sample_user["tenant_id"],
            "email": sample_user["email"],
            "token_digest": token_digest,
            "requested_at": datetime.utcnow().isoformat()
        }

        mock_cache_service.get.return_value = reset_payload
        mock_user_repository.get_by_id.return_value = sample_user
        mock_user_repository.update.return_value = sample_user

        request = PasswordResetConfirm(
            token=reset_token,
            new_password="NewPassword123!"
        )

        # Execute
        with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}
            with patch("app.utils.security.password_manager.hash_password", return_value="new_hash"):
                result = await auth_service.confirm_password_reset(request)

        # Verify
        assert result is True
        mock_user_repository.update.assert_called_once()
        # Cache should be cleared
        mock_cache_service.delete.assert_called()

    async def test_confirm_password_reset_invalid_token(
        self, auth_service, mock_cache_service
    ):
        """Test password reset with invalid/expired token"""
        # Setup
        mock_cache_service.get.return_value = None  # Token not found

        request = PasswordResetConfirm(
            token="invalid_token",
            new_password="NewPassword123!"
        )

        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid or expired reset token"):
            await auth_service.confirm_password_reset(request)


# =============================================================================
# SESSION MANAGEMENT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestSessionManagement:
    """Tests for user session creation, tracking, and termination"""

    async def test_list_sessions(
        self, auth_service, mock_cache_service
    ):
        """Test listing user sessions"""
        # Setup
        user_id = str(uuid4())
        sessions = [
            SessionInfo(
                session_id=str(uuid4()),
                user_id=user_id,
                tenant_id=str(uuid4()),
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
                created_at=datetime.utcnow().isoformat(),
                last_activity=datetime.utcnow().isoformat(),
                expires_at=(datetime.utcnow() + timedelta(days=7)).isoformat()
            )
        ]

        # Serialize sessions like cache would
        from app.utils.session_utils import serialize_sessions
        mock_cache_service.get.return_value = serialize_sessions(sessions)

        # Execute
        result = await auth_service.list_sessions(user_id)

        # Verify
        assert len(result) == 1
        assert result[0].user_id == user_id

    async def test_terminate_session(
        self, auth_service, mock_session_repository, mock_cache_service
    ):
        """Test terminating a specific session"""
        # Setup
        user_id = str(uuid4())
        session_id = str(uuid4())
        tenant_id = str(uuid4())

        session_record = {
            "id": session_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=7)
        }

        mock_session_repository.find_by_id.return_value = session_record
        mock_session_repository.delete.return_value = True
        mock_cache_service.get.return_value = None  # No cached sessions

        # Execute
        result = await auth_service.terminate_session(session_id, user_id)

        # Verify
        assert result is True
        mock_session_repository.delete.assert_called_once_with(session_id)

    async def test_terminate_session_wrong_user(
        self, auth_service, mock_session_repository
    ):
        """Test terminating session belonging to different user"""
        # Setup
        user_id = str(uuid4())
        different_user_id = str(uuid4())
        session_id = str(uuid4())

        session_record = {
            "id": session_id,
            "user_id": different_user_id,  # Different user!
            "tenant_id": str(uuid4())
        }

        mock_session_repository.find_by_id.return_value = session_record

        # Execute
        result = await auth_service.terminate_session(session_id, user_id)

        # Verify
        assert result is False
        mock_session_repository.delete.assert_not_called()


# =============================================================================
# USER REGISTRATION TESTS
# =============================================================================

@pytest.mark.asyncio
class TestUserRegistration:
    """Tests for user registration"""

    async def test_register_user_success(
        self, auth_service, mock_user_repository, mock_tenant_repository, sample_tenant
    ):
        """Test successful user registration"""
        # Setup
        mock_user_repository.get_by_email.return_value = None  # User doesn't exist
        mock_tenant_repository.get.return_value = sample_tenant

        new_user = {
            "id": str(uuid4()),
            "email": "newuser@example.com",
            "full_name": "New User",
            "tenant_id": sample_tenant["id"],
            "roles": ["user"],
            "is_active": True
        }
        mock_user_repository.create.return_value = new_user

        user_data = UserCreate(
            email="newuser@example.com",
            password="StrongPassword123!",
            full_name="New User",
            tenant_id=sample_tenant["id"]  # Already a string
        )

        # Execute
        with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}
            with patch("app.utils.security.password_manager.hash_password", return_value="hashed"):
                result = await auth_service.register_user(user_data)

        # Verify
        assert result is not None
        mock_user_repository.create.assert_called_once()

    async def test_register_user_duplicate_email(
        self, auth_service, mock_user_repository, mock_tenant_repository, sample_user, sample_tenant
    ):
        """Test registration with existing email"""
        # Setup
        mock_user_repository.get_by_email.return_value = sample_user  # User exists
        mock_tenant_repository.get.return_value = sample_tenant

        user_data = UserCreate(
            email=sample_user["email"],
            password="StrongPassword123!",
            full_name="Another User",
            tenant_id=sample_tenant["id"]  # Already a string
        )

        # Execute & Verify
        with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}

            with pytest.raises(ValueError, match="User already exists"):
                await auth_service.register_user(user_data)

    async def test_register_user_weak_password(
        self, auth_service, mock_tenant_repository, sample_tenant
    ):
        """Test registration with weak password"""
        # Setup
        mock_tenant_repository.get.return_value = sample_tenant

        user_data = UserCreate(
            email="newuser@example.com",
            password="weakpass",  # Meets min length but still weak
            full_name="New User",
            tenant_id=sample_tenant["id"]  # Already a string
        )

        # Execute & Verify
        with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "errors": ["Password too short"]
            }

            with pytest.raises(ValueError, match="Password validation failed"):
                await auth_service.register_user(user_data)

    async def test_register_user_invalid_tenant(
        self, auth_service, mock_user_repository, mock_tenant_repository
    ):
        """Test registration with invalid tenant"""
        # Setup
        mock_user_repository.get_by_email.return_value = None  # User doesn't exist yet
        mock_tenant_repository.get.return_value = None  # Tenant doesn't exist

        user_data = UserCreate(
            email="newuser@example.com",
            password="StrongPassword123!",
            full_name="New User",
            tenant_id=str(uuid4())
        )

        # Execute & Verify
        with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}

            with pytest.raises(ValueError, match="Invalid organization"):
                await auth_service.register_user(user_data)

    async def test_register_user_suspended_tenant(
        self, auth_service, mock_user_repository, mock_tenant_repository, sample_tenant
    ):
        """Test registration with suspended tenant"""
        # Setup
        mock_user_repository.get_by_email.return_value = None  # User doesn't exist yet
        sample_tenant["is_suspended"] = True
        mock_tenant_repository.get.return_value = sample_tenant

        user_data = UserCreate(
            email="newuser@example.com",
            password="StrongPassword123!",
            full_name="New User",
            tenant_id=sample_tenant["id"]  # Already a string
        )

        # Execute & Verify
        with patch("app.utils.security.password_manager.validate_password_strength") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}

            with pytest.raises(ValueError, match="Organization is currently suspended"):
                await auth_service.register_user(user_data)


# =============================================================================
# SECURITY & AUDIT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestSecurityAudit:
    """Tests for security audit logging and rate limiting"""

    async def test_failed_login_recorded(
        self, auth_service, mock_user_repository, mock_cache_service, sample_user
    ):
        """Test that failed login attempts are recorded"""
        # Setup
        mock_user_repository.get_by_email.return_value = sample_user
        mock_cache_service.get.return_value = 0  # No previous attempts

        credentials = UserLogin(
            email="test@example.com",
            password="wrongpassword",
            tenant_id=sample_user["tenant_id"]
        )

        # Execute
        with patch("app.utils.security.password_manager.verify_password", return_value=False):
            await auth_service.authenticate(credentials)

        # Verify - failed attempt should be cached
        # Check that set was called to increment attempt counter
        assert mock_cache_service.set.called

    async def test_tenant_isolation(
        self, auth_service, mock_cache_service, sample_user
    ):
        """Test that token verification enforces tenant isolation"""
        # Setup - token with different tenant than user
        different_tenant_id = str(uuid4())

        mock_cache_service.exists.return_value = False
        mock_cache_service.get.return_value = sample_user

        token_payload = {
            "sub": sample_user["id"],
            "email": sample_user["email"],
            "tenant_id": different_tenant_id,  # Different tenant!
            "roles": ["user"],
            "permissions": [],
            "token_type": "access",
            "exp": 9999999999
        }

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            with patch("app.utils.security.token_manager.extract_token_data") as mock_extract:
                mock_token_data = Mock(
                    sub=sample_user["id"],
                    tenant_id=different_tenant_id,
                    token_type="access",
                    roles=["user"],
                    permissions=[]
                )
                mock_extract.return_value = mock_token_data

                result = await auth_service.verify_token("token")

        # Verify - should reject due to tenant mismatch
        assert result is None

    async def test_super_admin_can_access_any_tenant(
        self, auth_service, mock_cache_service, sample_user
    ):
        """Test that super admins can bypass tenant restrictions"""
        # Setup
        sample_user["is_superuser"] = True
        different_tenant_id = str(uuid4())

        mock_cache_service.exists.return_value = False
        mock_cache_service.get.return_value = sample_user

        token_payload = {
            "sub": sample_user["id"],
            "email": sample_user["email"],
            "tenant_id": different_tenant_id,
            "roles": ["super_admin"],
            "permissions": [],
            "token_type": "access",
            "exp": 9999999999
        }

        # Execute
        with patch("app.utils.security.token_manager.verify_token", return_value=token_payload):
            with patch("app.utils.security.token_manager.extract_token_data") as mock_extract:
                mock_token_data = Mock(
                    sub=sample_user["id"],
                    tenant_id=different_tenant_id,
                    token_type="access",
                    roles=["super_admin"],
                    permissions=[]
                )
                mock_extract.return_value = mock_token_data

                result = await auth_service.verify_token("token")

        # Verify - super admin should have access
        assert result is not None
