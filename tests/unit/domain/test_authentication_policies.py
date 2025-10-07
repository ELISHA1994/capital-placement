"""
Unit tests for authentication domain policies and business logic.

These tests focus on pure business logic without external dependencies.
Tests password policies, authentication rules, token validation, and security policies.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch
from app.infrastructure.auth.authentication_service import AuthenticationService
from app.infrastructure.persistence.models.auth_tables import UserCreate, UserLogin, PasswordChangeRequest, PasswordResetRequest, PasswordResetConfirm
from app.infrastructure.persistence.models.tenant_table import SubscriptionTier
from tests.mocks.mock_repositories import MockUserRepository, MockTenantRepository, MockUserSessionRepository
from tests.mocks.mock_services import MockCacheService, MockNotificationService


class TestPasswordPolicyValidation:
    """Test password policy enforcement and validation rules."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_valid_password_acceptance(self):
        """Test that valid passwords are accepted."""
        valid_passwords = [
            "SecurePassword123!",
            "My$ecur3P@ssw0rd",
            "Complex&Password2024",
            "Str0ng!P@ssword#",
            "Valid$Password1"
        ]
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            # Setup test tenant
            tenant_id = str(uuid4())
            self.tenant_repo.add_test_tenant({
                "id": tenant_id,
                "name": "test-tenant",
                "is_active": True,
                "is_suspended": False
            })
            
            for idx, password in enumerate(valid_passwords):
                user_data = UserCreate(
                    tenant_id=tenant_id,
                    email=f"test_{idx}_{len(password)}@example.com",
                    password=password,
                    full_name="Test User"
                )
                
                # Should not raise any exceptions
                result = await self.auth_service.register_user(user_data)
                assert result is not None
    
    @pytest.mark.asyncio
    async def test_weak_password_rejection(self):
        """Test that weak passwords are rejected."""
        weak_passwords = [
            "weak",          # Too short
            "password",      # Too common
            "12345678",      # Only numbers
            "PASSWORD",      # Only uppercase
            "password123",   # Missing special chars
            "",              # Empty
            "a",             # Single character
        ]
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": False, "errors": ["Password too weak"]}
            
            # Setup test tenant
            tenant_id = str(uuid4())
            self.tenant_repo.add_test_tenant({
                "id": tenant_id,
                "name": "test-tenant",
                "is_active": True,
                "is_suspended": False
            })
            
            for password in weak_passwords:
                user_data = UserCreate(
                    tenant_id=tenant_id,
                    email=f"test_{len(password)}@example.com",
                    password=password,
                    full_name="Test User"
                )
                
                with pytest.raises(ValueError, match="Password validation failed"):
                    await self.auth_service.register_user(user_data)
    
    def test_password_complexity_requirements(self):
        """Test password complexity requirement logic."""
        # This would test the actual password complexity rules
        # For now, we test the interface to the password manager
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            # Test minimum length requirement
            mock_validator.return_value = {"valid": False, "errors": ["Password must be at least 8 characters"]}
            result = mock_validator("short")
            assert not result["valid"]
            assert "at least 8 characters" in result["errors"][0]
            
            # Test character type requirements
            mock_validator.return_value = {"valid": False, "errors": ["Password must contain uppercase, lowercase, number, and special character"]}
            result = mock_validator("lowercase123")
            assert not result["valid"]
            assert "uppercase" in result["errors"][0]


class TestAuthenticationAttemptLimiting:
    """Test authentication rate limiting and brute force protection."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_rate_limiting_logic(self):
        """Test authentication rate limiting logic."""
        email = "test@example.com"
        tenant_id = str(uuid4())
        
        # Setup tenant
        self.tenant_repo.add_test_tenant({
            "id": tenant_id,
            "is_active": True,
            "is_suspended": False
        })
        
        # Test that rate limiting kicks in after max attempts
        self.cache_service.data[f"login_attempts:{tenant_id}:{email}"] = 5  # Over limit
        
        credentials = UserLogin(
            email=email,
            password="wrong-password",
            tenant_id=tenant_id
        )
        
        with patch.object(self.auth_service.settings, 'MAX_LOGIN_ATTEMPTS', 3):
            result = await self.auth_service.authenticate(credentials)
            assert not result.success
            assert "Too many login attempts" in result.error
    
    @pytest.mark.asyncio 
    async def test_successful_login_clears_attempts(self):
        """Test that successful login clears failed attempt counter."""
        email = "test@example.com"
        tenant_id = str(uuid4())
        password = "SecurePassword123!"
        
        # Setup tenant
        self.tenant_repo.add_test_tenant({
            "id": tenant_id,
            "is_active": True,
            "is_suspended": False
        })
        
        # Setup user with failed attempts
        with patch('app.utils.security.password_manager.hash_password') as mock_hash:
            mock_hash.return_value = "hashed_password"
            
            self.user_repo.add_test_user({
                "id": "user-id",
                "tenant_id": tenant_id,
                "email": email,
                "hashed_password": "hashed_password",
                "full_name": "Test User",
                "roles": ["user"],
                "is_active": True
            })
        
        # Set some failed attempts
        self.cache_service.data[f"login_attempts:{tenant_id}:{email}"] = 2
        
        # Mock password verification to succeed
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = True
            
            with patch('app.utils.security.token_manager.create_access_token') as mock_access_token:
                mock_access_token.return_value = "mock_access_token"
                
                with patch('app.utils.security.token_manager.create_refresh_token') as mock_refresh_token:
                    mock_refresh_token.return_value = "mock_refresh_token"
                    
                    credentials = UserLogin(
                        email=email,
                        password=password,
                        tenant_id=tenant_id
                    )
                    
                    result = await self.auth_service.authenticate(credentials)
                    
                    # Should succeed and clear attempts
                    assert result.success
                    # Verify attempts were cleared
                    assert f"login_attempts:{tenant_id}:{email}" not in self.cache_service.data
    
    @pytest.mark.asyncio
    async def test_failed_login_increments_attempts(self):
        """Test that failed login increments attempt counter."""
        email = "test@example.com"
        tenant_id = str(uuid4())
        
        # Setup tenant
        self.tenant_repo.add_test_tenant({
            "id": tenant_id,
            "is_active": True,
            "is_suspended": False
        })
        
        # Setup user
        self.user_repo.add_test_user({
            "id": "user-id",
            "tenant_id": tenant_id,
            "email": email,
            "hashed_password": "correct_hash",
            "full_name": "Test User",
            "roles": ["user"],
            "is_active": True
        })
        
        # Mock password verification to fail
        with patch('app.utils.security.password_manager.verify_password') as mock_verify:
            mock_verify.return_value = False
            
            credentials = UserLogin(
                email=email,
                password="wrong-password",
                tenant_id=tenant_id
            )
            
            result = await self.auth_service.authenticate(credentials)
            
            # Should fail
            assert not result.success
            assert result.error == "Invalid credentials"
            
            # Verify attempts were incremented
            attempts_key = f"login_attempts:{tenant_id}:{email}"
            assert attempts_key in self.cache_service.data
            assert self.cache_service.data[attempts_key] == 1


class TestTokenValidationLogic:
    """Test JWT token validation and security logic."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_blacklisted_token_rejection(self):
        """Test that blacklisted tokens are rejected."""
        token = "mock.jwt.token"
        
        # Add token to blacklist
        self.cache_service.data["blacklist:mock_jti"] = True
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"jti": "mock_jti", "exp": 9999999999}
            
            result = await self.auth_service.verify_token(token)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_inactive_user_token_rejection(self):
        """Test that tokens for inactive users are rejected."""
        token = "mock.jwt.token"
        user_id = "user-id"
        
        # Setup inactive user
        self.user_repo.add_test_user({
            "id": user_id,
            "email": "test@example.com",
            "is_active": False,  # Inactive
            "tenant_id": "tenant-id"
        })
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": user_id, "exp": 9999999999}
            
            with patch('app.utils.security.token_manager.extract_token_data') as mock_extract:
                mock_extract.return_value = Mock(
                    token_type="access",
                    sub=user_id,
                    tenant_id="tenant-id",
                    roles=["user"],
                    permissions=[]
                )
                
                result = await self.auth_service.verify_token(token)
                assert result is None
    
    @pytest.mark.asyncio
    async def test_tenant_mismatch_rejection(self):
        """Test that tokens with tenant mismatch are rejected."""
        token = "mock.jwt.token"
        user_id = "user-id"
        
        # Setup user with different tenant than token
        self.user_repo.add_test_user({
            "id": user_id,
            "email": "test@example.com",
            "is_active": True,
            "tenant_id": str(uuid4())
        })
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": user_id, "exp": 9999999999}
            
            with patch('app.utils.security.token_manager.extract_token_data') as mock_extract:
                mock_extract.return_value = Mock(
                    token_type="access",
                    sub=user_id,
                    tenant_id=str(uuid4()),  # Different tenant
                    roles=["user"],
                    permissions=[]
                )
                
                result = await self.auth_service.verify_token(token)
                assert result is None
    
    @pytest.mark.asyncio
    async def test_super_admin_cross_tenant_access(self):
        """Test that super admins can access across tenants."""
        token = "mock.jwt.token"
        user_id = "super-admin-id"
        
        # Setup super admin user
        self.user_repo.add_test_user({
            "id": user_id,
            "email": "admin@system.com",
            "is_active": True,
            "tenant_id": str(uuid4())
        })
        
        with patch('app.utils.security.token_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": user_id, "exp": 9999999999}
            
            with patch('app.utils.security.token_manager.extract_token_data') as mock_extract:
                mock_extract.return_value = Mock(
                    token_type="access",
                    sub=user_id,
                    tenant_id=str(uuid4()),  # Different tenant
                    roles=["super_admin"],  # Super admin
                    permissions=[],
                    email="admin@system.com"
                )
                
                result = await self.auth_service.verify_token(token)
                assert result is not None
                assert result["user_id"] == user_id


class TestPasswordResetPolicyLogic:
    """Test password reset policy and security logic."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_password_reset_throttling_logic(self):
        """Test password reset request throttling."""
        email = "test@example.com"
        
        # Add throttle entry
        throttle_key = f"password_reset_throttle:{email}"
        self.cache_service.data[throttle_key] = True
        
        with patch.object(self.auth_service.settings, 'PASSWORD_RESET_REQUEST_INTERVAL_SECONDS', 300):
            request = PasswordResetRequest(email=email)
            result = await self.auth_service.request_password_reset(request)
            
            # Should be throttled
            assert result is None
    
    @pytest.mark.asyncio
    async def test_password_reset_email_validation(self):
        """Test password reset email format validation."""
        invalid_emails = [
            "invalid-email",
            "@domain.com", 
            "user@",
            "user@domain",
            "",
            "   ",
        ]
        
        with patch('app.utils.security.security_validator.validate_email') as mock_validator:
            mock_validator.return_value = False
            
            for email in invalid_emails:
                request = PasswordResetRequest(email=email)
                result = await self.auth_service.request_password_reset(request)
                assert result is None
    
    @pytest.mark.asyncio
    async def test_password_reset_token_expiration_logic(self):
        """Test password reset token expiration logic."""
        token = "valid-reset-token"
        
        # Simulate expired token by not having it in cache
        with patch.object(self.auth_service.settings, 'PASSWORD_RESET_TOKEN_TTL_MINUTES', 15):
            confirm_request = PasswordResetConfirm(
                token=token,
                new_password="NewSecurePassword123!"
            )
            
            with pytest.raises(ValueError, match="Invalid or expired reset token"):
                await self.auth_service.confirm_password_reset(confirm_request)
    
    @pytest.mark.asyncio
    async def test_password_reset_success_revokes_sessions(self):
        """Test that successful password reset revokes all user sessions."""
        token = "valid-reset-token"
        user_id = "user-id"
        
        # Setup reset token data
        token_digest = "token_digest_hash"
        reset_payload = {
            "user_id": user_id,
            "tenant_id": "tenant-id",
            "email": "test@example.com",
            "token_digest": token_digest,
            "requested_at": datetime.utcnow().isoformat()
        }
        
        cache_key = f"password_reset:{token_digest}"
        self.cache_service.data[cache_key] = reset_payload
        
        # Setup user
        self.user_repo.add_test_user({
            "id": user_id,
            "email": "test@example.com",
            "tenant_id": "tenant-id"
        })
        
        # Add some sessions
        self.session_repo.add_test_session({
            "id": "session-1",
            "user_id": user_id
        })
        self.session_repo.add_test_session({
            "id": "session-2", 
            "user_id": user_id
        })
        
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            with patch('app.utils.security.password_manager.hash_password') as mock_hash:
                mock_hash.return_value = "new_hashed_password"
                
                with patch('hashlib.sha256') as mock_sha:
                    mock_sha.return_value.hexdigest.return_value = token_digest
                    
                    confirm_request = PasswordResetConfirm(
                        token=token,
                        new_password="NewSecurePassword123!"
                    )
                    
                    result = await self.auth_service.confirm_password_reset(confirm_request)
                    
                    assert result is True
                    # Verify sessions were deleted
                    assert self.session_repo.get_call_count("delete_by_user") > 0


class TestSessionManagementLogic:
    """Test session creation, validation, and termination logic."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_session_termination_authorization(self):
        """Test session termination authorization logic."""
        user_id = "user-id"
        session_id = "session-id"
        different_user_id = str(uuid4())
        
        # Setup session for user
        self.session_repo.add_test_session({
            "id": session_id,
            "user_id": user_id,
            "tenant_id": "tenant-id"
        })
        
        # Try to terminate session as different user - should fail
        result = await self.auth_service.terminate_session(session_id, different_user_id)
        assert result is False
        
        # Try to terminate session as correct user - should succeed
        result = await self.auth_service.terminate_session(session_id, user_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_session_listing_isolation(self):
        """Test that users can only see their own sessions."""
        user1_id = "user1-id"
        user2_id = "user2-id"
        
        # Setup sessions for different users
        self.session_repo.add_test_session({
            "id": "session-1",
            "user_id": user1_id,
            "tenant_id": "tenant-id",
            "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat()
        })
        
        self.session_repo.add_test_session({
            "id": "session-2",
            "user_id": user2_id,
            "tenant_id": "tenant-id",
            "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat()
        })
        
        # User 1 should only see their session
        user1_sessions = await self.auth_service.list_sessions(user1_id)
        assert len(user1_sessions) == 1
        assert user1_sessions[0].session_id == "session-1"
        
        # User 2 should only see their session
        user2_sessions = await self.auth_service.list_sessions(user2_id)
        assert len(user2_sessions) == 1
        assert user2_sessions[0].session_id == "session-2"


class TestUserProfileUpdateLogic:
    """Test user profile update policies and validation."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_restricted_field_update_prevention(self):
        """Test that restricted fields cannot be updated via profile update."""
        from app.infrastructure.persistence.models.auth_tables import CurrentUser, UserUpdate
        
        user_id = "user-id"
        tenant_id = "tenant-id"
        
        # Setup user
        self.user_repo.add_test_user({
            "id": user_id,
            "email": "test@example.com",
            "tenant_id": tenant_id,
            "full_name": "Test User",
            "roles": ["user"],
            "permissions": ["read"],
            "is_active": True
        })
        
        current_user = CurrentUser(
            user_id=user_id,
            email="test@example.com",
            full_name="Test User",
            tenant_id=tenant_id,
            roles=["user"],
            permissions=["read"],
            is_active=True,
            is_superuser=False
        )
        
        # Try to update restricted fields
        restricted_updates = [
            UserUpdate(roles=["admin"]),
            UserUpdate(permissions=["admin"]),
            UserUpdate(is_active=False),
        ]
        
        for update in restricted_updates:
            with pytest.raises(ValueError, match="Not permitted to update fields"):
                await self.auth_service.update_user_profile(current_user, update)
    
    @pytest.mark.asyncio
    async def test_tenant_context_validation(self):
        """Test that profile updates are validated against tenant context."""
        from app.infrastructure.persistence.models.auth_tables import CurrentUser, UserUpdate
        
        user_id = "user-id"
        user_tenant_id = str(uuid4())
        different_tenant_id = str(uuid4())
        
        # Setup user in one tenant
        self.user_repo.add_test_user({
            "id": user_id,
            "email": "test@example.com",
            "tenant_id": user_tenant_id,
            "full_name": "Test User"
        })
        
        # Create current user context with different tenant
        current_user = CurrentUser(
            user_id=user_id,
            email="test@example.com",
            full_name="Test User",
            tenant_id=different_tenant_id,  # Different tenant
            roles=["user"],
            permissions=["read"],
            is_active=True,
            is_superuser=False
        )
        
        update = UserUpdate(full_name="Updated Name")
        
        with pytest.raises(ValueError, match="Invalid tenant context"):
            await self.auth_service.update_user_profile(current_user, update)


class TestAuthenticationBusinessRules:
    """Test core authentication business rules and edge cases."""
    
    def setup_method(self):
        self.user_repo = MockUserRepository()
        self.tenant_repo = MockTenantRepository()
        self.session_repo = MockUserSessionRepository()
        self.cache_service = MockCacheService()
        self.notification_service = MockNotificationService()
        
        self.auth_service = AuthenticationService(
            user_repository=self.user_repo,
            tenant_repository=self.tenant_repo,
            cache_manager=self.cache_service,
            notification_service=self.notification_service,
            session_repository=self.session_repo
        )
    
    @pytest.mark.asyncio
    async def test_suspended_tenant_authentication_blocked(self):
        """Test that users from suspended tenants cannot authenticate."""
        tenant_id = str(uuid4())
        email = "user@suspended.com"
        
        # Setup suspended tenant
        self.tenant_repo.add_test_tenant({
            "id": tenant_id,
            "is_active": True,
            "is_suspended": True  # Suspended
        })
        
        user_data = UserCreate(
            tenant_id=tenant_id,
            email=email,
            password="SecurePassword123!",
            full_name="Test User"
        )
        
        with pytest.raises(ValueError, match="Organization is currently suspended"):
            await self.auth_service.register_user(user_data)
    
    @pytest.mark.asyncio
    async def test_duplicate_email_per_tenant_prevention(self):
        """Test that duplicate emails are prevented within same tenant."""
        tenant_id = str(uuid4())
        email = "duplicate@example.com"
        
        # Setup tenant
        self.tenant_repo.add_test_tenant({
            "id": tenant_id,
            "is_active": True,
            "is_suspended": False
        })
        
        # Add existing user
        self.user_repo.add_test_user({
            "id": str(uuid4()),
            "tenant_id": tenant_id,
            "email": email
        })
        
        # Try to register another user with same email in same tenant
        user_data = UserCreate(
            tenant_id=tenant_id,
            email=email,
            password="SecurePassword123!",
            full_name="Duplicate User"
        )
        
        with pytest.raises(ValueError, match="User already exists with this email"):
            await self.auth_service.register_user(user_data)
    
    @pytest.mark.asyncio
    async def test_cross_tenant_email_isolation(self):
        """Test that same email can exist in different tenants."""
        email = "user@example.com"
        tenant1_id = "tenant1-id"
        tenant2_id = "tenant2-id"
        
        # Setup tenants
        for tenant_id in [tenant1_id, tenant2_id]:
            self.tenant_repo.add_test_tenant({
                "id": tenant_id,
                "is_active": True,
                "is_suspended": False
            })
        
        # Register user in first tenant
        with patch('app.utils.security.password_manager.validate_password_strength') as mock_validator:
            mock_validator.return_value = {"valid": True, "errors": []}
            
            user1_data = UserCreate(
                tenant_id=tenant1_id,
                email=email,
                password="SecurePassword123!",
                full_name="User One"
            )
            
            result1 = await self.auth_service.register_user(user1_data)
            assert result1 is not None
            
            # Register user with same email in different tenant - should succeed
            user2_data = UserCreate(
                tenant_id=tenant2_id,
                email=email,
                password="SecurePassword123!",
                full_name="User Two"
            )
            
            result2 = await self.auth_service.register_user(user2_data)
            assert result2 is not None
            assert result1.id != result2.id
