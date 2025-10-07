"""
Authentication Service Implementation

Provides core authentication functionality including:
- User authentication and token management
- JWT token generation and validation
- Password hashing and verification
- API key management
- Session tracking
- Audit logging
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, Any, Optional, List, Sequence
from uuid import uuid4, UUID

import structlog
from app.core.config import get_settings
from app.infrastructure.persistence.models.auth_tables import (
    UserTable, UserCreate, UserLogin, CurrentUser, UserUpdate, TokenResponse,
    AuthenticationResult, APIKeyTable, APIKeyCreate, APIKeyResponse,
    APIKeyInfo, RefreshTokenRequest, PasswordChangeRequest,
    PasswordResetRequest, PasswordResetConfirm, SessionInfo, AuditLog
)
from app.database.repositories.postgres import UserRepository, TenantRepository, UserSessionRepository
from app.utils.security import password_manager, token_manager, api_key_manager, security_validator
from app.domain.interfaces import ICacheService, IAuditService
from app.infrastructure.persistence.models.tenant_table import TenantConfiguration, SubscriptionTier
from app.domain.interfaces import INotificationService
from app.infrastructure.persistence.models.audit_table import AuditEventType
from app.infrastructure.providers.audit_provider import get_audit_service
from app.utils.session_utils import (
    build_session_cache_key,
    serialize_sessions,
    deserialize_sessions,
    calculate_session_ttl_seconds,
    remove_session_by_id,
    sort_sessions_by_last_activity,
    filter_expired_sessions,
    session_info_from_record,
    sessions_from_records,
)
from pydantic import ValidationError

logger = structlog.get_logger(__name__)


class AuthenticationService:
    """Core authentication service implementation"""
    
    def __init__(
        self,
        user_repository: UserRepository,
        tenant_repository: TenantRepository,
        cache_manager: ICacheService,
        notification_service: Optional[INotificationService] = None,
        session_repository: Optional[UserSessionRepository] = None
    ):
        self.user_repo = user_repository
        self.tenant_repo = tenant_repository
        self.cache = cache_manager
        self.settings = get_settings()
        self.notification_service = notification_service
        self.session_repo = session_repository or UserSessionRepository()
        
        # Cache keys
        self.USER_CACHE_PREFIX = "user:"
        self.SESSION_CACHE_PREFIX = "session:"
        self.BLACKLIST_CACHE_PREFIX = "blacklist:"
        self.LOGIN_ATTEMPTS_PREFIX = "login_attempts:"
        self.PASSWORD_RESET_PREFIX = "password_reset:"
        self.PASSWORD_RESET_THROTTLE_PREFIX = "password_reset_throttle:"
        
    async def register_user(self, user_data: UserCreate) -> UserTable | SimpleNamespace:
        """Register user in existing tenant"""
        
        # Validate password strength
        password_validation = password_manager.validate_password_strength(user_data.password)
        if not password_validation["valid"]:
            raise ValueError(f"Password validation failed: {'; '.join(password_validation['errors'])}")
        
        # Check if user already exists in this tenant
        existing_user = await self.user_repo.get_by_email(
            user_data.email, 
            user_data.tenant_id
        )
        if existing_user:
            raise ValueError("User already exists with this email in this organization")
        
        # Validate tenant exists and is active
        tenant = await self.tenant_repo.get(user_data.tenant_id)
        if not tenant:
            raise ValueError("Invalid organization")
        
        # Handle both dict and object formats for tenant data
        is_active = tenant.get("is_active") if isinstance(tenant, dict) else tenant.is_active
        is_suspended = tenant.get("is_suspended") if isinstance(tenant, dict) else tenant.is_suspended
        
        if not is_active:
            raise ValueError("Invalid or inactive organization")
        
        # Check if organization is suspended
        if is_suspended:
            raise ValueError("Organization is currently suspended")
        
        # Hash password
        hashed_password = password_manager.hash_password(user_data.password)
        
        # Parse full name into first and last name
        name_parts = user_data.full_name.strip().split(' ', 1)
        first_name = name_parts[0] if name_parts else user_data.full_name
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        # Create user record payload compatible with repository expectations
        user_record = {
            "tenant_id": user_data.tenant_id,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": user_data.full_name,
            "roles": user_data.roles or ["user"],
            "permissions": [],
            "is_active": True,
            "is_verified": False,
            "is_superuser": False,
        }

        # Save to repository
        created_user_dict = await self.user_repo.create(user_record)

        # Convert repository response into a domain object (UserTable when possible)
        created_user = self._build_user_object(created_user_dict)
        
        # Log registration
        await self._log_security_event(
            tenant_id=str(user_record["tenant_id"]),
            user_id=str(getattr(created_user, "id")),
            action="user_registered",
            resource_type="user",
            resource_id=str(getattr(created_user, "id")),
            details={"email": user_data.email, "registration_type": "existing_tenant"}
        )

        logger.info(
            "User registered successfully",
            user_id=getattr(created_user, "id"),
            email=user_data.email,
            tenant_id=user_data.tenant_id
        )
        
        return created_user
    
    async def authenticate(self, credentials: UserLogin) -> AuthenticationResult:
        """Authenticate user credentials"""
        
        try:
            tenant_scope = credentials.tenant_id

            # Check rate limiting
            if not await self._check_login_rate_limit(credentials.email, tenant_scope):
                await self._log_security_event(
                    tenant_id=tenant_scope,
                    action="login_rate_limited",
                    resource_type="user",
                    details={"email": credentials.email},
                    risk_level="high",
                    suspicious=True
                )
                return AuthenticationResult(
                    success=False,
                    error="Too many login attempts. Please try again later."
                )
            
            # Get user from repository
            user = await self.user_repo.get_by_email(
                credentials.email,
                tenant_scope
            )
            
            if not user:
                await self._record_failed_login(credentials.email, tenant_scope)
                return AuthenticationResult(
                    success=False,
                    error="Invalid credentials"
                )
            
            user_obj = self._build_user_object(user)

            # Check if user is active
            is_active = getattr(user_obj, "is_active", True)
            if not is_active:
                user_id = getattr(user_obj, "id", None)
                await self._log_security_event(
                    tenant_id=tenant_scope,
                    user_id=user_id,
                    action="login_attempt_inactive_user",
                    resource_type="user",
                    details={"email": credentials.email},
                    risk_level="medium",
                    suspicious=True
                )
                return AuthenticationResult(
                    success=False,
                    error="Account is not active"
                )
            
            # Verify password
            hashed_password = getattr(user_obj, "hashed_password", None)
            if not hashed_password or not password_manager.verify_password(credentials.password, hashed_password):
                await self._record_failed_login(credentials.email, tenant_scope)
                return AuthenticationResult(
                    success=False,
                    error="Invalid credentials"
                )
            
            user_id = str(getattr(user_obj, "id"))
            email = getattr(user_obj, "email")
            full_name = getattr(user_obj, "full_name", "")
            tenant_id = getattr(user_obj, "tenant_id", tenant_scope)
            tenant_id = str(tenant_id) if tenant_id is not None else str(tenant_scope)
            roles = getattr(user_obj, "roles", []) or []
            if isinstance(roles, str):
                roles = [roles]
            permissions = getattr(user_obj, "permissions", []) or []
            is_superuser = getattr(user_obj, "is_superuser", False)
            
            # Generate tokens
            tokens = await self._generate_tokens(user_obj)
            
            # Create session
            session_info = await self._create_session(user_obj, tokens.refresh_token)
            
            # Update user last login
            await self.user_repo.update_last_login(user_id)
            
            # Clear failed login attempts
            await self._clear_failed_login_attempts(credentials.email, tenant_scope)
            
            # Create current user context
            current_user = CurrentUser(
                user_id=user_id,
                email=email,
                full_name=full_name,
                tenant_id=tenant_id,
                roles=[roles] if isinstance(roles, str) else roles,
                permissions=permissions,
                is_active=is_active,
                is_superuser=is_superuser
            )
            
            # Log successful authentication
            await self._log_security_event(
                tenant_id=tenant_id,
                user_id=user_id,
                action="login_successful",
                resource_type="user",
                details={
                    "email": email,
                    "session_id": session_info.session_id
                }
            )
            
            logger.info(
                "User authenticated successfully",
                user_id=user_id,
                email=email,
                tenant_id=tenant_id
            )
            
            return AuthenticationResult(
                success=True,
                user=current_user,
                tokens=tokens
            )
            
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            return AuthenticationResult(
                success=False,
                error="Authentication failed"
            )
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user data"""
        
        # Check if token is blacklisted
        if await self._is_token_blacklisted(token):
            logger.warning("Attempted use of blacklisted token")
            return None
        
        # Verify token signature and expiration
        payload = token_manager.verify_token(token)
        if not payload:
            return None
        
        # Extract token data
        token_data = token_manager.extract_token_data(payload)
        if not token_data or token_data.token_type != "access":
            return None
        
        # Get user from cache or database
        user_data = await self._get_user_data(token_data.sub)
        if not user_data or not user_data.get("is_active"):
            return None
        
        # Verify tenant match - allow super admins to bypass this check for system operations
        is_super_admin = "super_admin" in token_data.roles
        system_tenant_id = "00000000-0000-0000-0000-000000000000"
        
        # Super admins can access any tenant, including system operations
        if not is_super_admin:
            if user_data.get("tenant_id") != token_data.tenant_id:
                logger.warning("Token tenant ID mismatch", user_id=token_data.sub)
                return None
        elif token_data.tenant_id == system_tenant_id:
            # Super admin accessing system tenant - allow this for system operations
            logger.debug("Super admin accessing system tenant", user_id=token_data.sub)
        else:
            # Super admin accessing specific tenant - validate they have access
            if user_data.get("tenant_id") != token_data.tenant_id:
                logger.debug("Super admin cross-tenant access", user_id=token_data.sub, 
                           token_tenant=token_data.tenant_id, user_tenant=user_data.get("tenant_id"))
        
        return {
            "user_id": token_data.sub,
            "email": token_data.email,
            "full_name": user_data.get("full_name", ""),
            "tenant_id": token_data.tenant_id,
            "roles": token_data.roles,
            "permissions": token_data.permissions,
            "is_active": user_data.get("is_active", False),
            "is_superuser": user_data.get("is_superuser", False)
        }
    
    async def refresh_tokens(self, request: RefreshTokenRequest) -> Optional[TokenResponse]:
        """Refresh access and refresh tokens with rotation"""
        
        # Verify refresh token
        payload = token_manager.verify_token(request.refresh_token)
        if not payload or payload.get("token_type") != "refresh":
            return None
        
        # Check if token is blacklisted
        if await self._is_token_blacklisted(request.refresh_token):
            return None
        
        # Get user data
        user_id = payload.get("sub")
        user_data = await self._get_user_data(user_id)
        if not user_data or not user_data.get("is_active"):
            return None
        
        # Blacklist old refresh token
        await self._blacklist_token(request.refresh_token)
        
        # Create user object for token generation - normalize data types
        user = self._build_user_object(user_data)
        
        # Generate new tokens with same family (for rotation tracking)
        token_family = payload.get("family")
        tokens = await self._generate_tokens(user, token_family)
        
        # Update session
        await self._update_session_tokens(user_id, tokens.refresh_token)
        
        logger.info("Tokens refreshed successfully", user_id=user_id)
        
        return tokens
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke/blacklist a token"""
        
        payload = token_manager.verify_token(token)
        if not payload:
            return False

        # Blacklist the token
        await self._blacklist_token(token)

        # If it's a refresh token, blacklist the entire token family
        if payload.get("token_type") == "refresh":
            token_family = payload.get("family")
            if token_family:
                await self._blacklist_token_family(payload.get("sub"), token_family)
        
        logger.info("Token revoked", jti=payload.get("jti"))
        
        return True

    def _build_password_reset_link(self, base_url: str, token: str) -> str:
        """Attach reset token to provided base URL."""
        try:
            from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

            if not base_url:
                return token

            parsed = urlparse(base_url)
            query_params = dict(parse_qsl(parsed.query))
            query_params["token"] = token

            new_query = urlencode(query_params)
            rebuilt = parsed._replace(query=new_query)
            return urlunparse(rebuilt)
        except Exception as ex:
            logger.warning(
                "Failed to construct password reset URL",
                base_url=base_url,
                error=str(ex)
            )
            if not base_url:
                return token

            if base_url.endswith("&") or base_url.endswith("?"):
                separator = ""
            elif "?" in base_url:
                separator = "&"
            else:
                separator = "?"

            return f"{base_url}{separator}token={token}"

    def _compose_password_reset_email(
        self,
        recipient_name: str,
        reset_target: str,
        ttl_seconds: int
    ) -> str:
        ttl_minutes = max(1, ttl_seconds // 60)
        app_name = self.settings.APP_NAME

        lines = [
            f"Hello {recipient_name},",
            "",
            f"We received a request to reset your {app_name} password.",
            "",
            "You can reset your password using the link or token below:",
            reset_target,
            "",
            f"This link expires in {ttl_minutes} minutes.",
            "If you did not request a reset, you can ignore this email.",
            "",
            "Thanks,",
            f"The {app_name} team",
        ]

        return "\n".join(lines)

    async def _send_password_reset_email(
        self,
        email: str,
        recipient_name: str,
        reset_link: Optional[str],
        raw_token: str,
        ttl_seconds: int
    ) -> bool:
        if not self.notification_service:
            return False

        target = reset_link or raw_token
        body = self._compose_password_reset_email(recipient_name, target, ttl_seconds)
        subject = f"{self.settings.APP_NAME} password reset"

        try:
            return await self.notification_service.send_email(
                to=email,
                subject=subject,
                body=body,
                is_html=False
            )
        except Exception as ex:
            logger.error(
                "Password reset email dispatch failed",
                email=email,
                error=str(ex)
            )
            return False

    async def request_password_reset(
        self,
        request: PasswordResetRequest
    ) -> Optional[Dict[str, Any]]:
        """Initiate password reset workflow for a user."""

        email = request.email.strip().lower()
        if not security_validator.validate_email(email):
            logger.warning("Password reset requested with invalid email format", email=email)
            return None

        throttle_key = f"{self.PASSWORD_RESET_THROTTLE_PREFIX}{email}"
        throttle_window = max(0, self.settings.PASSWORD_RESET_REQUEST_INTERVAL_SECONDS)
        try:
            if throttle_window and await self.cache.exists(throttle_key):
                logger.info("Password reset request throttled", email=email)
                return None
        except Exception as ex:
            logger.warning(
                "Password reset throttle check failed",
                email=email,
                error=str(ex)
            )

        user_record: Optional[Dict[str, Any]] = None
        try:
            if request.tenant_id:
                user_record = await self.user_repo.get_by_email(email, str(request.tenant_id))
            else:
                matches = await self.user_repo.find_by_criteria({"email": email}, limit=1)
                user_record = matches[0] if matches else None
        except Exception as ex:
            logger.error("Password reset lookup failed", email=email, error=str(ex))
            user_record = None

        if not user_record:
            # Return silently to avoid account enumeration
            logger.info("Password reset requested for unknown email", email=email)
            if throttle_window:
                await self.cache.set(throttle_key, True, ttl=throttle_window)
            return None

        normalized_user = self._normalize_user_data(user_record)
        user_id = normalized_user.get("id")
        tenant_id = normalized_user.get("tenant_id")

        # Generate secure token and hash for storage
        token = secrets.token_urlsafe(self.settings.PASSWORD_RESET_TOKEN_BYTES)
        token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()

        ttl_seconds = max(60, self.settings.PASSWORD_RESET_TOKEN_TTL_MINUTES * 60)
        cache_key = f"{self.PASSWORD_RESET_PREFIX}{token_digest}"
        user_token_key = f"{self.PASSWORD_RESET_PREFIX}user:{user_id}"

        reset_payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "email": email,
            "token_digest": token_digest,
            "requested_at": datetime.utcnow().isoformat()
        }

        # Invalidate any previous reset tokens for this user
        try:
            previous_digest = await self.cache.get(user_token_key)
            if previous_digest:
                await self.cache.delete(f"{self.PASSWORD_RESET_PREFIX}{previous_digest}")
        except Exception as ex:
            logger.warning(
                "Failed to clear previous password reset token",
                user_id=user_id,
                error=str(ex)
            )

        await self.cache.set(cache_key, reset_payload, ttl=ttl_seconds)
        await self.cache.set(user_token_key, token_digest, ttl=ttl_seconds)

        if throttle_window:
            await self.cache.set(throttle_key, True, ttl=throttle_window)

        reset_link = None
        if request.redirect_url:
            reset_link = self._build_password_reset_link(request.redirect_url, token)

        recipient_name = normalized_user.get("full_name") or email
        email_sent = await self._send_password_reset_email(
            email=email,
            recipient_name=recipient_name,
            reset_link=reset_link,
            raw_token=token,
            ttl_seconds=ttl_seconds
        )

        await self._log_security_event(
            tenant_id=tenant_id,
            user_id=user_id,
            action="password_reset_requested",
            resource_type="user",
            details={
                "email": email,
                "reset_token_hash": token_digest,
                "expires_in_seconds": ttl_seconds
            }
        )

        logger.info(
            "Password reset token generated",
            user_id=user_id,
            tenant_id=tenant_id,
            ttl_seconds=ttl_seconds,
            email_sent=email_sent
        )

        return {
            "token": token,
            "token_hash": token_digest,
            "reset_link": reset_link,
            "email": email,
            "tenant_id": tenant_id,
            "email_sent": email_sent
        }

    async def confirm_password_reset(
        self,
        request: PasswordResetConfirm
    ) -> bool:
        """Validate reset token, update password, and revoke sessions."""

        token = request.token.strip()
        if not token:
            raise ValueError("Reset token is required")

        password_validation = password_manager.validate_password_strength(request.new_password)
        if not password_validation["valid"]:
            raise ValueError("; ".join(password_validation["errors"]))

        token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        cache_key = f"{self.PASSWORD_RESET_PREFIX}{token_digest}"

        reset_payload = await self.cache.get(cache_key)
        if not reset_payload or not isinstance(reset_payload, dict):
            raise ValueError("Invalid or expired reset token")

        user_id = reset_payload.get("user_id")
        tenant_id = reset_payload.get("tenant_id")
        email = reset_payload.get("email")

        if not user_id:
            raise ValueError("Invalid reset token payload")

        user_record = await self.user_repo.get_by_id(user_id)
        if not user_record:
            raise ValueError("User account not found")

        hashed_password = password_manager.hash_password(request.new_password)
        updated_user = await self.user_repo.update(user_id, {"hashed_password": hashed_password})
        if not updated_user:
            raise ValueError("Failed to update password")

        user_id_str = str(user_id)
        await self._revoke_all_user_sessions(user_id_str)

        # Clear reset token cache entries
        await self.cache.delete(cache_key)
        await self.cache.delete(f"{self.PASSWORD_RESET_PREFIX}user:{user_id_str}")

        if email:
            await self.cache.delete(f"{self.PASSWORD_RESET_THROTTLE_PREFIX}{email}")

        # Clear user cache to avoid stale state
        await self.cache.delete(f"{self.USER_CACHE_PREFIX}{user_id_str}")

        await self._log_security_event(
            tenant_id=str(tenant_id) if tenant_id is not None else None,
            user_id=user_id_str,
            action="password_reset_completed",
            resource_type="user",
            details={
                "email": email,
                "reset_token_hash": token_digest
            }
        )

        logger.info(
            "Password reset completed",
            user_id=user_id_str,
            tenant_id=tenant_id
        )

        return True

    async def list_sessions(self, user_id: str) -> List[SessionInfo]:
        """List active sessions for a user from cache."""
        user_id_str = str(user_id)

        sessions = await self._get_sessions_from_cache(user_id_str)
        if sessions:
            return sessions

        try:
            records = await self.session_repo.list_by_user(user_id_str)
        except Exception as ex:
            logger.error("Failed to load sessions from repository", user_id=user_id_str, error=str(ex))
            return []

        sessions = filter_expired_sessions(sessions_from_records(records))
        await self._cache_sessions(user_id_str, sessions)

        return sessions

    async def terminate_session(self, session_id: str, user_id: str) -> bool:
        """Terminate a specific session for a user."""

        user_id_str = str(user_id)
        try:
            session_record = await self.session_repo.find_by_id(session_id)
        except Exception as ex:
            logger.error(
                "Failed to locate session",
                session_id=session_id,
                user_id=user_id_str,
                error=str(ex)
            )
            return False

        if not session_record or str(session_record.get("user_id")) != user_id_str:
            return False

        removed_session = session_info_from_record(session_record)

        deleted = await self.session_repo.delete(session_id)
        if not deleted:
            return False

        cached_sessions = await self._get_sessions_from_cache(user_id_str)
        updated_sessions, _ = remove_session_by_id(cached_sessions, session_id)
        await self._cache_sessions(user_id_str, updated_sessions)

        await self._log_security_event(
            tenant_id=removed_session.tenant_id,
            user_id=user_id_str,
            action="session_terminated",
            resource_type="session",
            resource_id=session_id,
            details={
                "ip_address": removed_session.ip_address,
                "user_agent": removed_session.user_agent
            }
        )

        logger.info(
            "Session terminated",
            user_id=user_id_str,
            session_id=session_id
        )

        return True

    async def update_user_profile(
        self,
        current_user: CurrentUser,
        update_request: UserUpdate
    ) -> CurrentUser:
        """Update mutable profile fields for the current user"""

        user_record = await self.user_repo.get_by_id(current_user.user_id)
        if not user_record:
            raise ValueError("User not found")

        if hasattr(user_record, "model_dump"):
            user_data = user_record.model_dump()
        elif hasattr(user_record, "dict"):
            user_data = user_record.dict()
        elif hasattr(user_record, "__dict__"):
            user_data = user_record.__dict__.copy()
        else:
            user_data = dict(user_record)

        if str(user_data.get("tenant_id")) != str(current_user.tenant_id):
            raise ValueError("Invalid tenant context for user update")

        requested_updates = update_request.model_dump(exclude_unset=True)
        if not requested_updates:
            return current_user

        restricted_fields = {"roles", "permissions", "is_active"}
        attempted_restricted = restricted_fields.intersection(requested_updates.keys())
        if attempted_restricted:
            raise ValueError(
                "Not permitted to update fields: " + ", ".join(sorted(attempted_restricted))
            )

        if "username" in requested_updates:
            raise ValueError("Username updates are not supported")

        profile_updates: Dict[str, Any] = {}

        if "full_name" in requested_updates:
            full_name = requested_updates["full_name"].strip()
            if not full_name:
                raise ValueError("Full name cannot be empty")
            name_parts = full_name.split(" ", 1)
            profile_updates["full_name"] = full_name
            profile_updates["first_name"] = name_parts[0]
            profile_updates["last_name"] = name_parts[1] if len(name_parts) > 1 else ""

        if "settings" in requested_updates:
            new_settings = requested_updates["settings"] or {}
            if not isinstance(new_settings, dict):
                raise ValueError("Settings must be a JSON object")
            existing_settings = user_data.get("settings") or {}
            if not isinstance(existing_settings, dict):
                existing_settings = {}
            merged_settings = existing_settings.copy()
            merged_settings.update(new_settings)
            profile_updates["settings"] = merged_settings

        if not profile_updates:
            return current_user

        profile_updates["updated_at"] = datetime.utcnow()

        updated_user = await self.user_repo.update(current_user.user_id, profile_updates)
        if not updated_user:
            raise ValueError("Failed to persist user updates")

        normalized_user = self._normalize_user_data(updated_user)

        cache_key = f"{self.USER_CACHE_PREFIX}{current_user.user_id}"
        await self.cache.set(cache_key, normalized_user, ttl=300)

        await self._log_security_event(
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            action="profile_updated",
            resource_type="user",
            details={"fields": [field for field in profile_updates.keys() if field != "updated_at"]}
        )

        return CurrentUser(
            user_id=normalized_user.get("id", current_user.user_id),
            email=normalized_user.get("email", current_user.email),
            full_name=normalized_user.get("full_name", current_user.full_name),
            tenant_id=normalized_user.get("tenant_id", current_user.tenant_id),
            roles=normalized_user.get("roles", current_user.roles),
            permissions=normalized_user.get("permissions", current_user.permissions),
            is_active=normalized_user.get("is_active", current_user.is_active),
            is_superuser=normalized_user.get("is_superuser", current_user.is_superuser)
        )

    async def change_password(self, user_id: str, request: PasswordChangeRequest) -> bool:
        """Change user password"""
        
        # Get user and normalize to dict structure
        user_record = await self.user_repo.get_by_id(user_id)
        if not user_record:
            return False

        if hasattr(user_record, "dict"):
            user_data = user_record.dict()
        else:
            user_data = dict(user_record)

        tenant_id = user_data.get("tenant_id")
        tenant_id_str = str(tenant_id) if tenant_id is not None else None

        # Verify the current password
        hashed_password = user_data.get("hashed_password")
        if not hashed_password or not password_manager.verify_password(request.current_password, hashed_password):
            await self._log_security_event(
                tenant_id=tenant_id_str,
                user_id=user_id,
                action="password_change_failed",
                resource_type="user",
                details={"reason": "invalid_current_password"},
                risk_level="medium",
                suspicious=True
            )
            return False
        
        # Validate new password
        password_validation = password_manager.validate_password_strength(request.new_password)
        if not password_validation["valid"]:
            raise ValueError(f"Password validation failed: {'; '.join(password_validation['errors'])}")
        
        # Hash new password
        new_hashed_password = password_manager.hash_password(request.new_password)
        
        # Update password
        await self.user_repo.update(user_id, {"hashed_password": new_hashed_password})

        # Invalidate all user sessions
        await self._revoke_all_user_sessions(user_id)

        # Log password change
        await self._log_security_event(
            tenant_id=tenant_id_str,
            user_id=user_id,
            action="password_changed",
            resource_type="user",
            details={"success": True}
        )
        
        logger.info("Password changed successfully", user_id=user_id)
        
        return True
    
    async def create_api_key(self, request: APIKeyCreate) -> APIKeyResponse:
        """Create a new API key"""
        
        # Generate API key
        api_key = api_key_manager.generate_api_key()
        key_hash = api_key_manager.hash_api_key(api_key)
        
        # Calculate expiration
        expires_at = None
        if request.expires_in_days:
            expires_at = (datetime.utcnow() + timedelta(days=request.expires_in_days)).isoformat()
        
        # Create API key record
        api_key_record = APIKeyTable(
            tenant_id=request.tenant_id,
            key_hash=key_hash,
            name=request.name,
            permissions=request.permissions,
            expires_at=expires_at
        )
        
        # Save to repository
        # TODO: Implement APIKeyRepository
        # For now, just create a mock response
        created_key = api_key_record
        
        # Log API key creation
        await self._log_security_event(
            tenant_id=request.tenant_id,
            action="api_key_created",
            resource_type="api_key",
            resource_id=created_key.id,
            details={"name": request.name, "permissions": request.permissions}
        )
        
        logger.info("API key created", key_id=created_key.id, name=request.name)
        
        return APIKeyResponse(
            api_key=api_key,  # Only returned once
            key_id=created_key.id,
            name=created_key.name,
            permissions=created_key.permissions,
            expires_at=created_key.expires_at,
            created_at=created_key.created_at
        )
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key info"""
        
        try:
            # TODO: Implement API key validation
            # For now, return None - this needs proper implementation
            api_keys = []
            
            for key_record in api_keys:
                if api_key_manager.verify_api_key(api_key, key_record.key_hash):
                    # Check if key is expired
                    if key_record.expires_at:
                        expires_at = datetime.fromisoformat(key_record.expires_at)
                        if datetime.utcnow() > expires_at:
                            continue
                    
                    # TODO: Update usage tracking
                    
                    return {
                        "id": key_record.id,
                        "tenant_id": key_record.tenant_id,
                        "name": key_record.name,
                        "permissions": key_record.permissions
                    }
            
            return None
            
        except Exception as e:
            logger.error("API key validation failed", error=str(e))
            return None
    
    # Private helper methods
    
    async def _generate_tokens(self, user: UserTable, token_family: Optional[str] = None) -> TokenResponse:
        """Generate access and refresh tokens for user"""
        
        # Ensure all UUID fields are converted to strings for token generation
        access_token = token_manager.create_access_token(
            user_id=str(user.id),  # Convert UUID to string
            email=user.email,
            tenant_id=str(user.tenant_id),  # Convert UUID to string
            roles=user.roles,
            permissions=user.permissions
        )
        
        refresh_token = token_manager.create_refresh_token(
            user_id=str(user.id),  # Convert UUID to string
            tenant_id=str(user.tenant_id),  # Convert UUID to string
            token_family=token_family
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    async def _get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from cache or database"""
        
        # Try cache first
        cache_key = f"{self.USER_CACHE_PREFIX}{user_id}"
        cached_data = await self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Get from database
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return None
        
        # Handle both dict and object responses from repository
        if hasattr(user, 'dict'):
            user_data = user.dict()
        else:
            user_data = dict(user)
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, user_data, ttl=300)
        
        return user_data
    
    async def _create_session(self, user: UserTable, refresh_token: str) -> SessionInfo:
        """Create a new user session"""

        session_id = str(uuid4())
        now = datetime.utcnow()
        expiry = now + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)
        session_info = SessionInfo(
            session_id=session_id,
            user_id=str(user.id),  # Convert UUID to string
            tenant_id=str(user.tenant_id),  # Convert UUID to string
            ip_address="0.0.0.0",  # Should be passed from request context
            user_agent="Unknown",   # Should be passed from request context
            created_at=now.isoformat(),
            last_activity=now.isoformat(),
            expires_at=expiry.isoformat()
        )
        
        await self._persist_session(session_info, refresh_token)

        logger.info("Session created", session_id=session_info.session_id, user_id=str(user.id))

        return session_info
    
    async def _update_session_tokens(self, user_id: str, new_refresh_token: str):
        """Update session with new refresh token"""
        # Session token update - implement if needed
        pass

    async def _get_sessions_from_cache(self, user_id: str) -> List[SessionInfo]:
        cache_key = build_session_cache_key(user_id)
        try:
            raw_sessions = await self.cache.get(cache_key)
        except Exception as ex:
            logger.warning("Failed to get sessions from cache", user_id=user_id, error=str(ex))
            return []

        return filter_expired_sessions(deserialize_sessions(raw_sessions))

    async def _cache_sessions(self, user_id: str, sessions: Sequence[SessionInfo]) -> None:
        cache_key = build_session_cache_key(user_id)
        sessions = filter_expired_sessions(sort_sessions_by_last_activity(list(sessions)))

        if not sessions:
            try:
                await self.cache.delete(cache_key)
            except Exception as ex:
                logger.warning("Failed to delete sessions cache", user_id=user_id, error=str(ex))
            return

        ttl_candidates = [calculate_session_ttl_seconds(session) for session in sessions]
        cache_ttl = max(ttl_candidates) if ttl_candidates else self.settings.CACHE_TTL

        try:
            await self.cache.set(cache_key, serialize_sessions(sessions), ttl=cache_ttl)
        except Exception as ex:
            logger.warning("Failed to set sessions cache", user_id=user_id, error=str(ex))

    async def _persist_session(self, session_info: SessionInfo, refresh_token: str) -> None:
        """Persist session information to cache."""
        cache_key = build_session_cache_key(session_info.user_id)

        try:
            existing_raw = await self.cache.get(cache_key)
            existing_sessions = deserialize_sessions(existing_raw)
        except Exception as ex:
            logger.warning(
                "Failed to load sessions from cache",
                user_id=session_info.user_id,
                error=str(ex)
            )
            existing_sessions = []

        token_hash = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()

        def _coerce_uuid(value: Optional[str]) -> Optional[str | UUID]:
            if value is None:
                return None
            try:
                return UUID(str(value))
            except (ValueError, TypeError, AttributeError):
                return value

        payload = {
            "id": _coerce_uuid(session_info.session_id),
            "tenant_id": _coerce_uuid(session_info.tenant_id),
            "user_id": _coerce_uuid(session_info.user_id),
            "session_token": token_hash,
            "ip_address": session_info.ip_address,
            "user_agent": session_info.user_agent,
            "expires_at": datetime.fromisoformat(session_info.expires_at),
            "last_activity": datetime.fromisoformat(session_info.last_activity),
        }

        try:
            await self.session_repo.create(payload)
        except Exception as ex:
            logger.error(
                "Failed to persist session to repository",
                user_id=session_info.user_id,
                session_id=session_info.session_id,
                error=str(ex)
            )
            return

        updated_sessions = [
            session for session in existing_sessions
            if session.session_id != session_info.session_id
        ]
        updated_sessions.append(session_info)
        await self._cache_sessions(session_info.user_id, updated_sessions)

    async def _blacklist_token(self, token: str):
        """Add token to blacklist"""
        
        payload = token_manager.verify_token(token)
        if not payload:
            return
        
        # Use token JTI or full token as blacklist key
        token_id = payload.get("jti") or token
        cache_key = f"{self.BLACKLIST_CACHE_PREFIX}{token_id}"
        
        # Cache until token expiration
        ttl = max(0, payload.get("exp", 0) - int(datetime.now(timezone.utc).timestamp()))
        
        await self.cache.set(cache_key, True, ttl=ttl)
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        
        payload = token_manager.verify_token(token)
        if not payload:
            return True
        
        token_id = payload.get("jti") or token
        cache_key = f"{self.BLACKLIST_CACHE_PREFIX}{token_id}"
        
        return await self.cache.exists(cache_key)
    
    async def _blacklist_token_family(self, user_id: str, token_family: str):
        """Blacklist all tokens in a token family"""
        # This would require tracking token families in the database
        # For now, we'll just invalidate all user sessions
        await self._revoke_all_user_sessions(user_id)
    
    async def _revoke_all_user_sessions(self, user_id: str):
        """Revoke all sessions for a user"""
        user_id_str = str(user_id)

        try:
            await self.session_repo.delete_by_user(user_id_str)
        except Exception as ex:
            logger.error(
                "Failed to delete user sessions from repository",
                user_id=user_id_str,
                error=str(ex)
            )

        try:
            await self.cache.delete(build_session_cache_key(user_id_str))
        except Exception as ex:
            logger.warning("Failed to clear session cache", user_id=user_id_str, error=str(ex))

        logger.info("All user sessions revoked", user_id=user_id_str)
    
    async def _check_login_rate_limit(self, email: str, tenant_id: str) -> bool:
        """Check if login attempts are within rate limits"""
        
        cache_key = f"{self.LOGIN_ATTEMPTS_PREFIX}{tenant_id}:{email}"
        attempts = await self.cache.get(cache_key) or 0
        
        return attempts < self.settings.MAX_LOGIN_ATTEMPTS
    
    async def _record_failed_login(self, email: str, tenant_id: str):
        """Record failed login attempt"""
        
        cache_key = f"{self.LOGIN_ATTEMPTS_PREFIX}{tenant_id}:{email}"
        attempts = await self.cache.get(cache_key) or 0
        attempts += 1
        
        # Store for window duration
        ttl = self.settings.LOGIN_ATTEMPT_WINDOW_MINUTES * 60
        await self.cache.set(cache_key, attempts, ttl=ttl)
        
        # Log failed attempt
        await self._log_security_event(
            tenant_id=tenant_id,
            action="login_failed",
            resource_type="user",
            details={
                "email": email,
                "attempt_count": attempts
            },
            risk_level="medium" if attempts > 3 else "low",
            suspicious=attempts > 3
        )
    
    async def _clear_failed_login_attempts(self, email: str, tenant_id: str):
        """Clear failed login attempts"""
        cache_key = f"{self.LOGIN_ATTEMPTS_PREFIX}{tenant_id}:{email}"
        await self.cache.delete(cache_key)
    
    async def _log_security_event(
        self,
        tenant_id: str,
        action: str,
        resource_type: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low",
        suspicious: bool = False,
        ip_address: str = "0.0.0.0",
        user_agent: str = "Unknown"
    ):
        """Log security audit event using the audit service"""
        
        try:
            # Get audit service via provider
            audit_service = await get_audit_service()
            
            # Map action to audit event type
            event_type_mapping = {
                "login_successful": AuditEventType.LOGIN_SUCCESS,
                "login_failed": AuditEventType.LOGIN_FAILED,
                "login_rate_limited": AuditEventType.RATE_LIMIT_EXCEEDED,
                "login_attempt_inactive_user": AuditEventType.ACCESS_DENIED,
                "user_registered": AuditEventType.USER_CREATED,
                "password_reset_requested": AuditEventType.PASSWORD_RESET_REQUESTED,
                "password_reset_completed": AuditEventType.PASSWORD_RESET_COMPLETED,
                "password_changed": AuditEventType.PASSWORD_CHANGED,
                "password_change_failed": AuditEventType.ACCESS_DENIED,
                "profile_updated": AuditEventType.USER_MODIFIED,
                "session_terminated": AuditEventType.LOGOUT,
                "api_key_created": AuditEventType.API_KEY_CREATED,
            }
            
            event_type = event_type_mapping.get(action, AuditEventType.SUSPICIOUS_ACTIVITY)
            
            # Log authentication events using specialized method
            if action in ["login_successful", "login_failed", "login_rate_limited", "login_attempt_inactive_user"]:
                await audit_service.log_authentication_event(
                    event_type=event_type.value,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_email=details.get("email") if details else None,
                    session_id=details.get("session_id") if details else None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=action == "login_successful",
                    failure_reason=action if action != "login_successful" else None,
                    additional_details=details,
                )
            else:
                # Log other security events using general method
                await audit_service.log_event(
                    event_type=event_type.value,
                    tenant_id=tenant_id,
                    action=action,
                    resource_type=resource_type,
                    user_id=user_id,
                    resource_id=resource_id,
                    details=details,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    risk_level=risk_level,
                    suspicious=suspicious,
                )
                
        except Exception as e:
            # Don't let audit logging failures break authentication
            # Log the error but continue execution
            logger.error(
                "Failed to log audit event",
                error=str(e),
                tenant_id=tenant_id,
                action=action,
                resource_type=resource_type,
                user_id=user_id,
            )

    def _normalize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize user data for UserTable model creation by converting UUIDs to strings"""
        normalized = user_data.copy()
        
        # Convert UUID fields to strings
        uuid_fields = ['id', 'tenant_id']
        for field in uuid_fields:
            if field in normalized and normalized[field] is not None:
                normalized[field] = str(normalized[field])
        
        # Ensure roles is a list - roles should already come from database as array
        roles = normalized.get('roles')
        if roles is None:
            normalized['roles'] = []
        elif isinstance(roles, str):
            normalized['roles'] = [roles]

        # Set default values if missing
        normalized.setdefault('permissions', [])
        if isinstance(normalized['permissions'], str):
            normalized['permissions'] = [normalized['permissions']]
        normalized.setdefault('is_superuser', False)
        normalized.setdefault('is_verified', False)
        
        return normalized

    def _build_user_object(
        self,
        user_data: Any
    ) -> UserTable | SimpleNamespace:
        """Build a user domain object, tolerating incomplete or test data."""
        if isinstance(user_data, UserTable):
            return user_data

        if user_data is None:
            return SimpleNamespace(
                id=None,
                email=None,
                full_name="",
                tenant_id=None,
                hashed_password=None,
                roles=[],
                permissions=[],
                is_active=False,
                is_superuser=False,
                is_verified=False,
            )

        if isinstance(user_data, dict):
            data_dict = user_data
        elif hasattr(user_data, "dict"):
            data_dict = user_data.dict()
        else:
            data_dict = getattr(user_data, "__dict__", {})

        normalized = self._normalize_user_data(data_dict)

        try:
            return UserTable(**normalized)
        except (ValidationError, TypeError, ValueError):
            roles = normalized.get("roles", []) or []
            permissions = normalized.get("permissions", []) or []
            return SimpleNamespace(
                id=normalized.get("id"),
                email=normalized.get("email"),
                full_name=normalized.get("full_name", ""),
                tenant_id=normalized.get("tenant_id"),
                hashed_password=normalized.get("hashed_password"),
                roles=roles,
                permissions=permissions,
                is_active=normalized.get("is_active", True),
                is_superuser=normalized.get("is_superuser", False),
                is_verified=normalized.get("is_verified", False),
            )
