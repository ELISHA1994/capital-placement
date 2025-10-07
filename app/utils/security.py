"""
Security utilities for authentication and password management
"""

import secrets
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

import structlog
from app.core.config import get_settings
from app.infrastructure.persistence.models.auth_tables import TokenData, CurrentUser

logger = structlog.get_logger(__name__)


class PasswordManager:
    """Password hashing and validation utilities"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt"""
        salt = bcrypt.gensalt(rounds=self.settings.BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error("Password verification failed", error=str(e))
            return False
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password meets security requirements"""
        errors = []
        
        if len(password) < self.settings.PASSWORD_MIN_LENGTH:
            errors.append(
                f"Password length must be at least {self.settings.PASSWORD_MIN_LENGTH} characters"
            )
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


class TokenManager:
    """JWT token management utilities"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def create_access_token(
        self,
        user_id: str,
        email: str,
        tenant_id: str,
        roles: List[str],
        permissions: List[str],
        extra_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create JWT access token"""
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "sub": user_id,
            "email": email,
            "tenant_id": tenant_id,
            "roles": roles,
            "permissions": permissions,
            "exp": expires_at,
            "iat": now,
            "jti": str(uuid4()),  # Unique token ID for revocation
            "token_type": "access"
        }
        
        # Add any extra claims
        if extra_claims:
            payload.update(extra_claims)
        
        return jwt.encode(
            payload,
            self.settings.SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )
    
    def create_refresh_token(
        self,
        user_id: str,
        tenant_id: str,
        token_family: Optional[str] = None
    ) -> str:
        """Create JWT refresh token"""
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "family": token_family or str(uuid4()),  # For token rotation
            "exp": expires_at,
            "iat": now,
            "token_type": "refresh"
        }
        
        return jwt.encode(
            payload,
            self.settings.SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.settings.SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except (jwt.PyJWTError, jwt.InvalidTokenError, Exception) as e:
            logger.warning("Token validation failed", error=str(e))
            return None
    
    def extract_token_data(self, payload: Dict[str, Any]) -> Optional[TokenData]:
        """Extract structured token data from payload"""
        try:
            return TokenData(
                sub=payload["sub"],
                email=payload["email"],
                tenant_id=payload["tenant_id"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                exp=payload["exp"],
                iat=payload["iat"],
                token_type=payload.get("token_type", "access")
            )
        except KeyError as e:
            logger.error("Missing required token field", field=str(e))
            return None
    
    def token_to_current_user(self, token_data: TokenData, user_data: Dict[str, Any]) -> CurrentUser:
        """Convert token data and user data to CurrentUser"""
        return CurrentUser(
            user_id=token_data.sub,
            email=token_data.email,
            full_name=user_data.get("full_name", ""),
            tenant_id=token_data.tenant_id,
            roles=token_data.roles,
            permissions=token_data.permissions,
            is_active=user_data.get("is_active", True),
            is_superuser=user_data.get("is_superuser", False)
        )
    
    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired without raising exceptions"""
        try:
            payload = jwt.decode(
                token,
                self.settings.SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM]
            )
            return False
        except jwt.ExpiredSignatureError:
            return True
        except (jwt.PyJWTError, jwt.InvalidTokenError, Exception):
            return True  # Invalid tokens are considered expired


class APIKeyManager:
    """API key generation and validation utilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self.password_manager = PasswordManager()
    
    def generate_api_key(self) -> str:
        """Generate a cryptographically secure API key"""
        return secrets.token_urlsafe(self.settings.API_KEY_LENGTH)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return self.password_manager.hash_password(api_key)
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash"""
        return self.password_manager.verify_password(api_key, hashed_key)


class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        if not email or "@" not in email:
            return False
        
        parts = email.split("@")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return False
        
        domain = parts[1]
        if "." not in domain or domain.startswith(".") or domain.endswith("."):
            return False
        
        return True
    
    @staticmethod
    def validate_organization_slug(slug: str) -> Dict[str, Any]:
        """Validate organization slug format and availability"""
        import re
        
        if not slug:
            return {"valid": False, "error": "Organization slug is required"}
        
        # Convert to lowercase and trim
        slug = slug.lower().strip()
        
        # Length validation
        if len(slug) < 2 or len(slug) > 50:
            return {"valid": False, "error": "Organization slug must be between 2 and 50 characters"}
        
        # Format validation - alphanumeric and hyphens only, can't start/end with hyphen
        if len(slug) == 1:
            if not re.match(r'^[a-z0-9]$', slug):
                return {"valid": False, "error": "Single character slug must be alphanumeric"}
        else:
            if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', slug):
                return {"valid": False, "error": "Slug must contain only lowercase letters, numbers, and hyphens. Cannot start or end with hyphen."}
        
        # Check for consecutive hyphens
        if '--' in slug:
            return {"valid": False, "error": "Slug cannot contain consecutive hyphens"}
        
        # Reserved words check
        reserved_words = {
            'api', 'www', 'admin', 'app', 'mail', 'ftp', 'blog', 'dev', 'test', 
            'staging', 'support', 'help', 'docs', 'status', 'login', 'signup',
            'dashboard', 'settings', 'profile', 'account', 'billing', 'security',
            'webhook', 'webhooks', 'callback', 'oauth', 'sso', 'health', 'metrics',
            'assets', 'static', 'public', 'private', 'internal', 'system', 'root',
            'user', 'users', 'tenant', 'tenants', 'org', 'organization', 'organizations'
        }
        
        if slug in reserved_words:
            return {"valid": False, "error": f"'{slug}' is a reserved organization slug"}
        
        return {"valid": True, "slug": slug}
    
    @staticmethod
    def generate_slug_from_name(name: str) -> str:
        """Generate a URL-safe slug from organization name"""
        import re
        
        if not name:
            return ""
        
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', name.lower())
        slug = re.sub(r'[\s-]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        # Truncate if too long
        if len(slug) > 50:
            slug = slug[:50].rstrip('-')
        
        # Ensure minimum length
        if len(slug) < 2:
            slug = f"org-{slug}" if slug else "organization"
        
        return slug
    
    @staticmethod
    def sanitize_input(value: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if not isinstance(value, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = "".join(char for char in value if ord(char) >= 32 or char in ['\n', '\r', '\t'])
        
        # Truncate to max length
        return sanitized[:max_length]
    
    @staticmethod
    def is_safe_redirect_url(url: str, allowed_hosts: Optional[List[str]] = None) -> bool:
        """Check if URL is safe for redirect"""
        if not url:
            return False
        
        # Prevent javascript: and data: URLs
        if url.lower().startswith(('javascript:', 'data:', 'vbscript:')):
            return False
        
        # Allow relative URLs
        if url.startswith('/') and not url.startswith('//'):
            return True
        
        # Check allowed hosts if provided
        if allowed_hosts:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(url)
                return parsed.hostname in allowed_hosts
            except Exception:
                return False
        
        return True


# Global instances
password_manager = PasswordManager()
token_manager = TokenManager()
api_key_manager = APIKeyManager()
security_validator = SecurityValidator()
