"""
Bootstrap Service

Handles system initialization and first-run setup:
- Creates system tenant on first run
- Provides endpoint for creating first super admin
- Manages system initialization state
"""

import secrets
import json
from datetime import datetime
from typing import Optional, Dict, Any

import structlog
from app.core.system_constants import (
    SYSTEM_TENANT_ID, SYSTEM_TENANT_NAME, SYSTEM_TENANT_DISPLAY_NAME,
    SUPER_ADMIN_ROLE
)
# Database schema matches simpler structure
from app.models.auth import User, CurrentUser, UserCreate, UserLogin
from app.database.repositories.postgres import TenantRepository, UserRepository
from app.services.auth import AuthenticationService
from app.utils.security import password_manager
from app.core.transaction_manager import transactional

logger = structlog.get_logger(__name__)


class BootstrapService:
    """Handles system initialization and bootstrap operations"""
    
    def __init__(
        self,
        tenant_repository: TenantRepository,
        user_repository: UserRepository,
        auth_service: AuthenticationService
    ):
        self.tenant_repo = tenant_repository
        self.user_repo = user_repository
        self.auth_service = auth_service
        
    async def ensure_system_tenant(self) -> Dict[str, Any]:
        """
        Ensure system tenant exists, create if not
        
        Returns:
            Dict: The system tenant record
        """

        system_tenant = await self.tenant_repo.get(SYSTEM_TENANT_ID)
        
        if system_tenant:
            logger.info("System tenant already exists")
            return system_tenant
        
        system_tenant_data = {
            "id": SYSTEM_TENANT_ID,
            "name": SYSTEM_TENANT_NAME,
            "slug": "system", 
            "display_name": SYSTEM_TENANT_DISPLAY_NAME,
            "description": "System administration tenant",
            "subscription_tier": "enterprise",
            "is_active": True,
            "is_system_tenant": True,
            "primary_contact_email": "system@cv-platform.local",
            "data_region": "us-central",
            "timezone": "UTC",
            "locale": "en-US",
            "date_format": "YYYY-MM-DD"
        }
        
        try:
            created_tenant = await self.tenant_repo.create(system_tenant_data)
            logger.info(
                "System tenant created successfully",
                tenant_id=SYSTEM_TENANT_ID,
                tenant_name=SYSTEM_TENANT_NAME
            )
            return created_tenant
        except Exception as e:
            logger.error(f"Failed to create system tenant: {str(e)}")
            raise
    
    async def has_super_admin(self) -> bool:
        """
        Check if any super admin exists in the system
        
        Returns:
            bool: True if at least one super admin exists
        """
        try:
            users = await self.user_repo.get_by_tenant(SYSTEM_TENANT_ID)
            
            for user in users:
                if SUPER_ADMIN_ROLE in user.get("roles", []):
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking for super admin: {str(e)}")
            return False
    
    async def create_super_admin(
        self, 
        email: str,
        password: str,
        full_name: str
    ) -> Dict[str, Any]:
        """
        Create the first super admin user with proper transaction isolation
        
        Args:
            email: Admin email
            password: Admin password
            full_name: Admin full name
            
        Returns:
            Dict containing user info and access token
        """
        from app.core.transaction_manager import get_transaction_manager
        
        tx_manager = get_transaction_manager()
        
        async with tx_manager.transaction(tenant_id=str(SYSTEM_TENANT_ID), isolation_level="READ COMMITTED") as session:
            try:
                existing_users = await self.user_repo.get_by_tenant(str(SYSTEM_TENANT_ID), session=session)
                for user in existing_users:
                    if SUPER_ADMIN_ROLE in user.get("roles", []):
                        raise ValueError("Super admin already exists")
                
                system_tenant = await self.tenant_repo.get(SYSTEM_TENANT_ID, session=session)
                if not system_tenant:
                    system_tenant_data = {
                        "id": SYSTEM_TENANT_ID,
                        "name": SYSTEM_TENANT_NAME,
                        "slug": "system", 
                        "display_name": SYSTEM_TENANT_DISPLAY_NAME,
                        "description": "System administration tenant",
                        "subscription_tier": "enterprise",
                        "is_active": True,
                        "is_system_tenant": True,
                        "primary_contact_email": "system@cv-platform.local",
                        "data_region": "us-central",
                        "timezone": "UTC",
                        "locale": "en-US",
                        "date_format": "YYYY-MM-DD"
                    }
                    await self.tenant_repo.create(system_tenant_data, session=session)
                    logger.info("System tenant created successfully", tenant_id=SYSTEM_TENANT_ID)
                
                password_validation = password_manager.validate_password_strength(password)
                if not password_validation["valid"]:
                    raise ValueError(f"Password validation failed: {password_validation['errors']}")
                
                hashed_password = password_manager.hash_password(password)
                
                name_parts = full_name.split(' ', 1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                
                super_admin_data = {
                    "tenant_id": str(SYSTEM_TENANT_ID),
                    "email": email,
                    "hashed_password": hashed_password,
                    "first_name": first_name,
                    "last_name": last_name,
                    "full_name": full_name,
                    "roles": [SUPER_ADMIN_ROLE],
                    "is_active": True,
                    "is_verified": True,
                }
                
                # Create the user within the transaction
                created_user = await self.user_repo.create(super_admin_data, session=session)
                
                logger.info(
                    "Super admin created successfully",
                    email=email,
                    user_id=str(created_user.get("id"))
                )
                
                # Extract user data for use outside transaction
                user_id = str(created_user.get("id"))
                user_email = created_user.get("email")
                user_full_name = f"{created_user.get('first_name', '')} {created_user.get('last_name', '')}".strip()
                tenant_id = str(created_user.get("tenant_id"))
                
                # Prepare user info for response
                current_user = CurrentUser(
                    user_id=user_id,
                    email=user_email,
                    full_name=user_full_name,
                    tenant_id=tenant_id,
                    roles=created_user.get("roles", [SUPER_ADMIN_ROLE]),
                    permissions=[],  # Database doesn't have permissions field yet
                    is_active=created_user.get("is_active", True),
                    is_superuser=(SUPER_ADMIN_ROLE in created_user.get("roles", []))
                )

            except Exception as e:
                logger.error(
                    "Failed to create super admin in transaction",
                    error=str(e),
                    email=email
                )
                # Transaction will auto-rollback due to exception
                raise ValueError(f"Failed to create super admin: {str(e)}")
        
        return {
            "user": current_user.model_dump(),
            "message": "Super admin created successfully. Use the login endpoint to authenticate."
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system initialization status
        
        Returns:
            Dict with system status information
        """
        has_system_tenant = await self.tenant_repo.get(SYSTEM_TENANT_ID) is not None
        has_super_admin = await self.has_super_admin()
        
        all_tenants = await self.tenant_repo.list_all()
        tenant_count = len([t for t in all_tenants if t.get("id") != SYSTEM_TENANT_ID])
        
        try:
            total_users = 0
            for tenant in all_tenants:
                users = await self.user_repo.get_by_tenant(tenant.get("id"))
                total_users += len(users)
        except:
            total_users = 0
        
        return {
            "initialized": has_system_tenant and has_super_admin,
            "has_system_tenant": has_system_tenant,
            "has_super_admin": has_super_admin,
            "tenant_count": tenant_count,
            "total_users": total_users,
            "system_tenant_id": SYSTEM_TENANT_ID if has_system_tenant else None
        }
    
    async def initialize_system(self) -> bool:
        """
        Initialize the system on startup
        
        Returns:
            bool: True if initialization successful
        """
        try:
            await self.ensure_system_tenant()
            has_admin = await self.has_super_admin()
            
            if not has_admin:
                logger.warning(
                    "No super admin found. Please use /api/v1/setup/initialize to create one"
                )
            else:
                logger.info("System initialized with existing super admin")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False