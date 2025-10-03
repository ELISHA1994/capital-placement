"""PostgreSQL implementation of IUserRepository using UserMapper."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from app.domain.entities.user import User, UserRole, UserStatus
from app.domain.repositories.user_repository import IUserRepository
from app.domain.value_objects import UserId, TenantId, EmailAddress
from app.infrastructure.persistence.mappers.user_mapper import UserMapper
from app.models.auth import UserTable
from app.infrastructure.providers.postgres_provider import get_postgres_adapter


class PostgresUserRepository(IUserRepository):
    """PostgreSQL adapter implementation of IUserRepository."""

    def __init__(self):
        self._adapter = None

    async def _get_adapter(self):
        """Get database adapter (lazy initialization)."""
        if self._adapter is None:
            self._adapter = await get_postgres_adapter()
        return self._adapter

    async def save(self, user: User) -> User:
        """Save user to database and return updated domain entity."""
        adapter = await self._get_adapter()
        
        try:
            # Check if user exists
            existing = await self.find_by_id(user.id)
            
            if existing:
                # Update existing user
                user_table = UserMapper.to_persistence(user)
                
                await adapter.execute(
                    """
                    UPDATE users 
                    SET tenant_id = $2, email = $3, hashed_password = $4, 
                        first_name = $5, last_name = $6, full_name = $7,
                        is_active = $8, is_verified = $9, is_superuser = $10,
                        roles = $11, permissions = $12, last_login_at = $13,
                        failed_login_attempts = $14, locked_until = $15,
                        settings = $16, ai_preferences = $17, updated_at = $18
                    WHERE id = $1
                    """,
                    user_table.id, user_table.tenant_id, user_table.email,
                    user_table.hashed_password, user_table.first_name,
                    user_table.last_name, user_table.full_name, user_table.is_active,
                    user_table.is_verified, user_table.is_superuser, user_table.roles,
                    user_table.permissions, user_table.last_login_at,
                    user_table.failed_login_attempts, user_table.locked_until,
                    user_table.settings, user_table.ai_preferences, user_table.updated_at
                )
            else:
                # Insert new user
                user_table = UserMapper.to_persistence(user)
                
                await adapter.execute(
                    """
                    INSERT INTO users (
                        id, tenant_id, email, hashed_password, first_name, last_name,
                        full_name, is_active, is_verified, is_superuser, roles,
                        permissions, last_login_at, failed_login_attempts, locked_until,
                        settings, ai_preferences, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                        $15, $16, $17, $18, $19
                    )
                    """,
                    user_table.id, user_table.tenant_id, user_table.email,
                    user_table.hashed_password, user_table.first_name,
                    user_table.last_name, user_table.full_name, user_table.is_active,
                    user_table.is_verified, user_table.is_superuser, user_table.roles,
                    user_table.permissions, user_table.last_login_at,
                    user_table.failed_login_attempts, user_table.locked_until,
                    user_table.settings, user_table.ai_preferences,
                    user_table.created_at, user_table.updated_at
                )
            
            return user
            
        except Exception as e:
            raise Exception(f"Failed to save user: {str(e)}")

    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        """Find user by ID."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one(
                "SELECT * FROM users WHERE id = $1",
                user_id.value
            )
            
            if not record:
                return None
            
            user_table = UserTable(**dict(record))
            return UserMapper.to_domain(user_table)
            
        except Exception as e:
            raise Exception(f"Failed to find user by ID: {str(e)}")

    async def find_by_email(self, email: EmailAddress) -> Optional[User]:
        """Find user by email address."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one(
                "SELECT * FROM users WHERE email = $1",
                str(email)
            )
            
            if not record:
                return None
            
            user_table = UserTable(**dict(record))
            return UserMapper.to_domain(user_table)
            
        except Exception as e:
            raise Exception(f"Failed to find user by email: {str(e)}")

    async def find_by_tenant_id(self, tenant_id: TenantId, limit: int = 100, offset: int = 0) -> List[User]:
        """Find users by tenant ID with pagination."""
        adapter = await self._get_adapter()
        
        try:
            records = await adapter.fetch_all(
                "SELECT * FROM users WHERE tenant_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                tenant_id.value, limit, offset
            )
            
            users = []
            for record in records:
                user_table = UserTable(**dict(record))
                users.append(UserMapper.to_domain(user_table))
            
            return users
            
        except Exception as e:
            raise Exception(f"Failed to find users by tenant ID: {str(e)}")

    async def find_by_role(self, tenant_id: TenantId, role: UserRole) -> List[User]:
        """Find users by role within a tenant."""
        adapter = await self._get_adapter()
        
        try:
            records = await adapter.fetch_all(
                "SELECT * FROM users WHERE tenant_id = $1 AND roles @> $2 ORDER BY created_at DESC",
                tenant_id.value, [role.value]
            )
            
            users = []
            for record in records:
                user_table = UserTable(**dict(record))
                users.append(UserMapper.to_domain(user_table))
            
            return users
            
        except Exception as e:
            raise Exception(f"Failed to find users by role: {str(e)}")

    async def find_by_status(self, tenant_id: TenantId, status: UserStatus) -> List[User]:
        """Find users by status within a tenant."""
        adapter = await self._get_adapter()
        
        try:
            # Map status to database fields
            where_conditions = ["tenant_id = $1"]
            params = [tenant_id.value]
            
            if status == UserStatus.ACTIVE:
                where_conditions.append("is_active = true")
            elif status == UserStatus.INACTIVE:
                where_conditions.append("is_active = false")
            elif status == UserStatus.SUSPENDED:
                where_conditions.append("locked_until IS NOT NULL")
            elif status == UserStatus.PENDING_VERIFICATION:
                where_conditions.append("is_verified = false")
            
            where_clause = " AND ".join(where_conditions)
            
            records = await adapter.fetch_all(
                f"SELECT * FROM users WHERE {where_clause} ORDER BY created_at DESC",
                *params
            )
            
            users = []
            for record in records:
                user_table = UserTable(**dict(record))
                users.append(UserMapper.to_domain(user_table))
            
            return users
            
        except Exception as e:
            raise Exception(f"Failed to find users by status: {str(e)}")

    async def delete_by_id(self, user_id: UserId) -> bool:
        """Delete user by ID."""
        adapter = await self._get_adapter()
        
        try:
            result = await adapter.execute(
                "DELETE FROM users WHERE id = $1",
                user_id.value
            )
            
            return result and result.split()[-1] != '0'
            
        except Exception as e:
            raise Exception(f"Failed to delete user: {str(e)}")

    async def count_by_tenant_id(self, tenant_id: TenantId) -> int:
        """Count users by tenant ID."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one(
                "SELECT COUNT(*) as count FROM users WHERE tenant_id = $1",
                tenant_id.value
            )
            
            return record['count'] if record else 0
            
        except Exception as e:
            raise Exception(f"Failed to count users: {str(e)}")

    async def find_admins_by_tenant_id(self, tenant_id: TenantId) -> List[User]:
        """Find admin users within a tenant."""
        adapter = await self._get_adapter()
        
        try:
            records = await adapter.fetch_all(
                """
                SELECT * FROM users 
                WHERE tenant_id = $1 
                AND (roles @> $2 OR roles @> $3 OR is_superuser = true)
                ORDER BY created_at DESC
                """,
                tenant_id.value, [UserRole.ADMIN.value], [UserRole.SUPER_ADMIN.value]
            )
            
            users = []
            for record in records:
                user_table = UserTable(**dict(record))
                users.append(UserMapper.to_domain(user_table))
            
            return users
            
        except Exception as e:
            raise Exception(f"Failed to find admin users: {str(e)}")

    async def update_last_login(self, user_id: UserId) -> None:
        """Update user's last login timestamp."""
        adapter = await self._get_adapter()
        
        try:
            await adapter.execute(
                """
                UPDATE users 
                SET last_login_at = NOW(), 
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    updated_at = NOW()
                WHERE id = $1
                """,
                user_id.value
            )
            
        except Exception as e:
            raise Exception(f"Failed to update last login: {str(e)}")

    async def increment_failed_login(self, user_id: UserId) -> None:
        """Increment user's failed login attempts."""
        adapter = await self._get_adapter()
        
        try:
            await adapter.execute(
                """
                UPDATE users 
                SET failed_login_attempts = failed_login_attempts + 1,
                    updated_at = NOW()
                WHERE id = $1
                """,
                user_id.value
            )
            
        except Exception as e:
            raise Exception(f"Failed to increment failed login: {str(e)}")

    async def lock_user_account(self, user_id: UserId, until_timestamp) -> None:
        """Lock user account until specified timestamp."""
        adapter = await self._get_adapter()
        
        try:
            await adapter.execute(
                """
                UPDATE users 
                SET locked_until = $2,
                    updated_at = NOW()
                WHERE id = $1
                """,
                user_id.value, until_timestamp
            )
            
        except Exception as e:
            raise Exception(f"Failed to lock user account: {str(e)}")

    async def unlock_user_account(self, user_id: UserId) -> None:
        """Unlock user account."""
        adapter = await self._get_adapter()
        
        try:
            await adapter.execute(
                """
                UPDATE users 
                SET locked_until = NULL,
                    failed_login_attempts = 0,
                    updated_at = NOW()
                WHERE id = $1
                """,
                user_id.value
            )
            
        except Exception as e:
            raise Exception(f"Failed to unlock user account: {str(e)}")


__all__ = ["PostgresUserRepository"]