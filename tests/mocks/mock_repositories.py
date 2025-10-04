"""
Mock repository implementations for testing.

These mocks implement the repository interfaces and maintain
test data in memory while tracking method calls for verification.
"""

import asyncio
from typing import Dict, List, Optional, Any
from uuid import uuid4
from datetime import datetime, timedelta, timezone


class MockTenantRepository:
    """Mock tenant repository for testing."""
    
    def __init__(self):
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self.call_log: List[tuple] = []
        self.should_fail_on_create = False
        self.should_fail_on_update = False
        self.slug_availability: Dict[str, bool] = {}
    
    async def create(self, tenant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a tenant."""
        self.call_log.append(("create", tenant_data.copy()))
        
        if self.should_fail_on_create:
            raise Exception("Mock create failure")
        
        # Generate ID if not provided
        if "id" not in tenant_data:
            tenant_data["id"] = str(uuid4())
        
        # Set timestamps
        now = datetime.now(timezone.utc).isoformat()
        tenant_data.setdefault("created_at", now)
        tenant_data.setdefault("updated_at", now)
        
        # Store tenant
        tenant_id = tenant_data["id"]
        self.tenants[tenant_id] = tenant_data.copy()
        
        return tenant_data
    
    async def get(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant by ID."""
        self.call_log.append(("get", tenant_id))
        return self.tenants.get(tenant_id)
    
    async def update(self, tenant_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update tenant."""
        self.call_log.append(("update", tenant_id, updates.copy()))
        
        if self.should_fail_on_update:
            raise Exception("Mock update failure")
        
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Apply updates
        self.tenants[tenant_id].update(updates)
        self.tenants[tenant_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        return self.tenants[tenant_id].copy()
    
    async def delete(self, tenant_id: str) -> bool:
        """Delete tenant."""
        self.call_log.append(("delete", tenant_id))
        
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            return True
        return False
    
    async def list_all(self) -> List[Dict[str, Any]]:
        """List all tenants."""
        self.call_log.append(("list_all",))
        return list(self.tenants.values())
    
    async def find_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find tenants by criteria."""
        self.call_log.append(("find_by_criteria", criteria))
        
        results = []
        for tenant in self.tenants.values():
            match = True
            for key, value in criteria.items():
                if tenant.get(key) != value:
                    match = False
                    break
            if match:
                results.append(tenant.copy())
        
        return results
    
    async def check_slug_availability(self, slug: str) -> bool:
        """Check if slug is available."""
        self.call_log.append(("check_slug_availability", slug))
        
        # Check manual overrides first
        if slug in self.slug_availability:
            return self.slug_availability[slug]
        
        # Check if any tenant has this slug
        for tenant in self.tenants.values():
            if tenant.get("slug") == slug or tenant.get("name") == slug:
                return False
        
        return True
    
    async def get_active_tenants(self) -> List[Dict[str, Any]]:
        """Get all active tenants."""
        self.call_log.append(("get_active_tenants",))
        
        return [
            tenant for tenant in self.tenants.values()
            if tenant.get("is_active", True) and not tenant.get("is_suspended", False)
        ]
    
    # Test helper methods
    
    def set_slug_availability(self, slug: str, available: bool):
        """Set slug availability for testing."""
        self.slug_availability[slug] = available
    
    def was_called_with(self, method: str, *expected_args) -> bool:
        """Check if method was called with specific arguments."""
        expected_call = (method,) + expected_args
        return expected_call in self.call_log
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def clear_call_log(self):
        """Clear call log for testing."""
        self.call_log.clear()
    
    def add_test_tenant(self, tenant_data: Dict[str, Any]) -> str:
        """Add a test tenant directly (bypass create)."""
        tenant_id = tenant_data.get("id", str(uuid4()))
        tenant_data["id"] = tenant_id
        self.tenants[tenant_id] = tenant_data
        return tenant_id


class MockUserRepository:
    """Mock user repository for testing."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.call_log: List[tuple] = []
        self.should_fail_on_create = False
        self.should_fail_on_update = False
        self.email_lookup: Dict[str, str] = {}  # email -> user_id
    
    async def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a user."""
        self.call_log.append(("create", user_data.copy()))
        
        if self.should_fail_on_create:
            raise Exception("Mock user create failure")
        
        # Generate ID if not provided
        if "id" not in user_data:
            user_data["id"] = str(uuid4())
        
        # Set timestamps
        now = datetime.now(timezone.utc).isoformat()
        user_data.setdefault("created_at", now)
        user_data.setdefault("updated_at", now)
        
        # Store user
        user_id = user_data["id"]
        self.users[user_id] = user_data.copy()
        
        # Index by email for lookup
        email = user_data.get("email")
        if email:
            self.email_lookup[email] = user_id
        
        return user_data
    
    async def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        self.call_log.append(("get_by_id", user_id))
        return self.users.get(user_id)
    
    async def get_by_email(self, email: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        self.call_log.append(("get_by_email", email, tenant_id))
        
        user_id = self.email_lookup.get(email)
        if not user_id:
            return None
        
        user = self.users.get(user_id)
        if not user:
            return None
        
        # Filter by tenant if specified
        if tenant_id and user.get("tenant_id") != tenant_id:
            return None
        
        return user.copy()
    
    async def update(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user."""
        self.call_log.append(("update", user_id, updates.copy()))
        
        if self.should_fail_on_update:
            raise Exception("Mock user update failure")
        
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        # Update email index if email changed
        old_email = self.users[user_id].get("email")
        new_email = updates.get("email")
        if new_email and new_email != old_email:
            if old_email:
                self.email_lookup.pop(old_email, None)
            self.email_lookup[new_email] = user_id
        
        # Apply updates
        self.users[user_id].update(updates)
        self.users[user_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        return self.users[user_id].copy()
    
    async def delete(self, user_id: str) -> bool:
        """Delete user."""
        self.call_log.append(("delete", user_id))
        
        if user_id not in self.users:
            return False
        
        # Remove from email index
        email = self.users[user_id].get("email")
        if email:
            self.email_lookup.pop(email, None)
        
        del self.users[user_id]
        return True
    
    async def get_by_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all users in a tenant."""
        self.call_log.append(("get_by_tenant", tenant_id))
        
        return [
            user.copy() for user in self.users.values()
            if user.get("tenant_id") == tenant_id
        ]
    
    async def find_by_criteria(self, criteria: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find users by criteria."""
        self.call_log.append(("find_by_criteria", criteria, limit))
        
        results = []
        for user in self.users.values():
            match = True
            for key, value in criteria.items():
                if user.get(key) != value:
                    match = False
                    break
            if match:
                results.append(user.copy())
        
        if limit:
            results = results[:limit]
        
        return results
    
    async def update_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        self.call_log.append(("update_last_login", user_id))
        
        if user_id in self.users:
            self.users[user_id]["last_login_at"] = datetime.now(timezone.utc).isoformat()
            self.users[user_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    # Test helper methods
    
    def add_test_user(self, user_data: Dict[str, Any]) -> str:
        """Add a test user directly (bypass create)."""
        user_id = user_data.get("id", str(uuid4()))
        user_data["id"] = user_id
        self.users[user_id] = user_data
        
        # Index by email
        email = user_data.get("email")
        if email:
            self.email_lookup[email] = user_id
        
        return user_id
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def was_called_with(self, method: str, *expected_args) -> bool:
        """Check if method was called with specific arguments."""
        expected_call = (method,) + expected_args
        return expected_call in self.call_log
    
    def clear_call_log(self):
        """Clear call log for testing."""
        self.call_log.clear()


class MockUserSessionRepository:
    """Mock user session repository for testing."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.call_log: List[tuple] = []
        self.should_fail_on_create = False
    
    async def create(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a session."""
        self.call_log.append(("create", session_data.copy()))
        
        if self.should_fail_on_create:
            raise Exception("Mock session create failure")
        
        # Generate ID if not provided
        if "id" not in session_data:
            session_data["id"] = str(uuid4())
        
        session_id = session_data["id"]
        self.sessions[session_id] = session_data.copy()
        
        return session_data
    
    async def find_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Find session by ID."""
        self.call_log.append(("find_by_id", session_id))
        return self.sessions.get(session_id)
    
    async def list_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List sessions for a user."""
        self.call_log.append(("list_by_user", user_id))
        
        return [
            session.copy() for session in self.sessions.values()
            if str(session.get("user_id")) == str(user_id)
        ]
    
    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        self.call_log.append(("delete", session_id))
        
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def delete_by_user(self, user_id: str) -> int:
        """Delete all sessions for a user."""
        self.call_log.append(("delete_by_user", user_id))
        
        deleted_count = 0
        sessions_to_delete = [
            session_id for session_id, session in self.sessions.items()
            if str(session.get("user_id")) == str(user_id)
        ]
        
        for session_id in sessions_to_delete:
            del self.sessions[session_id]
            deleted_count += 1
        
        return deleted_count
    
    # Test helper methods
    
    def add_test_session(self, session_data: Dict[str, Any]) -> str:
        """Add a test session directly."""
        session_id = session_data.get("id", str(uuid4()))
        session_data["id"] = session_id
        self.sessions[session_id] = session_data
        return session_id
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def clear_call_log(self):
        """Clear call log for testing."""
        self.call_log.clear()


class MockAPIKeyRepository:
    """Mock API key repository for testing."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.call_log: List[tuple] = []
        self.key_hash_lookup: Dict[str, str] = {}  # key_hash -> api_key_id
    
    async def create(self, api_key_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an API key."""
        self.call_log.append(("create", api_key_data.copy()))
        
        # Generate ID if not provided
        if "id" not in api_key_data:
            api_key_data["id"] = str(uuid4())
        
        # Set timestamps
        now = datetime.now(timezone.utc).isoformat()
        api_key_data.setdefault("created_at", now)
        api_key_data.setdefault("updated_at", now)
        
        # Store API key
        key_id = api_key_data["id"]
        self.api_keys[key_id] = api_key_data.copy()
        
        # Index by key hash for lookup
        key_hash = api_key_data.get("key_hash")
        if key_hash:
            self.key_hash_lookup[key_hash] = key_id
        
        return api_key_data
    
    async def get_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get API key by hash."""
        self.call_log.append(("get_by_hash", key_hash))
        
        key_id = self.key_hash_lookup.get(key_hash)
        if key_id:
            return self.api_keys.get(key_id)
        return None
    
    async def list_by_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """List API keys for a tenant."""
        self.call_log.append(("list_by_tenant", tenant_id))
        
        return [
            key.copy() for key in self.api_keys.values()
            if key.get("tenant_id") == tenant_id
        ]
    
    async def update_usage(self, key_id: str) -> bool:
        """Update API key usage."""
        self.call_log.append(("update_usage", key_id))
        
        if key_id in self.api_keys:
            self.api_keys[key_id]["usage_count"] = self.api_keys[key_id].get("usage_count", 0) + 1
            self.api_keys[key_id]["last_used_at"] = datetime.now(timezone.utc).isoformat()
            return True
        return False
    
    # Test helper methods
    
    def add_test_api_key(self, api_key_data: Dict[str, Any]) -> str:
        """Add a test API key directly."""
        key_id = api_key_data.get("id", str(uuid4()))
        api_key_data["id"] = key_id
        self.api_keys[key_id] = api_key_data
        
        # Index by hash
        key_hash = api_key_data.get("key_hash")
        if key_hash:
            self.key_hash_lookup[key_hash] = key_id
        
        return key_id
    
    def clear_call_log(self):
        """Clear call log for testing."""
        self.call_log.clear()