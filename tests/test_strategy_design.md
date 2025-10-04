# Hexagonal Architecture Test Strategy for Tenant & Authentication Services

## Overview

This document outlines a comprehensive testing strategy for the tenant and authentication services following hexagonal architecture principles. The strategy focuses on testing business logic in isolation while properly testing infrastructure adapters.

## Architecture Analysis

### Current State
- **Domain Interfaces**: Well-defined in `app/domain/interfaces.py`
- **Services**: Mixed business logic and infrastructure concerns
- **Providers**: Good dependency injection pattern
- **Missing**: Dedicated application services for orchestration

### Testing Challenges
1. Services tightly coupled to infrastructure
2. Complex tenant creation workflows with transactions
3. Authentication flows with external dependencies
4. Multi-tenant data isolation requirements

## Testing Strategy by Layer

### 1. Domain Layer Tests (60% of tests)

**Purpose**: Test pure business logic without external dependencies

**What to Test**:
- Tenant name validation rules
- Password security policies  
- Role-based authorization logic
- Permission inheritance rules
- Business rule validation

**Characteristics**:
- Fast execution (< 1ms per test)
- No mocks needed
- Pure functions
- High coverage of edge cases

### 2. Application Layer Tests (25% of tests)

**Purpose**: Test workflow orchestration and service coordination

**What to Test**:
- Multi-step tenant creation workflows
- Authentication flows end-to-end
- Error handling and rollback scenarios
- Cross-service coordination

**Characteristics**:
- Use test doubles for infrastructure
- Focus on workflow logic
- Test error propagation
- Validate state transitions

### 3. Infrastructure Layer Tests (15% of tests)

**Purpose**: Test adapters and external integrations

**What to Test**:
- Repository implementations
- Cache behavior
- Provider pattern functionality
- Database integration
- Notification service integration

**Characteristics**:
- Use real or in-memory implementations
- Test adapter contracts
- Verify infrastructure behavior
- Performance characteristics

## Test Organization Structure

```
tests/
├── unit/                           # Fast, isolated tests
│   ├── domain/                     # Pure business logic
│   │   ├── test_tenant_rules.py
│   │   ├── test_auth_policies.py
│   │   └── test_authorization_logic.py
│   ├── application/                # Workflow orchestration
│   │   ├── test_tenant_workflows.py
│   │   ├── test_auth_workflows.py
│   │   └── test_error_handling.py
│   └── services/                   # Service behavior (with mocks)
│       ├── test_tenant_service.py
│       ├── test_auth_service.py
│       └── test_authorization_service.py
│
├── integration/                    # Component interaction tests
│   ├── repositories/               # Database layer
│   │   ├── test_tenant_repository.py
│   │   ├── test_user_repository.py
│   │   └── test_tenant_isolation.py
│   ├── adapters/                   # Infrastructure adapters
│   │   ├── test_cache_adapter.py
│   │   ├── test_notification_adapter.py
│   │   └── test_database_adapter.py
│   ├── providers/                  # Provider integrations
│   │   ├── test_auth_provider.py
│   │   ├── test_cache_provider.py
│   │   └── test_provider_lifecycle.py
│   └── workflows/                  # End-to-end workflows
│       ├── test_tenant_creation_flow.py
│       ├── test_authentication_flow.py
│       └── test_multi_tenant_isolation.py
│
├── fixtures/                       # Test data and utilities
│   ├── auth_fixtures.py
│   ├── tenant_fixtures.py
│   ├── database_fixtures.py
│   └── test_builders.py
│
├── mocks/                          # Test doubles
│   ├── mock_repositories.py
│   ├── mock_services.py
│   ├── mock_providers.py
│   └── mock_external_services.py
│
└── utils/                          # Test utilities
    ├── test_database.py
    ├── assertion_helpers.py
    └── tenant_test_utils.py
```

## Test Data Strategy

### Test Builders Pattern
```python
class TenantTestBuilder:
    def __init__(self):
        self.tenant_data = {
            "name": "test-tenant",
            "display_name": "Test Tenant",
            "primary_contact_email": "admin@test.com",
            "subscription_tier": SubscriptionTier.FREE
        }
    
    def with_name(self, name: str):
        self.tenant_data["name"] = name
        return self
    
    def with_admin_user(self, email: str, password: str):
        self.tenant_data["admin_user"] = {
            "email": email,
            "password": password,
            "full_name": "Test Admin"
        }
        return self
    
    def build(self) -> dict:
        return self.tenant_data.copy()
```

### Fixture Strategy
```python
@pytest.fixture
def sample_tenant():
    return TenantTestBuilder().build()

@pytest.fixture  
def tenant_with_admin():
    return (TenantTestBuilder()
            .with_admin_user("admin@test.com", "SecurePassword123!")
            .build())

@pytest.fixture
def multiple_tenants():
    return [
        TenantTestBuilder().with_name(f"tenant-{i}").build()
        for i in range(3)
    ]
```

## Mock Strategy

### 1. Repository Mocks
```python
class MockTenantRepository:
    def __init__(self):
        self.tenants = {}
        self.call_log = []
    
    async def create(self, tenant_data: dict) -> dict:
        self.call_log.append(("create", tenant_data))
        tenant_id = str(uuid4())
        tenant_data["id"] = tenant_id
        self.tenants[tenant_id] = tenant_data
        return tenant_data
    
    async def get(self, tenant_id: str) -> Optional[dict]:
        self.call_log.append(("get", tenant_id))
        return self.tenants.get(tenant_id)
    
    def was_called_with(self, method: str, expected_args):
        return (method, expected_args) in self.call_log
```

### 2. Service Interface Mocks
```python
class MockCacheService(ICacheService):
    def __init__(self):
        self.cache = {}
        self.operations = []
    
    async def get(self, key: str) -> Optional[Any]:
        self.operations.append(("get", key))
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        self.operations.append(("set", key, value, ttl))
        self.cache[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        self.operations.append(("delete", key))
        return self.cache.pop(key, None) is not None
```

### 3. External Service Mocks
```python
class MockNotificationService(INotificationService):
    def __init__(self):
        self.sent_emails = []
        self.should_fail = False
    
    async def send_email(self, to: str, subject: str, body: str, is_html: bool = False) -> bool:
        if self.should_fail:
            return False
        
        self.sent_emails.append({
            "to": to,
            "subject": subject, 
            "body": body,
            "is_html": is_html
        })
        return True
    
    def verify_email_sent(self, to: str, subject_contains: str = None):
        for email in self.sent_emails:
            if email["to"] == to:
                if subject_contains is None:
                    return True
                if subject_contains in email["subject"]:
                    return True
        return False
```

## Tenant Isolation Testing Strategy

### 1. Database Isolation Tests
```python
async def test_tenant_data_isolation():
    # Create two tenants with users
    tenant1 = await create_test_tenant("tenant-1")
    tenant2 = await create_test_tenant("tenant-2")
    
    user1 = await create_test_user(tenant1["id"], "user1@tenant1.com")
    user2 = await create_test_user(tenant2["id"], "user2@tenant2.com")
    
    # Verify tenant 1 cannot access tenant 2 data
    tenant1_users = await user_repo.get_by_tenant(tenant1["id"])
    tenant2_users = await user_repo.get_by_tenant(tenant2["id"])
    
    assert len(tenant1_users) == 1
    assert len(tenant2_users) == 1
    assert user1["id"] in [u["id"] for u in tenant1_users]
    assert user1["id"] not in [u["id"] for u in tenant2_users]
```

### 2. Cache Isolation Tests  
```python
async def test_cache_isolation():
    tenant1_id = "tenant-1"
    tenant2_id = "tenant-2"
    
    # Set data for both tenants
    await cache_service.set(f"tenant:{tenant1_id}:data", "tenant1-data")
    await cache_service.set(f"tenant:{tenant2_id}:data", "tenant2-data")
    
    # Verify isolation
    tenant1_data = await cache_service.get(f"tenant:{tenant1_id}:data")
    tenant2_data = await cache_service.get(f"tenant:{tenant2_id}:data")
    
    assert tenant1_data == "tenant1-data"
    assert tenant2_data == "tenant2-data"
    assert tenant1_data != tenant2_data
```

### 3. Authorization Isolation Tests
```python
async def test_cross_tenant_authorization_denied():
    # Create users in different tenants
    user1 = create_current_user(tenant_id="tenant-1", roles=["admin"])
    user2_data = create_test_user_data(tenant_id="tenant-2")
    
    # Verify user1 cannot access tenant-2 resources
    auth_service = AuthorizationService(mock_repos...)
    
    has_access = await auth_service.check_tenant_access(
        user_tenant_id="tenant-1",
        resource_tenant_id="tenant-2",
        user_roles=user1.roles
    )
    
    assert has_access is False
```

## Performance Testing Strategy

### 1. Load Testing for Tenant Operations
```python
@pytest.mark.performance
async def test_concurrent_tenant_creation():
    import asyncio
    
    async def create_tenant_task(i):
        return await tenant_service.create_tenant(
            name=f"load-test-{i}",
            display_name=f"Load Test Tenant {i}",
            primary_contact_email=f"admin{i}@loadtest.com"
        )
    
    # Test concurrent creation
    tasks = [create_tenant_task(i) for i in range(50)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify all succeeded
    successful_creations = [r for r in results if not isinstance(r, Exception)]
    assert len(successful_creations) == 50
```

### 2. Database Performance Tests
```python
@pytest.mark.performance
async def test_tenant_query_performance():
    # Create many tenants
    for i in range(1000):
        await create_test_tenant(f"perf-tenant-{i}")
    
    # Time tenant listing
    start_time = time.time()
    tenants = await tenant_service.list_tenants(limit=100)
    duration = time.time() - start_time
    
    assert duration < 0.5  # Should complete in under 500ms
    assert len(tenants) == 100
```

## Error Handling Test Strategy

### 1. Transactional Rollback Tests
```python
async def test_tenant_creation_rollback_on_admin_failure():
    mock_user_repo = MockUserRepository()
    mock_user_repo.should_fail_on_create = True
    
    tenant_service = TenantService(
        tenant_repository=MockTenantRepository(),
        user_repository=mock_user_repo,
        cache_manager=MockCacheService()
    )
    
    with pytest.raises(TransactionError):
        await tenant_service.create_tenant(
            name="test-tenant",
            display_name="Test Tenant", 
            primary_contact_email="admin@test.com",
            admin_user_data={
                "email": "admin@test.com",
                "password": "SecurePassword123!",
                "full_name": "Test Admin"
            }
        )
    
    # Verify no tenant was created
    tenants = await tenant_service.list_tenants()
    assert len(tenants) == 0
```

### 2. External Service Failure Tests
```python
async def test_password_reset_email_failure_handling():
    mock_notification = MockNotificationService()
    mock_notification.should_fail = True
    
    auth_service = AuthenticationService(
        user_repository=mock_user_repo,
        tenant_repository=mock_tenant_repo,
        cache_manager=mock_cache,
        notification_service=mock_notification
    )
    
    result = await auth_service.request_password_reset(
        PasswordResetRequest(email="user@test.com")
    )
    
    # Should still generate token but mark email as failed
    assert result is not None
    assert result["email_sent"] is False
    assert "token" in result
```

## Integration Test Examples

### 1. Complete Authentication Flow
```python
async def test_complete_authentication_flow():
    # Setup: Create tenant with admin user
    tenant_data = TenantTestBuilder().with_admin_user(
        "admin@test.com", "SecurePassword123!"
    ).build()
    
    created_tenant = await tenant_service.create_tenant(**tenant_data)
    
    # Test: Authenticate the admin user
    login_request = UserLogin(
        email="admin@test.com",
        password="SecurePassword123!",
        tenant_id=created_tenant.id
    )
    
    auth_result = await auth_service.authenticate(login_request)
    
    # Verify: Successful authentication
    assert auth_result.success is True
    assert auth_result.user is not None
    assert auth_result.tokens is not None
    assert auth_result.user.email == "admin@test.com"
    assert "admin" in auth_result.user.roles
```

### 2. Multi-Tenant User Management Flow
```python
async def test_multi_tenant_user_management():
    # Create two tenants
    tenant1 = await create_test_tenant("tenant-1")
    tenant2 = await create_test_tenant("tenant-2")
    
    # Create admin users for each tenant
    admin1 = await tenant_service.create_tenant_user(
        tenant_id=tenant1.id,
        email="admin1@test.com", 
        password="SecurePassword123!",
        full_name="Admin One",
        roles=["admin"]
    )
    
    admin2 = await tenant_service.create_tenant_user(
        tenant_id=tenant2.id,
        email="admin2@test.com",
        password="SecurePassword123!", 
        full_name="Admin Two",
        roles=["admin"]
    )
    
    # Verify tenant isolation
    tenant1_users = await tenant_service.get_tenant_users(tenant1.id)
    tenant2_users = await tenant_service.get_tenant_users(tenant2.id)
    
    assert len(tenant1_users) == 1
    assert len(tenant2_users) == 1
    assert admin1.user_id in [u.user_id for u in tenant1_users]
    assert admin1.user_id not in [u.user_id for u in tenant2_users]
```

## Test Database Strategy

### 1. Test Database Setup
```python
@pytest.fixture(scope="session")
async def test_database():
    """Create isolated test database."""
    test_db_name = f"test_db_{uuid4().hex[:8]}"
    
    # Create test database
    await create_test_database(test_db_name)
    
    try:
        # Run migrations
        await run_migrations(test_db_name)
        yield test_db_name
    finally:
        # Cleanup
        await drop_test_database(test_db_name)

@pytest.fixture
async def clean_database(test_database):
    """Ensure clean state for each test."""
    await truncate_all_tables(test_database)
    yield test_database
    await truncate_all_tables(test_database)
```

### 2. Transaction Isolation
```python
@pytest.fixture
async def db_transaction():
    """Provide database transaction that rolls back after test."""
    async with get_db_connection() as conn:
        tx = await conn.begin()
        try:
            yield conn
        finally:
            await tx.rollback()
```

## Continuous Integration Strategy

### 1. Test Categories for CI
```yaml
# .github/workflows/tests.yml
test-matrix:
  - name: "Unit Tests"
    command: "pytest tests/unit -v --tb=short"
    timeout: 5
    
  - name: "Integration Tests" 
    command: "pytest tests/integration -v --tb=short"
    timeout: 30
    
  - name: "Performance Tests"
    command: "pytest tests -m performance -v"
    timeout: 60
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

### 2. Test Reporting
```python
# pytest.ini
[tool:pytest]
addopts = 
    --strict-markers
    --strict-config
    --cov=app
    --cov-report=html
    --cov-report=xml
    --junit-xml=test-results.xml
    --tb=short

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, real dependencies)  
    performance: Performance tests (slow, load testing)
    slow: Slow tests (skip in CI by default)
```

## Summary

This testing strategy provides:

1. **Clear separation** of concerns following hexagonal architecture
2. **Comprehensive coverage** of business logic through unit tests
3. **Proper isolation** testing for multi-tenant requirements
4. **Realistic integration** testing with proper mocks
5. **Performance validation** for critical tenant operations
6. **Robust error handling** test scenarios
7. **Maintainable test structure** with builders and fixtures

The strategy prioritizes fast unit tests for business logic while ensuring critical integrations are properly tested with minimal external dependencies.