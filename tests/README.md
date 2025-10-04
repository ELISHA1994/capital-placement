# Testing Strategy for Capital Placement

This document outlines the comprehensive testing strategy for the Capital Placement AI-powered CV matching platform, following hexagonal architecture principles.

## Test Organization

Our tests are organized by architectural layers to ensure proper separation of concerns and maintainable test suites:

```
tests/
├── unit/                           # Fast, isolated tests (60% of tests)
│   ├── domain/                     # Pure business logic
│   ├── application/                # Workflow orchestration  
│   └── services/                   # Service behavior (with mocks)
├── integration/                    # Component interaction tests (25% of tests)
│   ├── repositories/               # Database layer
│   ├── adapters/                   # Infrastructure adapters
│   ├── providers/                  # Provider integrations
│   └── workflows/                  # End-to-end workflows
├── fixtures/                       # Test data and utilities
├── mocks/                          # Test doubles
└── utils/                          # Test utilities
```

## Running Tests

### All Tests
```bash
pytest
```

### By Architecture Layer
```bash
# Unit tests (fast, isolated)
pytest -m unit

# Integration tests
pytest -m integration

# Domain logic only
pytest -m domain

# Infrastructure tests
pytest -m infrastructure
```

### By Feature Area
```bash
# Tenant-related tests
pytest -m tenant_isolation

# Authentication flows
pytest -m auth_flow

# Performance tests
pytest -m performance
```

### Specific Test Files
```bash
# Tenant domain rules
pytest tests/unit/domain/test_tenant_rules.py

# Authentication workflows
pytest tests/integration/workflows/test_authentication_flow.py

# Tenant isolation
pytest tests/integration/repositories/test_tenant_isolation.py
```

## Test Strategy by Layer

### 1. Domain Layer Tests (Unit Tests)

**Purpose**: Test pure business logic without external dependencies

**Files**:
- `tests/unit/domain/test_tenant_rules.py`
- `tests/unit/domain/test_auth_policies.py`
- `tests/unit/domain/test_authorization_logic.py`

**Characteristics**:
- Fast execution (< 1ms per test)
- No mocks needed for business logic
- High coverage of edge cases
- Test business rules in isolation

**Example**:
```python
def test_tenant_name_validation():
    """Test that tenant names follow business rules."""
    service = TenantService(None, None, None)  # No dependencies needed
    
    assert service._validate_tenant_name("valid-name") is True
    assert service._validate_tenant_name("Invalid Name!") is False
```

### 2. Application Layer Tests (Unit/Integration)

**Purpose**: Test workflow orchestration and service coordination

**Files**:
- `tests/integration/workflows/test_tenant_creation_flow.py`
- `tests/integration/workflows/test_authentication_flow.py`

**Characteristics**:
- Use test doubles for infrastructure
- Focus on workflow logic
- Test error handling and rollback scenarios
- Validate state transitions

**Example**:
```python
@pytest.mark.asyncio
async def test_tenant_creation_with_admin_user():
    """Test complete tenant creation workflow."""
    # Uses mock repositories but tests real workflow logic
    result = await tenant_service.create_tenant(
        name="test-company",
        admin_user_data={"email": "admin@test.com", ...}
    )
    
    assert result.name == "test-company"
    assert mock_user_repo.was_called_with("create")
```

### 3. Infrastructure Layer Tests (Integration)

**Purpose**: Test adapters and external integrations

**Files**:
- `tests/integration/repositories/test_tenant_isolation.py`
- `tests/integration/adapters/test_cache_adapter.py`

**Characteristics**:
- Use real or in-memory implementations
- Test adapter contracts
- Verify infrastructure behavior
- Test tenant isolation

**Example**:
```python
@pytest.mark.asyncio
async def test_users_isolated_by_tenant():
    """Test that tenant data isolation works correctly."""
    # Create users in different tenants
    user1 = await user_repo.create({...tenant_id: "tenant-1"...})
    user2 = await user_repo.create({...tenant_id: "tenant-2"...})
    
    # Verify isolation
    tenant1_users = await user_repo.get_by_tenant("tenant-1")
    assert len(tenant1_users) == 1
    assert user2["id"] not in [u["id"] for u in tenant1_users]
```

## Mock Strategy

### Repository Mocks
- **MockTenantRepository**: In-memory tenant storage with call logging
- **MockUserRepository**: User management with email indexing
- **MockUserSessionRepository**: Session tracking and management

### Service Mocks
- **MockCacheService**: Redis-like behavior with TTL support
- **MockNotificationService**: Email/webhook sending simulation
- **MockAnalyticsService**: Event tracking and metrics

### Test Builders
- **TenantTestBuilder**: Fluent interface for creating test tenant data
- **TenantScenarioBuilder**: Complex multi-tenant scenarios

## Tenant Isolation Testing

Our tenant isolation strategy ensures data security:

### 1. Repository Level Isolation
```python
async def test_cross_tenant_access_denied():
    """Verify users cannot access other tenants' data."""
    # Creates users in separate tenants
    # Verifies queries are tenant-scoped
    # Tests email uniqueness per tenant
```

### 2. Cache Isolation
```python
async def test_cache_tenant_keys():
    """Test cache keys include tenant ID."""
    # Verifies cache keys are tenant-prefixed
    # Tests cache invalidation is tenant-scoped
```

### 3. Authorization Isolation
```python
async def test_authorization_tenant_context():
    """Test authorization respects tenant boundaries."""
    # Verifies role checks include tenant context
    # Tests super admin cross-tenant access
```

## Performance Testing

### Load Testing
```python
@pytest.mark.performance
async def test_concurrent_tenant_creation():
    """Test system handles concurrent operations."""
    # Creates 50 tenants concurrently
    # Verifies no race conditions
    # Measures response times
```

### Database Performance
```python
@pytest.mark.performance  
async def test_large_tenant_queries():
    """Test queries perform well with large datasets."""
    # Creates 1000+ records per tenant
    # Measures query response times
    # Verifies sub-2 second target
```

## Error Handling Testing

### Transaction Rollback
```python
async def test_tenant_creation_rollback():
    """Test atomic operations roll back on failure."""
    # Simulates user creation failure
    # Verifies tenant creation is rolled back
    # Tests data consistency
```

### External Service Failures
```python
async def test_email_service_failure_handling():
    """Test graceful degradation when email fails."""
    # Simulates notification service failure
    # Verifies core operation still succeeds
    # Tests error logging and reporting
```

## CI/CD Integration

### GitHub Actions
```yaml
test-matrix:
  - name: "Unit Tests"
    command: "pytest tests/unit -v"
    timeout: 5
    
  - name: "Integration Tests"
    command: "pytest tests/integration -v"
    timeout: 30
```

### Coverage Requirements
- **Minimum coverage**: 80%
- **Domain logic**: 95%+
- **Critical paths**: 100%

## Test Data Management

### Fixtures
- **sample_tenant**: Basic tenant for simple tests
- **multi_tenant_scenario**: Complex scenarios with relationships
- **enterprise_tenant**: High-tier tenant with all features

### Builders
- **TenantTestBuilder**: Fluent API for tenant creation
- **UserTestBuilder**: User creation with role management
- **ScenarioBuilder**: Multi-entity test scenarios

## Best Practices

### 1. Test Naming
- Use descriptive names that explain the scenario
- Include expected outcome in the name
- Group related tests in classes

### 2. Test Structure
- **Arrange**: Set up test data and mocks
- **Act**: Execute the operation being tested
- **Assert**: Verify expected outcomes

### 3. Mock Usage
- Mock external dependencies, not business logic
- Use real objects for domain logic testing
- Verify mock interactions when relevant

### 4. Async Testing
- Use `pytest.mark.asyncio` for async tests
- Properly handle async context managers
- Test concurrent scenarios when relevant

### 5. Error Testing
- Test both success and failure paths
- Verify error messages and types
- Test edge cases and boundary conditions

## Continuous Improvement

### Metrics Tracking
- Test execution time trends
- Coverage percentage over time
- Flaky test identification

### Regular Reviews
- Monthly test strategy reviews
- Performance regression analysis
- Coverage gap identification

This testing strategy ensures our hexagonal architecture maintains proper separation of concerns while providing comprehensive coverage of all business scenarios and edge cases.