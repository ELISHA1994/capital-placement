# CV Matching Platform

Intelligent CV/Resume matching system with semantic search capabilities and multi-tenant support.

## Features

- **Semantic Search**: Advanced AI-powered matching beyond keyword matching
- **Multi-tenant Architecture**: Complete data isolation between organizations  
- **Scalable Design**: Handle 2M+ profiles with sub-2 second search response
- **Cloud-Agnostic AI**: Direct OpenAI integration with local development fallbacks
- **Modern Tech Stack**: FastAPI, Python 3.12+, async/await patterns

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy environment file
cp .env.local .env

# Generate secret key and add to .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Set up database (ensure PostgreSQL is running)
# Database URL: postgresql://cv_user:cv_password@localhost:5432/cv-analytic

# Run Alembic database migrations
alembic upgrade head

# Start the application (migrations also run automatically on startup)
python -m uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for API documentation.

## System Initialization

After running the application for the first time, initialize the system with a super admin:

```bash
# Check system status
curl -X GET "http://localhost:8000/api/v1/setup/status"

# Initialize system with super admin
curl -X POST "http://localhost:8000/api/v1/setup/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@system.local",
    "password": "SuperAdmin123!",
    "full_name": "System Administrator"
  }'
```

**Default Super Admin Credentials:**
- **Email**: `admin@system.local`
- **Password**: `SuperAdmin123!`
- **Role**: `super_admin`
- **Tenant ID**: `00000000-0000-0000-0000-000000000000` (System tenant)

Use these credentials to:
- Create new tenants
- Manage system-wide settings
- Create additional admin users
- Monitor system health

## Documentation

- [Local Setup Guide](docs/LOCAL_SETUP.md) - Detailed setup instructions
- [Architecture Overview](docs/ARCHITECTURE_SOLUTION.md) - System design and patterns
- [Implementation Guide](docs/implementation-guide.md) - Detailed implementation details
- [Deployment Guide](docs/deployment-guide.md) - Cloud deployment instructions

## Development

The system automatically detects your environment and uses appropriate services:
- **Local**: OpenAI API, Memory Cache, Local Files, PostgreSQL
- **Development**: OpenAI API, Redis Cache, Local Files, PostgreSQL
- **Production**: OpenAI API, Redis Cache, Cloud Storage, PostgreSQL

Cloud-agnostic design - same codebase works with any infrastructure!

## Hexagonal Architecture

- **Domain contracts** live in `app/domain/interfaces.py` (e.g., `ICacheService`, `IEventPublisher`).
- **Application services** orchestrate use cases in `app/application/` (search and upload flows today).
- **Infrastructure providers** under `app/infrastructure/providers/` expose async `get_*` helpers that return singleton adapters.
- **FastAPI dependencies** resolve services through those providers (`app/core/dependencies.py`), so routers never touch containers directly.
- **Tests** reset provider singletons via `tests/conftest.py` to keep scenarios isolated.

## Database Architecture

The system uses a modern **FastAPI + SQLModel + Alembic** architecture for database management:

### Key Features
- ✅ **Industry Standard**: Following 2024-2025 FastAPI best practices
- ✅ **Foreign Key Constraints**: Automatic CASCADE DELETE for multi-tenant data isolation
- ✅ **Type Safety**: SQLModel provides full type checking and autocomplete
- ✅ **Migration Management**: Alembic handles all schema changes with versioning
- ✅ **Auto-generation**: Migrations automatically generated from SQLModel changes

### Database Commands

```bash
# Check current migration status
alembic current

# Generate new migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View migration history
alembic history
```

### Multi-tenant Data Isolation

The system implements **complete tenant isolation** with CASCADE DELETE constraints:

- When a tenant is deleted, ALL related data is automatically removed:
  - Users → CASCADE DELETE
  - User Sessions → CASCADE DELETE  
  - API Keys → CASCADE DELETE
  - Profiles → CASCADE DELETE
  - Embeddings → CASCADE DELETE

This ensures perfect data isolation and prevents orphaned records across tenant boundaries.

### Schema Changes

1. **Modify SQLModel models** in `app/models/`
2. **Generate migration**: `alembic revision --autogenerate -m "Add new field"`
3. **Review migration** in `migrations/versions/`
4. **Apply migration**: `alembic upgrade head`

Alembic automatically detects:
- New tables and columns
- Index changes
- Foreign key constraints
- Data type modifications

## Testing

The project uses **pytest** with a comprehensive test suite following hexagonal architecture principles.

### Test Structure

```
tests/
├── unit/domain/                    # Pure business logic tests (fast, isolated)
│   ├── test_authentication_policies.py  # Auth business rules
│   ├── test_authorization_policies.py   # RBAC & permissions  
│   └── test_tenant_rules.py            # Tenant validation logic
├── integration/workflows/          # End-to-end workflow tests
│   ├── test_comprehensive_authentication_workflows.py
│   └── test_comprehensive_tenant_workflows.py
├── mocks/                         # Production-ready mock infrastructure
│   ├── mock_repositories.py       # Data layer mocks
│   └── mock_services.py          # Service layer mocks
├── fixtures/                     # Test data builders and utilities
└── conftest.py                   # Pytest configuration and fixtures
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit                         # Unit tests only (fast)
pytest -m integration                  # Integration tests only
pytest -m "not slow"                   # Skip slow tests

# Run tests by layer (hexagonal architecture)
pytest -m domain                       # Domain logic tests
pytest -m application                  # Application workflow tests
pytest -m infrastructure               # Infrastructure adapter tests

# Run specific test types
pytest -m auth_flow                     # Authentication/authorization
pytest -m tenant_isolation             # Multi-tenant isolation
pytest -m security                     # Security-related tests

# Run single test file
pytest tests/unit/domain/test_authentication_policies.py

# Run specific test method
pytest tests/unit/domain/test_authentication_policies.py::TestPasswordPolicyValidation::test_valid_password_acceptance

# Run with verbose output
pytest -v

# Run without coverage (faster)
pytest --cov=''

# Generate HTML coverage report
pytest --cov-report=html
# Open htmlcov/index.html in browser

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

### Test Configuration

The test suite is configured via `pytest.ini` with:

- ✅ **80% minimum coverage requirement**
- ✅ **Async test support** with `asyncio_mode = auto`
- ✅ **Test markers** for architectural layers and test types
- ✅ **HTML coverage reports** generated in `htmlcov/`
- ✅ **JUnit XML output** for CI/CD integration
- ✅ **Hexagonal architecture compliance** testing

### Test Quality Standards

- **Unit Tests**: Fast (< 1ms), isolated, no external dependencies
- **Integration Tests**: Test complete workflows and component interactions  
- **Mock Infrastructure**: Production-ready mocks for all external services
- **Tenant Isolation**: Comprehensive multi-tenant data boundary testing
- **Security Testing**: Authentication, authorization, and input validation
- **Error Handling**: Graceful degradation and resilience testing

### Continuous Integration

The test suite is designed for CI/CD with:

```bash
# CI-optimized test run
pytest --junit-xml=test-results.xml --cov-report=xml:coverage.xml

# Quick feedback (unit tests only)
pytest -m unit --tb=short

# Full test suite with performance tests
pytest -m "not performance" --maxfail=3
```

### Test Markers Reference

| Marker | Description |
|--------|-------------|
| `unit` | Fast, isolated unit tests |
| `integration` | Component integration tests |
| `domain` | Pure business logic tests |
| `application` | Application workflow tests |
| `infrastructure` | Infrastructure adapter tests |
| `auth_flow` | Authentication/authorization tests |
| `tenant_isolation` | Multi-tenant isolation tests |
| `security` | Security-related tests |
| `slow` | Slow tests (performance, large datasets) |
| `performance` | Performance and load tests |
