# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AI-powered CV matching platform** with semantic search capabilities, built using **Hexagonal Architecture** and following 2025 software architecture best practices. The system supports multi-tenant organizations with complete data isolation and can handle 2M+ profiles with sub-2 second search response times.

**Core Technologies**: FastAPI, Python 3.12+, PostgreSQL with pgvector, OpenAI SDK, LangChain/LangGraph, Redis

## Development Commands

### Environment Setup
```bash
# Copy environment file and configure
cp .env.local .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Install dependencies (choose installation type)
pip install -e .                        # Core only
pip install -e ".[ai,local]"           # AI + local dev
pip install -e ".[full]"               # Everything
pip install -e ".[dev]"                # Development tools
```

### Running the Application
```bash
# Development server with auto-reload
python -m uvicorn app.main:app --reload --port 8000

# Or using the module directly
python -m app.main

# Access API docs at: http://localhost:8000/docs
```

### Database Operations
```bash
# Initialize database with SQLModel auto-migration
python -m uvicorn app.main:app --reload --port 8000

# Run SQL migrations manually (if needed)
python run_migrations.py

# Check database health
curl -X GET "http://localhost:8000/health"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit                         # Unit tests only
pytest -m integration                  # Integration tests only
pytest -m "not slow"                   # Skip slow tests

# Run single test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=app --cov-report=html
```

### Code Quality
```bash
# Run linting
ruff check app/

# Run type checking
mypy app/

# Fix auto-fixable issues
ruff check --fix app/
```

## Architecture Overview

### Hexagonal Architecture (Ports & Adapters)

The codebase follows **Hexagonal Architecture** principles with clear separation between business logic and infrastructure:

```
app/
├── api/                    # API layer (HTTP interface)
├── core/                   # Core business domain
│   ├── interfaces.py       # Abstract service contracts
│   ├── service_factory.py  # Dynamic service instantiation
│   └── config.py          # Cloud-agnostic configuration
├── services/               # Domain services organized by capability
│   ├── adapters/           # Infrastructure adapters (Hexagonal Architecture)
│   ├── ai/                 # AI/ML domain services
│   ├── auth/               # Authentication domain
│   ├── document/           # Document processing domain
│   └── search/             # Search domain services
├── database/               # Data persistence layer
│   ├── migrations/         # SQL migrations
│   └── repositories/       # Data access patterns
└── main.py                # FastAPI application entry point
```

### Service Factory Pattern

The system uses a **Service Factory** (`app/core/service_factory.py`) that dynamically selects implementations based on environment:

- **Local**: Memory cache, local file storage, PostgreSQL
- **Development**: Redis cache, local file storage, PostgreSQL  
- **Production**: Redis cache, cloud storage, PostgreSQL

All services implement abstract interfaces (`app/core/interfaces.py`) enabling seamless switching between implementations.

### AI Services Architecture

The AI services follow a **cloud-agnostic design** with LangChain orchestration:

- **`app/services/ai/openai_service.py`**: Direct OpenAI SDK integration (supports both OpenAI and Azure OpenAI)
- **`app/services/ai/embedding_service.py`**: pgvector integration for similarity search
- **`app/services/ai/cache_manager.py`**: Multi-tier semantic caching (memory + Redis)
- **`app/services/document/pdf_processor.py`**: Document processing with LangChain

### Database Design

PostgreSQL with **pgvector extension** for vector similarity search:

- **Multi-tenant isolation**: All tables include `tenant_id` for data separation
- **Vector embeddings**: Stored using pgvector for sub-second similarity queries
- **AI tables**: Dedicated tables for embeddings, search cache, and AI analytics
- **SQLModel ORM**: Uses SQLModel for type-safe database operations with automatic schema generation
- **Auto-migration**: SQLModel automatically creates tables on startup; manual SQL migrations available for complex schema changes

### Authentication & Multi-tenancy

- **JWT-based authentication** with tenant context injection
- **Dependency injection**: `CurrentUserDep`, `TenantContextDep` for request context
- **Role-based access control**: Organization admins, users, and system roles
- **Complete tenant isolation**: All operations are tenant-aware

## Configuration Management

### Environment Variables

The system uses **pydantic-settings** for type-safe configuration (`app/core/config.py`):

- **Required**: `SECRET_KEY` (minimum 32 characters)
- **Database**: `POSTGRES_URL` or individual connection parameters
- **Cache**: `REDIS_URL` (optional, falls back to memory cache)
- **AI**: `OPENAI_API_KEY` or Azure OpenAI credentials
- **Environment**: `ENVIRONMENT` (local/development/production)

### Service Selection Strategy

The `ServiceStrategy` enum in `app/core/environment.py` controls which implementations are used:

```python
# Local development
ENVIRONMENT=local -> Memory cache, local storage, PostgreSQL

# Development/staging  
ENVIRONMENT=development -> Redis cache, local storage, PostgreSQL

# Production
ENVIRONMENT=production -> Redis cache, cloud storage, PostgreSQL
```

## Key Architectural Patterns

### Adapter Pattern
All infrastructure services use the `*_adapter.py` naming convention and implement abstract interfaces, enabling easy swapping of implementations.

### Repository Pattern  
Database access is abstracted through repositories (`app/database/repositories/`) that handle tenant-aware queries and connection management.

### Strategy Pattern
The service factory uses environment-based strategy selection to choose appropriate service implementations without changing business logic.

### Dependency Injection
FastAPI's dependency system provides tenant context, user authentication, and service instances throughout the request lifecycle.

## AI & Search Capabilities

### Semantic Search Pipeline
1. **Document ingestion**: PDF processing with quality analysis
2. **Embedding generation**: OpenAI text-embedding-3-large model
3. **Vector storage**: PostgreSQL with pgvector for similarity search  
4. **Hybrid search**: Combines text search with vector similarity
5. **AI reranking**: GPT-4 powered result reranking for relevance

### Caching Strategy
- **L1 Cache**: In-memory for frequently accessed data
- **L2 Cache**: Redis for distributed caching
- **Semantic Cache**: Similar query detection using cosine similarity
- **Cache invalidation**: Tenant-aware cache clearing

### Performance Targets
- **Sub-2 second search response** for 2M+ profiles
- **60%+ cache hit rate** through semantic similarity matching
- **37% cost reduction** through intelligent caching and batching

## Important Implementation Details

### Service Lifecycle Management
Services implement `IHealthCheck` interface and are managed through the FastAPI lifespan context (`app/main.py`). Database connections, cache pools, and AI services are initialized on startup and properly closed on shutdown.

### Error Handling  
Comprehensive error handling through `app/database/error_handling.py` with structured logging, retry logic, and graceful degradation for external service failures.

### Migration Safety
Database migrations are versioned, validated, and support rollback. The migration system tracks applied migrations and prevents accidental data loss through checksum validation.

**IMPORTANT**: Once a migration has been applied and committed to version control, it should **NEVER** be modified. Follow these migration best practices:

- **Never edit applied migrations**: Modifying a migration file after it's been applied will cause checksum mismatches and potential database inconsistencies across environments
- **Create new migrations for changes**: If you need to modify database schema after a migration is applied, create a new migration file with the additional changes
- **Use rollback migrations**: Include DOWN sections in migrations for safe rollback capability
- **Test migrations thoroughly**: Test both UP and DOWN migrations in development before applying to production
- **Sequential naming**: Use sequential numbering (001, 002, 003) to ensure proper migration order
- **Descriptive names**: Use clear, descriptive names that explain what the migration does

If you accidentally modify an applied migration and get checksum warnings, the safest approach is to:
1. Revert the migration file to its original state
2. Create a new migration file with your intended changes
3. Apply the new migration

### Multi-tenant Security
All database queries are automatically tenant-scoped through repository patterns. API endpoints use dependency injection to enforce tenant isolation at the request level.

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

## Current Implementation Status

**✅ Recently Completed Migration:**
- **AsyncPG to SQLModel**: Successfully migrated from AsyncPG to SQLModel ORM for better type safety and automatic schema generation
- **Database Modernization**: Updated connection pooling and error handling for SQLModel compatibility
- **Field Name Conflicts**: Resolved Pydantic metadata field conflicts by renaming to avoid shadowing BaseModel attributes

**🔧 Database Schema:**
- Uses SQLModel with PostgreSQL backend
- Automatic table creation on startup
- Legacy SQL migrations in `app/database/migrations/` for complex schema changes (001-009)
- All models include tenant isolation with `tenant_id` fields