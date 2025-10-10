# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AI-powered CV matching platform** with semantic search capabilities, built following **Hexagonal Architecture** principles with ongoing refactoring toward pure ports & adapters. The system supports multi-tenant organizations with complete data isolation and can handle 2M+ profiles with sub-2 second search response times.

**Core Technologies**: FastAPI, Python 3.12+, PostgreSQL with pgvector, OpenAI SDK, SQLModel, Alembic, Redis

## Development Commands

### Environment Setup
```bash
# Install dependencies (choose installation type)
pip install -e .                        # Core only
pip install -e ".[ai,local]"           # AI + local dev
pip install -e ".[full]"               # Everything
pip install -e ".[dev]"                # Development tools

# Copy environment file and configure
cp .env.local .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
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

### Pure Hexagonal Architecture Implementation ✅ COMPLETE

The codebase follows **Pure Hexagonal Architecture** with complete separation of concerns:

```
app/
├── domain/                          # DOMAIN LAYER (Pure business logic)
│   ├── entities/                    # Domain entities (dataclasses, NO SQLModel)
│   │   ├── profile.py              # Profile aggregate root
│   │   └── document_processing.py  # DocumentProcessing entity
│   ├── value_objects.py            # Value objects (ProfileId, TenantId, etc.)
│   ├── repositories/               # Repository interfaces (ports)
│   ├── interfaces.py               # Port definitions (IDatabase, ICacheService, etc.)
│   └── exceptions.py               # Domain exceptions
│
├── application/                     # APPLICATION LAYER (Use cases/orchestration)
│   ├── search_service.py           # Search workflow orchestration
│   ├── upload_service.py           # Upload workflow orchestration
│   └── dependencies/               # Application service dependencies
│
├── infrastructure/                  # INFRASTRUCTURE LAYER (Adapters)
│   ├── persistence/
│   │   ├── models/                 # SQLModel persistence tables
│   │   │   ├── base.py            # Base table models
│   │   │   ├── profile_table.py
│   │   │   ├── document_processing_table.py
│   │   │   ├── audit_table.py
│   │   │   ├── auth_tables.py
│   │   │   ├── tenant_table.py
│   │   │   ├── webhook_table.py
│   │   │   ├── retry_table.py
│   │   │   └── embedding_table.py
│   │   ├── mappers/                # Domain ↔ Table converters
│   │   │   ├── profile_mapper.py
│   │   │   └── document_processing_mapper.py
│   │   └── repositories/           # Repository implementations
│   │       ├── profile_repository.py
│   │       ├── user_repository.py
│   │       └── tenant_repository.py
│   └── providers/                  # Service provider modules (DI)
│       ├── ai_provider.py
│       ├── search_provider.py
│       ├── postgres_provider.py
│       └── cache_provider.py
│
├── api/                            # API LAYER (HTTP interface)
│   ├── v1/                         # API version 1 routes
│   │   ├── profiles.py
│   │   ├── search.py
│   │   ├── upload.py
│   │   ├── auth.py
│   │   └── webhooks.py
│   └── schemas/                    # API DTOs (request/response models)
│       ├── profile_schemas.py
│       ├── search_schemas.py
│       ├── upload_schemas.py
│       ├── analytics_schemas.py
│       ├── document_schemas.py
│       ├── job_schemas.py
│       ├── notification_schemas.py
│       └── webhook_schemas.py
│
└── database/                       # Database utilities
    └── sqlmodel_engine.py          # SQLModel session management
```

**Key Architectural Achievements:**
- ✅ **No app/models folder** - Completely removed
- ✅ **Pure domain entities** - Zero infrastructure dependencies
- ✅ **Mapper pattern** - Clean separation via ProfileMapper, DocumentProcessingMapper
- ✅ **API schemas** - DTOs in api/schemas/ separate from domain
- ✅ **Persistence tables** - All in infrastructure/persistence/models/
- ✅ **Zero circular dependencies** - Clean layered architecture

### Architecture Restrictions & Boundaries

Claude *must* preserve the following constraints when making changes:

1. **Domain Layer purity**
   - No imports from `app.infrastructure`, `app.application`, `app.api`, or external adapters.
   - Domain code can only depend on Python stdlib + other domain modules.

2. **Application Layer boundaries**
   - Application modules may only consume domain ports (`app.domain.interfaces`, `app.domain.repositories`, value objects, domain services).
   - Do **not** import concrete infrastructure classes (adapters, persistence models, providers) in the application layer; always request dependencies through providers/factories defined under `app.infrastructure.providers` or explicit dependency DTOs.

3. **Infrastructure Layer scope**
   - Infrastructure adapters implement domain interfaces only; avoid referencing application-level modules.
   - Providers may depend on other providers, but they must return implementations that satisfy the domain port they advertise.

4. **API Layer isolation**
   - API routers/controllers should orchestrate application services only; never reach into infrastructure adapters directly.
   - **CRITICAL**: Business logic must NOT reside in API routers. API endpoints should be thin controllers that only handle HTTP concerns (request parsing, response formatting, status codes).
   - All business logic, orchestration, and decision-making must be in the application service layer.
   - API routers should call a single service method and map the result to HTTP responses.

5. **Cross-layer rules**
   - No shared mutable state across layers; use providers/factories for lifecycle management.
   - Keep DTOs (FastAPI schemas) out of domain/application logic.
   - Any new service must expose a port in domain/application and register its adapter through a provider.

6. **Testing boundary awareness**
   - When stubbing dependencies in tests, prefer domain interfaces and providers instead of instantiating infrastructure adapters directly.

### Service Provider Pattern

The system uses a **Provider Pattern** for dependency injection, with all services accessed through provider modules in `app/infrastructure/providers/`:

- **ai_provider.py**: OpenAI, embedding, and prompt management services
- **search_provider.py**: Vector search, hybrid search, and reranking services
- **postgres_provider.py**: Database adapter
- **cache_provider.py**: Redis/memory cache services
- **analytics_provider.py**: Analytics and metrics services

Services are accessed via provider functions that ensure singleton instances:
```python
from app.infrastructure.providers.ai_provider import get_openai_service
openai_service = await get_openai_service()
```

### Application Services

Complex workflows are orchestrated through application services:
- **SearchApplicationService**: Handles search requests, AI processing, and analytics
- **UploadApplicationService**: Manages document upload, processing, and embedding generation

These services keep API endpoints thin and testable.

### Database Design

**SQLModel + PostgreSQL** with Alembic migrations:
- All persistence tables in `app/infrastructure/persistence/models/` use SQLModel for type safety
- Automatic table creation on startup via SQLModel
- Alembic for versioned schema migrations in `/migrations/env.py`
- pgvector extension for similarity search with 3072-dimension embeddings
- Multi-tenant isolation with `tenant_id` fields and CASCADE DELETE constraints
- Mappers in `app/infrastructure/persistence/mappers/` convert between domain entities and tables

### AI Services Architecture

Cloud-agnostic design with OpenAI SDK:
- **OpenAIService**: Direct OpenAI/Azure OpenAI integration
- **EmbeddingService**: pgvector integration for similarity search  
- **CacheManager**: Multi-tier semantic caching (memory + Redis)
- **QueryProcessor**: Query expansion and optimization
- **ResultRerankerService**: GPT-4 powered result reranking

### Authentication & Multi-tenancy

- **JWT-based authentication** with tenant context injection
- **Dependency injection**: `CurrentUserDep`, `TenantContextDep`
- **Complete tenant isolation**: All operations are tenant-aware
- **Super admin support**: System tenant for platform management

## Configuration Management

### Environment Variables

The system uses **pydantic-settings** for type-safe configuration (`app/core/config.py`):

- **Required**: `SECRET_KEY` (minimum 32 characters)
- **Database**: `POSTGRES_URL` or individual connection parameters
- **Cache**: `REDIS_URL` (optional, falls back to memory cache)
- **AI**: `OPENAI_API_KEY` or Azure OpenAI credentials
- **Environment**: `ENVIRONMENT` (local/development/production)

### Service Strategy

Environment determines service implementations:
```
ENVIRONMENT=local → Memory cache, local storage, PostgreSQL
ENVIRONMENT=development → Redis cache, local storage, PostgreSQL  
ENVIRONMENT=production → Redis cache, cloud storage, PostgreSQL
```

## Key Architectural Patterns

### Provider Pattern
All services accessed through provider modules ensuring consistent singleton management and proper initialization.

### Application Layer Orchestration
Complex workflows handled by application services, keeping API endpoints focused on HTTP concerns.

### Adapter Pattern
Infrastructure services implement domain interfaces, enabling easy swapping of implementations.

### Repository Pattern
Database access abstracted through repositories with tenant-aware queries.

## Code Organization and Style Standards

### Static Method Declaration Pattern

**CRITICAL**: All application services must follow a strict pattern for declaring and organizing static methods.

#### Static Method Requirements

1. **Use @staticmethod decorator**: All static helper methods MUST use the `@staticmethod` decorator
2. **No self parameter**: Static methods should NOT have `self` as a parameter (only accept parameters they need)
3. **Bottom placement**: ALL static methods MUST be placed at the bottom of the class, before `__all__` exports
4. **Clear separation**: Static methods handle pure data transformation, validation, or parsing without requiring instance state
5. **Calling pattern**: Instance methods should call static methods using `self.static_method()` for consistency; static methods calling other static methods must use `ClassName.static_method()` (since they don't have `self`)

#### When to Make a Method Static

A method should be static when it:
- Does NOT access `self._deps` or any instance attributes
- Performs pure data transformation or parsing
- Provides utility/helper functionality
- Does NOT require any injected dependencies
- Can be tested independently without class instantiation

#### Correct Pattern

```python
class UploadApplicationService:
    """Application service for document upload workflows."""

    def __init__(self, deps: UploadServiceDependencies):
        """Initialize with injected dependencies."""
        self._deps = deps

    async def process_upload(self, file_content: bytes) -> Profile:
        """Instance method using dependencies."""
        # Uses self._deps.openai_service, etc.
        email = self._find_first_email(text)  # Call static method
        return profile

    # === STATIC HELPER METHODS (at bottom of class) ===

    @staticmethod
    def _find_first_email(text: str) -> Optional[str]:
        """Find the first email address in text.

        Args:
            text: Input text to search

        Returns:
            First email found or None
        """
        if not text:
            return None
        matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return matches[0] if matches else None

    @staticmethod
    def _parse_skills(skills_data: Any) -> list[Skill]:
        """Parse skills from raw data into domain objects.

        Args:
            skills_data: Raw skills data from parser

        Returns:
            List of Skill domain objects
        """
        if not skills_data or not isinstance(skills_data, list):
            return []
        return [Skill(name=item.get("skill")) for item in skills_data if item.get("skill")]

    @staticmethod
    def _normalize_phone(phone: str) -> Optional[str]:
        """Normalize phone number to E.164 format.

        Args:
            phone: Raw phone number string

        Returns:
            Normalized phone number or None
        """
        if not phone:
            return None
        # Normalization logic here
        return normalized_phone
```

#### Common Mistakes to Avoid

❌ **WRONG**: Instance method when no dependencies are used
```python
def _parse_skills(self, skills_data: Any) -> list[Skill]:  # ❌ Has self but doesn't use it
    return [Skill(name=item.get("skill")) for item in skills_data]
```

❌ **WRONG**: Static method placed at top or middle of class
```python
class UploadApplicationService:
    @staticmethod
    def _helper_method():  # ❌ Should be at bottom
        pass

    def __init__(self):
        pass
```

❌ **WRONG**: Missing @staticmethod decorator
```python
def _parse_skills(skills_data: Any) -> list[Skill]:  # ❌ Missing decorator
    return [Skill(name=item.get("skill")) for item in skills_data]
```

✅ **CORRECT**: Static method with decorator at bottom
```python
class UploadApplicationService:
    def __init__(self, deps: UploadServiceDependencies):
        self._deps = deps

    async def instance_method(self):
        """Uses self._deps"""
        pass

    # === STATIC HELPER METHODS ===

    @staticmethod
    def _helper_method(param: str) -> str:
        """Pure helper function."""
        return param.upper()
```

#### Class Organization Template

```python
class YourApplicationService:
    """Service documentation."""

    # 1. Class-level constants (if any)
    MAX_SIZE = 1000

    # 2. __init__ method
    def __init__(self, deps: YourServiceDependencies):
        """Initialize with dependencies."""
        self._deps = deps

    # 3. Public methods (instance methods using dependencies)
    async def public_method(self, param: str) -> Result:
        """Public API method."""
        # Uses self._deps
        pass

    # 4. Private instance methods (using dependencies)
    async def _private_instance_method(self) -> None:
        """Private method using dependencies."""
        # Uses self._deps
        pass

    # === STATIC HELPER METHODS ===
    # (All static methods grouped at bottom)

    @staticmethod
    def _static_helper_1(param: str) -> str:
        """Static helper method."""
        return result

    @staticmethod
    def _static_helper_2(data: dict) -> Model:
        """Another static helper."""
        return Model(**data)
```

#### Benefits of This Pattern

✅ **Clear separation**: Instance methods vs pure functions clearly separated
✅ **Better testability**: Static methods can be tested without mocking dependencies
✅ **Improved readability**: Easy to identify which methods require dependencies
✅ **Consistent structure**: All service classes follow same organization pattern
✅ **Type safety**: Static methods are self-contained and easier to type-check
✅ **Reusability**: Static methods can be easily extracted to utility modules if needed

#### Reference Implementations

See these files for correct static method patterns:
- `app/application/profile_service.py` - ProfileApplicationService (8 static methods)
- `app/application/upload_service.py` - UploadApplicationService (15+ static methods)
- `app/infrastructure/persistence/mappers/profile_mapper.py` - All mapper methods are static

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

## Database Model Creation and Migration Pattern

**CRITICAL**: This codebase follows a strict pattern for creating database tables. **NEVER** create migrations manually. Always follow this 3-step process:

### Step 1: Create SQLModel Table Class

Create a new model file in `app/models/` following this exact pattern:

```python
from sqlalchemy import Column, String, DateTime, Integer, Index
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID, JSONB
from sqlmodel import Field, Relationship
from .base import TenantModel, create_tenant_id_column

class YourTableName(TenantModel, table=True):
    """Your table documentation."""
    __tablename__ = "your_table_name"

    # REQUIRED: Tenant isolation using create_tenant_id_column()
    tenant_id: UUID = Field(
        sa_column=create_tenant_id_column(),
        description="Tenant identifier for multi-tenant isolation"
    )

    # Relationships (if needed)
    tenant: Optional["TenantTable"] = Relationship(back_populates="your_table_name")

    # Your fields - ALWAYS use sa_column=Column() for explicit column definitions
    your_field: str = Field(
        sa_column=Column(String(100), nullable=False, index=True),
        description="Field description"
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_your_table_tenant_field", "tenant_id", "your_field"),
    )
```

**Key Requirements:**
- Inherit from `TenantModel` (provides id, created_at, updated_at, tenant_id base)
- Set `table=True` to make it a database table
- Define `__tablename__` explicitly
- Use `create_tenant_id_column()` for tenant_id field (ensures CASCADE DELETE)
- Use `sa_column=Column()` for ALL field definitions
- Add composite indexes in `__table_args__` for common query patterns
- Reference pattern: See `app/models/audit.py` or `app/models/document_processing.py`

### Step 2: Generate Migration with Alembic Autogenerate

**NEVER create migration files manually.** Always use autogenerate:

```bash
# Generate migration from your SQLModel
alembic revision --autogenerate -m "Add your_table_name table"

# Review the generated migration file in migrations/versions/
# Alembic will detect your new table and generate the migration
```

### Step 3: Apply Migration

```bash
# Apply the migration
alembic upgrade head

# Verify it worked
alembic current
psql -U cv_user -d cv_matching -c "\d your_table_name"
```

### Migration Best Practices

**IMPORTANT**: Never modify applied migrations. Follow these rules:

- **NEVER manually create migrations**: Always use `alembic revision --autogenerate`
- **NEVER edit applied migrations**: Creates checksum mismatches across environments
- **Always create SQLModel first**: Define the table model, then generate migration
- **Test migrations thoroughly**: Test both UP and DOWN migrations before applying
- **Review auto-generated migrations**: Always review generated code before applying
- **Sequential naming**: Alembic auto-generates sequential identifiers

### Common Mistakes to Avoid

❌ **WRONG**: Creating migration file manually
❌ **WRONG**: Using raw SQL in migration without SQLModel
❌ **WRONG**: Not using `create_tenant_id_column()` for tenant_id
❌ **WRONG**: Not using `sa_column=Column()` for field definitions
❌ **WRONG**: Inheriting from `BaseModel` instead of `TenantModel`

✅ **CORRECT**: Create SQLModel → Run autogenerate → Review → Apply


## Pure Hexagonal Architecture: Domain-Infrastructure Separation

**CRITICAL**: This codebase follows **Pure Hexagonal Architecture** with complete separation between domain entities and persistence tables. This pattern ensures the domain layer has NO dependencies on infrastructure concerns.

### Architecture Layers

```
app/
├── domain/
│   ├── entities/              # Pure domain entities (dataclasses, NO SQLModel)
│   │   ├── profile.py         # Profile aggregate root with business logic
│   │   └── document_processing.py  # DocumentProcessing entity
│   ├── value_objects.py       # Value objects (ProfileId, TenantId, etc.)
│   ├── repositories/          # Repository interfaces (ports)
│   └── interfaces.py          # Other port definitions
├── infrastructure/
│   └── persistence/
│       ├── models/            # SQLModel tables (database-specific)
│       │   ├── profile_table.py         # ProfileTable for database
│       │   └── document_processing_table.py  # DocumentProcessingTable
│       ├── mappers/           # Domain ↔ Persistence converters
│       │   ├── profile_mapper.py        # ProfileMapper
│       │   └── document_processing_mapper.py
│       └── repositories/      # Repository implementations (adapters)
│           └── profile_repository.py    # PostgresProfileRepository
└── application/               # Application services (use domain entities)
    └── upload_service.py      # Uses Profile entity, not ProfileTable
```

### The Mapper Pattern

**Mappers** convert between domain entities and persistence tables, maintaining complete separation:

#### ProfileMapper Example

```python
from app.infrastructure.persistence.mappers import ProfileMapper
from app.domain.entities.profile import Profile
from app.infrastructure.persistence.models.profile_table import ProfileTable

# Domain → Table (for saving to database)
profile_entity = Profile(...)  # Pure domain entity
profile_table = ProfileMapper.to_table(profile_entity)
session.add(profile_table)
session.commit()

# Table → Domain (for loading from database)
profile_table = session.get(ProfileTable, profile_id)
profile_entity = ProfileMapper.to_domain(profile_table)

# Now use domain business logic
profile_entity.activate()
profile_entity.calculate_completeness_score()
```

### Domain Entity Pattern

**Domain entities** are pure Python dataclasses with business logic and NO database annotations:

```python
# app/domain/entities/profile.py
from dataclasses import dataclass, field
from datetime import datetime
from app.domain.value_objects import ProfileId, TenantId, EmailAddress

@dataclass
class Profile:
    """Aggregate root representing a candidate profile."""

    id: ProfileId                    # Value object, not UUID
    tenant_id: TenantId              # Value object, not UUID
    status: ProfileStatus            # Enum
    profile_data: ProfileData        # Value object
    created_at: datetime = field(default_factory=datetime.utcnow)

    def activate(self) -> None:
        """Business logic method."""
        if self.status == ProfileStatus.DELETED:
            raise ValueError("Cannot activate deleted profile")
        self.status = ProfileStatus.ACTIVE
        self.updated_at = datetime.utcnow()

    def calculate_completeness_score(self) -> float:
        """Calculate profile completeness (0-100)."""
        # Business logic here
        return score
```

**Key characteristics:**
- Uses `@dataclass` decorator, NOT SQLModel
- Uses value objects (ProfileId, EmailAddress) instead of primitives
- Contains business logic methods
- NO database annotations (`table=True`, `sa_column`, etc.)
- NO imports from SQLModel or SQLAlchemy

### Persistence Table Pattern

**Persistence tables** are SQLModel classes with database-specific concerns:

```python
# app/infrastructure/persistence/models/profile_table.py
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, String, Index
from app.models.base import TenantModel, create_tenant_id_column

class ProfileTable(TenantModel, table=True):
    """Database table for profile persistence."""
    __tablename__ = "profiles"

    # Database columns with explicit SQLAlchemy definitions
    name: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    email: str = Field(sa_column=Column(String(255), nullable=False, index=True))

    # JSONB for complex nested data
    profile_data: Dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))

    # Denormalized fields for query performance
    normalized_skills: List[str] = Field(sa_column=Column(ARRAY(String)))

    # Indexes
    __table_args__ = (
        Index("ix_profiles_tenant_email", "tenant_id", "email"),
    )
```

**Key characteristics:**
- Inherits from `TenantModel` with `table=True`
- Uses `sa_column=Column()` for all field definitions
- Stores complex objects as JSONB
- Denormalizes fields for query performance
- Contains database indexes and constraints
- Located in `app/infrastructure/persistence/models/`

### Repository Implementation Pattern

**Repositories** use mappers to convert between domain and persistence:

```python
# app/infrastructure/persistence/repositories/profile_repository.py
from app.domain.repositories.profile_repository import IProfileRepository
from app.infrastructure.persistence.mappers.profile_mapper import ProfileMapper
from app.infrastructure.persistence.models.profile_table import ProfileTable

class PostgresProfileRepository(IProfileRepository):
    """PostgreSQL implementation of IProfileRepository."""

    async def save(self, profile: Profile) -> Profile:
        """Save domain entity to database."""
        # Convert domain → table
        profile_table = ProfileMapper.to_table(profile)

        # Persist to database
        await adapter.execute(
            "INSERT INTO profiles (...) VALUES (...)",
            profile_table.id, profile_table.name, ...
        )

        return profile

    async def get_by_id(self, profile_id: ProfileId) -> Optional[Profile]:
        """Load domain entity from database."""
        # Fetch from database
        record = await adapter.fetch_one(
            "SELECT * FROM profiles WHERE id = $1",
            profile_id.value
        )

        if not record:
            return None

        # Convert table → domain
        profile_table = ProfileTable(**dict(record))
        return ProfileMapper.to_domain(profile_table)
```

### Application Service Pattern

**Application services** work ONLY with domain entities, never with tables:

```python
# app/application/upload_service.py
from app.domain.entities.profile import Profile, ProfileStatus
from app.domain.value_objects import ProfileId, TenantId

class UploadApplicationService:
    """Application service for document uploads."""

    async def process_upload(self, file_content: bytes) -> Profile:
        """Process document and return domain entity."""

        # Work with domain entities
        profile = Profile(
            id=ProfileId(uuid4()),
            tenant_id=TenantId(tenant_uuid),
            status=ProfileStatus.ACTIVE,
            profile_data=parsed_data
        )

        # Apply business logic
        profile.activate()
        score = profile.calculate_completeness_score()

        # Save via repository (repository handles mapping)
        await self.profile_repository.save(profile)

        return profile  # Return domain entity, not table
```

### Creating New Domain Models

When adding a new domain concept, follow this pattern:

#### Step 1: Create Domain Entity

```bash
# Create pure domain entity (NO SQLModel)
touch app/domain/entities/your_entity.py
```

```python
from dataclasses import dataclass
from app.domain.value_objects import YourId

@dataclass
class YourEntity:
    """Pure domain entity with business logic."""
    id: YourId
    name: str

    def your_business_method(self) -> None:
        """Business logic here."""
        pass
```

#### Step 2: Create Persistence Table

```bash
# Create SQLModel table
touch app/infrastructure/persistence/models/your_entity_table.py
```

```python
from sqlmodel import Field
from app.models.base import TenantModel

class YourEntityTable(TenantModel, table=True):
    """Database table for YourEntity."""
    __tablename__ = "your_entities"

    name: str = Field(sa_column=Column(String(255)))
```

#### Step 3: Create Mapper

```bash
# Create mapper for conversion
touch app/infrastructure/persistence/mappers/your_entity_mapper.py
```

```python
class YourEntityMapper:
    """Convert between YourEntity and YourEntityTable."""

    @staticmethod
    def to_domain(table: YourEntityTable) -> YourEntity:
        """Convert table to domain entity."""
        return YourEntity(
            id=YourId(table.id),
            name=table.name
        )

    @staticmethod
    def to_table(entity: YourEntity) -> YourEntityTable:
        """Convert domain entity to table."""
        return YourEntityTable(
            id=entity.id.value,
            name=entity.name
        )
```

#### Step 4: Generate Migration

```bash
# Generate Alembic migration from SQLModel
alembic revision --autogenerate -m "Add your_entities table"
alembic upgrade head
```

### Benefits of This Pattern

✅ **Domain Independence**: Domain layer has zero dependencies on infrastructure
✅ **Testability**: Domain entities can be tested without database
✅ **Flexibility**: Can swap database implementations without changing domain
✅ **Business Logic Isolation**: All business rules in domain entities
✅ **Type Safety**: Value objects provide compile-time safety
✅ **Clean Architecture**: Clear separation of concerns across layers

### Common Mistakes to Avoid

❌ **WRONG**: Using SQLModel in domain entities
❌ **WRONG**: Importing from `app/models/` in domain layer
❌ **WRONG**: Putting business logic in table classes
❌ **WRONG**: Application services working directly with tables
❌ **WRONG**: Skipping mappers and mixing domain/persistence

✅ **CORRECT**: Domain entities use dataclasses
✅ **CORRECT**: Tables live in `app/infrastructure/persistence/models/`
✅ **CORRECT**: Mappers handle all domain ↔ table conversions
✅ **CORRECT**: Application services use domain entities only
✅ **CORRECT**: Repositories hide all persistence details

### Import Rules

**Domain Layer** (`app/domain/`):
- ✅ Can import: Other domain entities, value objects, domain interfaces
- ❌ Cannot import: SQLModel, infrastructure, adapters, services

**Infrastructure Layer** (`app/infrastructure/`):
- ✅ Can import: Domain entities, domain interfaces, SQLModel
- ✅ Implements: Domain interfaces (repositories, services)

**Application Layer** (`app/application/`):
- ✅ Can import: Domain entities, domain interfaces, repositories
- ❌ Cannot import: Persistence tables directly

**API Layer** (`app/api/`):
- ✅ Can import: Application services, API schemas (DTOs), domain value objects for type conversion
- ❌ Cannot import: Infrastructure adapters, persistence tables directly
- ❌ Cannot contain: Business logic, orchestration, or decision-making

### API Layer Best Practices

**CRITICAL**: API routers must be thin controllers that only handle HTTP concerns.

#### ❌ WRONG: Business Logic in API Router

```python
@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    permanent: bool,
    profile_service: ProfileServiceDep,
    background_tasks: BackgroundTasks
):
    # ❌ BAD: Business logic branching in API router
    if permanent:
        # Permanent deletion logic
        success = await profile_service.permanently_delete_profile(...)
        if not success:
            return JSONResponse(status_code=404, ...)
        response = ProfileDeletionResponse(
            success=True,
            deletion_type="permanent_delete",
            message="Profile permanently deleted",
            can_restore=False
        )
    else:
        # Soft deletion logic
        deleted_profile = await profile_service.soft_delete_profile(...)
        if not deleted_profile:
            return JSONResponse(status_code=404, ...)
        response = ProfileDeletionResponse(
            success=True,
            deletion_type="soft_delete",
            message="Profile deleted (can be restored)",
            can_restore=True
        )
    return JSONResponse(content=response.model_dump())
```

**Problems with this approach:**
- Business logic decision (soft vs permanent) made in router
- Duplicated response construction logic
- Router knows about deletion types and business rules
- Harder to test business logic separately from HTTP layer
- Cannot reuse deletion logic from other entry points (CLI, GraphQL, etc.)

#### ✅ CORRECT: Thin API Router Delegating to Service

```python
@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    permanent: bool,
    profile_service: ProfileServiceDep,
    background_tasks: BackgroundTasks
):
    # ✅ GOOD: Just call service and map result to HTTP
    result = await profile_service.delete_profile(
        profile_id=ProfileId(profile_id),
        tenant_id=TenantId(current_user.tenant_id),
        user_id=current_user.user_id,
        permanent=permanent,  # Pass parameter, don't decide what to do
        schedule_task=background_tasks
    )

    # Map service result to HTTP response
    if not result.success:
        return JSONResponse(status_code=404, content={"detail": result.message})

    # Convert service result to API schema
    response = ProfileDeletionResponse(
        success=result.success,
        deletion_type=result.deletion_type,
        profile_id=result.profile_id,
        message=result.message,
        can_restore=result.can_restore
    )

    return JSONResponse(status_code=200, content=response.model_dump())
```

**Benefits of this approach:**
- ✅ Router only handles HTTP concerns (status codes, request/response formatting)
- ✅ All business logic in application service
- ✅ Service returns standardized result object
- ✅ Easy to test business logic without HTTP layer
- ✅ Reusable service method across different interfaces
- ✅ Single source of truth for deletion business rules

#### API Router Responsibilities (HTTP Concerns Only)

**What API routers SHOULD do:**
- Parse and validate HTTP request parameters
- Convert request DTOs to domain value objects
- Call application service methods
- Map service results to HTTP responses
- Set appropriate HTTP status codes
- Handle HTTP-specific errors (404, 400, 500)
- Add background tasks for non-critical operations

**What API routers should NOT do:**
- Make business decisions (if/else based on business rules)
- Orchestrate multiple service calls
- Contain conditional logic based on business state
- Construct complex domain objects
- Implement workflows or multi-step processes
- Duplicate logic across multiple endpoints

## Architectural Enforcement

The codebase uses **Ruff linting rules** to prevent architectural regressions:

```toml
[tool.ruff.lint.flake8-tidy-imports]
banned-modules = { "app.core.interfaces" = "Import from app.domain.interfaces instead.", "app.services.providers" = "Use app.infrastructure.providers instead." }
```

These rules automatically prevent imports from deprecated locations during development.

## Architecture Migration Complete ✅

### Recently Completed (Latest Migration)
- ✅ **Complete hexagonal architecture**: Pure separation of domain, application, infrastructure, and API layers
- ✅ **Deleted app/models folder**: Completely removed - all models migrated to proper layers
- ✅ **API schemas layer**: All DTOs moved to `app/api/schemas/` (8 schema modules, 55+ classes)
- ✅ **Infrastructure persistence**: All tables in `app/infrastructure/persistence/models/` (8 table modules)
- ✅ **Mapper pattern**: ProfileMapper and DocumentProcessingMapper for domain ↔ table conversion
- ✅ **Domain entities**: Pure dataclasses in `app/domain/entities/` with zero infrastructure dependencies
- ✅ **Import migration**: Updated 80+ imports across 50+ files to use new architecture
- ✅ **Zero circular dependencies**: Clean layered architecture with proper import rules

### Previous Milestones
- **Complete provider migration**: All services accessed via `app/infrastructure/providers/`
- **Removed Container class**: Replaced with provider pattern throughout
- **Application services**: Search and upload workflows orchestrated by dedicated services
- **Domain interfaces**: All ports moved to `app/domain/interfaces.py`
- **FastAPI dependencies**: Updated to use provider-backed helpers

### Outstanding Work
- **Auth application services**: Authentication/authorization flows still use direct service dependencies
- **Event publishing integration**: `IEventPublisher` exists but not yet integrated into domain workflows
- **Provider test coverage**: Need more comprehensive testing of provider-aware flows
- **Legacy services migration**: Services in `app/services/` to be migrated to infrastructure layer

## System Initialization

After first run, initialize with super admin:
```bash
curl -X POST "http://localhost:8000/api/v1/setup/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@system.local",
    "password": "SuperAdmin123!",
    "full_name": "System Administrator"
  }'
```

Default credentials:
- Email: `admin@system.local`
- Password: `SuperAdmin123!`
- Tenant ID: `00000000-0000-0000-0000-000000000000` (System tenant)
- Add to memory "always follow the pure hexagonal architecture setup in the code base
- Prefer ORM/SQLModel methods instead of raw SQL when interacting with the database; reach for direct SQL only when no repository/provider abstraction exists.
