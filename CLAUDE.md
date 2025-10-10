# CLAUDE.md — Guidance for Claude Code Agent

## Project: AI-Powered CV Matching Platform (Hexagonal Architecture)

- **Stack:** FastAPI · Python 3.12+ · PostgreSQL+pgvector · OpenAI · SQLModel · Alembic · Redis
- **Context:** Multi-tenant; each org's data is isolated. Search must be sub-2s for 2M+ profiles.

---

## 1. Core Constraints (ALWAYS ENFORCE)

1. **Hexagonal architecture**: Keep domain (`app/domain/`) pure; no SQLModel or infra.
2. **Business logic/testability**:
    - All orchestration in application services only.
    - API routers/controllers: handle HTTP only, thin and stateless.
    - No business logic in API endpoints.
3. **Layer boundaries**:
    - Domain ↔ Application ↔ Infrastructure ↔ API; no skipping.
    - Mappers convert domain ↔ persistence.
    - DTOs only in API schemas (`app/api/schemas/`).
4. **Providers only**: Dependency injection via `app/infrastructure/providers/[…].py`.
5. **Testing**: Prefer interface/provider stubs over direct infra adapter instantiation.
6. **Strict static method pattern** (see #7 below).

---

## 2. Essential Commands

- **Install:**  
  `pip install -e ".[full]"` # Full setup  
  `cp .env.local .env`  
  `python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"`

- **Run:**  
  `python -m uvicorn app.main:app --reload --port 8000`

- **Migrations:**
    - `alembic revision --autogenerate`
    - `alembic upgrade head`

- **Test:**
    - `pytest -m unit`
    - `pytest --cov=app --cov-report=html`

- **Quality:**
    - `ruff check app/`
    - `mypy app/`

---

## 3. Architectural Boundaries

- **Domain (`app/domain/`)**: Only dataclasses, value objects, interfaces, ports.
- **Infrastructure (`app/infrastructure/`)**: Only adapters, tables, mappers, providers.
- **Application (`app/application/`)**: Only orchestrates; no infra or DB logic.
- **API Layer:** Only calls application; presents HTTP DTOs, no logic.

---

## 4. Service & Provider Pattern

- **All services and external calls through provider modules:**  
  `from app.infrastructure.providers.ai_provider import get_openai_service`
  Example: openai_service = await get_openai_service()


---

## 5. Migration & Model Best Practice

1. **Create SQLModel table in infra/models/**
2. **NEVER create migrations by hand** — always use Alembic autogenerate.
3. **Mandatory fields/patterns**:
- Inherit from TenantModel
- Use `sa_column=Column()` for fields
- Composite indexes in `__table_args__`
- Define `__tablename__`
- Use `create_tenant_id_column()` for isolation
4. **Review, then apply migrations only after test coverage**

---

## 6. Static Method Declaration Pattern

- All static helpers go **at the end of class**, decorated with `@staticmethod`.
- Static if: No instance state access, pure transform/parse/validate, no deps.
- **Instance methods**: Use `self._deps` only for side effects/services.
- See examples in: `app/application/upload_service.py`, `app/domain/entities/profile.py`

---

## 7. Architecture Enforcement

- **Linting:**
- Use Ruff to block architectural regressions via `[tool.ruff.lint.flake8-tidy-imports]` config.
- **No raw SQL unless no repo/provider abstraction exists. Prefer SQLModel.**
- **Always follow pure hexagonal architecture and directory setup.**
- **Never mix domain and infra concerns.**
- **Update CLAUDE.md when introducing major new patterns or architectural changes.**

---

## 8. Reference File/Directory Patterns

- Domain: `app/domain/entities/`, `app/domain/value_objects.py`
- Infra: `app/infrastructure/persistence/models`, `app/infrastructure/persistence/mappers/`
- Application: `app/application/` (service orchestration)
- API: `app/api/schemas/`, `app/api/v1/`

---

## 9. Performance & Testing

- Target: Sub-2s response for 2M+ profiles, test `pytest`
- Multi-tenant: Always respect tenant boundaries.
- Semantic cache: Use Redis+memory; invalidate on updates.

---

## 10. Example Prompt for Claude

> Follow the project architecture in CLAUDE.md—do NOT deviate.  
> Always clarify where code changes will occur before making them.  
> If a workflow is unclear, propose a numbered, step-by-step refactor plan.

