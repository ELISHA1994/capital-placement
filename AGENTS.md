# Repository Guidelines

## Project Structure & Module Organization
- `app/api/` hosts FastAPI routers, while orchestration sits in `app/application/` and contract definitions in `app/domain/`.
- Infrastructure providers live in `app/infrastructure/providers/`; persistence wiring is in `app/services/adapters/` and raw database access in `app/database/`.
- SQLModel entities reside in `app/models/`, with shared config and factories in `app/core/`.
- Tests mirror this layout under `tests/`, and operational scripts live in `scripts/`.

## Build, Test, and Development Commands
- `pip install -e .[local,ai]` installs the editable package with local tooling and AI extras.
- `python -m uvicorn app.main:app --reload --port 8000` starts a hot-reloading API server on port 8000.
- `alembic upgrade head` applies the latest migrations against PostgreSQL.
- `pytest` runs the full suite; use `pytest -m "not slow"` for a quicker pass during iteration.
- `ruff check .` and `ruff format .` enforce linting and formatting, while `mypy app` validates typing.

## Coding Style & Naming Conventions
- Target Python 3.12 with four-space indentation and 88-character lines.
- Follow `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Prefer dependency providers from `app/core/dependencies.py` (e.g., `get_*_service`) instead of instantiating adapters directly.
- Run `ruff` before sending changes to keep imports ordered and formatting synchronized.

## Testing Guidelines
- Pytest is the primary framework; markers are configured in `pyproject.toml`.
- Name files `test_<area>.py` and test functions `test_<scenario>`.
- Share fixtures from `tests/conftest.py`, especially `reset_provider_state`, to avoid singleton leakage.
- Focus coverage on new `app/` modules and add integration tests for new application services when feasible.

## Commit & Pull Request Guidelines
- Write imperative, sentence-case commit messages (e.g., "Introduce event publisher provider").
- Scope commits to related changes and reference issue IDs when available.
- PRs should state intent, list validation commands (pytest, ruff, mypy, migrations), and attach API traces or screenshots for visible changes.
- Call out schema or configuration updates and include the relevant Alembic revision ID in the description.

## Security & Configuration Tips
- Never commit secrets; base local setup on `.env.local` and generate a unique `SECRET_KEY`.
- Preserve tenant and auth guards when touching related services, and review migrations for cascading effects before deployment.
- Keep environment credentials out of version control, and rotate keys immediately if exposure is suspected.

## Hexagonal Architecture Boundaries
- Domain modules (`app/domain/`) must stay free of application, API, or infrastructure imports.
- Application services can depend only on domain ports/value objects; resolve implementations via `app/infrastructure/providers/` rather than direct adapter imports.
- Infrastructure adapters implement domain interfaces; avoid referencing application logic.
- **API controllers call application services only—never reach directly into infrastructure.**
  - **CRITICAL**: Business logic must NOT reside in API routers. API endpoints should be thin controllers that only handle HTTP concerns (request parsing, response formatting, status codes).
  - All business logic, orchestration, and decision-making must be in the application service layer.
  - API routers should call a single service method and map the result to HTTP responses.
- Any new service should expose a port in domain/application and register its adapter in a provider to keep dependencies directional.
- Prefer ORM/SQLModel queries over raw SQL for database access unless a provider explicitly exposes a raw command helper.
