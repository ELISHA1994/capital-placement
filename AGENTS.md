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

## Static Method Declaration Pattern - MANDATORY

**All application services MUST follow this strict pattern for declaring and organizing static methods.**

### Requirements

1. **Use `@staticmethod` decorator**: All static helper methods MUST use the `@staticmethod` decorator
2. **No self parameter**: Static methods should NOT have `self` as a parameter (only accept parameters they need)
3. **Bottom placement**: ALL static methods MUST be placed at the bottom of the class, before `__all__` exports
4. **Clear separation**: Static methods handle pure data transformation, validation, or parsing without requiring instance state
5. **Calling pattern**: Instance methods should call static methods using `self.static_method()` for consistency; static methods calling other static methods must use `ClassName.static_method()` (since they don't have `self`)

### When to Make a Method Static

A method should be static when it:
- Does NOT access `self._deps` or any instance attributes
- Performs pure data transformation or parsing
- Provides utility/helper functionality
- Does NOT require any injected dependencies
- Can be tested independently without class instantiation

### Correct Pattern

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
```

### Common Mistakes to Avoid

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

### Class Organization Template

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

### Benefits of This Pattern

✅ **Clear separation**: Instance methods vs pure functions clearly separated
✅ **Better testability**: Static methods can be tested without mocking dependencies
✅ **Improved readability**: Easy to identify which methods require dependencies
✅ **Consistent structure**: All service classes follow same organization pattern
✅ **Type safety**: Static methods are self-contained and easier to type-check
✅ **Reusability**: Static methods can be easily extracted to utility modules if needed

### Reference Implementations

See these files for correct static method patterns:
- `app/application/profile_service.py` - ProfileApplicationService (8 static methods at bottom)
- `app/application/upload_service.py` - UploadApplicationService (15+ static methods at bottom)
- `app/infrastructure/persistence/mappers/profile_mapper.py` - All mapper methods are static
