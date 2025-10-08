"""Tests to enforce hexagonal architecture boundaries."""

import ast
import os
from pathlib import Path
from typing import Set, List, Dict
import pytest


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to collect import statements, excluding TYPE_CHECKING blocks."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.in_type_checking = False

    def visit_If(self, node):
        """Handle if statements, specifically TYPE_CHECKING blocks."""
        # Check if this is a TYPE_CHECKING block
        is_type_checking = False
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            is_type_checking = True

        if is_type_checking:
            # Skip visiting children of TYPE_CHECKING blocks
            return
        else:
            # Continue visiting non-TYPE_CHECKING if blocks
            self.generic_visit(node)

    def visit_Import(self, node):
        if not self.in_type_checking:
            for alias in node.names:
                self.imports.add(alias.name)

    def visit_ImportFrom(self, node):
        if not self.in_type_checking and node.module:
            self.imports.add(node.module)


def get_imports_from_file(file_path: Path) -> Set[str]:
    """Extract imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except (SyntaxError, UnicodeDecodeError):
        return set()


def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in a directory recursively."""
    if not directory.exists():
        return []
    
    python_files = []
    for file_path in directory.rglob("*.py"):
        if file_path.name != "__init__.py":  # Skip __init__.py files
            python_files.append(file_path)
    return python_files


class TestHexagonalBoundaries:
    """Test hexagonal architecture boundary violations."""
    
    @pytest.fixture(scope="class")
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent / "app"
    
    def test_domain_layer_purity(self, project_root):
        """Test that domain layer has no infrastructure dependencies."""
        domain_dir = project_root / "domain"
        if not domain_dir.exists():
            pytest.skip("Domain directory not found")
        
        domain_files = get_python_files(domain_dir)
        violations = []
        
        for file_path in domain_files:
            imports = get_imports_from_file(file_path)
            
            # Check for violations
            for import_stmt in imports:
                # Domain should not import from infrastructure or services
                if (import_stmt.startswith("app.infrastructure") or 
                    import_stmt.startswith("app.services") or
                    import_stmt.startswith("app.models") or
                    import_stmt.startswith("app.api")):
                    
                    relative_path = file_path.relative_to(project_root)
                    violations.append(f"{relative_path}: imports {import_stmt}")
        
        if violations:
            pytest.fail(
                f"Domain layer boundary violations found:\n" + 
                "\n".join(violations)
            )
    
    def test_application_layer_depends_only_on_abstractions(self, project_root):
        """Test that application layer only depends on domain interfaces."""
        application_dir = project_root / "application"
        if not application_dir.exists():
            pytest.skip("Application directory not found")
        
        application_files = get_python_files(application_dir)
        violations = []
        
        for file_path in application_files:
            imports = get_imports_from_file(file_path)
            
            # Check for violations
            for import_stmt in imports:
                # Application should not import concrete implementations
                if (import_stmt.startswith("app.services") and 
                    not import_stmt.startswith("app.services.adapters") and
                    "provider" not in import_stmt):
                    
                    # Allow imports from specific service modules that are interfaces
                    allowed_service_imports = {
                        "app.services.search.hybrid_search",  # For type annotations
                        "app.services.search.result_reranker",  # For type annotations
                        "app.services.search.vector_search",  # For type annotations
                    }
                    
                    if import_stmt not in allowed_service_imports:
                        relative_path = file_path.relative_to(project_root)
                        violations.append(f"{relative_path}: imports {import_stmt}")
        
        if violations:
            pytest.fail(
                f"Application layer boundary violations found:\n" + 
                "\n".join(violations)
            )
    
    def test_infrastructure_layer_isolation(self, project_root):
        """Test that infrastructure implementations don't leak to domain."""
        infrastructure_dir = project_root / "infrastructure"
        if not infrastructure_dir.exists():
            pytest.skip("Infrastructure directory not found")
        
        # Infrastructure can import from anywhere, but domain should not import infrastructure
        # This is covered by test_domain_layer_purity
        pass
    
    def test_no_circular_dependencies(self, project_root):
        """Test for circular dependencies between layers."""
        # This is a simplified check - a full dependency analysis would be more complex
        
        layer_dirs = {
            "domain": project_root / "domain",
            "application": project_root / "application", 
            "infrastructure": project_root / "infrastructure",
            "api": project_root / "api"
        }
        
        layer_imports = {}
        
        for layer_name, layer_dir in layer_dirs.items():
            if layer_dir.exists():
                files = get_python_files(layer_dir)
                imports = set()
                
                for file_path in files:
                    file_imports = get_imports_from_file(file_path)
                    imports.update(file_imports)
                
                layer_imports[layer_name] = imports
        
        # Check dependency rules
        violations = []
        
        # Domain should not depend on any other application layers
        if "domain" in layer_imports:
            for import_stmt in layer_imports["domain"]:
                if (import_stmt.startswith("app.application") or
                    import_stmt.startswith("app.infrastructure") or
                    import_stmt.startswith("app.api")):
                    violations.append(f"Domain imports {import_stmt}")
        
        # Application should not depend on infrastructure (except through interfaces)
        if "application" in layer_imports:
            for import_stmt in layer_imports["application"]:
                if (import_stmt.startswith("app.infrastructure") and
                    not import_stmt.startswith("app.infrastructure.providers")):
                    violations.append(f"Application imports {import_stmt}")
        
        if violations:
            pytest.fail(
                f"Circular dependency violations found:\n" + 
                "\n".join(violations)
            )
    
    def test_deprecated_imports_not_used(self, project_root):
        """Test that deprecated imports are not used."""
        all_files = []
        for root_dir in [project_root / "domain", project_root / "application", 
                        project_root / "infrastructure", project_root / "api"]:
            if root_dir.exists():
                all_files.extend(get_python_files(root_dir))
        
        violations = []
        deprecated_imports = {
            "app.core.interfaces",  # Should use app.domain.interfaces
            "app.services.providers",  # Should use app.infrastructure.providers
            "app.core.container",  # Removed in favor of provider pattern
        }
        
        for file_path in all_files:
            imports = get_imports_from_file(file_path)
            
            for import_stmt in imports:
                if import_stmt in deprecated_imports:
                    relative_path = file_path.relative_to(project_root)
                    violations.append(f"{relative_path}: imports deprecated {import_stmt}")
        
        if violations:
            pytest.fail(
                f"Deprecated import violations found:\n" + 
                "\n".join(violations)
            )
    
    def test_provider_pattern_compliance(self, project_root):
        """Test that services are accessed through providers."""
        application_dir = project_root / "application"
        api_dir = project_root / "api"
        
        if not application_dir.exists() and not api_dir.exists():
            pytest.skip("Application or API directories not found")
        
        files_to_check = []
        if application_dir.exists():
            files_to_check.extend(get_python_files(application_dir))
        if api_dir.exists():
            files_to_check.extend(get_python_files(api_dir))
        
        violations = []
        
        for file_path in files_to_check:
            imports = get_imports_from_file(file_path)
            
            # Check for direct service imports (should use providers instead)
            for import_stmt in imports:
                if (import_stmt.startswith("app.services") and 
                    not import_stmt.startswith("app.services.providers") and
                    not import_stmt.startswith("app.services.adapters") and
                    "provider" not in import_stmt):
                    
                    # Allow some exceptions for type hints and configurations
                    allowed_exceptions = {
                        "app.services.search.hybrid_search",  # For config classes
                        "app.services.search.result_reranker",  # For config classes
                        "app.services.search.vector_search",  # For filter classes
                        "app.services.document",  # Document processing classes
                    }
                    
                    is_exception = any(import_stmt.startswith(exc) for exc in allowed_exceptions)
                    
                    if not is_exception:
                        relative_path = file_path.relative_to(project_root)
                        violations.append(f"{relative_path}: direct service import {import_stmt}")
        
        if violations:
            pytest.fail(
                f"Provider pattern violations found:\n" + 
                "\n".join(violations)
            )


class TestDomainModelPurity:
    """Test domain model purity and isolation."""
    
    @pytest.fixture(scope="class")
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent / "app"
    
    def test_domain_entities_are_pure(self, project_root):
        """Test that domain entities don't import SQLModel or database concerns."""
        entities_dir = project_root / "domain" / "entities"
        if not entities_dir.exists():
            pytest.skip("Domain entities directory not found")
        
        entity_files = get_python_files(entities_dir)
        violations = []
        
        forbidden_imports = {
            "sqlmodel", "sqlalchemy", "asyncpg", "psycopg2", 
            "redis", "pymongo", "app.models"
        }
        
        for file_path in entity_files:
            imports = get_imports_from_file(file_path)
            
            for import_stmt in imports:
                for forbidden in forbidden_imports:
                    if forbidden in import_stmt:
                        relative_path = file_path.relative_to(project_root)
                        violations.append(f"{relative_path}: imports {import_stmt}")
        
        if violations:
            pytest.fail(
                f"Domain entity purity violations found:\n" + 
                "\n".join(violations)
            )
    
    def test_value_objects_are_immutable(self, project_root):
        """Test that value objects follow immutability principles."""
        value_objects_file = project_root / "domain" / "value_objects.py"
        if not value_objects_file.exists():
            pytest.skip("Value objects file not found")
        
        # This is a basic check - a full immutability test would require more analysis
        imports = get_imports_from_file(value_objects_file)
        
        # Value objects should use dataclasses with frozen=True or similar
        has_dataclass_import = any("dataclass" in imp for imp in imports)
        
        if not has_dataclass_import:
            pytest.fail("Value objects should use dataclasses for immutability")


if __name__ == "__main__":
    pytest.main([__file__])