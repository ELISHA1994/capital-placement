"""Upload dependency provider for creating UploadDependencies with all required services."""

from __future__ import annotations

from typing import Optional

from app.application.dependencies.upload_dependencies import UploadDependencies, IUploadDependencyFactory
from app.domain.interfaces import IFileResourceManager

# Import all required service providers
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.resource_provider import get_file_resource_service
from app.infrastructure.providers.storage_provider import get_file_storage
from app.infrastructure.providers.validation_provider import (
    get_file_content_validator_sync,
    get_webhook_validator_sync
)

# Import document processing services
from app.infrastructure.providers.document_provider import (
    get_document_processor,
    get_content_extractor,
    get_quality_analyzer,
    get_embedding_generator
)
from app.infrastructure.providers.ai_provider import get_embedding_service
from app.infrastructure.providers.audit_provider import get_audit_service
from app.infrastructure.task_manager import get_task_manager

# Mock implementations for missing services
class MockContentExtractor:
    async def extract_cv_data(self, text_content: str) -> dict:
        return {"mock": True, "text_length": len(text_content)}
    
    def prepare_text_for_embedding(self, text: str, structured_data: dict) -> str:
        return text[:1000]  # Truncate for embedding
    
    def hash_content(self, content: str) -> str:
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()

class MockQualityAnalyzer:
    async def analyze_document_quality(
        self,
        text: str,
        document_type: str,
        structured_data: Optional[dict] = None,
        use_ai: bool = True
    ) -> dict:
        return {
            "overall_score": 0.8,
            "quality_indicators": ["text_length"],
            "document_type": document_type,
            "is_acceptable": True
        }

class MockEmbeddingService:
    async def generate_embedding(self, text: str, **kwargs) -> list[float]:
        # Return mock embedding vector
        return [0.1] * 1536

class MockTenantManagerService:
    def __init__(self, **kwargs):
        pass
    
    async def get_tenant_configuration(self, tenant_id: str) -> dict:
        return {"allowed_file_extensions": [".pdf", ".doc", ".docx", ".txt"]}
    
    async def check_quota_limit(self, tenant_id: str, resource_type: str, **kwargs) -> dict:
        return {"allowed": True, "remaining": 1000}
    
    async def update_usage_metrics(self, tenant_id: str, metrics_update: dict) -> None:
        pass

class MockNotificationService:
    def __init__(self, **kwargs):
        pass
    
    async def send_webhook(self, url: str, payload: dict) -> None:
        pass
    
    async def send_email(self, to: str, subject: str, body: str) -> None:
        pass

class MockEventPublisher:
    def __init__(self, **kwargs):
        pass
    
    async def publish(self, event) -> None:
        pass

class MockRepository:
    def __init__(self, database_adapter):
        self.database_adapter = database_adapter



class UploadDependencyFactory(IUploadDependencyFactory):
    """Factory for creating upload dependencies with all required services."""
    
    def __init__(self):
        """Initialize the dependency factory."""
        self._dependencies_cache: Optional[UploadDependencies] = None
    
    async def create_dependencies(self) -> UploadDependencies:
        """
        Create and return upload dependencies.
        
        Returns:
            UploadDependencies: All dependencies required by UploadApplicationService
        """
        if self._dependencies_cache is not None:
            return self._dependencies_cache
        
        # Get database and core services
        database_adapter = await get_postgres_adapter()

        # Get resource management services
        file_resource_manager = await get_file_resource_service()

        # Get file storage service (uses hexagonal architecture adapter)
        storage_service = await get_file_storage()

        # Create service instances using mock implementations
        content_extractor = MockContentExtractor()
        quality_analyzer = MockQualityAnalyzer()
        embedding_service = MockEmbeddingService()

        tenant_manager = MockTenantManagerService()
        notification_service = MockNotificationService()
        event_publisher = MockEventPublisher()

        # Create repositories using mock implementation
        profile_repository = MockRepository(database_adapter)
        user_repository = MockRepository(database_adapter)
        tenant_repository = MockRepository(database_adapter)

        # Get validation services from providers
        webhook_validator = get_webhook_validator_sync()
        file_content_validator = get_file_content_validator_sync()

        # Get audit service
        audit_service = await get_audit_service()

        # Get task manager
        task_manager = get_task_manager()

        # Create document processor
        document_processor = DocumentProcessor()

        # Create dependencies
        self._dependencies_cache = UploadDependencies(
            # Repositories
            profile_repository=profile_repository,
            user_repository=user_repository,
            tenant_repository=tenant_repository,

            # Document processing services
            document_processor=document_processor,
            content_extractor=content_extractor,
            quality_analyzer=quality_analyzer,
            embedding_service=embedding_service,

            # Infrastructure services
            file_storage=storage_service,
            notification_service=notification_service,
            tenant_manager=tenant_manager,
            database_adapter=database_adapter,
            event_publisher=event_publisher,
            audit_service=audit_service,
            task_manager=task_manager,

            # Validation services
            webhook_validator=webhook_validator,
            file_content_validator=file_content_validator,

            # Resource management
            file_resource_manager=file_resource_manager,
        )
        
        return self._dependencies_cache
    
    def clear_cache(self) -> None:
        """Clear the cached dependencies (useful for testing)."""
        self._dependencies_cache = None


# Global factory instance
_upload_dependency_factory: Optional[UploadDependencyFactory] = None


async def get_upload_dependencies() -> UploadDependencies:
    """
    Get upload dependencies instance.
    
    Returns:
        UploadDependencies: All dependencies required by UploadApplicationService
    """
    global _upload_dependency_factory
    
    if _upload_dependency_factory is None:
        _upload_dependency_factory = UploadDependencyFactory()
    
    return await _upload_dependency_factory.create_dependencies()


def clear_upload_dependencies_cache() -> None:
    """Clear the upload dependencies cache."""
    global _upload_dependency_factory
    if _upload_dependency_factory is not None:
        _upload_dependency_factory.clear_cache()


__all__ = [
    "UploadDependencyFactory",
    "get_upload_dependencies", 
    "clear_upload_dependencies_cache",
]