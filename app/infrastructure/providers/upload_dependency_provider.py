"""Upload dependency provider for creating UploadDependencies with all required services."""

from __future__ import annotations

from typing import Optional

from app.application.dependencies.upload_dependencies import UploadDependencies, IUploadDependencyFactory
from app.domain.interfaces import IFileResourceManager

# Import all required service providers
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.resource_provider import get_file_resource_service

# Create mock implementations for services that might not exist
from app.services.validation.webhook_validator import WebhookValidator
from app.services.document.document_processor import DocumentProcessor
from app.services.storage.local_storage import LocalStorageService

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
    async def analyze_quality(self, extracted_text: str, structured_data: dict, **kwargs) -> dict:
        return {"overall_score": 0.8, "quality_indicators": ["text_length"]}

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

class MockFileContentValidator:
    async def validate_upload_file(self, file, tenant_config=None):
        # Mock validation result
        class MockValidationResult:
            def __init__(self):
                self.is_valid = True
                self.file_size = getattr(file, 'size', len(getattr(file, 'file', b'')))
                self.validation_errors = []
                self.security_warnings = []
                self.confidence_score = 1.0
                self.validation_details = {}
        
        return MockValidationResult()


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
        
        # Create validation services
        webhook_validator = WebhookValidator()
        file_content_validator = MockFileContentValidator()
        
        # Create document processor and storage service
        document_processor = DocumentProcessor()
        storage_service = LocalStorageService()
        
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
            storage_service=storage_service,
            notification_service=notification_service,
            tenant_manager=tenant_manager,
            database_adapter=database_adapter,
            event_publisher=event_publisher,
            
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