"""Dependencies interface for UploadApplicationService."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

from app.domain.repositories.profile_repository import IProfileRepository
from app.domain.repositories.user_repository import IUserRepository
from app.domain.repositories.tenant_repository import ITenantRepository


@runtime_checkable
class IDocumentProcessor(Protocol):
    """Interface for document processing services."""
    
    async def process_document(self, file_content: bytes, filename: str, **kwargs) -> dict:
        """Process document and extract content."""
        ...


@runtime_checkable
class IContentExtractor(Protocol):
    """Interface for content extraction services."""
    
    async def extract_cv_data(self, text_content: str) -> dict:
        """Extract structured data from CV text."""
        ...
    
    def prepare_text_for_embedding(self, text: str, structured_data: dict) -> str:
        """Prepare text for embedding generation."""
        ...
    
    def hash_content(self, content: str) -> str:
        """Generate content hash."""
        ...


@runtime_checkable
class IQualityAnalyzer(Protocol):
    """Interface for document quality analysis."""
    
    async def analyze_quality(self, extracted_text: str, structured_data: dict, **kwargs) -> dict:
        """Analyze document quality."""
        ...


@runtime_checkable
class IEmbeddingService(Protocol):
    """Interface for embedding generation services."""
    
    async def generate_embedding(self, text: str, **kwargs) -> list[float]:
        """Generate embedding vector for text."""
        ...


@runtime_checkable
class IStorageService(Protocol):
    """Interface for file storage services."""
    
    async def store_file(self, file_content: bytes, filename: str, **kwargs) -> str:
        """Store file and return storage path."""
        ...
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        ...


@runtime_checkable
class INotificationService(Protocol):
    """Interface for notification services."""
    
    async def send_webhook(self, url: str, payload: dict) -> None:
        """Send webhook notification."""
        ...
    
    async def send_email(self, to: str, subject: str, body: str) -> None:
        """Send email notification."""
        ...


@runtime_checkable
class ITenantManagerService(Protocol):
    """Interface for tenant management services."""
    
    async def get_tenant_configuration(self, tenant_id: str) -> dict:
        """Get tenant configuration."""
        ...
    
    async def check_quota_limit(self, tenant_id: str, resource_type: str, **kwargs) -> dict:
        """Check quota limits."""
        ...
    
    async def update_usage_metrics(self, tenant_id: str, metrics_update: dict) -> None:
        """Update tenant usage metrics."""
        ...


@runtime_checkable
class IDatabaseAdapter(Protocol):
    """Interface for database operations."""
    
    async def execute(self, query: str, *params) -> Any:
        """Execute database query."""
        ...
    
    async def fetch_one(self, query: str, *params) -> Any:
        """Fetch single record."""
        ...
    
    async def fetch_all(self, query: str, *params) -> list:
        """Fetch multiple records."""
        ...


@runtime_checkable
class IEventPublisher(Protocol):
    """Interface for domain event publishing."""
    
    async def publish(self, event: Any) -> None:
        """Publish domain event."""
        ...


@dataclass
class UploadDependencies:
    """Dependencies required by UploadApplicationService."""
    
    # Repositories (domain layer)
    profile_repository: IProfileRepository
    user_repository: IUserRepository
    tenant_repository: ITenantRepository
    
    # Document processing services
    document_processor: IDocumentProcessor
    content_extractor: IContentExtractor
    quality_analyzer: IQualityAnalyzer
    embedding_service: IEmbeddingService
    
    # Infrastructure services
    storage_service: IStorageService
    notification_service: INotificationService
    tenant_manager: ITenantManagerService
    database_adapter: IDatabaseAdapter
    event_publisher: IEventPublisher


class IUploadDependencyFactory(ABC):
    """Abstract factory for creating upload dependencies."""
    
    @abstractmethod
    async def create_dependencies(self) -> UploadDependencies:
        """Create and return upload dependencies."""
        pass