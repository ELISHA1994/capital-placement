"""Concrete factory for creating UploadApplicationService dependencies."""

from __future__ import annotations

from app.application.dependencies import IUploadDependencyFactory, UploadDependencies
from app.infrastructure.adapters.notification_adapter import NotificationAdapter
from app.infrastructure.persistence.repositories import (
    PostgresProfileRepository,
    PostgresTenantRepository,
    PostgresUserRepository,
)
from app.infrastructure.providers.ai_provider import (
    get_embedding_service,
)
from app.infrastructure.providers.audit_provider import get_audit_service
from app.infrastructure.providers.document_provider import (
    get_content_extractor,
    get_pdf_processor,
    get_quality_analyzer,
)
from app.infrastructure.providers.event_provider import get_event_publisher
from app.infrastructure.providers.postgres_provider import get_postgres_adapter
from app.infrastructure.providers.resource_provider import get_file_resource_service
from app.infrastructure.providers.storage_provider import get_file_storage
from app.infrastructure.providers.tenant_provider import (
    get_tenant_service as get_tenant_manager,
)
from app.infrastructure.providers.validation_provider import (
    get_file_content_validator,
    get_webhook_validator,
)
from app.infrastructure.task_manager import get_task_manager


class UploadDependencyFactory(IUploadDependencyFactory):
    """Concrete factory for creating upload dependencies using current providers."""

    async def create_dependencies(self) -> UploadDependencies:
        """Create and return upload dependencies."""

        # Repository implementations
        profile_repository = PostgresProfileRepository()
        user_repository = PostgresUserRepository()
        tenant_repository = PostgresTenantRepository()

        # Document processing services (via providers)
        pdf_processor = await get_pdf_processor()
        content_extractor = await get_content_extractor()
        quality_analyzer = await get_quality_analyzer()

        # AI services
        embedding_service = await get_embedding_service()

        # Infrastructure services
        notification_service = NotificationAdapter()
        tenant_manager = await get_tenant_manager()
        database_adapter = await get_postgres_adapter()
        file_storage = await get_file_storage()

        # Event publisher
        event_publisher = await get_event_publisher()

        # Validation services
        webhook_validator = await get_webhook_validator()
        file_content_validator = await get_file_content_validator()

        # Resource management
        file_resource_manager = await get_file_resource_service()

        # Audit service
        audit_service = await get_audit_service()

        # Task manager
        task_manager = get_task_manager()

        # Wrap document processor to use PDFProcessor
        document_processor = DocumentProcessorAdapter(
            pdf_processor=pdf_processor,
            content_extractor=content_extractor
        )

        return UploadDependencies(
            # Repositories
            profile_repository=profile_repository,
            user_repository=user_repository,
            tenant_repository=tenant_repository,

            # Document processing
            document_processor=document_processor,
            content_extractor=content_extractor,
            quality_analyzer=quality_analyzer,
            embedding_service=embedding_service,

            # Infrastructure
            notification_service=notification_service,
            tenant_manager=tenant_manager,
            database_adapter=database_adapter,
            event_publisher=event_publisher,
            file_storage=file_storage,
            audit_service=audit_service,
            task_manager=task_manager,

            # Validation services
            webhook_validator=webhook_validator,
            file_content_validator=file_content_validator,

            # Resource management
            file_resource_manager=file_resource_manager
        )


class DocumentProcessorAdapter:
    """Adapter to make PDFProcessor compatible with IDocumentProcessor interface."""

    def __init__(self, pdf_processor, content_extractor):
        self.pdf_processor = pdf_processor
        self.content_extractor = content_extractor

    async def process_document(self, file_content: bytes, filename: str, **kwargs) -> dict:
        """Process document and extract content."""
        file_extension = filename.lower().split(".")[-1] if "." in filename else "unknown"

        if file_extension == "pdf":
            # Call process_pdf method which returns a PDFDocument object
            pdf_document = await self.pdf_processor.process_pdf(
                pdf_content=file_content,
                filename=filename
            )
            return {
                "text": pdf_document.full_text,
                "metadata": {
                    **pdf_document.metadata,
                    **pdf_document.processing_info
                }
            }
        else:
            # Handle text files
            try:
                text_content = file_content.decode("utf-8")
                metadata = {"file_type": file_extension}
                return {
                    "text": text_content,
                    "metadata": metadata
                }
            except UnicodeDecodeError:
                text_content = file_content.decode("utf-8", errors="ignore")
                metadata = {"file_type": file_extension, "encoding_issues": True}
                return {
                    "text": text_content,
                    "metadata": metadata
                }


# Singleton instance for global usage
_upload_dependency_factory: UploadDependencyFactory | None = None


async def get_upload_dependency_factory() -> UploadDependencyFactory:
    """Get singleton instance of upload dependency factory."""
    global _upload_dependency_factory
    if _upload_dependency_factory is None:
        _upload_dependency_factory = UploadDependencyFactory()
    return _upload_dependency_factory


async def get_upload_dependencies() -> UploadDependencies:
    """Helper function to get upload dependencies directly."""
    factory = await get_upload_dependency_factory()
    return await factory.create_dependencies()


__all__ = ["UploadDependencyFactory", "get_upload_dependency_factory", "get_upload_dependencies", "DocumentProcessorAdapter"]
