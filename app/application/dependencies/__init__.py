"""Application service dependencies and factories."""

from .search_dependencies import (
    SearchDependencies,
    ISearchDependencyFactory,
    IHybridSearchService,
    IResultRerankerService,
    ITenantManagerService as ISearchTenantManagerService,
)

from .upload_dependencies import (
    UploadDependencies,
    IUploadDependencyFactory,
    IDocumentProcessor,
    IContentExtractor,
    IQualityAnalyzer,
    IEmbeddingService,
    INotificationService,
    ITenantManagerService as IUploadTenantManagerService,
    IDatabaseAdapter,
    IEventPublisher as IUploadEventPublisher
)

from .profile_dependencies import ProfileDependencies

__all__ = [
    # Search dependencies
    "SearchDependencies",
    "ISearchDependencyFactory",
    "IHybridSearchService",
    "IResultRerankerService",
    "ISearchTenantManagerService",

    # Upload dependencies
    "UploadDependencies",
    "IUploadDependencyFactory",
    "IDocumentProcessor",
    "IContentExtractor",
    "IQualityAnalyzer",
    "IEmbeddingService",
    "INotificationService",
    "IUploadTenantManagerService",
    "IDatabaseAdapter",
    "IUploadEventPublisher",

    # Profile dependencies
    "ProfileDependencies",
]
