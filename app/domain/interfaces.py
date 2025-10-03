"""Domain-layer service interfaces.

These abstractions define the stable contracts that the application layer relies on,
while infrastructure adapters provide concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Union


class IHealthCheck:
    """Health check interface mixin."""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        pass


class ICacheService(IHealthCheck, ABC):
    """Cache service interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        pass


class IDocumentStore(IHealthCheck, ABC):
    """Document storage interface."""

    @abstractmethod
    async def store_document(self, container: str, path: str, content: bytes) -> str:
        """Store document and return URL."""
        pass

    @abstractmethod
    async def get_document(self, container: str, path: str) -> bytes:
        """Get document content."""
        pass

    @abstractmethod
    async def delete_document(self, container: str, path: str) -> bool:
        """Delete document."""
        pass

    @abstractmethod
    async def list_documents(self, container: str, prefix: str = "") -> List[str]:
        """List documents with optional prefix filter."""
        pass

    @abstractmethod
    async def get_document_url(self, container: str, path: str, expires_in: int = 3600) -> str:
        """Get signed URL for document access."""
        pass


class IDatabase(IHealthCheck, ABC):
    """Database interface for document operations."""

    @abstractmethod
    async def create_item(self, container: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item."""
        pass

    @abstractmethod
    async def get_item(self, container: str, item_id: str, partition_key: str) -> Optional[Dict[str, Any]]:
        """Get item by ID and partition key."""
        pass

    @abstractmethod
    async def update_item(self, container: str, item_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing item."""
        pass

    @abstractmethod
    async def delete_item(self, container: str, item_id: str, partition_key: str) -> bool:
        """Delete item."""
        pass

    @abstractmethod
    async def query_items(
        self,
        container: str,
        query: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query items with SQL-like syntax."""
        pass


class ISearchService(IHealthCheck, ABC):
    """Search service interface."""

    @abstractmethod
    async def create_index(self, index_name: str, fields: List[Dict[str, Any]]) -> bool:
        """Create search index."""
        pass

    @abstractmethod
    async def index_document(self, index_name: str, document: Dict[str, Any]) -> bool:
        """Index a single document."""
        pass

    @abstractmethod
    async def index_documents_batch(self, index_name: str, documents: List[Dict[str, Any]]) -> int:
        """Index multiple documents, return count of successful indexes."""
        pass

    @abstractmethod
    async def search(
        self,
        index_name: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top: int = 50,
    ) -> Dict[str, Any]:
        """Search documents."""
        pass

    @abstractmethod
    async def vector_search(self, index_name: str, vector: List[float], top: int = 50) -> Dict[str, Any]:
        """Vector similarity search."""
        pass

    @abstractmethod
    async def delete_document(self, index_name: str, document_id: str) -> bool:
        """Delete document from index."""
        pass


class IAIService(IHealthCheck, ABC):
    """AI service interface for embeddings and chat."""

    @abstractmethod
    async def generate_embedding(self, text: str, model: str = "default") -> List[float]:
        """Generate text embedding."""
        pass

    @abstractmethod
    async def generate_embeddings_batch(
        self, texts: List[str], model: str = "default"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        pass

    @abstractmethod
    async def extract_text_from_document(
        self,
        document_content: bytes,
        document_type: str = "pdf",
    ) -> Dict[str, Any]:
        """Extract text and metadata from document."""
        pass

    @abstractmethod
    def get_embedding_dimension(self, model: str = "default") -> int:
        """Get embedding vector dimension."""
        pass


@dataclass
class Message:
    """Lightweight message envelope used by queue adapters."""

    id: str
    body: str
    properties: Dict[str, Any] = field(default_factory=dict)
    delivery_count: int = 0


class IMessageQueue(IHealthCheck, ABC):
    """Message queue interface."""

    @abstractmethod
    async def send_message(self, queue_name: str, message: Dict[str, Any], delay: int = 0) -> str:
        """Send message to queue, return message ID."""
        pass

    @abstractmethod
    async def receive_messages(
        self,
        queue_name: str,
        max_messages: int = 1,
        visibility_timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        """Receive messages from queue."""
        pass

    @abstractmethod
    async def complete_message(self, queue_name: str, message_id: str, receipt_handle: str) -> bool:
        """Mark message as processed."""
        pass

    @abstractmethod
    async def abandon_message(self, queue_name: str, message_id: str, receipt_handle: str) -> bool:
        """Return message to queue."""
        pass

    @abstractmethod
    async def dead_letter_message(
        self,
        queue_name: str,
        message_id: str,
        receipt_handle: str,
        reason: str,
    ) -> bool:
        """Send message to dead letter queue."""
        pass


@dataclass
class BlobMetadata:
    """Metadata describing a stored blob."""

    name: str
    size: int
    content_type: Optional[str] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IBlobStorage(IHealthCheck, ABC):
    """Blob storage interface."""

    @abstractmethod
    async def upload_blob(
        self,
        container: str,
        blob_name: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload a blob and return its identifier or URL."""
        pass

    @abstractmethod
    async def download_blob(self, container: str, blob_name: str) -> bytes:
        """Download a blob's contents."""
        pass

    @abstractmethod
    async def delete_blob(self, container: str, blob_name: str) -> bool:
        """Delete a blob."""
        pass

    @abstractmethod
    async def list_blobs(
        self, container: str, prefix: Optional[str] = None
    ) -> List[BlobMetadata]:
        """List blobs within a container."""
        pass

    @abstractmethod
    async def get_blob_url(
        self,
        container: str,
        blob_name: str,
        expires_in: Optional[timedelta] = None,
    ) -> str:
        """Return an access URL for the blob."""
        pass


class IEventPublisher(IHealthCheck, ABC):
    """Publish events to downstream subscribers."""

    @abstractmethod
    async def publish_event(self, topic: str, event_data: Dict[str, Any]) -> bool:
        """Publish a single event."""
        pass

    @abstractmethod
    async def publish_events(self, topic: str, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events."""
        pass


class INotificationService(IHealthCheck, ABC):
    """Notification service interface."""

    @abstractmethod
    async def send_email(self, to: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email notification."""
        pass

    @abstractmethod
    async def send_webhook(
        self, url: str, payload: Dict[str, Any], secret: Optional[str] = None
    ) -> bool:
        """Send webhook notification."""
        pass

    @abstractmethod
    async def send_push_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send push notification."""
        pass


class IAnalyticsService(IHealthCheck, ABC):
    """Analytics and metrics interface."""

    @abstractmethod
    async def track_event(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> bool:
        """Track analytics event."""
        pass

    @abstractmethod
    async def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Increment counter metric."""
        pass

    @abstractmethod
    async def record_timing(
        self,
        metric_name: str,
        duration_ms: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Record timing metric."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get metrics data."""
        pass


class ISecretManager(IHealthCheck, ABC):
    """Secret management interface."""

    @abstractmethod
    async def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve a secret value."""
        pass

    @abstractmethod
    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Store or update a secret value."""
        pass

    @abstractmethod
    async def delete_secret(self, secret_name: str) -> bool:
        """Delete a stored secret."""
        pass

    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List available secret identifiers."""
        pass


class IServiceRegistry(ABC):
    """Service registry for managing service instances."""

    @abstractmethod
    def register_service(
        self, interface_type: type, implementation: Any, priority: int = 0
    ) -> None:
        """Register service implementation."""
        pass

    @abstractmethod
    def get_service(self, interface_type: type) -> Any:
        """Get service implementation."""
        pass

    @abstractmethod
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all registered services."""
        pass


class ISearchAnalyticsService(IHealthCheck, ABC):
    """Search analytics service interface."""

    @abstractmethod
    async def track_search_event(
        self,
        event_type: str,
        search_data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Track a search event."""
        pass

    @abstractmethod
    async def get_search_metrics(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get search metrics for a time period."""
        pass

    @abstractmethod
    async def get_popular_queries(
        self,
        tenant_id: str,
        limit: int = 10,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get popular search queries."""
        pass


class IContentExtractor(IHealthCheck, ABC):
    """Content extraction service interface."""

    @abstractmethod
    async def extract_cv_data(self, text_content: str) -> Dict[str, Any]:
        """Extract structured data from CV text."""
        pass

    @abstractmethod
    def prepare_text_for_embedding(
        self, text_content: str, structured_data: Dict[str, Any]
    ) -> str:
        """Prepare text for embedding generation."""
        pass

    @abstractmethod
    def hash_content(self, content: str) -> str:
        """Generate content hash."""
        pass


class IPDFProcessor(IHealthCheck, ABC):
    """PDF processing service interface."""

    @abstractmethod
    async def extract_content(self, file_content: bytes) -> Dict[str, Any]:
        """Extract text and metadata from PDF."""
        pass

    @abstractmethod
    async def validate_pdf(self, file_content: bytes) -> bool:
        """Validate PDF file format."""
        pass


class IQualityAnalyzer(IHealthCheck, ABC):
    """Document quality analysis service interface."""

    @abstractmethod
    async def analyze_quality(
        self,
        extracted_text: str,
        structured_data: Dict[str, Any],
        document_type: str = "cv",
    ) -> Dict[str, Any]:
        """Analyze document quality."""
        pass

    @abstractmethod
    async def get_quality_score(
        self,
        text_content: str,
        metadata: Dict[str, Any],
    ) -> float:
        """Get quality score for content."""
        pass


class IAuthenticationService(IHealthCheck, ABC):
    """Authentication service interface."""

    @abstractmethod
    async def register_user(self, user_data: Any) -> Any:
        """Register a new user."""
        pass

    @abstractmethod
    async def authenticate(self, credentials: Any) -> Any:
        """Authenticate user credentials."""
        pass

    @abstractmethod
    async def refresh_tokens(self, request_data: Any) -> Any:
        """Refresh authentication tokens."""
        pass

    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token."""
        pass

    @abstractmethod
    async def update_user_profile(self, current_user: Any, update_request: Any) -> Any:
        """Update user profile."""
        pass

    @abstractmethod
    async def change_password(self, user_id: str, request_data: Any) -> bool:
        """Change user password."""
        pass

    @abstractmethod
    async def request_password_reset(self, request_data: Any) -> Optional[Dict[str, Any]]:
        """Request password reset."""
        pass

    @abstractmethod
    async def confirm_password_reset(self, request_data: Any) -> bool:
        """Confirm password reset."""
        pass

    @abstractmethod
    async def create_api_key(self, request_data: Any) -> Any:
        """Create API key."""
        pass

    @abstractmethod
    async def list_sessions(self, user_id: str) -> List[Any]:
        """List user sessions."""
        pass

    @abstractmethod
    async def terminate_session(self, session_id: str, user_id: str) -> bool:
        """Terminate user session."""
        pass


class IAuthorizationService(IHealthCheck, ABC):
    """Authorization service interface."""

    @abstractmethod
    async def check_permission(
        self,
        user_roles: List[str],
        required_permission: str,
        tenant_id: str,
    ) -> bool:
        """Check if user has required permission."""
        pass

    @abstractmethod
    async def get_user_permissions(
        self,
        user_roles: List[str],
        tenant_id: str,
    ) -> List[str]:
        """Get all user permissions."""
        pass


class IBootstrapService(IHealthCheck, ABC):
    """Bootstrap service interface."""

    @abstractmethod
    async def initialize_system(self, admin_data: Any) -> Any:
        """Initialize system with admin user."""
        pass

    @abstractmethod
    async def is_system_initialized(self) -> bool:
        """Check if system is initialized."""
        pass


class IUsageService(IHealthCheck, ABC):
    """Usage tracking service interface."""

    @abstractmethod
    async def track_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Track resource usage."""
        pass

    @abstractmethod
    async def get_usage_stats(
        self,
        tenant_id: str,
        resource_type: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get usage statistics."""
        pass

    @abstractmethod
    async def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        requested_amount: int,
    ) -> Dict[str, Any]:
        """Check quota limits."""
        pass


__all__ = [
    "IHealthCheck",
    "ICacheService",
    "IDocumentStore",
    "IDatabase",
    "ISearchService",
    "IAIService",
    "Message",
    "BlobMetadata",
    "IMessageQueue",
    "IEventPublisher",
    "IBlobStorage",
    "ISecretManager",
    "INotificationService",
    "IAnalyticsService",
    "IServiceRegistry",
    "ISearchAnalyticsService",
    "IContentExtractor",
    "IPDFProcessor",
    "IQualityAnalyzer",
    "IAuthenticationService",
    "IAuthorizationService",
    "IBootstrapService",
    "IUsageService",
]
