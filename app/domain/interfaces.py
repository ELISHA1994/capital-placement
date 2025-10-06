"""Domain-layer service interfaces.

These abstractions define the stable contracts that the application layer relies on,
while infrastructure adapters provide concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Union
from enum import Enum


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
    async def process_pdf(
        self,
        pdf_content: bytes,
        filename: Optional[str] = None,
        extract_metadata: bool = True,
        validate_content: bool = True
    ) -> Any:  # Returns PDFDocument object
        """Process PDF document and extract content."""
        pass

    @abstractmethod
    async def validate_pdf(self, file_content: bytes) -> bool:
        """Validate PDF file format."""
        pass


class IQualityAnalyzer(IHealthCheck, ABC):
    """Document quality analysis service interface."""

    @abstractmethod
    async def analyze_document_quality(
        self,
        text: str,
        document_type: str,
        structured_data: Optional[Dict[str, Any]] = None,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality analysis on a document.

        Args:
            text: Document text content
            document_type: Type of document being analyzed (e.g., 'cv', 'job_description')
            structured_data: Optional structured data from content extraction
            use_ai: Whether to use AI for advanced quality assessment

        Returns:
            Quality assessment dictionary with scores and recommendations
        """
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


class IWebhookValidator(ABC):
    """Webhook URL validation service interface."""

    @abstractmethod
    def validate_webhook_url(self, url: str) -> None:
        """
        Validate webhook URL against security policies.
        
        Raises WebhookValidationError if URL is invalid or poses security risks.
        
        Args:
            url: The webhook URL to validate
            
        Raises:
            WebhookValidationError: If URL validation fails
        """
        pass

    @abstractmethod
    def is_allowed_scheme(self, scheme: str) -> bool:
        """Check if URL scheme is allowed."""
        pass

    @abstractmethod  
    def is_blocked_host(self, host: str) -> bool:
        """Check if host/domain is blocked."""
        pass

    @abstractmethod
    def is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private/internal."""
        pass


class IAuditService(IHealthCheck, ABC):
    """
    Audit logging service interface for security compliance and tamper-resistant logging.
    
    Provides comprehensive audit logging capabilities with tamper resistance,
    compliance reporting, and security event tracking.
    """

    @abstractmethod
    async def log_event(
        self,
        event_type: str,
        tenant_id: str,
        action: str,
        resource_type: str,
        *,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        risk_level: str = "low",
        suspicious: bool = False,
        correlation_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Log an audit event with comprehensive details.
        
        Args:
            event_type: Type of audit event (from AuditEventType enum)
            tenant_id: Tenant identifier
            action: Specific action performed
            resource_type: Type of resource affected
            user_id: User who performed the action (optional)
            user_email: Email of user who performed the action (optional)
            session_id: Session identifier (optional)
            api_key_id: API key used (optional)
            resource_id: Identifier of affected resource (optional)
            details: Additional event details (optional)
            ip_address: Client IP address
            user_agent: Client user agent string
            risk_level: Risk level (low, medium, high, critical)
            suspicious: Whether event is flagged as suspicious
            correlation_id: ID for correlating related events (optional)
            batch_id: Batch identifier for grouped operations (optional)
            error_code: Error code if operation failed (optional)
            error_message: Error message if operation failed (optional)
            
        Returns:
            Audit log entry ID
            
        Raises:
            AuditServiceError: If audit logging fails
        """
        pass

    @abstractmethod
    async def log_authentication_event(
        self,
        event_type: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        success: bool = True,
        failure_reason: Optional[str] = None,
        additional_details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log authentication-specific events with standardized details.
        
        Args:
            event_type: Authentication event type
            tenant_id: Tenant identifier
            user_id: User identifier (optional)
            user_email: User email (optional)
            session_id: Session identifier (optional)
            ip_address: Client IP address
            user_agent: Client user agent
            success: Whether authentication was successful
            failure_reason: Reason for failure (if applicable)
            additional_details: Additional event details
            
        Returns:
            Audit log entry ID
        """
        pass

    @abstractmethod
    async def log_file_upload_event(
        self,
        event_type: str,
        tenant_id: str,
        user_id: str,
        filename: str,
        file_size: int,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        validation_errors: Optional[List[str]] = None,
        security_warnings: Optional[List[str]] = None,
        processing_duration_ms: Optional[int] = None,
        batch_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Log file upload and processing events with file-specific details.
        
        Args:
            event_type: Upload event type
            tenant_id: Tenant identifier
            user_id: User who uploaded the file
            filename: Name of uploaded file
            file_size: Size of uploaded file in bytes
            upload_id: Unique upload identifier
            session_id: Session identifier (optional)
            ip_address: Client IP address
            user_agent: Client user agent
            validation_errors: File validation errors (optional)
            security_warnings: Security warnings detected (optional)
            processing_duration_ms: Processing time in milliseconds (optional)
            batch_id: Batch identifier for batch uploads (optional)
            error_message: Error message if upload failed (optional)
            
        Returns:
            Audit log entry ID
        """
        pass

    @abstractmethod
    async def log_security_event(
        self,
        event_type: str,
        tenant_id: str,
        threat_type: str,
        severity: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        threat_details: Optional[Dict[str, Any]] = None,
        mitigation_actions: Optional[List[str]] = None,
    ) -> str:
        """
        Log security-related events with threat analysis details.
        
        Args:
            event_type: Security event type
            tenant_id: Tenant identifier
            threat_type: Type of security threat detected
            severity: Severity level of the threat
            user_id: User involved (optional)
            session_id: Session identifier (optional)
            resource_id: Affected resource (optional)
            ip_address: Client IP address
            user_agent: Client user agent
            threat_details: Detailed threat information (optional)
            mitigation_actions: Actions taken to mitigate threat (optional)
            
        Returns:
            Audit log entry ID
        """
        pass

    @abstractmethod
    async def query_audit_logs(
        self,
        tenant_id: str,
        *,
        user_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        suspicious_only: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        page: int = 1,
        size: int = 50,
    ) -> Dict[str, Any]:
        """
        Query audit logs with filtering and pagination.
        
        Args:
            tenant_id: Tenant identifier
            user_id: Filter by user ID (optional)
            event_types: Filter by event types (optional)
            resource_type: Filter by resource type (optional)
            resource_id: Filter by resource ID (optional)
            risk_level: Filter by risk level (optional)
            suspicious_only: Show only suspicious events
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            correlation_id: Filter by correlation ID (optional)
            batch_id: Filter by batch ID (optional)
            ip_address: Filter by IP address (optional)
            page: Page number for pagination
            size: Page size for pagination
            
        Returns:
            Dictionary containing audit logs and pagination info
        """
        pass

    @abstractmethod
    async def get_audit_statistics(
        self,
        tenant_id: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get audit log statistics for compliance reporting.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Start time for statistics (optional)
            end_time: End time for statistics (optional)
            
        Returns:
            Dictionary containing audit statistics
        """
        pass

    @abstractmethod
    async def verify_log_integrity(
        self,
        tenant_id: str,
        log_id: str,
    ) -> Dict[str, Any]:
        """
        Verify the integrity of a specific audit log entry.
        
        Args:
            tenant_id: Tenant identifier
            log_id: Audit log entry ID
            
        Returns:
            Dictionary containing integrity verification results
        """
        pass

    @abstractmethod
    async def export_audit_logs(
        self,
        tenant_id: str,
        format: str = "json",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
    ) -> bytes:
        """
        Export audit logs for compliance reporting.
        
        Args:
            tenant_id: Tenant identifier
            format: Export format (json, csv, xml)
            start_time: Start time for export (optional)
            end_time: End time for export (optional)
            event_types: Filter by event types (optional)
            
        Returns:
            Exported audit logs as bytes
        """
        pass


@dataclass
class FileValidationResult:
    """Result of comprehensive file content validation."""
    
    is_valid: bool
    filename: str
    file_size: int
    detected_mime_type: Optional[str] = None
    detected_extension: Optional[str] = None
    file_signature: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileTypeConfig:
    """Configuration for allowed file types with comprehensive validation rules."""
    
    extension: str
    mime_types: List[str]
    magic_bytes_patterns: List[bytes]
    max_size_mb: Optional[int] = None
    description: str = ""
    security_level: str = "standard"  # standard, strict, permissive
    allow_binary_content: bool = True


class IFileContentValidator(IHealthCheck, ABC):
    """
    Comprehensive file content validation service interface.
    
    Provides multi-layered validation including:
    - MIME type validation from Content-Type header
    - File signature validation using magic bytes
    - Extension vs content consistency checks  
    - Security threat detection
    - Malicious file pattern detection
    """

    @abstractmethod
    async def validate_file_content(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> FileValidationResult:
        """
        Perform comprehensive file content validation.
        
        Args:
            file_content: Raw file content bytes
            filename: Original filename with extension
            content_type: MIME type from Content-Type header
            tenant_config: Tenant-specific validation configuration
            
        Returns:
            FileValidationResult with validation details and results
        """
        pass

    @abstractmethod
    async def validate_upload_file(
        self,
        file: Any,  # UploadFile from FastAPI
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> FileValidationResult:
        """
        Validate uploaded file without loading entire content into memory.
        
        Args:
            file: FastAPI UploadFile instance
            tenant_config: Tenant-specific validation configuration
            
        Returns:
            FileValidationResult with validation details
        """
        pass

    @abstractmethod
    def detect_file_type_from_content(self, file_content: bytes) -> Optional[str]:
        """
        Detect file type from magic bytes/file signature.
        
        Args:
            file_content: Raw file content bytes
            
        Returns:
            Detected file extension or None if unknown
        """
        pass

    @abstractmethod
    def detect_mime_type_from_content(self, file_content: bytes) -> Optional[str]:
        """
        Detect MIME type from file content analysis.
        
        Args:
            file_content: Raw file content bytes
            
        Returns:
            Detected MIME type or None if unknown
        """
        pass

    @abstractmethod
    def validate_file_signature(
        self,
        file_content: bytes,
        expected_extension: str,
    ) -> bool:
        """
        Validate file signature matches expected extension.
        
        Args:
            file_content: Raw file content bytes
            expected_extension: Expected file extension (e.g., '.pdf')
            
        Returns:
            True if signature matches, False otherwise
        """
        pass

    @abstractmethod
    def cross_validate_file_properties(
        self,
        filename: str,
        content_type: Optional[str],
        detected_type: Optional[str],
        detected_mime: Optional[str],
    ) -> List[str]:
        """
        Cross-validate filename extension, MIME type, and detected type for consistency.
        
        Args:
            filename: Original filename
            content_type: MIME type from Content-Type header
            detected_type: File type detected from content
            detected_mime: MIME type detected from content
            
        Returns:
            List of validation error messages (empty if all consistent)
        """
        pass

    @abstractmethod
    def scan_for_security_threats(
        self,
        file_content: bytes,
        filename: str,
    ) -> List[str]:
        """
        Scan file content for potential security threats.
        
        Args:
            file_content: Raw file content bytes
            filename: Original filename
            
        Returns:
            List of security warnings/threats detected
        """
        pass

    @abstractmethod
    def get_supported_file_types(
        self,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> List[FileTypeConfig]:
        """
        Get list of supported file types with validation rules.
        
        Args:
            tenant_config: Tenant-specific configuration
            
        Returns:
            List of supported file type configurations
        """
        pass

    @abstractmethod
    def is_file_type_allowed(
        self,
        extension: str,
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if file type is allowed based on extension.
        
        Args:
            extension: File extension (e.g., '.pdf')
            tenant_config: Tenant-specific configuration
            
        Returns:
            True if file type is allowed, False otherwise
        """
        pass


class IResourceManager(IHealthCheck, ABC):
    """Resource management interface for file cleanup and memory management."""

    @abstractmethod
    async def track_resource(
        self,
        resource_id: str,
        resource_type: str,
        size_bytes: int,
        *,
        tenant_id: str,
        upload_id: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Track a resource for cleanup management.
        
        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource (file_content, temp_file, etc.)
            size_bytes: Size of the resource in bytes
            tenant_id: Tenant identifier
            upload_id: Associated upload ID if applicable
            file_path: File path if applicable
            metadata: Additional metadata
            
        Returns:
            True if tracking was successful
        """
        pass

    @abstractmethod
    async def release_resource(
        self,
        resource_id: str,
        *,
        force: bool = False,
    ) -> bool:
        """
        Release and cleanup a tracked resource.
        
        Args:
            resource_id: Resource identifier to release
            force: Force cleanup even if resource is marked as in-use
            
        Returns:
            True if resource was successfully released
        """
        pass

    @abstractmethod
    async def release_upload_resources(
        self,
        upload_id: str,
        *,
        exclude_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Release all resources associated with an upload.
        
        Args:
            upload_id: Upload identifier
            exclude_types: Resource types to exclude from cleanup
            
        Returns:
            Dictionary with cleanup statistics
        """
        pass

    @abstractmethod
    async def cleanup_orphaned_resources(
        self,
        *,
        older_than_minutes: int = 60,
        max_cleanup_count: int = 100,
    ) -> Dict[str, Any]:
        """
        Clean up orphaned resources older than specified time.
        
        Args:
            older_than_minutes: Cleanup resources older than this
            max_cleanup_count: Maximum number of resources to cleanup in one pass
            
        Returns:
            Dictionary with cleanup statistics
        """
        pass

    @abstractmethod
    async def get_resource_stats(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Args:
            tenant_id: Tenant to get stats for, or None for all tenants
            
        Returns:
            Dictionary with resource statistics
        """
        pass

    @abstractmethod
    async def mark_resource_in_use(self, resource_id: str) -> bool:
        """Mark a resource as currently in use."""
        pass

    @abstractmethod
    async def mark_resource_available(self, resource_id: str) -> bool:
        """Mark a resource as available for cleanup."""
        pass


class IFileResourceManager(IResourceManager):
    """Specialized resource manager for file operations."""

    @abstractmethod
    async def track_file_content(
        self,
        content: bytes,
        filename: str,
        *,
        upload_id: str,
        tenant_id: str,
        auto_cleanup_after: Optional[int] = None,
    ) -> str:
        """
        Track file content in memory for cleanup.
        
        Args:
            content: File content bytes
            filename: Original filename
            upload_id: Associated upload ID
            tenant_id: Tenant identifier
            auto_cleanup_after: Auto cleanup after N seconds
            
        Returns:
            Resource ID for tracking
        """
        pass

    @abstractmethod
    async def get_file_content(self, resource_id: str) -> Optional[bytes]:
        """Get tracked file content by resource ID."""
        pass

    @abstractmethod
    async def track_temp_file(
        self,
        file_path: str,
        *,
        upload_id: str,
        tenant_id: str,
        auto_cleanup_after: Optional[int] = None,
    ) -> str:
        """
        Track temporary file for cleanup.
        
        Args:
            file_path: Path to temporary file
            upload_id: Associated upload ID
            tenant_id: Tenant identifier
            auto_cleanup_after: Auto cleanup after N seconds
            
        Returns:
            Resource ID for tracking
        """
        pass

    @abstractmethod
    async def cleanup_file_handles(self, upload_id: str) -> int:
        """Cleanup file handles for an upload."""
        pass


class RateLimitType(str, Enum):
    """Types of rate limits that can be applied."""
    USER = "user"
    TENANT = "tenant"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


class TimeWindow(str, Enum):
    """Time windows for rate limiting."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


@dataclass
class RateLimitRule:
    """Configuration for a rate limit rule."""
    
    limit_type: RateLimitType
    time_window: TimeWindow
    max_requests: int
    identifier: Optional[str] = None  # specific identifier (e.g., endpoint pattern)
    description: str = ""
    priority: int = 0  # higher priority rules are checked first


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    
    allowed: bool
    limit_type: RateLimitType
    time_window: TimeWindow
    max_requests: int
    current_usage: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    identifier: str = ""


@dataclass
class RateLimitViolation:
    """Details of a rate limit violation for audit logging."""
    
    limit_type: RateLimitType
    time_window: TimeWindow
    max_requests: int
    actual_requests: int
    identifier: str
    tenant_id: str
    user_id: Optional[str] = None
    ip_address: str = "unknown"
    user_agent: str = "unknown"
    endpoint: str = ""
    violation_time: datetime = field(default_factory=datetime.utcnow)


class IRateLimitService(IHealthCheck, ABC):
    """Rate limiting service interface."""

    @abstractmethod
    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: RateLimitType,
        time_window: TimeWindow,
        max_requests: int,
        *,
        tenant_id: Optional[str] = None,
        increment: bool = True
    ) -> RateLimitResult:
        """
        Check if a request is within rate limits.
        
        Args:
            identifier: Unique identifier for the limit (user_id, IP, etc.)
            limit_type: Type of rate limit being checked
            time_window: Time window for the limit
            max_requests: Maximum requests allowed in the time window
            tenant_id: Tenant context for isolation
            increment: Whether to increment the counter (set False for checks only)
            
        Returns:
            RateLimitResult with limit status and details
        """
        pass

    @abstractmethod
    async def check_multiple_limits(
        self,
        identifiers: Dict[RateLimitType, str],
        rules: List[RateLimitRule],
        *,
        tenant_id: Optional[str] = None,
        increment: bool = True
    ) -> List[RateLimitResult]:
        """
        Check multiple rate limits in a single operation.
        
        Args:
            identifiers: Map of limit types to their identifiers
            rules: List of rate limit rules to check
            tenant_id: Tenant context for isolation
            increment: Whether to increment counters
            
        Returns:
            List of RateLimitResult for each rule checked
        """
        pass

    @abstractmethod
    async def reset_rate_limit(
        self,
        identifier: str,
        limit_type: RateLimitType,
        time_window: TimeWindow,
        *,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Reset rate limit counter for an identifier.
        
        Args:
            identifier: Identifier to reset
            limit_type: Type of limit to reset
            time_window: Time window to reset
            tenant_id: Tenant context
            
        Returns:
            True if reset was successful
        """
        pass

    @abstractmethod
    async def get_rate_limit_status(
        self,
        identifier: str,
        limit_type: RateLimitType,
        time_window: TimeWindow,
        *,
        tenant_id: Optional[str] = None
    ) -> Optional[RateLimitResult]:
        """
        Get current rate limit status without incrementing.
        
        Args:
            identifier: Identifier to check
            limit_type: Type of limit
            time_window: Time window
            tenant_id: Tenant context
            
        Returns:
            Current rate limit status or None if no limit exists
        """
        pass

    @abstractmethod
    async def cleanup_expired_limits(
        self,
        *,
        batch_size: int = 1000
    ) -> int:
        """
        Clean up expired rate limit entries.
        
        Args:
            batch_size: Number of entries to clean in one batch
            
        Returns:
            Number of entries cleaned up
        """
        pass

    @abstractmethod
    async def get_rate_limit_stats(
        self,
        *,
        tenant_id: Optional[str] = None,
        limit_type: Optional[RateLimitType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get rate limiting statistics.
        
        Args:
            tenant_id: Filter by tenant
            limit_type: Filter by limit type
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            Dictionary with statistics
        """
        pass

    @abstractmethod
    async def is_whitelisted(
        self,
        identifier: str,
        limit_type: RateLimitType,
        *,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Check if an identifier is whitelisted from rate limits.
        
        Args:
            identifier: Identifier to check
            limit_type: Type of limit
            tenant_id: Tenant context
            
        Returns:
            True if whitelisted
        """
        pass

    @abstractmethod
    async def add_to_whitelist(
        self,
        identifier: str,
        limit_type: RateLimitType,
        *,
        tenant_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Add identifier to whitelist.
        
        Args:
            identifier: Identifier to whitelist
            limit_type: Type of limit
            tenant_id: Tenant context
            expires_at: When whitelist entry expires
            
        Returns:
            True if successfully whitelisted
        """
        pass

    @abstractmethod
    async def remove_from_whitelist(
        self,
        identifier: str,
        limit_type: RateLimitType,
        *,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Remove identifier from whitelist.
        
        Args:
            identifier: Identifier to remove
            limit_type: Type of limit
            tenant_id: Tenant context
            
        Returns:
            True if successfully removed
        """
        pass


@dataclass 
class WebhookDeliveryResult:
    """Result of a webhook delivery attempt."""
    
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    failure_reason: Optional[str] = None
    signature_verified: Optional[bool] = None


@dataclass
class WebhookRetrySchedule:
    """Webhook retry schedule calculation result."""
    
    should_retry: bool
    next_attempt_at: Optional[datetime] = None
    delay_seconds: Optional[float] = None
    attempt_number: int = 1
    reason: str = ""


class IWebhookDeliveryService(IHealthCheck, ABC):
    """Webhook delivery service with retry mechanisms and reliability features."""
    
    @abstractmethod
    async def deliver_webhook(
        self,
        endpoint_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        tenant_id: str,
        event_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """
        Queue a webhook for delivery with retry mechanism.
        
        Args:
            endpoint_id: Webhook endpoint identifier
            event_type: Type of event triggering the webhook
            payload: Webhook payload data
            tenant_id: Tenant identifier
            event_id: Unique event identifier
            correlation_id: Correlation ID for tracking
            priority: Delivery priority (higher = more urgent)
            
        Returns:
            Delivery ID for tracking
        """
        pass
    
    @abstractmethod
    async def deliver_webhook_immediate(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        secret: Optional[str] = None,
        timeout_seconds: int = 30,
        signature_header: str = "X-Webhook-Signature",
        correlation_id: Optional[str] = None
    ) -> WebhookDeliveryResult:
        """
        Deliver webhook immediately without queuing.
        
        Args:
            url: Webhook URL
            payload: Webhook payload
            secret: Secret for signature generation
            timeout_seconds: Request timeout
            signature_header: Header name for signature
            correlation_id: Correlation ID for tracking
            
        Returns:
            WebhookDeliveryResult with delivery outcome
        """
        pass
    
    @abstractmethod
    async def retry_failed_delivery(
        self,
        delivery_id: str,
        *,
        override_max_attempts: bool = False,
        new_max_attempts: Optional[int] = None,
        admin_user_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Manually retry a failed webhook delivery.
        
        Args:
            delivery_id: Delivery ID to retry
            override_max_attempts: Override the max attempts limit
            new_max_attempts: New max attempts if overriding
            admin_user_id: Admin user performing the retry
            notes: Notes about the retry
            
        Returns:
            True if retry was scheduled successfully
        """
        pass
    
    @abstractmethod
    async def cancel_delivery(
        self,
        delivery_id: str,
        *,
        reason: str,
        admin_user_id: Optional[str] = None
    ) -> bool:
        """
        Cancel a pending webhook delivery.
        
        Args:
            delivery_id: Delivery ID to cancel
            reason: Cancellation reason
            admin_user_id: Admin user performing the cancellation
            
        Returns:
            True if cancellation was successful
        """
        pass
    
    @abstractmethod
    async def get_delivery_status(
        self,
        delivery_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the status of a webhook delivery.
        
        Args:
            delivery_id: Delivery ID to check
            
        Returns:
            Delivery status details or None if not found
        """
        pass
    
    @abstractmethod
    async def process_delivery_queue(
        self,
        *,
        max_deliveries: int = 50,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Process pending webhook deliveries from queue.
        
        Args:
            max_deliveries: Maximum deliveries to process
            timeout_seconds: Processing timeout
            
        Returns:
            Processing summary with statistics
        """
        pass
    
    @abstractmethod
    async def calculate_retry_schedule(
        self,
        delivery_id: str,
        failure_reason: str
    ) -> WebhookRetrySchedule:
        """
        Calculate when to retry a failed delivery.
        
        Args:
            delivery_id: Delivery ID that failed
            failure_reason: Reason for failure
            
        Returns:
            WebhookRetrySchedule with retry timing
        """
        pass


class IWebhookCircuitBreakerService(IHealthCheck, ABC):
    """Circuit breaker service for webhook endpoints."""
    
    @abstractmethod
    async def should_allow_request(
        self,
        endpoint_id: str
    ) -> bool:
        """
        Check if requests should be allowed to an endpoint.
        
        Args:
            endpoint_id: Webhook endpoint ID
            
        Returns:
            True if requests are allowed
        """
        pass
    
    @abstractmethod
    async def record_success(
        self,
        endpoint_id: str,
        response_time_ms: int
    ) -> None:
        """
        Record a successful webhook delivery.
        
        Args:
            endpoint_id: Webhook endpoint ID
            response_time_ms: Response time in milliseconds
        """
        pass
    
    @abstractmethod
    async def record_failure(
        self,
        endpoint_id: str,
        failure_reason: str
    ) -> None:
        """
        Record a failed webhook delivery.
        
        Args:
            endpoint_id: Webhook endpoint ID
            failure_reason: Reason for failure
        """
        pass
    
    @abstractmethod
    async def get_circuit_state(
        self,
        endpoint_id: str
    ) -> Dict[str, Any]:
        """
        Get circuit breaker state for an endpoint.
        
        Args:
            endpoint_id: Webhook endpoint ID
            
        Returns:
            Circuit breaker state information
        """
        pass
    
    @abstractmethod
    async def force_circuit_state(
        self,
        endpoint_id: str,
        state: str,
        *,
        admin_user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Manually set circuit breaker state.
        
        Args:
            endpoint_id: Webhook endpoint ID
            state: Target state (open, closed, half_open)
            admin_user_id: Admin user making the change
            reason: Reason for manual override
            
        Returns:
            True if state change was successful
        """
        pass


class IWebhookSignatureService(ABC):
    """Webhook signature generation and verification service."""
    
    @abstractmethod
    def generate_signature(
        self,
        payload: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        Generate webhook signature for payload.
        
        Args:
            payload: Webhook payload as string
            secret: Secret key for signing
            algorithm: Signature algorithm
            
        Returns:
            Generated signature
        """
        pass
    
    @abstractmethod
    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Webhook payload as string
            signature: Provided signature
            secret: Secret key for verification
            algorithm: Signature algorithm
            
        Returns:
            True if signature is valid
        """
        pass
    
    @abstractmethod
    def generate_timestamp_signature(
        self,
        payload: str,
        timestamp: int,
        secret: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        Generate timestamped signature for replay protection.
        
        Args:
            payload: Webhook payload as string
            timestamp: Unix timestamp
            secret: Secret key for signing
            algorithm: Signature algorithm
            
        Returns:
            Generated timestamped signature
        """
        pass


class IWebhookDeadLetterService(IHealthCheck, ABC):
    """Dead letter queue service for failed webhook deliveries."""
    
    @abstractmethod
    async def move_to_dead_letter(
        self,
        delivery_id: str,
        *,
        final_failure_reason: str,
        final_error_message: Optional[str] = None,
        moved_by: str = "system"
    ) -> str:
        """
        Move a failed delivery to dead letter queue.
        
        Args:
            delivery_id: Original delivery ID
            final_failure_reason: Final reason for failure
            final_error_message: Final error message
            moved_by: Who moved it to dead letter
            
        Returns:
            Dead letter ID
        """
        pass
    
    @abstractmethod
    async def retry_dead_letter(
        self,
        dead_letter_id: str,
        *,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Retry a dead letter delivery.
        
        Args:
            dead_letter_id: Dead letter ID to retry
            admin_user_id: Admin performing the retry
            notes: Retry notes
            
        Returns:
            New delivery ID for retry
        """
        pass
    
    @abstractmethod
    async def resolve_dead_letter(
        self,
        dead_letter_id: str,
        *,
        resolution_action: str,
        admin_user_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Mark a dead letter as resolved.
        
        Args:
            dead_letter_id: Dead letter ID to resolve
            resolution_action: Action taken to resolve
            admin_user_id: Admin resolving the issue
            notes: Resolution notes
            
        Returns:
            True if resolution was successful
        """
        pass
    
    @abstractmethod
    async def get_dead_letters(
        self,
        *,
        tenant_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get dead letter queue entries.
        
        Args:
            tenant_id: Filter by tenant
            endpoint_id: Filter by endpoint
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum entries to return
            offset: Offset for pagination
            
        Returns:
            Dead letter entries with pagination info
        """
        pass
    
    @abstractmethod
    async def cleanup_old_dead_letters(
        self,
        *,
        older_than_days: int = 30,
        batch_size: int = 100
    ) -> int:
        """
        Clean up old dead letter entries.
        
        Args:
            older_than_days: Remove entries older than this
            batch_size: Batch size for cleanup
            
        Returns:
            Number of entries cleaned up
        """
        pass


class IWebhookStatsService(IHealthCheck, ABC):
    """Webhook delivery statistics and monitoring service."""
    
    @abstractmethod
    async def get_delivery_stats(
        self,
        *,
        tenant_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get webhook delivery statistics.
        
        Args:
            tenant_id: Filter by tenant
            endpoint_id: Filter by endpoint
            event_type: Filter by event type
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            Delivery statistics
        """
        pass
    
    @abstractmethod
    async def get_endpoint_health(
        self,
        endpoint_id: str,
        *,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get health metrics for a specific endpoint.
        
        Args:
            endpoint_id: Webhook endpoint ID
            time_window_hours: Time window for metrics
            
        Returns:
            Endpoint health metrics
        """
        pass
    
    @abstractmethod
    async def get_tenant_webhook_summary(
        self,
        tenant_id: str,
        *,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get webhook summary for a tenant.
        
        Args:
            tenant_id: Tenant ID
            time_window_hours: Time window for summary
            
        Returns:
            Tenant webhook summary
        """
        pass
    
    @abstractmethod
    async def generate_stats_digest(
        self,
        tenant_id: str,
        *,
        period_days: int = 1
    ) -> Dict[str, Any]:
        """
        Generate webhook statistics digest.
        
        Args:
            tenant_id: Tenant ID
            period_days: Period for digest
            
        Returns:
            Statistics digest
        """
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
    "IWebhookValidator",
    "IAuditService",
    "IFileContentValidator",
    "FileValidationResult",
    "FileTypeConfig",
    "IResourceManager",
    "IFileResourceManager",
    "RateLimitType",
    "TimeWindow",
    "RateLimitRule",
    "RateLimitResult",
    "RateLimitViolation",
    "IRateLimitService",
    "WebhookDeliveryResult",
    "WebhookRetrySchedule",
    "IWebhookDeliveryService",
    "IWebhookCircuitBreakerService",
    "IWebhookSignatureService",
    "IWebhookDeadLetterService",
    "IWebhookStatsService",
]
