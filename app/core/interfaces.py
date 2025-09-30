"""
Abstract Service Interfaces - Clean contracts that work for both local and Azure implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime


class IHealthCheck:
    """Health check interface mixin"""
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        pass


class ICacheService(IHealthCheck, ABC):
    """Cache service interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern"""
        pass


class IDocumentStore(IHealthCheck, ABC):
    """Document storage interface"""
    
    @abstractmethod
    async def store_document(self, container: str, path: str, content: bytes) -> str:
        """Store document and return URL"""
        pass
    
    @abstractmethod
    async def get_document(self, container: str, path: str) -> bytes:
        """Get document content"""
        pass
    
    @abstractmethod
    async def delete_document(self, container: str, path: str) -> bool:
        """Delete document"""
        pass
    
    @abstractmethod
    async def list_documents(self, container: str, prefix: str = "") -> List[str]:
        """List documents with optional prefix filter"""
        pass
    
    @abstractmethod
    async def get_document_url(self, container: str, path: str, expires_in: int = 3600) -> str:
        """Get signed URL for document access"""
        pass


class IDatabase(IHealthCheck, ABC):
    """Database interface for document operations"""
    
    @abstractmethod
    async def create_item(self, container: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        pass
    
    @abstractmethod
    async def get_item(self, container: str, item_id: str, partition_key: str) -> Optional[Dict[str, Any]]:
        """Get item by ID and partition key"""
        pass
    
    @abstractmethod
    async def update_item(self, container: str, item_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing item"""
        pass
    
    @abstractmethod
    async def delete_item(self, container: str, item_id: str, partition_key: str) -> bool:
        """Delete item"""
        pass
    
    @abstractmethod
    async def query_items(self, container: str, query: str, parameters: List[Dict[str, Any]] = None, partition_key: str = None) -> List[Dict[str, Any]]:
        """Query items with SQL-like syntax"""
        pass


class ISearchService(IHealthCheck, ABC):
    """Search service interface"""
    
    @abstractmethod
    async def create_index(self, index_name: str, fields: List[Dict[str, Any]]) -> bool:
        """Create search index"""
        pass
    
    @abstractmethod
    async def index_document(self, index_name: str, document: Dict[str, Any]) -> bool:
        """Index a single document"""
        pass
    
    @abstractmethod
    async def index_documents_batch(self, index_name: str, documents: List[Dict[str, Any]]) -> int:
        """Index multiple documents, return count of successful indexes"""
        pass
    
    @abstractmethod
    async def search(self, index_name: str, query: str, filters: Dict[str, Any] = None, top: int = 50) -> Dict[str, Any]:
        """Search documents"""
        pass
    
    @abstractmethod
    async def vector_search(self, index_name: str, vector: List[float], top: int = 50) -> Dict[str, Any]:
        """Vector similarity search"""
        pass
    
    @abstractmethod
    async def delete_document(self, index_name: str, document_id: str) -> bool:
        """Delete document from index"""
        pass


class IAIService(IHealthCheck, ABC):
    """AI service interface for embeddings and chat"""
    
    @abstractmethod
    async def generate_embedding(self, text: str, model: str = "default") -> List[float]:
        """Generate text embedding"""
        pass
    
    @abstractmethod
    async def generate_embeddings_batch(self, texts: List[str], model: str = "default") -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], model: str = "default", max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def extract_text_from_document(self, document_content: bytes, document_type: str = "pdf") -> Dict[str, Any]:
        """Extract text and metadata from document"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self, model: str = "default") -> int:
        """Get embedding vector dimension"""
        pass


class IMessageQueue(IHealthCheck, ABC):
    """Message queue interface"""
    
    @abstractmethod
    async def send_message(self, queue_name: str, message: Dict[str, Any], delay: int = 0) -> str:
        """Send message to queue, return message ID"""
        pass
    
    @abstractmethod
    async def receive_messages(self, queue_name: str, max_messages: int = 1, visibility_timeout: int = 30) -> List[Dict[str, Any]]:
        """Receive messages from queue"""
        pass
    
    @abstractmethod
    async def complete_message(self, queue_name: str, message_id: str, receipt_handle: str) -> bool:
        """Mark message as processed"""
        pass
    
    @abstractmethod
    async def abandon_message(self, queue_name: str, message_id: str, receipt_handle: str) -> bool:
        """Return message to queue"""
        pass
    
    @abstractmethod
    async def dead_letter_message(self, queue_name: str, message_id: str, receipt_handle: str, reason: str) -> bool:
        """Send message to dead letter queue"""
        pass


class INotificationService(IHealthCheck, ABC):
    """Notification service interface"""
    
    @abstractmethod
    async def send_email(self, to: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email notification"""
        pass
    
    @abstractmethod
    async def send_webhook(self, url: str, payload: Dict[str, Any], secret: str = None) -> bool:
        """Send webhook notification"""
        pass
    
    @abstractmethod
    async def send_push_notification(self, user_id: str, title: str, message: str, data: Dict[str, Any] = None) -> bool:
        """Send push notification"""
        pass


class IAnalyticsService(IHealthCheck, ABC):
    """Analytics and metrics interface"""
    
    @abstractmethod
    async def track_event(self, event_name: str, properties: Dict[str, Any], user_id: str = None) -> bool:
        """Track analytics event"""
        pass
    
    @abstractmethod
    async def increment_counter(self, metric_name: str, value: int = 1, tags: Dict[str, str] = None) -> bool:
        """Increment counter metric"""
        pass
    
    @abstractmethod
    async def record_timing(self, metric_name: str, duration_ms: int, tags: Dict[str, str] = None) -> bool:
        """Record timing metric"""
        pass
    
    @abstractmethod
    async def get_metrics(self, metric_names: List[str], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get metrics data"""
        pass


# Service registry interface
class IServiceRegistry(ABC):
    """Service registry for managing service instances"""
    
    @abstractmethod
    def register_service(self, interface_type: type, implementation: Any, priority: int = 0) -> None:
        """Register service implementation"""
        pass
    
    @abstractmethod
    def get_service(self, interface_type: type) -> Any:
        """Get service implementation"""
        pass
    
    @abstractmethod
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all registered services"""
        pass