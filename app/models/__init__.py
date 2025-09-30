"""Pydantic models for data validation and serialization."""

from .base import BaseModel, TimestampedModel, TenantModel
from .auth import UserTable, UserCreate, UserUpdate, TokenResponse, APIKeyTable, Permission, UserRole
from .tenant_models import TenantConfiguration
from .profile import Profile, ProfileCreate, ProfileUpdate, ProfileResponse
from .document import Document, DocumentCreate, DocumentUpdate, DocumentStatus
from .search_models import SearchRequest, SearchResult, SearchFilter
from .notification import Notification, NotificationCreate, NotificationUpdate
from .analytics import Usage, UsageEvent
from .job import Job, JobCreate, JobUpdate, JobStatus

__all__ = [
    # Base models
    "BaseModel",
    "TimestampedModel", 
    "TenantModel",
    
    # Authentication models
    "UserTable",
    "UserCreate",
    "UserUpdate", 
    "TokenResponse",
    "APIKeyTable",
    "Permission",
    "UserRole",
    
    # Tenant models
    "TenantConfiguration",
    
    # Profile models
    "Profile",
    "ProfileCreate",
    "ProfileUpdate",
    "ProfileResponse",
    
    # Document models
    "Document",
    "DocumentCreate", 
    "DocumentUpdate",
    "DocumentStatus",
    
    # Search models
    "SearchRequest",
    "SearchResult",
    "SearchFilter",
    
    # Notification models
    "Notification",
    "NotificationCreate",
    "NotificationUpdate",
    
    # Analytics models
    "Usage",
    "UsageEvent",
    
    # Job models
    "Job",
    "JobCreate",
    "JobUpdate", 
    "JobStatus",
]