"""
Notification models for system messaging.

This module defines models for notifications, delivery channels,
and notification preferences.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import EmailStr, Field, HttpUrl

from .base import BaseModel, TimestampedModel


class NotificationType(str, Enum):
    """Notification type enumeration."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ALERT = "alert"


class NotificationChannel(str, Enum):
    """Notification delivery channel enumeration."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SLACK = "slack"
    TEAMS = "teams"


class NotificationStatus(str, Enum):
    """Notification delivery status enumeration."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"


class NotificationPriority(str, Enum):
    """Notification priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationTemplate(BaseModel):
    """Notification template model."""
    
    id: UUID = Field(..., description="Template ID")
    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    type: NotificationType = Field(..., description="Notification type")
    channel: NotificationChannel = Field(..., description="Delivery channel")
    
    # Content templates
    subject_template: Optional[str] = Field(None, description="Subject line template")
    body_template: str = Field(..., description="Message body template")
    html_template: Optional[str] = Field(None, description="HTML body template")
    
    # Template variables
    required_variables: List[str] = Field(
        default_factory=list,
        description="Required template variables"
    )
    optional_variables: List[str] = Field(
        default_factory=list,
        description="Optional template variables"
    )
    
    # Localization
    language: str = Field(default="en", description="Template language")
    locale_variants: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Localized template variants"
    )
    
    # Settings
    is_active: bool = Field(default=True, description="Template is active")
    tenant_id: Optional[UUID] = Field(None, description="Tenant-specific template")


class NotificationPreferences(BaseModel):
    """User notification preferences model."""
    
    user_id: UUID = Field(..., description="User ID")
    
    # Channel preferences
    email_enabled: bool = Field(default=True, description="Email notifications enabled")
    sms_enabled: bool = Field(default=False, description="SMS notifications enabled")
    push_enabled: bool = Field(default=True, description="Push notifications enabled")
    in_app_enabled: bool = Field(default=True, description="In-app notifications enabled")
    
    # Contact information
    email_address: Optional[EmailStr] = Field(None, description="Email address")
    phone_number: Optional[str] = Field(
        None,
        pattern="^\\+?[1-9]\\d{1,14}$",
        description="Phone number for SMS"
    )
    
    # Event-specific preferences
    event_preferences: Dict[str, Dict[str, bool]] = Field(
        default_factory=dict,
        description="Per-event notification preferences"
    )
    
    # Frequency settings
    digest_frequency: Optional[str] = Field(
        None,
        pattern="^(immediate|hourly|daily|weekly)$",
        description="Digest notification frequency"
    )
    quiet_hours_start: Optional[str] = Field(
        None,
        pattern="^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$",
        description="Quiet hours start time (HH:MM)"
    )
    quiet_hours_end: Optional[str] = Field(
        None,
        pattern="^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$",
        description="Quiet hours end time (HH:MM)"
    )
    timezone: str = Field(default="UTC", description="User timezone")


class NotificationDelivery(BaseModel):
    """Notification delivery record."""
    
    channel: NotificationChannel = Field(..., description="Delivery channel")
    status: NotificationStatus = Field(..., description="Delivery status")
    
    # Delivery details
    recipient: str = Field(..., description="Delivery recipient (email, phone, etc.)")
    sent_at: Optional[datetime] = Field(None, description="Sent timestamp")
    delivered_at: Optional[datetime] = Field(None, description="Delivered timestamp")
    opened_at: Optional[datetime] = Field(None, description="Opened timestamp")
    clicked_at: Optional[datetime] = Field(None, description="Clicked timestamp")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    next_retry_at: Optional[datetime] = Field(None, description="Next retry timestamp")
    
    # Provider details
    provider: Optional[str] = Field(None, description="Delivery provider")
    provider_message_id: Optional[str] = Field(
        None,
        description="Provider's message ID"
    )
    provider_response: Optional[Dict[str, Any]] = Field(
        None,
        description="Provider response data"
    )


class Notification(TimestampedModel):
    """Notification model."""
    
    id: UUID = Field(..., description="Notification ID")
    tenant_id: UUID = Field(..., description="Tenant ID")
    
    # Recipients
    user_id: Optional[UUID] = Field(None, description="Target user ID")
    recipient_emails: List[EmailStr] = Field(
        default_factory=list,
        description="Additional email recipients"
    )
    recipient_phones: List[str] = Field(
        default_factory=list,
        description="SMS recipients"
    )
    
    # Content
    type: NotificationType = Field(..., description="Notification type")
    priority: NotificationPriority = Field(
        default=NotificationPriority.NORMAL,
        description="Notification priority"
    )
    title: str = Field(..., min_length=1, max_length=200, description="Notification title")
    message: str = Field(..., min_length=1, max_length=2000, description="Notification message")
    html_message: Optional[str] = Field(None, description="HTML formatted message")
    
    # Metadata
    event_type: str = Field(..., description="Event that triggered the notification")
    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data"
    )
    template_id: Optional[UUID] = Field(None, description="Template used")
    template_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables used for templating"
    )
    
    # Delivery
    channels: List[NotificationChannel] = Field(..., description="Delivery channels")
    delivery_records: List[NotificationDelivery] = Field(
        default_factory=list,
        description="Delivery attempt records"
    )
    
    # Scheduling
    scheduled_for: Optional[datetime] = Field(
        None,
        description="Scheduled delivery time"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Notification expiration time"
    )
    
    # Interaction
    read_at: Optional[datetime] = Field(None, description="Read timestamp")
    dismissed_at: Optional[datetime] = Field(None, description="Dismissed timestamp")
    action_taken: Optional[str] = Field(None, description="Action taken by user")
    action_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Action-specific data"
    )
    
    # Links and actions
    action_url: Optional[HttpUrl] = Field(None, description="Action URL")
    action_label: Optional[str] = Field(None, description="Action button label")
    deep_link: Optional[str] = Field(None, description="Mobile app deep link")
    
    @property
    def is_read(self) -> bool:
        """Check if notification has been read."""
        return self.read_at is not None
    
    @property
    def is_dismissed(self) -> bool:
        """Check if notification has been dismissed."""
        return self.dismissed_at is not None
    
    @property
    def is_expired(self) -> bool:
        """Check if notification is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_scheduled(self) -> bool:
        """Check if notification is scheduled for future delivery."""
        if not self.scheduled_for:
            return False
        return datetime.utcnow() < self.scheduled_for
    
    def get_delivery_status(self, channel: NotificationChannel) -> Optional[NotificationStatus]:
        """Get delivery status for specific channel."""
        delivery = next(
            (d for d in self.delivery_records if d.channel == channel), None
        )
        return delivery.status if delivery else None
    
    def mark_as_read(self) -> None:
        """Mark notification as read."""
        if not self.is_read:
            self.read_at = datetime.utcnow()
            self.update_timestamp()
    
    def dismiss(self) -> None:
        """Dismiss notification."""
        if not self.is_dismissed:
            self.dismissed_at = datetime.utcnow()
            self.update_timestamp()


class NotificationCreate(BaseModel):
    """Model for notification creation."""
    
    # Recipients
    user_id: Optional[UUID] = None
    recipient_emails: List[EmailStr] = Field(default_factory=list)
    recipient_phones: List[str] = Field(default_factory=list)
    
    # Content
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.NORMAL
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=2000)
    html_message: Optional[str] = None
    
    # Metadata
    event_type: str
    event_data: Dict[str, Any] = Field(default_factory=dict)
    template_id: Optional[UUID] = None
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Delivery
    channels: List[NotificationChannel]
    
    # Scheduling
    scheduled_for: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Actions
    action_url: Optional[HttpUrl] = None
    action_label: Optional[str] = None
    deep_link: Optional[str] = None


class NotificationUpdate(BaseModel):
    """Model for notification updates."""
    
    read_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None
    action_taken: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None


class BulkNotificationRequest(BaseModel):
    """Model for bulk notification sending."""
    
    user_ids: List[UUID] = Field(..., description="Target user IDs")
    notification: NotificationCreate = Field(..., description="Notification to send")
    personalize: bool = Field(
        default=True,
        description="Personalize notifications per user"
    )
    send_async: bool = Field(
        default=True,
        description="Send notifications asynchronously"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for processing"
    )


class NotificationStats(BaseModel):
    """Notification statistics model."""
    
    total_notifications: int = Field(default=0, description="Total notifications")
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Notifications by type"
    )
    by_channel: Dict[str, int] = Field(
        default_factory=dict,
        description="Notifications by channel"
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Notifications by status"
    )
    
    # Delivery metrics
    delivery_rate: float = Field(default=0.0, description="Overall delivery rate")
    open_rate: float = Field(default=0.0, description="Open rate")
    click_rate: float = Field(default=0.0, description="Click-through rate")
    
    # Performance metrics
    avg_delivery_time: Optional[float] = Field(
        None,
        description="Average delivery time in seconds"
    )
    failed_deliveries: int = Field(default=0, description="Failed delivery count")


class NotificationAnalytics(BaseModel):
    """Notification analytics model."""
    
    # Time-based metrics
    delivery_trends: Dict[str, int] = Field(
        default_factory=dict,
        description="Delivery trends by date"
    )
    engagement_trends: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Engagement metrics by date"
    )
    
    # Channel performance
    channel_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Performance metrics by channel"
    )
    
    # Content analysis
    popular_templates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most used notification templates"
    )
    event_frequency: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequency of notification events"
    )
    
    # User engagement
    top_engaged_users: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Users with highest engagement"
    )
    unsubscribe_rate: float = Field(default=0.0, description="Unsubscribe rate")


class WebhookNotification(BaseModel):
    """Webhook notification payload model."""
    
    event: str = Field(..., description="Event name")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    tenant_id: UUID = Field(..., description="Tenant ID")
    
    # Event data
    data: Dict[str, Any] = Field(..., description="Event-specific data")
    
    # Metadata
    version: str = Field(default="1.0", description="Webhook payload version")
    request_id: str = Field(..., description="Unique request identifier")
    
    # Delivery info
    delivery_id: UUID = Field(..., description="Webhook delivery ID")
    attempt: int = Field(default=1, description="Delivery attempt number")
    
    # Signature for verification
    signature: Optional[str] = Field(None, description="Payload signature")


class NotificationDigest(BaseModel):
    """Notification digest model."""
    
    user_id: UUID = Field(..., description="User ID")
    period_start: datetime = Field(..., description="Digest period start")
    period_end: datetime = Field(..., description="Digest period end")
    
    # Digest content
    notifications: List[Notification] = Field(
        default_factory=list,
        description="Notifications included in digest"
    )
    
    # Summary
    total_notifications: int = Field(default=0, description="Total notification count")
    unread_count: int = Field(default=0, description="Unread notification count")
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Notifications by type"
    )
    
    # Actions required
    action_required_count: int = Field(
        default=0,
        description="Notifications requiring action"
    )
    urgent_count: int = Field(default=0, description="Urgent notifications")
    
    @property
    def has_urgent(self) -> bool:
        """Check if digest contains urgent notifications."""
        return self.urgent_count > 0
    
    @property
    def has_actions_required(self) -> bool:
        """Check if digest contains notifications requiring action."""
        return self.action_required_count > 0