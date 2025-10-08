"""Domain layer task types and value objects.

These are domain concepts that belong in the domain layer, not infrastructure.
Task management is a domain concern when it comes to tracking document processing workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TaskType(str, Enum):
    """Types of background tasks in the domain."""

    DOCUMENT_PROCESSING = "document_processing"
    BATCH_PROCESSING = "batch_processing"
    REPROCESSING = "reprocessing"
    CLEANUP = "cleanup"
    RESOURCE_CLEANUP = "resource_cleanup"


class TaskStatus(str, Enum):
    """Status of background tasks in the domain."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Domain value object containing information about a background task.

    This is a domain concept because it represents the state of a business
    workflow (document processing, batch uploads, etc.) rather than
    infrastructure concerns.
    """

    task_id: str
    upload_id: str
    tenant_id: str
    user_id: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message

    def mark_cancelled(self, reason: Optional[str] = None) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.error_message = reason or "Cancelled by user"


__all__ = ["TaskType", "TaskStatus", "TaskInfo"]
