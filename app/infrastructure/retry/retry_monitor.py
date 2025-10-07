"""Retry monitoring and alerting service for comprehensive system health tracking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import structlog

from app.domain.interfaces import IHealthCheck
from app.domain.retry import ErrorCategory, RetryResult
from app.infrastructure.providers.retry_provider import (
    get_retry_service, get_dead_letter_service, health_check_retry_services
)


logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    HIGH_FAILURE_RATE = "high_failure_rate"
    DEAD_LETTER_BACKLOG = "dead_letter_backlog"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    RETRY_QUEUE_BUILDUP = "retry_queue_buildup"
    SYSTEM_DEGRADATION = "system_degradation"
    THRESHOLD_BREACH = "threshold_breach"


class RetryMonitoringService(IHealthCheck):
    """Service for monitoring retry operations and generating alerts."""
    
    def __init__(self):
        self._logger = structlog.get_logger(__name__)
        
        # Monitoring thresholds
        self._thresholds = {
            "failure_rate_warning": 0.15,    # 15% failure rate warning
            "failure_rate_critical": 0.30,   # 30% failure rate critical
            "dead_letter_warning": 50,       # 50 unresolved dead letters
            "dead_letter_critical": 200,     # 200 unresolved dead letters
            "retry_queue_warning": 100,      # 100 pending retries
            "retry_queue_critical": 500,     # 500 pending retries
            "response_time_warning": 2000,   # 2 second response time
            "response_time_critical": 5000,  # 5 second response time
        }
        
        # Alert history and suppression
        self._alert_history: List[Dict[str, Any]] = []
        self._suppression_window = timedelta(minutes=15)  # Suppress duplicate alerts for 15 minutes
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 60  # Check every minute
        self._shutdown = False
    
    async def start_monitoring(self) -> None:
        """Start the monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._logger.info("Retry monitoring service started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        self._shutdown = True
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Retry monitoring service stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown:
            try:
                await self._perform_monitoring_checks()
                await asyncio.sleep(self._monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(self._monitoring_interval)
    
    async def _perform_monitoring_checks(self) -> None:
        """Perform all monitoring checks."""
        
        try:
            # Check retry system health
            await self._check_retry_system_health()
            
            # Check failure rates
            await self._check_failure_rates()
            
            # Check dead letter queue
            await self._check_dead_letter_queue()
            
            # Check retry queue buildup
            await self._check_retry_queue_buildup()
            
            # Check circuit breaker states
            await self._check_circuit_breakers()
            
            # Cleanup old alerts
            await self._cleanup_old_alerts()
            
        except Exception as e:
            self._logger.error("Error during monitoring checks", error=str(e))
    
    async def _check_retry_system_health(self) -> None:
        """Check overall retry system health."""
        
        health_status = await health_check_retry_services()
        
        unhealthy_services = [
            service for service, status in health_status.items()
            if isinstance(status, dict) and status.get("status") != "healthy"
        ]
        
        if unhealthy_services:
            await self._create_alert(
                alert_type=AlertType.SYSTEM_DEGRADATION,
                severity=AlertSeverity.HIGH,
                title="Retry System Degradation",
                description=f"Unhealthy services: {', '.join(unhealthy_services)}",
                metadata={
                    "unhealthy_services": unhealthy_services,
                    "health_status": health_status
                }
            )
    
    async def _check_failure_rates(self) -> None:
        """Check retry failure rates for different operation types."""
        
        retry_service = await get_retry_service()
        
        # Check failure rates for the last hour
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        # Get overall statistics
        overall_stats = await retry_service.get_retry_statistics(
            start_time=start_time,
            end_time=end_time
        )
        
        total_retries = overall_stats.get("total_retry_states", 0)
        if total_retries == 0:
            return
        
        failed_retries = overall_stats.get("failed_retries", 0)
        failure_rate = failed_retries / total_retries
        
        # Check overall failure rate
        if failure_rate >= self._thresholds["failure_rate_critical"]:
            await self._create_alert(
                alert_type=AlertType.HIGH_FAILURE_RATE,
                severity=AlertSeverity.CRITICAL,
                title="Critical Retry Failure Rate",
                description=f"Overall failure rate: {failure_rate:.1%} ({failed_retries}/{total_retries})",
                metadata={
                    "failure_rate": failure_rate,
                    "failed_retries": failed_retries,
                    "total_retries": total_retries,
                    "time_window": "1 hour"
                }
            )
        elif failure_rate >= self._thresholds["failure_rate_warning"]:
            await self._create_alert(
                alert_type=AlertType.HIGH_FAILURE_RATE,
                severity=AlertSeverity.MEDIUM,
                title="High Retry Failure Rate",
                description=f"Overall failure rate: {failure_rate:.1%} ({failed_retries}/{total_retries})",
                metadata={
                    "failure_rate": failure_rate,
                    "failed_retries": failed_retries,
                    "total_retries": total_retries,
                    "time_window": "1 hour"
                }
            )
        
        # Check failure rates by operation type
        by_operation_type = overall_stats.get("by_operation_type", {})
        for operation_type, count in by_operation_type.items():
            if count < 10:  # Skip if not enough data
                continue
            
            op_stats = await retry_service.get_retry_statistics(
                operation_type=operation_type,
                start_time=start_time,
                end_time=end_time
            )
            
            op_failed = op_stats.get("failed_retries", 0)
            op_total = op_stats.get("total_retry_states", 0)
            
            if op_total > 0:
                op_failure_rate = op_failed / op_total
                
                if op_failure_rate >= self._thresholds["failure_rate_critical"]:
                    await self._create_alert(
                        alert_type=AlertType.HIGH_FAILURE_RATE,
                        severity=AlertSeverity.HIGH,
                        title=f"Critical Failure Rate: {operation_type}",
                        description=f"{operation_type} failure rate: {op_failure_rate:.1%} ({op_failed}/{op_total})",
                        metadata={
                            "operation_type": operation_type,
                            "failure_rate": op_failure_rate,
                            "failed_retries": op_failed,
                            "total_retries": op_total
                        }
                    )
    
    async def _check_dead_letter_queue(self) -> None:
        """Check dead letter queue for backlogs."""
        
        dead_letter_service = await get_dead_letter_service()
        
        # Get unresolved dead letters
        queue_result = await dead_letter_service.get_dead_letter_queue(
            resolved=False,
            limit=1000  # Get up to 1000 to get accurate count
        )
        
        unresolved_count = queue_result.get("total_count", 0)
        
        if unresolved_count >= self._thresholds["dead_letter_critical"]:
            await self._create_alert(
                alert_type=AlertType.DEAD_LETTER_BACKLOG,
                severity=AlertSeverity.CRITICAL,
                title="Critical Dead Letter Queue Backlog",
                description=f"{unresolved_count} unresolved dead letter entries",
                metadata={
                    "unresolved_count": unresolved_count,
                    "threshold": self._thresholds["dead_letter_critical"]
                }
            )
        elif unresolved_count >= self._thresholds["dead_letter_warning"]:
            await self._create_alert(
                alert_type=AlertType.DEAD_LETTER_BACKLOG,
                severity=AlertSeverity.MEDIUM,
                title="Dead Letter Queue Backlog",
                description=f"{unresolved_count} unresolved dead letter entries",
                metadata={
                    "unresolved_count": unresolved_count,
                    "threshold": self._thresholds["dead_letter_warning"]
                }
            )
        
        # Check for old unresolved entries
        old_entries = await dead_letter_service.get_dead_letter_queue(
            resolved=False,
            end_date=datetime.utcnow() - timedelta(hours=24),
            limit=100
        )
        
        old_count = old_entries.get("total_count", 0)
        if old_count > 0:
            await self._create_alert(
                alert_type=AlertType.DEAD_LETTER_BACKLOG,
                severity=AlertSeverity.HIGH,
                title="Stale Dead Letter Entries",
                description=f"{old_count} dead letter entries older than 24 hours",
                metadata={
                    "old_entries_count": old_count,
                    "age_threshold_hours": 24
                }
            )
    
    async def _check_retry_queue_buildup(self) -> None:
        """Check for retry queue buildup."""
        
        retry_service = await get_retry_service()
        
        # Get ready retries (pending execution)
        ready_retries = await retry_service.get_ready_retries(limit=1000)
        ready_count = len(ready_retries)
        
        if ready_count >= self._thresholds["retry_queue_critical"]:
            await self._create_alert(
                alert_type=AlertType.RETRY_QUEUE_BUILDUP,
                severity=AlertSeverity.CRITICAL,
                title="Critical Retry Queue Buildup",
                description=f"{ready_count} retries pending execution",
                metadata={
                    "pending_retries": ready_count,
                    "threshold": self._thresholds["retry_queue_critical"]
                }
            )
        elif ready_count >= self._thresholds["retry_queue_warning"]:
            await self._create_alert(
                alert_type=AlertType.RETRY_QUEUE_BUILDUP,
                severity=AlertSeverity.MEDIUM,
                title="Retry Queue Buildup",
                description=f"{ready_count} retries pending execution",
                metadata={
                    "pending_retries": ready_count,
                    "threshold": self._thresholds["retry_queue_warning"]
                }
            )
        
        # Check for old pending retries
        old_retries = [
            retry for retry in ready_retries
            if retry.created_at < datetime.utcnow() - timedelta(hours=2)
        ]
        
        if old_retries:
            await self._create_alert(
                alert_type=AlertType.RETRY_QUEUE_BUILDUP,
                severity=AlertSeverity.HIGH,
                title="Stale Retry Queue Entries",
                description=f"{len(old_retries)} retries pending for over 2 hours",
                metadata={
                    "stale_retries": len(old_retries),
                    "age_threshold_hours": 2
                }
            )
    
    async def _check_circuit_breakers(self) -> None:
        """Check circuit breaker states."""
        
        # This would typically check webhook circuit breakers
        # For now, we'll create a placeholder check
        
        # If we had access to webhook service:
        # webhook_service = await get_webhook_service()
        # endpoints = await webhook_service.get_all_endpoints()
        # 
        # for endpoint in endpoints:
        #     circuit_state = await webhook_service.get_circuit_state(endpoint.id)
        #     if circuit_state["circuit_state"] == "open":
        #         await self._create_alert(...)
        
        pass
    
    async def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create and process an alert."""
        
        alert = {
            "id": f"alert_{len(self._alert_history) + 1}",
            "type": alert_type.value,
            "severity": severity.value,
            "title": title,
            "description": description,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "suppressed": False
        }
        
        # Check if this alert should be suppressed
        if self._should_suppress_alert(alert):
            alert["suppressed"] = True
            self._logger.debug(
                "Alert suppressed due to recent similar alert",
                alert_type=alert_type.value,
                title=title
            )
            return
        
        # Add to history
        self._alert_history.append(alert)
        
        # Log the alert
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self._logger.error(
                "Retry system alert",
                alert_type=alert_type.value,
                severity=severity.value,
                title=title,
                description=description,
                metadata=metadata
            )
        else:
            self._logger.warning(
                "Retry system alert",
                alert_type=alert_type.value,
                severity=severity.value,
                title=title,
                description=description,
                metadata=metadata
            )
        
        # Send alert via notification service
        await self._send_alert_notification(alert)
    
    def _should_suppress_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed due to recent similar alerts."""
        
        cutoff_time = datetime.utcnow() - self._suppression_window
        
        for existing_alert in self._alert_history:
            if (existing_alert["created_at"] > cutoff_time and
                existing_alert["type"] == alert["type"] and
                existing_alert["title"] == alert["title"] and
                not existing_alert["suppressed"]):
                return True
        
        return False
    
    async def _send_alert_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert notification via configured channels."""
        
        try:
            # Import here to avoid circular dependencies
            from app.infrastructure.providers.notification_provider import get_notification_service
            
            notification_service = await get_notification_service()
            
            # Determine notification channels based on severity
            if alert["severity"] in ["high", "critical"]:
                # Send to multiple channels for high/critical alerts
                channels = ["email", "slack", "webhook"]
            else:
                # Send to slack only for lower severity
                channels = ["slack"]
            
            # Format alert message
            message = f"**{alert['title']}**\n{alert['description']}"
            if alert.get("metadata"):
                message += f"\n\nDetails: {alert['metadata']}"
            
            # Send notifications
            for channel in channels:
                try:
                    if channel == "email":
                        await notification_service.send_email(
                            to="alerts@company.com",
                            subject=f"[{alert['severity'].upper()}] {alert['title']}",
                            body=message,
                            is_html=False
                        )
                    elif channel == "webhook":
                        await notification_service.send_webhook(
                            url="https://alerts.company.com/webhooks/retry-alerts",
                            payload={
                                "alert": alert,
                                "timestamp": alert["created_at"].isoformat(),
                                "service": "retry-monitoring"
                            }
                        )
                    # Add other notification channels as needed
                    
                except Exception as e:
                    self._logger.error(
                        "Failed to send alert notification",
                        channel=channel,
                        alert_id=alert["id"],
                        error=str(e)
                    )
        
        except Exception as e:
            self._logger.error(
                "Failed to send alert notifications",
                alert_id=alert["id"],
                error=str(e)
            )
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts from memory."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        old_count = len(self._alert_history)
        self._alert_history = [
            alert for alert in self._alert_history
            if alert["created_at"] > cutoff_time
        ]
        
        cleaned_count = old_count - len(self._alert_history)
        if cleaned_count > 0:
            self._logger.debug("Cleaned up old alerts", count=cleaned_count)
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        
        retry_service = await get_retry_service()
        dead_letter_service = await get_dead_letter_service()
        
        # Get retry statistics for last 24 hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        retry_stats = await retry_service.get_retry_statistics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get dead letter statistics
        dead_letter_stats = await dead_letter_service.get_dead_letter_statistics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self._alert_history
            if alert["created_at"] > datetime.utcnow() - timedelta(hours=6)
        ]
        
        # Get health status
        health_status = await health_check_retry_services()
        
        return {
            "monitoring_status": {
                "active": not self._shutdown,
                "last_check": datetime.utcnow().isoformat(),
                "monitoring_interval_seconds": self._monitoring_interval
            },
            "retry_statistics": retry_stats,
            "dead_letter_statistics": dead_letter_stats,
            "recent_alerts": recent_alerts,
            "alert_summary": {
                "total_alerts_24h": len([a for a in self._alert_history if a["created_at"] > start_time]),
                "critical_alerts_24h": len([a for a in self._alert_history if a["created_at"] > start_time and a["severity"] == "critical"]),
                "suppressed_alerts_24h": len([a for a in self._alert_history if a["created_at"] > start_time and a["suppressed"]])
            },
            "health_status": health_status,
            "thresholds": self._thresholds
        }
    
    async def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update monitoring thresholds."""
        
        for key, value in new_thresholds.items():
            if key in self._thresholds:
                old_value = self._thresholds[key]
                self._thresholds[key] = value
                
                self._logger.info(
                    "Updated monitoring threshold",
                    threshold=key,
                    old_value=old_value,
                    new_value=value
                )
    
    async def get_alert_history(
        self,
        *,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None
    ) -> List[Dict[str, Any]]:
        """Get alert history with filtering."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_alerts = []
        for alert in self._alert_history:
            if alert["created_at"] < cutoff_time:
                continue
            
            if severity and alert["severity"] != severity.value:
                continue
            
            if alert_type and alert["type"] != alert_type.value:
                continue
            
            filtered_alerts.append(alert)
        
        return sorted(filtered_alerts, key=lambda x: x["created_at"], reverse=True)
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the monitoring service."""
        
        return {
            "status": "healthy" if not self._shutdown else "stopped",
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "alert_history_size": len(self._alert_history),
            "monitoring_interval_seconds": self._monitoring_interval,
            "thresholds_configured": len(self._thresholds)
        }


# Global monitoring service instance
_monitoring_service: Optional[RetryMonitoringService] = None


async def get_retry_monitoring_service() -> RetryMonitoringService:
    """Get the retry monitoring service instance."""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = RetryMonitoringService()
    
    return _monitoring_service


async def start_retry_monitoring() -> None:
    """Start retry monitoring service."""
    monitoring_service = await get_retry_monitoring_service()
    await monitoring_service.start_monitoring()


async def stop_retry_monitoring() -> None:
    """Stop retry monitoring service."""
    global _monitoring_service
    
    if _monitoring_service:
        await _monitoring_service.stop_monitoring()


__all__ = [
    "RetryMonitoringService",
    "AlertSeverity",
    "AlertType",
    "get_retry_monitoring_service",
    "start_retry_monitoring",
    "stop_retry_monitoring"
]