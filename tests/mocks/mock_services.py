"""
Mock service implementations for testing.

These mocks implement the service interfaces and maintain
test data in memory while tracking method calls for verification.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone


class MockCacheService:
    """Mock cache service for testing."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.call_log: List[tuple] = []
        self.operations: List[tuple] = []  # Alias for call_log for compatibility
        self.should_fail_on_get = False
        self.should_fail_on_set = False
        self.ttl_data: Dict[str, datetime] = {}  # Track TTL for keys
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.call_log.append(("get", key))
        self.operations.append(("get", key))
        
        if self.should_fail_on_get:
            raise Exception("Mock cache get failure")
        
        # Check TTL
        if key in self.ttl_data:
            if datetime.now(timezone.utc) > self.ttl_data[key]:
                # Expired, remove from cache
                self.data.pop(key, None)
                self.ttl_data.pop(key, None)
                return None
        
        return self.data.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        self.call_log.append(("set", key, value, ttl))
        self.operations.append(("set", key, value, ttl))
        
        if self.should_fail_on_set:
            raise Exception("Mock cache set failure")
        
        self.data[key] = value
        
        # Set expiration time
        if ttl > 0:
            self.ttl_data[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        self.call_log.append(("delete", key))
        self.operations.append(("delete", key))
        
        existed = key in self.data
        self.data.pop(key, None)
        self.ttl_data.pop(key, None)
        return existed
    
    def get_operation_count(self, operation_type: str) -> int:
        """Get count of specific operation type."""
        return len([op for op in self.operations if op[0] == operation_type])
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        self.call_log.append(("exists", key))
        
        # Check TTL first
        if key in self.ttl_data:
            if datetime.now(timezone.utc) > self.ttl_data[key]:
                # Expired, remove from cache
                self.data.pop(key, None)
                self.ttl_data.pop(key, None)
                return False
        
        return key in self.data
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        self.call_log.append(("clear", pattern))
        
        if pattern == "*":
            count = len(self.data)
            self.data.clear()
            self.ttl_data.clear()
            return count
        
        # Simple pattern matching for testing
        keys_to_delete = []
        for key in self.data.keys():
            if pattern.replace("*", "") in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.data.pop(key, None)
            self.ttl_data.pop(key, None)
        
        return len(keys_to_delete)
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        return await self.clear(pattern)
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {"status": "healthy", "type": "mock_cache"}
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def was_called_with(self, method: str, *expected_args) -> bool:
        """Check if method was called with specific arguments."""
        expected_call = (method,) + expected_args
        return expected_call in self.call_log
    
    def clear_call_log(self):
        """Clear call log for testing."""
        self.call_log.clear()
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data.clear()
        self.ttl_data.clear()


class MockNotificationService:
    """Mock notification service for testing."""
    
    def __init__(self):
        self.emails_sent: List[Dict[str, Any]] = []
        self.webhooks_sent: List[Dict[str, Any]] = []
        self.push_notifications_sent: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail_on_send_email = False
        self.should_fail_on_send_webhook = False
        self.should_fail_on_send_push = False
    
    async def send_email(self, to: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email notification."""
        self.call_log.append(("send_email", to, subject, body, is_html))
        
        if self.should_fail_on_send_email:
            raise Exception("Mock email send failure")
        
        email_data = {
            "to": to,
            "subject": subject,
            "body": body,
            "is_html": is_html,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        self.emails_sent.append(email_data)
        return True
    
    async def send_webhook(
        self, url: str, payload: Dict[str, Any], secret: Optional[str] = None
    ) -> bool:
        """Send webhook notification."""
        self.call_log.append(("send_webhook", url, payload, secret))
        
        if self.should_fail_on_send_webhook:
            raise Exception("Mock webhook send failure")
        
        webhook_data = {
            "url": url,
            "payload": payload,
            "secret": secret,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        self.webhooks_sent.append(webhook_data)
        return True
    
    async def send_push_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send push notification."""
        self.call_log.append(("send_push_notification", user_id, title, message, data))
        
        if self.should_fail_on_send_push:
            raise Exception("Mock push notification send failure")
        
        push_data = {
            "user_id": user_id,
            "title": title,
            "message": message,
            "data": data or {},
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        self.push_notifications_sent.append(push_data)
        return True
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {
            "status": "healthy",
            "type": "mock_notification",
            "emails_sent": len(self.emails_sent),
            "webhooks_sent": len(self.webhooks_sent),
            "push_notifications_sent": len(self.push_notifications_sent)
        }
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def get_last_email(self) -> Optional[Dict[str, Any]]:
        """Get the last email sent."""
        return self.emails_sent[-1] if self.emails_sent else None
    
    def get_emails_to(self, recipient: str) -> List[Dict[str, Any]]:
        """Get all emails sent to a specific recipient."""
        return [email for email in self.emails_sent if email["to"] == recipient]
    
    def clear_sent_items(self):
        """Clear all sent items for testing."""
        self.emails_sent.clear()
        self.webhooks_sent.clear()
        self.push_notifications_sent.clear()
        self.call_log.clear()
    
    @property
    def sent_emails(self) -> List[Dict[str, Any]]:
        """Alias for emails_sent for test compatibility."""
        return self.emails_sent
    
    def verify_email_sent(self, to: str, subject_contains: str = None) -> bool:
        """Verify that an email was sent to a specific recipient."""
        emails_to_recipient = self.get_emails_to(to)
        if not emails_to_recipient:
            return False
        
        if subject_contains:
            return any(subject_contains in email["subject"] for email in emails_to_recipient)
        
        return True


class MockAnalyticsService:
    """Mock analytics service for testing."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.counters: Dict[str, int] = {}
        self.timings: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail_on_track = False
    
    async def track_event(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> bool:
        """Track analytics event."""
        self.call_log.append(("track_event", event_name, properties, user_id))
        
        if self.should_fail_on_track:
            raise Exception("Mock analytics track failure")
        
        event_data = {
            "event_name": event_name,
            "properties": properties,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.events.append(event_data)
        return True
    
    async def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Increment counter metric."""
        self.call_log.append(("increment_counter", metric_name, value, tags))
        
        if self.should_fail_on_track:
            raise Exception("Mock counter increment failure")
        
        # Create key with tags if provided
        counter_key = metric_name
        if tags:
            tag_string = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            counter_key = f"{metric_name}[{tag_string}]"
        
        self.counters[counter_key] = self.counters.get(counter_key, 0) + value
        return True
    
    async def record_timing(
        self,
        metric_name: str,
        duration_ms: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Record timing metric."""
        self.call_log.append(("record_timing", metric_name, duration_ms, tags))
        
        if self.should_fail_on_track:
            raise Exception("Mock timing record failure")
        
        timing_data = {
            "metric_name": metric_name,
            "duration_ms": duration_ms,
            "tags": tags or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.timings.append(timing_data)
        return True
    
    async def get_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get metrics data."""
        self.call_log.append(("get_metrics", metric_names, start_time, end_time))
        
        # Return mock metrics data
        metrics = {}
        for metric_name in metric_names:
            if metric_name in self.counters:
                metrics[metric_name] = {
                    "value": self.counters[metric_name],
                    "type": "counter"
                }
            else:
                # Find timing metrics
                relevant_timings = [
                    t for t in self.timings 
                    if t["metric_name"] == metric_name
                ]
                if relevant_timings:
                    durations = [t["duration_ms"] for t in relevant_timings]
                    metrics[metric_name] = {
                        "count": len(durations),
                        "avg": sum(durations) / len(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "type": "timing"
                    }
        
        return metrics
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {
            "status": "healthy",
            "type": "mock_analytics",
            "events_tracked": len(self.events),
            "counters": len(self.counters),
            "timings_recorded": len(self.timings)
        }
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def get_events_by_name(self, event_name: str) -> List[Dict[str, Any]]:
        """Get all events with a specific name."""
        return [event for event in self.events if event["event_name"] == event_name]
    
    def get_counter_value(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        counter_key = metric_name
        if tags:
            tag_string = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            counter_key = f"{metric_name}[{tag_string}]"
        
        return self.counters.get(counter_key, 0)
    
    def clear_analytics_data(self):
        """Clear all analytics data for testing."""
        self.events.clear()
        self.counters.clear()
        self.timings.clear()
        self.call_log.clear()


class MockSearchAnalyticsService:
    """Mock search analytics service for testing."""
    
    def __init__(self):
        self.search_events: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail_on_track = False
    
    async def track_search_event(
        self,
        event_type: str,
        search_data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Track a search event."""
        self.call_log.append(("track_search_event", event_type, search_data, tenant_id, user_id))
        
        if self.should_fail_on_track:
            raise Exception("Mock search analytics track failure")
        
        event_data = {
            "event_type": event_type,
            "search_data": search_data,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.search_events.append(event_data)
        return True
    
    async def get_search_metrics(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get search metrics for a time period."""
        self.call_log.append(("get_search_metrics", tenant_id, start_time, end_time))
        
        # Filter events by tenant and time range
        filtered_events = [
            event for event in self.search_events
            if event["tenant_id"] == tenant_id
        ]
        
        return {
            "total_searches": len(filtered_events),
            "unique_users": len(set(e["user_id"] for e in filtered_events if e["user_id"])),
            "search_types": {},
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    async def get_popular_queries(
        self,
        tenant_id: str,
        limit: int = 10,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get popular search queries."""
        self.call_log.append(("get_popular_queries", tenant_id, limit, days))
        
        # Mock popular queries
        return [
            {"query": "python developer", "count": 25},
            {"query": "react engineer", "count": 18},
            {"query": "senior backend", "count": 12},
            {"query": "frontend developer", "count": 10},
            {"query": "data scientist", "count": 8}
        ][:limit]
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {
            "status": "healthy",
            "type": "mock_search_analytics",
            "events_tracked": len(self.search_events)
        }
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type."""
        return [event for event in self.search_events if event["event_type"] == event_type]
    
    def get_events_by_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific tenant."""
        return [event for event in self.search_events if event["tenant_id"] == tenant_id]
    
    def clear_search_events(self):
        """Clear all search events for testing."""
        self.search_events.clear()
        self.call_log.clear()


class MockSecretManager:
    """Mock secret manager for testing."""
    
    def __init__(self):
        self.secrets: Dict[str, str] = {}
        self.call_log: List[tuple] = []
        self.should_fail_on_get = False
        self.should_fail_on_set = False
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve a secret value."""
        self.call_log.append(("get_secret", secret_name))
        
        if self.should_fail_on_get:
            raise Exception("Mock secret get failure")
        
        return self.secrets.get(secret_name)
    
    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Store or update a secret value."""
        self.call_log.append(("set_secret", secret_name, secret_value))
        
        if self.should_fail_on_set:
            raise Exception("Mock secret set failure")
        
        self.secrets[secret_name] = secret_value
        return True
    
    async def delete_secret(self, secret_name: str) -> bool:
        """Delete a stored secret."""
        self.call_log.append(("delete_secret", secret_name))
        
        existed = secret_name in self.secrets
        self.secrets.pop(secret_name, None)
        return existed
    
    async def list_secrets(self) -> List[str]:
        """List available secret identifiers."""
        self.call_log.append(("list_secrets",))
        return list(self.secrets.keys())
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {
            "status": "healthy",
            "type": "mock_secret_manager",
            "secrets_count": len(self.secrets)
        }
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def add_test_secret(self, name: str, value: str):
        """Add a test secret directly."""
        self.secrets[name] = value
    
    def clear_secrets(self):
        """Clear all secrets for testing."""
        self.secrets.clear()
        self.call_log.clear()


class MockEventPublisher:
    """Mock event publisher for testing."""
    
    def __init__(self):
        self.published_events: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail_on_publish = False
    
    async def publish_event(self, topic: str, event_data: Dict[str, Any]) -> bool:
        """Publish a single event."""
        self.call_log.append(("publish_event", topic, event_data))
        
        if self.should_fail_on_publish:
            raise Exception("Mock event publish failure")
        
        event_record = {
            "topic": topic,
            "event_data": event_data,
            "published_at": datetime.now(timezone.utc).isoformat()
        }
        self.published_events.append(event_record)
        return True
    
    async def publish_events(self, topic: str, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events."""
        self.call_log.append(("publish_events", topic, events))
        
        if self.should_fail_on_publish:
            raise Exception("Mock events publish failure")
        
        for event_data in events:
            event_record = {
                "topic": topic,
                "event_data": event_data,
                "published_at": datetime.now(timezone.utc).isoformat()
            }
            self.published_events.append(event_record)
        
        return True
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {
            "status": "healthy",
            "type": "mock_event_publisher",
            "events_published": len(self.published_events)
        }
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def get_events_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all events for a specific topic."""
        return [event for event in self.published_events if event["topic"] == topic]
    
    def clear_published_events(self):
        """Clear all published events for testing."""
        self.published_events.clear()
        self.call_log.clear()


class MockUsageService:
    """Mock usage service for testing."""
    
    def __init__(self):
        self.usage_records: List[Dict[str, Any]] = []
        self.call_log: List[tuple] = []
        self.should_fail_on_track = False
    
    async def track_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Track resource usage."""
        self.call_log.append(("track_usage", tenant_id, resource_type, amount, metadata))
        
        if self.should_fail_on_track:
            raise Exception("Mock usage track failure")
        
        usage_record = {
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "amount": amount,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.usage_records.append(usage_record)
        return True
    
    async def get_usage_stats(
        self,
        tenant_id: str,
        resource_type: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get usage statistics."""
        self.call_log.append(("get_usage_stats", tenant_id, resource_type, start_time, end_time))
        
        # Filter usage records
        relevant_records = [
            record for record in self.usage_records
            if record["tenant_id"] == tenant_id and record["resource_type"] == resource_type
        ]
        
        total_usage = sum(record["amount"] for record in relevant_records)
        
        return {
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "total_usage": total_usage,
            "record_count": len(relevant_records),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    async def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        requested_amount: int,
    ) -> Dict[str, Any]:
        """Check quota limits."""
        self.call_log.append(("check_quota", tenant_id, resource_type, requested_amount))
        
        # Mock quota check
        current_usage = sum(
            record["amount"] for record in self.usage_records
            if record["tenant_id"] == tenant_id and record["resource_type"] == resource_type
        )
        
        # Mock quota limits
        quota_limits = {
            "profiles": 1000,
            "searches": 500,
            "storage": 10000,
            "api_requests": 10000
        }
        
        limit = quota_limits.get(resource_type, 100)
        remaining = max(0, limit - current_usage)
        can_fulfill = remaining >= requested_amount
        
        return {
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "current_usage": current_usage,
            "limit": limit,
            "remaining": remaining,
            "requested": requested_amount,
            "can_fulfill": can_fulfill
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Return health check details."""
        return {
            "status": "healthy",
            "type": "mock_usage_service",
            "usage_records": len(self.usage_records)
        }
    
    # Test helper methods
    
    def get_call_count(self, method: str) -> int:
        """Get number of times method was called."""
        return len([call for call in self.call_log if call[0] == method])
    
    def get_usage_by_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all usage records for a tenant."""
        return [record for record in self.usage_records if record["tenant_id"] == tenant_id]
    
    def clear_usage_records(self):
        """Clear all usage records for testing."""
        self.usage_records.clear()
        self.call_log.clear()