"""
Webhook delivery statistics and monitoring service.

This service provides comprehensive statistics, monitoring, and health metrics
for webhook deliveries including success rates, performance metrics, and trends.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import statistics

import structlog

from app.domain.interfaces import IWebhookStatsService, IDatabase
from app.models.webhook_models import WebhookDeliveryStatus, WebhookFailureReason

logger = structlog.get_logger(__name__)


class WebhookStatsService(IWebhookStatsService):
    """Service for webhook delivery statistics and monitoring."""

    def __init__(self, database: IDatabase):
        """
        Initialize webhook stats service.
        
        Args:
            database: Database interface for persistence
        """
        self.database = database
        
    async def check_health(self) -> Dict[str, Any]:
        """Return webhook stats service health."""
        try:
            # Get basic statistics
            total_deliveries = await self._get_total_deliveries()
            active_endpoints = await self._get_active_endpoints_count()
            
            return {
                "status": "healthy",
                "service": "WebhookStatsService",
                "total_deliveries_tracked": total_deliveries,
                "active_endpoints": active_endpoints
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "WebhookStatsService",
                "error": str(e)
            }
    
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
        try:
            # Set default time range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=7)
            
            # Build query conditions
            conditions = ["created_at BETWEEN %s AND %s"]
            parameters = [
                {"name": "@start_date", "value": start_date},
                {"name": "@end_date", "value": end_date}
            ]
            
            if tenant_id:
                conditions.append("tenant_id = %s")
                parameters.append({"name": "@tenant_id", "value": tenant_id})
            
            if endpoint_id:
                conditions.append("endpoint_id = %s")
                parameters.append({"name": "@endpoint_id", "value": endpoint_id})
            
            if event_type:
                conditions.append("event_type = %s")
                parameters.append({"name": "@event_type", "value": event_type})
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get overall statistics
            overall_stats = await self._get_overall_stats(where_clause, parameters)
            
            # Get status breakdown
            status_breakdown = await self._get_status_breakdown(where_clause, parameters)
            
            # Get failure reason breakdown
            failure_breakdown = await self._get_failure_breakdown(where_clause, parameters)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics(where_clause, parameters)
            
            # Get daily trends
            daily_trends = await self._get_daily_trends(where_clause, parameters, start_date, end_date)
            
            return {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": (end_date - start_date).days
                },
                "filters": {
                    "tenant_id": tenant_id,
                    "endpoint_id": endpoint_id,
                    "event_type": event_type
                },
                "overall_stats": overall_stats,
                "status_breakdown": status_breakdown,
                "failure_breakdown": failure_breakdown,
                "performance_metrics": performance_metrics,
                "daily_trends": daily_trends
            }
            
        except Exception as e:
            logger.error(
                "Failed to get delivery statistics",
                tenant_id=tenant_id,
                endpoint_id=endpoint_id,
                event_type=event_type,
                start_date=start_date,
                end_date=end_date,
                error=str(e)
            )
            
            return {
                "error": str(e),
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
    
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
        try:
            start_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            end_time = datetime.utcnow()
            
            # Get endpoint configuration
            endpoint_config = await self._get_endpoint_config(endpoint_id)
            if not endpoint_config:
                return {
                    "endpoint_id": endpoint_id,
                    "error": "Endpoint not found"
                }
            
            # Get delivery statistics for this endpoint
            stats = await self.get_delivery_stats(
                endpoint_id=endpoint_id,
                start_date=start_time,
                end_date=end_time
            )
            
            # Get circuit breaker information
            circuit_info = await self._get_circuit_breaker_info(endpoint_id)
            
            # Calculate health score
            health_score = await self._calculate_endpoint_health_score(endpoint_id, stats["overall_stats"])
            
            # Get recent delivery attempts
            recent_deliveries = await self._get_recent_deliveries(endpoint_id, limit=10)
            
            return {
                "endpoint_id": endpoint_id,
                "endpoint_url": endpoint_config.get("url"),
                "endpoint_name": endpoint_config.get("name"),
                "time_window_hours": time_window_hours,
                "health_score": health_score,
                "circuit_breaker": circuit_info,
                "statistics": stats["overall_stats"],
                "performance": stats["performance_metrics"],
                "recent_deliveries": recent_deliveries,
                "status_summary": {
                    "enabled": endpoint_config.get("enabled", True),
                    "last_successful_delivery": endpoint_config.get("last_successful_delivery"),
                    "total_deliveries": endpoint_config.get("total_deliveries", 0),
                    "successful_deliveries": endpoint_config.get("successful_deliveries", 0)
                }
            }
            
        except Exception as e:
            logger.error(
                "Failed to get endpoint health",
                endpoint_id=endpoint_id,
                time_window_hours=time_window_hours,
                error=str(e)
            )
            
            return {
                "endpoint_id": endpoint_id,
                "time_window_hours": time_window_hours,
                "error": str(e)
            }
    
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
        try:
            start_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            end_time = datetime.utcnow()
            
            # Get overall statistics for tenant
            stats = await self.get_delivery_stats(
                tenant_id=tenant_id,
                start_date=start_time,
                end_date=end_time
            )
            
            # Get endpoint summary
            endpoints_summary = await self._get_tenant_endpoints_summary(tenant_id)
            
            # Get event type breakdown
            event_breakdown = await self._get_event_type_breakdown(tenant_id, start_time, end_time)
            
            # Get top failing endpoints
            failing_endpoints = await self._get_top_failing_endpoints(tenant_id, start_time, end_time)
            
            # Calculate overall tenant health
            tenant_health = await self._calculate_tenant_health(tenant_id, stats["overall_stats"])
            
            return {
                "tenant_id": tenant_id,
                "time_window_hours": time_window_hours,
                "health_score": tenant_health,
                "overall_stats": stats["overall_stats"],
                "performance_metrics": stats["performance_metrics"],
                "endpoints_summary": endpoints_summary,
                "event_type_breakdown": event_breakdown,
                "top_failing_endpoints": failing_endpoints,
                "daily_trends": stats["daily_trends"]
            }
            
        except Exception as e:
            logger.error(
                "Failed to get tenant webhook summary",
                tenant_id=tenant_id,
                time_window_hours=time_window_hours,
                error=str(e)
            )
            
            return {
                "tenant_id": tenant_id,
                "time_window_hours": time_window_hours,
                "error": str(e)
            }
    
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
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=period_days)
            
            # Get current period stats
            current_stats = await self.get_delivery_stats(
                tenant_id=tenant_id,
                start_date=start_time,
                end_date=end_time
            )
            
            # Get previous period stats for comparison
            prev_end_time = start_time
            prev_start_time = prev_end_time - timedelta(days=period_days)
            
            previous_stats = await self.get_delivery_stats(
                tenant_id=tenant_id,
                start_date=prev_start_time,
                end_date=prev_end_time
            )
            
            # Calculate changes
            changes = self._calculate_period_changes(
                current_stats["overall_stats"],
                previous_stats["overall_stats"]
            )
            
            # Get alerts and issues
            alerts = await self._generate_alerts(tenant_id, current_stats)
            
            # Get recommendations
            recommendations = await self._generate_recommendations(tenant_id, current_stats, alerts)
            
            return {
                "tenant_id": tenant_id,
                "period_days": period_days,
                "current_period": {
                    "start_date": start_time,
                    "end_date": end_time,
                    "stats": current_stats["overall_stats"]
                },
                "previous_period": {
                    "start_date": prev_start_time,
                    "end_date": prev_end_time,
                    "stats": previous_stats["overall_stats"]
                },
                "changes": changes,
                "performance_summary": current_stats["performance_metrics"],
                "alerts": alerts,
                "recommendations": recommendations,
                "top_events": current_stats.get("event_breakdown", [])[:5],
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(
                "Failed to generate stats digest",
                tenant_id=tenant_id,
                period_days=period_days,
                error=str(e)
            )
            
            return {
                "tenant_id": tenant_id,
                "period_days": period_days,
                "error": str(e),
                "generated_at": datetime.utcnow()
            }
    
    async def _get_overall_stats(self, where_clause: str, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get overall delivery statistics."""
        query = f"""
            SELECT 
                COUNT(*) as total_deliveries,
                COUNT(CASE WHEN status = 'delivered' THEN 1 END) as successful_deliveries,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_deliveries,
                COUNT(CASE WHEN status = 'dead_letter' THEN 1 END) as dead_letter_deliveries,
                COUNT(CASE WHEN attempt_number = 1 AND status = 'delivered' THEN 1 END) as first_attempt_successes,
                AVG(CASE WHEN status = 'delivered' THEN response_time_ms END) as avg_response_time_ms,
                AVG(attempt_number) as avg_attempts_per_delivery,
                COUNT(DISTINCT endpoint_id) as unique_endpoints,
                COUNT(DISTINCT event_type) as unique_event_types
            FROM webhook_deliveries {where_clause}
        """
        
        result = await self.database.query_items("webhook_deliveries", query, parameters)
        stats = result[0] if result else {}
        
        # Calculate derived metrics
        total = stats.get("total_deliveries", 0)
        successful = stats.get("successful_deliveries", 0)
        first_attempt_successes = stats.get("first_attempt_successes", 0)
        
        success_rate = (successful / total * 100) if total > 0 else 0
        first_attempt_success_rate = (first_attempt_successes / total * 100) if total > 0 else 0
        
        return {
            "total_deliveries": total,
            "successful_deliveries": successful,
            "failed_deliveries": stats.get("failed_deliveries", 0),
            "dead_letter_deliveries": stats.get("dead_letter_deliveries", 0),
            "success_rate": round(success_rate, 2),
            "first_attempt_success_rate": round(first_attempt_success_rate, 2),
            "avg_response_time_ms": round(stats.get("avg_response_time_ms", 0) or 0, 2),
            "avg_attempts_per_delivery": round(stats.get("avg_attempts_per_delivery", 0) or 0, 2),
            "unique_endpoints": stats.get("unique_endpoints", 0),
            "unique_event_types": stats.get("unique_event_types", 0)
        }
    
    async def _get_status_breakdown(self, where_clause: str, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get breakdown by delivery status."""
        query = f"""
            SELECT 
                status,
                COUNT(*) as count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
            FROM webhook_deliveries {where_clause}
            GROUP BY status
            ORDER BY count DESC
        """
        
        return await self.database.query_items("webhook_deliveries", query, parameters)
    
    async def _get_failure_breakdown(self, where_clause: str, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get breakdown by failure reason."""
        modified_where = where_clause + " AND failure_reason IS NOT NULL"
        
        query = f"""
            SELECT 
                failure_reason,
                COUNT(*) as count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
            FROM webhook_deliveries {modified_where}
            GROUP BY failure_reason
            ORDER BY count DESC
        """
        
        return await self.database.query_items("webhook_deliveries", query, parameters)
    
    async def _get_performance_metrics(self, where_clause: str, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get performance metrics."""
        query = f"""
            SELECT 
                MIN(response_time_ms) as min_response_time,
                MAX(response_time_ms) as max_response_time,
                AVG(response_time_ms) as avg_response_time,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as median_response_time,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_response_time
            FROM webhook_deliveries {where_clause}
            AND response_time_ms IS NOT NULL
        """
        
        result = await self.database.query_items("webhook_deliveries", query, parameters)
        metrics = result[0] if result else {}
        
        return {
            "min_response_time_ms": metrics.get("min_response_time"),
            "max_response_time_ms": metrics.get("max_response_time"),
            "avg_response_time_ms": round(metrics.get("avg_response_time", 0) or 0, 2),
            "median_response_time_ms": round(metrics.get("median_response_time", 0) or 0, 2),
            "p95_response_time_ms": round(metrics.get("p95_response_time", 0) or 0, 2),
            "p99_response_time_ms": round(metrics.get("p99_response_time", 0) or 0, 2)
        }
    
    async def _get_daily_trends(
        self,
        where_clause: str,
        parameters: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get daily delivery trends."""
        query = f"""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total_deliveries,
                COUNT(CASE WHEN status = 'delivered' THEN 1 END) as successful_deliveries,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_deliveries,
                AVG(CASE WHEN status = 'delivered' THEN response_time_ms END) as avg_response_time
            FROM webhook_deliveries {where_clause}
            GROUP BY DATE(created_at)
            ORDER BY date
        """
        
        trends = await self.database.query_items("webhook_deliveries", query, parameters)
        
        # Calculate success rate for each day
        for trend in trends:
            total = trend.get("total_deliveries", 0)
            successful = trend.get("successful_deliveries", 0)
            trend["success_rate"] = (successful / total * 100) if total > 0 else 0
        
        return trends
    
    async def _get_endpoint_config(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get endpoint configuration."""
        try:
            return await self.database.get_item("webhook_endpoints", endpoint_id, endpoint_id)
        except Exception as e:
            logger.warning("Failed to get endpoint config", endpoint_id=endpoint_id, error=str(e))
            return None
    
    async def _get_circuit_breaker_info(self, endpoint_id: str) -> Dict[str, Any]:
        """Get circuit breaker information for endpoint."""
        try:
            endpoint_config = await self._get_endpoint_config(endpoint_id)
            if not endpoint_config:
                return {"state": "unknown"}
            
            return {
                "state": endpoint_config.get("circuit_state", "closed"),
                "failure_count": endpoint_config.get("failure_count", 0),
                "last_failure_at": endpoint_config.get("last_failure_at"),
                "circuit_opened_at": endpoint_config.get("circuit_opened_at")
            }
        except Exception as e:
            logger.warning("Failed to get circuit breaker info", endpoint_id=endpoint_id, error=str(e))
            return {"state": "unknown", "error": str(e)}
    
    async def _calculate_endpoint_health_score(self, endpoint_id: str, stats: Dict[str, Any]) -> float:
        """Calculate health score for endpoint (0-100)."""
        try:
            success_rate = stats.get("success_rate", 0)
            total_deliveries = stats.get("total_deliveries", 0)
            avg_response_time = stats.get("avg_response_time_ms", 0)
            
            # Base score from success rate
            score = success_rate
            
            # Penalty for slow response times (>5s)
            if avg_response_time > 5000:
                score *= 0.8
            elif avg_response_time > 2000:
                score *= 0.9
            
            # Bonus for sufficient volume (confidence)
            if total_deliveries >= 100:
                score *= 1.0
            elif total_deliveries >= 10:
                score *= 0.95
            elif total_deliveries > 0:
                score *= 0.8
            else:
                score = 0
            
            return round(min(100, max(0, score)), 1)
            
        except Exception:
            return 0.0
    
    async def _get_recent_deliveries(self, endpoint_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent delivery attempts for endpoint."""
        try:
            query = """
                SELECT id, event_type, status, attempt_number, response_time_ms, 
                       created_at, delivered_at, failure_reason
                FROM webhook_deliveries 
                WHERE endpoint_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """
            
            return await self.database.query_items(
                "webhook_deliveries",
                query,
                [
                    {"name": "@endpoint_id", "value": endpoint_id},
                    {"name": "@limit", "value": limit}
                ]
            )
        except Exception as e:
            logger.warning("Failed to get recent deliveries", endpoint_id=endpoint_id, error=str(e))
            return []
    
    async def _get_tenant_endpoints_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get summary of endpoints for tenant."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_endpoints,
                    COUNT(CASE WHEN enabled = true THEN 1 END) as enabled_endpoints,
                    COUNT(CASE WHEN circuit_state = 'open' THEN 1 END) as circuit_open_endpoints
                FROM webhook_endpoints 
                WHERE tenant_id = %s
            """
            
            result = await self.database.query_items(
                "webhook_endpoints",
                query,
                [{"name": "@tenant_id", "value": tenant_id}]
            )
            
            return result[0] if result else {
                "total_endpoints": 0,
                "enabled_endpoints": 0,
                "circuit_open_endpoints": 0
            }
        except Exception as e:
            logger.warning("Failed to get tenant endpoints summary", tenant_id=tenant_id, error=str(e))
            return {}
    
    async def _get_event_type_breakdown(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get breakdown by event type for tenant."""
        try:
            query = """
                SELECT 
                    event_type,
                    COUNT(*) as count,
                    COUNT(CASE WHEN status = 'delivered' THEN 1 END) as successful_count
                FROM webhook_deliveries 
                WHERE tenant_id = %s AND created_at BETWEEN %s AND %s
                GROUP BY event_type
                ORDER BY count DESC
            """
            
            return await self.database.query_items(
                "webhook_deliveries",
                query,
                [
                    {"name": "@tenant_id", "value": tenant_id},
                    {"name": "@start_time", "value": start_time},
                    {"name": "@end_time", "value": end_time}
                ]
            )
        except Exception as e:
            logger.warning("Failed to get event type breakdown", tenant_id=tenant_id, error=str(e))
            return []
    
    async def _get_top_failing_endpoints(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top failing endpoints for tenant."""
        try:
            query = """
                SELECT 
                    endpoint_id,
                    COUNT(*) as total_deliveries,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_deliveries,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) * 100.0 / COUNT(*) as failure_rate
                FROM webhook_deliveries 
                WHERE tenant_id = %s AND created_at BETWEEN %s AND %s
                GROUP BY endpoint_id
                HAVING COUNT(*) >= 5  -- Only endpoints with sufficient volume
                ORDER BY failure_rate DESC, failed_deliveries DESC
                LIMIT %s
            """
            
            return await self.database.query_items(
                "webhook_deliveries",
                query,
                [
                    {"name": "@tenant_id", "value": tenant_id},
                    {"name": "@start_time", "value": start_time},
                    {"name": "@end_time", "value": end_time},
                    {"name": "@limit", "value": limit}
                ]
            )
        except Exception as e:
            logger.warning("Failed to get top failing endpoints", tenant_id=tenant_id, error=str(e))
            return []
    
    async def _calculate_tenant_health(self, tenant_id: str, stats: Dict[str, Any]) -> float:
        """Calculate overall health score for tenant."""
        return await self._calculate_endpoint_health_score("tenant", stats)
    
    def _calculate_period_changes(
        self,
        current_stats: Dict[str, Any],
        previous_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate changes between periods."""
        changes = {}
        
        metrics = [
            "total_deliveries",
            "successful_deliveries", 
            "failed_deliveries",
            "success_rate",
            "avg_response_time_ms"
        ]
        
        for metric in metrics:
            current = current_stats.get(metric, 0)
            previous = previous_stats.get(metric, 0)
            
            if previous > 0:
                change_percent = ((current - previous) / previous) * 100
            else:
                change_percent = 100 if current > 0 else 0
            
            changes[metric] = {
                "current": current,
                "previous": previous,
                "change": current - previous,
                "change_percent": round(change_percent, 2)
            }
        
        return changes
    
    async def _generate_alerts(self, tenant_id: str, stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts based on statistics."""
        alerts = []
        overall_stats = stats["overall_stats"]
        
        # Low success rate alert
        success_rate = overall_stats.get("success_rate", 100)
        if success_rate < 90:
            severity = "critical" if success_rate < 50 else "warning"
            alerts.append({
                "type": "low_success_rate",
                "severity": severity,
                "message": f"Success rate is {success_rate:.1f}%, below threshold",
                "value": f"{success_rate:.1f}%"
            })
        
        # High response time alert
        avg_response_time = overall_stats.get("avg_response_time_ms", 0)
        if avg_response_time > 5000:
            alerts.append({
                "type": "high_response_time",
                "severity": "warning",
                "message": f"Average response time is {avg_response_time:.0f}ms",
                "value": f"{avg_response_time:.0f}ms"
            })
        
        # Dead letters alert
        dead_letters = overall_stats.get("dead_letter_deliveries", 0)
        if dead_letters > 0:
            alerts.append({
                "type": "dead_letters",
                "severity": "warning",
                "message": f"{dead_letters} deliveries in dead letter queue",
                "value": str(dead_letters)
            })
        
        return alerts
    
    async def _generate_recommendations(
        self,
        tenant_id: str,
        stats: Dict[str, Any],
        alerts: List[Dict[str, str]]
    ) -> List[str]:
        """Generate recommendations based on statistics and alerts."""
        recommendations = []
        overall_stats = stats["overall_stats"]
        
        # Recommendations based on alerts
        for alert in alerts:
            if alert["type"] == "low_success_rate":
                recommendations.append(
                    "Review failing webhook endpoints and check endpoint URLs and configurations"
                )
            elif alert["type"] == "high_response_time":
                recommendations.append(
                    "Consider increasing timeout values or optimizing webhook endpoint performance"
                )
            elif alert["type"] == "dead_letters":
                recommendations.append(
                    "Review dead letter queue and manually retry failed deliveries if appropriate"
                )
        
        # Performance recommendations
        avg_attempts = overall_stats.get("avg_attempts_per_delivery", 1)
        if avg_attempts > 2:
            recommendations.append(
                "High retry rate detected - consider reviewing endpoint reliability"
            )
        
        # General recommendations
        if overall_stats.get("total_deliveries", 0) == 0:
            recommendations.append(
                "No webhook deliveries in this period - ensure webhooks are properly configured"
            )
        
        return recommendations
    
    async def _get_total_deliveries(self) -> int:
        """Get total number of deliveries tracked."""
        try:
            query = "SELECT COUNT(*) as count FROM webhook_deliveries"
            result = await self.database.query_items("webhook_deliveries", query)
            return result[0]["count"] if result else 0
        except Exception:
            return 0
    
    async def _get_active_endpoints_count(self) -> int:
        """Get number of active endpoints."""
        try:
            query = "SELECT COUNT(*) as count FROM webhook_endpoints WHERE enabled = true"
            result = await self.database.query_items("webhook_endpoints", query)
            return result[0]["count"] if result else 0
        except Exception:
            return 0


__all__ = ["WebhookStatsService"]