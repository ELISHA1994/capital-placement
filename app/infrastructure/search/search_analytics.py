"""
Search Analytics Service for Performance Tracking and Insights

Comprehensive analytics and monitoring for search operations:
- Real-time performance metrics and monitoring
- Search pattern analysis and optimization insights
- User behavior tracking and click-through rates  
- AI operation cost analysis and token usage
- Cache performance and optimization recommendations
- Tenant-specific analytics and reporting
- Automated performance alerts and notifications
- Historical trend analysis and forecasting
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from uuid import UUID, uuid4
import statistics

from app.core.config import get_settings
from app.domain.interfaces import IHealthCheck, IAnalyticsService
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    SEARCH_PERFORMANCE = "search_performance"
    SEARCH_QUALITY = "search_quality"
    USER_ENGAGEMENT = "user_engagement"
    AI_OPERATIONS = "ai_operations"
    CACHE_PERFORMANCE = "cache_performance"
    COST_ANALYSIS = "cost_analysis"


class AggregationType(Enum):
    """Aggregation methods for metrics"""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE_95 = "percentile_95"
    MIN = "min"
    MAX = "max"


@dataclass
class SearchMetric:
    """Individual search metric data point"""
    metric_name: str
    value: Union[int, float]
    timestamp: datetime
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    search_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    report_id: str
    tenant_id: Optional[str]
    report_type: str
    time_period: Dict[str, datetime]
    metrics: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    generated_at: datetime
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceAlert:
    """Performance alert configuration"""
    alert_id: str
    name: str
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==
    tenant_id: Optional[str]
    enabled: bool
    last_triggered: Optional[datetime] = None
    notification_channels: List[str] = None


class SearchAnalyticsService(IAnalyticsService):
    """
    Advanced search analytics service for comprehensive monitoring.
    
    Features:
    - Real-time performance monitoring and alerting
    - Comprehensive search pattern analysis
    - User behavior tracking and engagement metrics
    - AI operation cost analysis and optimization
    - Cache performance monitoring and recommendations
    - Automated insights generation and reporting
    - Historical trend analysis and forecasting
    - Multi-tenant analytics with data isolation
    """
    
    def __init__(
        self,
        db_adapter: PostgresAdapter,
        notification_service: Optional[Any] = None
    ):
        self.settings = get_settings()
        self.db_adapter = db_adapter
        self.notification_service = notification_service
        
        # Analytics configuration
        self.default_retention_days = 90
        self.alert_check_interval = 300  # 5 minutes
        self.batch_size = 1000
        
        # In-memory metrics cache for real-time tracking
        self._recent_metrics = []
        self._max_recent_metrics = 10000
        
        # Performance alerts
        self._active_alerts: List[PerformanceAlert] = []
        
        # Service statistics
        self._stats = {
            "metrics_recorded": 0,
            "reports_generated": 0,
            "alerts_triggered": 0,
            "insights_generated": 0,
            "database_writes": 0,
            "cache_operations": 0,
            "errors": 0
        }
        
    async def track_search_event(
        self,
        event_type: str,
        search_data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Track search-related events for analytics.
        
        Args:
            event_type: Type of search event (search_executed, result_clicked, etc.)
            search_data: Search event data and metrics
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            Success status of tracking operation
        """
        try:
            timestamp = datetime.now()
            
            # Create base metrics from search data
            metrics = await self._extract_search_metrics(event_type, search_data, timestamp)
            
            # Add tenant and user context
            for metric in metrics:
                metric.tenant_id = tenant_id
                metric.user_id = user_id
            
            # Store in database
            await self._store_metrics_batch(metrics)
            
            # Add to recent metrics cache
            self._recent_metrics.extend(metrics)
            if len(self._recent_metrics) > self._max_recent_metrics:
                self._recent_metrics = self._recent_metrics[-self._max_recent_metrics:]
            
            # Check for performance alerts
            await self._check_performance_alerts(metrics)
            
            self._stats["metrics_recorded"] += len(metrics)
            
            logger.debug(
                "Search event tracked",
                event_type=event_type,
                metrics_count=len(metrics),
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to track search event: {e}")
            return False
    
    async def track_event(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: str = None
    ) -> bool:
        """Implementation of IAnalyticsService interface"""
        return await self.track_search_event(
            event_type=event_name,
            search_data=properties,
            tenant_id=properties.get("tenant_id"),
            user_id=user_id
        )
    
    async def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        tags: Dict[str, str] = None
    ) -> bool:
        """Implementation of IAnalyticsService interface"""
        try:
            metric = SearchMetric(
                metric_name=metric_name,
                value=value,
                timestamp=datetime.now(),
                tenant_id=tags.get("tenant_id") if tags else None,
                user_id=tags.get("user_id") if tags else None,
                metadata=tags or {}
            )
            
            await self._store_metrics_batch([metric])
            return True
            
        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
            return False
    
    async def record_timing(
        self,
        metric_name: str,
        duration_ms: int,
        tags: Dict[str, str] = None
    ) -> bool:
        """Implementation of IAnalyticsService interface"""
        try:
            metric = SearchMetric(
                metric_name=f"{metric_name}_duration_ms",
                value=duration_ms,
                timestamp=datetime.now(),
                tenant_id=tags.get("tenant_id") if tags else None,
                user_id=tags.get("user_id") if tags else None,
                metadata=tags or {}
            )
            
            await self._store_metrics_batch([metric])
            return True
            
        except Exception as e:
            logger.error(f"Failed to record timing: {e}")
            return False
    
    async def get_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Implementation of IAnalyticsService interface"""
        return await self.get_search_metrics(
            metric_names=metric_names,
            start_date=start_time,
            end_date=end_time
        )
    
    async def get_search_metrics(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        metric_names: Optional[List[str]] = None,
        aggregation: AggregationType = AggregationType.AVERAGE
    ) -> Dict[str, Any]:
        """
        Retrieve search metrics with flexible filtering and aggregation.
        
        Args:
            tenant_id: Filter by tenant ID
            start_date: Start date for metrics range
            end_date: End date for metrics range
            metric_names: Specific metrics to retrieve
            aggregation: Aggregation method for metrics
            
        Returns:
            Dictionary containing requested metrics and metadata
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=7)
            
            # Build query conditions
            conditions = ["timestamp >= $1 AND timestamp <= $2"]
            params = [start_date, end_date]
            param_count = 2
            
            if tenant_id:
                param_count += 1
                conditions.append(f"tenant_id = ${param_count}")
                params.append(tenant_id)
            
            if metric_names:
                param_count += 1
                conditions.append(f"metric_name = ANY(${param_count})")
                params.append(metric_names)
            
            where_clause = " AND ".join(conditions)
            
            # Execute query based on aggregation type
            async with self.db_adapter.get_connection() as conn:
                if aggregation == AggregationType.COUNT:
                    query = f"""
                        SELECT metric_name, COUNT(*) as value
                        FROM search_metrics 
                        WHERE {where_clause}
                        GROUP BY metric_name
                        ORDER BY metric_name
                    """
                elif aggregation == AggregationType.SUM:
                    query = f"""
                        SELECT metric_name, SUM(value) as value
                        FROM search_metrics 
                        WHERE {where_clause}
                        GROUP BY metric_name
                        ORDER BY metric_name
                    """
                elif aggregation == AggregationType.AVERAGE:
                    query = f"""
                        SELECT metric_name, AVG(value) as value
                        FROM search_metrics 
                        WHERE {where_clause}
                        GROUP BY metric_name
                        ORDER BY metric_name
                    """
                elif aggregation == AggregationType.PERCENTILE_95:
                    query = f"""
                        SELECT metric_name, PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as value
                        FROM search_metrics 
                        WHERE {where_clause}
                        GROUP BY metric_name
                        ORDER BY metric_name
                    """
                else:
                    query = f"""
                        SELECT metric_name, MIN(value) as min_value, MAX(value) as max_value, 
                               AVG(value) as avg_value, COUNT(*) as count
                        FROM search_metrics 
                        WHERE {where_clause}
                        GROUP BY metric_name
                        ORDER BY metric_name
                    """
                
                rows = await conn.fetch(query, *params)
                
                # Format results
                metrics = {}
                for row in rows:
                    if aggregation in [AggregationType.MIN, AggregationType.MAX]:
                        metrics[row['metric_name']] = {
                            "min": float(row['min_value']) if row['min_value'] else 0,
                            "max": float(row['max_value']) if row['max_value'] else 0,
                            "average": float(row['avg_value']) if row['avg_value'] else 0,
                            "count": int(row['count'])
                        }
                    else:
                        metrics[row['metric_name']] = float(row['value']) if row['value'] else 0
                
                return {
                    "metrics": metrics,
                    "time_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "aggregation": aggregation.value,
                    "tenant_id": tenant_id,
                    "total_metrics": len(metrics)
                }
                
        except Exception as e:
            logger.error(f"Failed to get search metrics: {e}")
            return {"metrics": {}, "error": str(e)}
    
    async def generate_performance_report(
        self,
        tenant_id: Optional[str] = None,
        report_type: str = "comprehensive",
        time_period_days: int = 7
    ) -> AnalyticsReport:
        """
        Generate comprehensive performance report with insights.
        
        Args:
            tenant_id: Tenant ID for report scope
            report_type: Type of report to generate
            time_period_days: Number of days to include in report
            
        Returns:
            AnalyticsReport with metrics, insights, and recommendations
        """
        try:
            report_id = str(uuid4())
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_period_days)
            
            # Gather comprehensive metrics
            metrics = {}
            
            # Performance metrics
            performance_metrics = await self.get_search_metrics(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                metric_names=[
                    "search_duration_ms", "vector_search_duration_ms",
                    "text_search_duration_ms", "reranking_duration_ms",
                    "cache_hit_rate", "results_returned"
                ]
            )
            metrics["performance"] = performance_metrics["metrics"]
            
            # Search quality metrics
            quality_metrics = await self.get_search_metrics(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                metric_names=[
                    "click_through_rate", "result_relevance_score",
                    "query_success_rate", "zero_results_rate"
                ]
            )
            metrics["quality"] = quality_metrics["metrics"]
            
            # User engagement metrics
            engagement_metrics = await self.get_search_metrics(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                metric_names=[
                    "unique_users", "searches_per_user",
                    "session_duration", "bounce_rate"
                ],
                aggregation=AggregationType.COUNT
            )
            metrics["engagement"] = engagement_metrics["metrics"]
            
            # AI operations cost analysis
            cost_metrics = await self._get_ai_cost_analysis(tenant_id, start_date, end_date)
            metrics["cost_analysis"] = cost_metrics
            
            # Generate insights
            insights = await self._generate_insights(metrics, tenant_id)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(metrics, insights)
            
            # Create report
            report = AnalyticsReport(
                report_id=report_id,
                tenant_id=tenant_id,
                report_type=report_type,
                time_period={
                    "start": start_date,
                    "end": end_date
                },
                metrics=metrics,
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.now(),
                metadata={
                    "time_period_days": time_period_days,
                    "total_metrics": sum(len(m) for m in metrics.values() if isinstance(m, dict))
                }
            )
            
            # Store report in database
            await self._store_analytics_report(report)
            
            self._stats["reports_generated"] += 1
            
            logger.info(
                "Performance report generated",
                report_id=report_id,
                tenant_id=tenant_id,
                report_type=report_type,
                time_period_days=time_period_days,
                insights_count=len(insights),
                recommendations_count=len(recommendations)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            raise
    
    async def get_real_time_metrics(
        self,
        tenant_id: Optional[str] = None,
        metric_types: Optional[List[MetricType]] = None
    ) -> Dict[str, Any]:
        """
        Get real-time metrics from in-memory cache and recent database records.
        
        Args:
            tenant_id: Filter by tenant ID
            metric_types: Types of metrics to include
            
        Returns:
            Real-time metrics summary
        """
        try:
            current_time = datetime.now()
            
            # Filter recent metrics by tenant and type
            filtered_metrics = self._recent_metrics
            if tenant_id:
                filtered_metrics = [m for m in filtered_metrics if m.tenant_id == tenant_id]
            
            # Group metrics by name and calculate real-time stats
            metric_groups = {}
            for metric in filtered_metrics:
                if metric.metric_name not in metric_groups:
                    metric_groups[metric.metric_name] = []
                metric_groups[metric.metric_name].append(metric)
            
            real_time_stats = {}
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) > 0:
                    values = [m.value for m in metric_list]
                    recent_values = [
                        m.value for m in metric_list 
                        if (current_time - m.timestamp).seconds < 300  # Last 5 minutes
                    ]
                    
                    real_time_stats[metric_name] = {
                        "current_value": values[-1] if values else 0,
                        "recent_average": sum(recent_values) / len(recent_values) if recent_values else 0,
                        "total_count": len(metric_list),
                        "recent_count": len(recent_values),
                        "last_updated": metric_list[-1].timestamp.isoformat() if metric_list else None
                    }
            
            # Add system health indicators
            system_health = await self._calculate_system_health(real_time_stats)
            
            return {
                "real_time_metrics": real_time_stats,
                "system_health": system_health,
                "timestamp": current_time.isoformat(),
                "tenant_id": tenant_id,
                "metrics_in_cache": len(filtered_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {"error": str(e)}
    
    async def setup_performance_alert(
        self,
        alert_config: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Set up automated performance alert.
        
        Args:
            alert_config: Alert configuration parameters
            tenant_id: Tenant ID for alert scope
            
        Returns:
            Alert ID for management
        """
        try:
            alert = PerformanceAlert(
                alert_id=str(uuid4()),
                name=alert_config["name"],
                metric_name=alert_config["metric_name"],
                threshold_value=float(alert_config["threshold"]),
                comparison_operator=alert_config.get("operator", ">"),
                tenant_id=tenant_id,
                enabled=alert_config.get("enabled", True),
                notification_channels=alert_config.get("notification_channels", [])
            )
            
            # Store alert configuration
            async with self.db_adapter.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO performance_alerts (
                        id, name, metric_name, threshold_value, comparison_operator,
                        tenant_id, enabled, notification_channels, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                alert.alert_id, alert.name, alert.metric_name,
                alert.threshold_value, alert.comparison_operator,
                alert.tenant_id, alert.enabled,
                json.dumps(alert.notification_channels or []),
                datetime.now()
                )
            
            # Add to active alerts
            self._active_alerts.append(alert)
            
            logger.info(
                "Performance alert created",
                alert_id=alert.alert_id,
                name=alert.name,
                metric_name=alert.metric_name,
                tenant_id=tenant_id
            )
            
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"Failed to setup performance alert: {e}")
            raise
    
    async def _extract_search_metrics(
        self,
        event_type: str,
        search_data: Dict[str, Any],
        timestamp: datetime
    ) -> List[SearchMetric]:
        """Extract metrics from search event data"""
        
        metrics = []
        
        try:
            # Basic search metrics
            if "search_duration_ms" in search_data:
                metrics.append(SearchMetric(
                    metric_name="search_duration_ms",
                    value=search_data["search_duration_ms"],
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={"event_type": event_type}
                ))
            
            if "results_count" in search_data:
                metrics.append(SearchMetric(
                    metric_name="results_returned",
                    value=search_data["results_count"],
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={"event_type": event_type}
                ))
            
            # Cache performance
            if "cache_hit" in search_data:
                metrics.append(SearchMetric(
                    metric_name="cache_hit_rate",
                    value=1.0 if search_data["cache_hit"] else 0.0,
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={"event_type": event_type}
                ))
            
            # AI operation metrics
            if "token_usage" in search_data and isinstance(search_data["token_usage"], dict):
                token_usage = search_data["token_usage"]
                if "total_tokens" in token_usage:
                    metrics.append(SearchMetric(
                        metric_name="ai_tokens_used",
                        value=token_usage["total_tokens"],
                        timestamp=timestamp,
                        search_id=search_data.get("search_id"),
                        metadata={"event_type": event_type, "ai_model": search_data.get("ai_model")}
                    ))
            
            # Vector search specific metrics
            if "vector_search_duration_ms" in search_data:
                metrics.append(SearchMetric(
                    metric_name="vector_search_duration_ms",
                    value=search_data["vector_search_duration_ms"],
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={"event_type": event_type}
                ))
            
            # Text search specific metrics
            if "text_search_duration_ms" in search_data:
                metrics.append(SearchMetric(
                    metric_name="text_search_duration_ms",
                    value=search_data["text_search_duration_ms"],
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={"event_type": event_type}
                ))
            
            # Reranking metrics
            if "reranking_duration_ms" in search_data:
                metrics.append(SearchMetric(
                    metric_name="reranking_duration_ms",
                    value=search_data["reranking_duration_ms"],
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={"event_type": event_type}
                ))
            
            # Click-through tracking
            if event_type == "result_clicked":
                metrics.append(SearchMetric(
                    metric_name="result_clicks",
                    value=1,
                    timestamp=timestamp,
                    search_id=search_data.get("search_id"),
                    metadata={
                        "event_type": event_type,
                        "result_position": search_data.get("position", 0),
                        "result_id": search_data.get("result_id")
                    }
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to extract search metrics: {e}")
            return []
    
    async def _store_metrics_batch(self, metrics: List[SearchMetric]) -> None:
        """Store a batch of metrics in the database"""
        
        try:
            if not metrics:
                return
            
            # First ensure the table exists (you might need to create it)
            async with self.db_adapter.get_connection() as conn:
                # Create table if it doesn't exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_metrics (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        metric_name VARCHAR(255) NOT NULL,
                        value DECIMAL NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        tenant_id UUID,
                        user_id UUID,
                        search_id VARCHAR(255),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create index for efficient querying
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_search_metrics_timestamp_metric 
                    ON search_metrics(timestamp, metric_name)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_search_metrics_tenant 
                    ON search_metrics(tenant_id, timestamp)
                """)
                
                # Batch insert metrics (asyncpg doesn't support executemany, use individual inserts)
                for metric in metrics:
                    await conn.execute("""
                        INSERT INTO search_metrics (
                            metric_name, value, timestamp, tenant_id,
                            user_id, search_id, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    metric.metric_name,
                    metric.value,
                    metric.timestamp,
                    metric.tenant_id,
                    metric.user_id,
                    metric.search_id,
                    json.dumps(metric.metadata or {})
                    )
                
                self._stats["database_writes"] += len(metrics)
                
        except Exception as e:
            logger.error(f"Failed to store metrics batch: {e}")
            raise
    
    async def _get_ai_cost_analysis(
        self,
        tenant_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get AI operation cost analysis"""
        
        try:
            async with self.db_adapter.get_connection() as conn:
                # Query AI analytics table
                conditions = ["created_at >= $1 AND created_at <= $2"]
                params = [start_date, end_date]
                
                if tenant_id:
                    conditions.append("tenant_id = $3")
                    params.append(tenant_id)
                
                where_clause = " AND ".join(conditions)
                
                cost_data = await conn.fetch(f"""
                    SELECT 
                        ai_model,
                        operation_type,
                        COUNT(*) as operation_count,
                        SUM(COALESCE((token_usage->>'total_tokens')::INTEGER, 0)) as total_tokens,
                        SUM(COALESCE(cost_estimate, 0)) as total_cost,
                        AVG(processing_time_ms) as avg_processing_time,
                        COUNT(CASE WHEN success THEN 1 END)::DECIMAL / COUNT(*) as success_rate
                    FROM ai_analytics
                    WHERE {where_clause}
                    GROUP BY ai_model, operation_type
                    ORDER BY total_cost DESC
                """, *params)
                
                cost_analysis = {
                    "total_operations": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "average_processing_time": 0.0,
                    "overall_success_rate": 0.0,
                    "by_model": {},
                    "by_operation": {}
                }
                
                for row in cost_data:
                    cost_analysis["total_operations"] += row["operation_count"]
                    cost_analysis["total_cost"] += float(row["total_cost"] or 0)
                    cost_analysis["total_tokens"] += row["total_tokens"]
                    
                    # Group by model
                    model = row["ai_model"]
                    if model not in cost_analysis["by_model"]:
                        cost_analysis["by_model"][model] = {
                            "operations": 0,
                            "cost": 0.0,
                            "tokens": 0,
                            "success_rate": 0.0
                        }
                    
                    cost_analysis["by_model"][model]["operations"] += row["operation_count"]
                    cost_analysis["by_model"][model]["cost"] += float(row["total_cost"] or 0)
                    cost_analysis["by_model"][model]["tokens"] += row["total_tokens"]
                    cost_analysis["by_model"][model]["success_rate"] = float(row["success_rate"])
                    
                    # Group by operation
                    operation = row["operation_type"]
                    if operation not in cost_analysis["by_operation"]:
                        cost_analysis["by_operation"][operation] = {
                            "operations": 0,
                            "cost": 0.0,
                            "tokens": 0,
                            "success_rate": 0.0
                        }
                    
                    cost_analysis["by_operation"][operation]["operations"] += row["operation_count"]
                    cost_analysis["by_operation"][operation]["cost"] += float(row["total_cost"] or 0)
                    cost_analysis["by_operation"][operation]["tokens"] += row["total_tokens"]
                    cost_analysis["by_operation"][operation]["success_rate"] = float(row["success_rate"])
                
                return cost_analysis
                
        except Exception as e:
            logger.error(f"Failed to get AI cost analysis: {e}")
            return {"error": str(e)}
    
    async def _generate_insights(
        self,
        metrics: Dict[str, Any],
        tenant_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate insights from metrics data"""
        
        insights = []
        
        try:
            # Performance insights
            performance_metrics = metrics.get("performance", {})
            if "search_duration_ms" in performance_metrics:
                avg_duration = performance_metrics["search_duration_ms"]
                if avg_duration > 2000:  # > 2 seconds
                    insights.append({
                        "type": "performance_warning",
                        "title": "Slow Search Performance",
                        "description": f"Average search duration is {avg_duration:.0f}ms, exceeding 2-second target",
                        "severity": "high",
                        "recommendation": "Consider optimizing vector indexes or increasing cache usage"
                    })
                elif avg_duration < 500:  # < 500ms
                    insights.append({
                        "type": "performance_good",
                        "title": "Excellent Search Performance",
                        "description": f"Average search duration is {avg_duration:.0f}ms, well within targets",
                        "severity": "info"
                    })
            
            # Cache effectiveness insights
            if "cache_hit_rate" in performance_metrics:
                cache_rate = performance_metrics["cache_hit_rate"]
                if cache_rate < 0.3:  # < 30%
                    insights.append({
                        "type": "cache_warning",
                        "title": "Low Cache Hit Rate",
                        "description": f"Cache hit rate is {cache_rate:.1%}, indicating potential for optimization",
                        "severity": "medium",
                        "recommendation": "Review cache TTL settings and query patterns"
                    })
                elif cache_rate > 0.7:  # > 70%
                    insights.append({
                        "type": "cache_good",
                        "title": "High Cache Efficiency",
                        "description": f"Cache hit rate is {cache_rate:.1%}, indicating effective caching strategy",
                        "severity": "info"
                    })
            
            # Cost optimization insights
            cost_analysis = metrics.get("cost_analysis", {})
            if "total_cost" in cost_analysis and cost_analysis["total_cost"] > 0:
                total_cost = cost_analysis["total_cost"]
                total_operations = cost_analysis.get("total_operations", 1)
                cost_per_operation = total_cost / total_operations
                
                if cost_per_operation > 0.10:  # > $0.10 per operation
                    insights.append({
                        "type": "cost_warning",
                        "title": "High AI Operation Costs",
                        "description": f"Average cost per operation is ${cost_per_operation:.3f}",
                        "severity": "medium",
                        "recommendation": "Consider using smaller models or implementing better caching"
                    })
                
                # Model efficiency insights
                by_model = cost_analysis.get("by_model", {})
                if len(by_model) > 1:
                    most_expensive_model = max(by_model.items(), key=lambda x: x[1]["cost"])
                    insights.append({
                        "type": "cost_analysis",
                        "title": "Model Cost Analysis",
                        "description": f"'{most_expensive_model[0]}' accounts for largest AI costs",
                        "severity": "info",
                        "metadata": {"model_costs": by_model}
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    async def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        try:
            # Performance recommendations
            high_severity_insights = [i for i in insights if i.get("severity") == "high"]
            if high_severity_insights:
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "title": "Immediate Performance Optimization Required",
                    "actions": [
                        "Review and optimize vector search indexes",
                        "Increase cache TTL for frequently accessed queries",
                        "Consider implementing query result pagination",
                        "Monitor database query performance"
                    ],
                    "expected_impact": "30-50% reduction in search latency"
                })
            
            # Cache optimization recommendations
            cache_warnings = [i for i in insights if i.get("type") == "cache_warning"]
            if cache_warnings:
                recommendations.append({
                    "type": "cache_optimization",
                    "priority": "medium",
                    "title": "Cache Strategy Optimization",
                    "actions": [
                        "Analyze query patterns to identify cacheable queries",
                        "Implement semantic similarity cache matching",
                        "Increase cache memory allocation",
                        "Review cache invalidation strategies"
                    ],
                    "expected_impact": "20-40% improvement in response times"
                })
            
            # Cost optimization recommendations
            cost_warnings = [i for i in insights if i.get("type") == "cost_warning"]
            if cost_warnings:
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "medium",
                    "title": "AI Operation Cost Reduction",
                    "actions": [
                        "Implement more aggressive caching for AI operations",
                        "Use smaller, more efficient models where appropriate",
                        "Batch AI operations to reduce API calls",
                        "Implement request deduplication"
                    ],
                    "expected_impact": "25-45% reduction in AI operation costs"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def _calculate_system_health(
        self,
        real_time_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall system health indicators"""
        
        try:
            health_score = 100.0  # Start with perfect health
            health_indicators = {}
            
            # Performance health
            if "search_duration_ms" in real_time_stats:
                avg_duration = real_time_stats["search_duration_ms"]["recent_average"]
                if avg_duration > 2000:
                    health_score -= 20
                    health_indicators["performance"] = "poor"
                elif avg_duration > 1000:
                    health_score -= 10
                    health_indicators["performance"] = "fair"
                else:
                    health_indicators["performance"] = "good"
            
            # Cache health
            if "cache_hit_rate" in real_time_stats:
                cache_rate = real_time_stats["cache_hit_rate"]["recent_average"]
                if cache_rate < 0.3:
                    health_score -= 15
                    health_indicators["cache"] = "poor"
                elif cache_rate < 0.6:
                    health_score -= 5
                    health_indicators["cache"] = "fair"
                else:
                    health_indicators["cache"] = "good"
            
            # Error rate health
            error_indicators = [
                key for key in real_time_stats.keys() 
                if "error" in key.lower() or "fail" in key.lower()
            ]
            
            if error_indicators:
                error_rate = sum(
                    real_time_stats[key]["recent_average"] 
                    for key in error_indicators
                ) / len(error_indicators)
                
                if error_rate > 0.05:  # > 5% error rate
                    health_score -= 25
                    health_indicators["errors"] = "high"
                elif error_rate > 0.01:  # > 1% error rate
                    health_score -= 10
                    health_indicators["errors"] = "medium"
                else:
                    health_indicators["errors"] = "low"
            
            # Overall health status
            if health_score >= 90:
                overall_status = "excellent"
            elif health_score >= 75:
                overall_status = "good"
            elif health_score >= 60:
                overall_status = "fair"
            else:
                overall_status = "poor"
            
            return {
                "overall_score": health_score,
                "overall_status": overall_status,
                "indicators": health_indicators,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate system health: {e}")
            return {"overall_status": "unknown", "error": str(e)}
    
    async def _check_performance_alerts(self, metrics: List[SearchMetric]) -> None:
        """Check metrics against configured performance alerts"""
        
        try:
            for alert in self._active_alerts:
                if not alert.enabled:
                    continue
                
                # Find relevant metrics for this alert
                relevant_metrics = [
                    m for m in metrics 
                    if m.metric_name == alert.metric_name
                    and (not alert.tenant_id or m.tenant_id == alert.tenant_id)
                ]
                
                if not relevant_metrics:
                    continue
                
                # Check if threshold is exceeded
                for metric in relevant_metrics:
                    triggered = False
                    
                    if alert.comparison_operator == ">" and metric.value > alert.threshold_value:
                        triggered = True
                    elif alert.comparison_operator == "<" and metric.value < alert.threshold_value:
                        triggered = True
                    elif alert.comparison_operator == ">=" and metric.value >= alert.threshold_value:
                        triggered = True
                    elif alert.comparison_operator == "<=" and metric.value <= alert.threshold_value:
                        triggered = True
                    elif alert.comparison_operator == "==" and metric.value == alert.threshold_value:
                        triggered = True
                    
                    if triggered:
                        await self._trigger_performance_alert(alert, metric)
                        
        except Exception as e:
            logger.error(f"Failed to check performance alerts: {e}")
    
    async def _trigger_performance_alert(
        self,
        alert: PerformanceAlert,
        metric: SearchMetric
    ) -> None:
        """Trigger a performance alert"""
        
        try:
            # Update alert last triggered time
            alert.last_triggered = datetime.now()
            
            # Log alert
            logger.warning(
                "Performance alert triggered",
                alert_name=alert.name,
                metric_name=alert.metric_name,
                threshold=alert.threshold_value,
                actual_value=metric.value,
                tenant_id=alert.tenant_id
            )
            
            # Send notifications if service is available
            if self.notification_service and alert.notification_channels:
                alert_message = (
                    f"Performance alert '{alert.name}' triggered. "
                    f"Metric '{alert.metric_name}' value {metric.value} "
                    f"{alert.comparison_operator} {alert.threshold_value}"
                )
                
                for channel in alert.notification_channels:
                    if channel.startswith("email:"):
                        email_address = channel[6:]  # Remove "email:" prefix
                        await self.notification_service.send_email(
                            to=email_address,
                            subject=f"Performance Alert: {alert.name}",
                            body=alert_message
                        )
                    elif channel.startswith("webhook:"):
                        webhook_url = channel[8:]  # Remove "webhook:" prefix
                        await self.notification_service.send_webhook(
                            url=webhook_url,
                            payload={
                                "alert": alert.name,
                                "metric": alert.metric_name,
                                "value": metric.value,
                                "threshold": alert.threshold_value,
                                "timestamp": metric.timestamp.isoformat()
                            }
                        )
            
            self._stats["alerts_triggered"] += 1
            
        except Exception as e:
            logger.error(f"Failed to trigger performance alert: {e}")
    
    async def _store_analytics_report(self, report: AnalyticsReport) -> None:
        """Store analytics report in database"""
        
        try:
            async with self.db_adapter.get_connection() as conn:
                # Create reports table if it doesn't exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_reports (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        report_id VARCHAR(255) UNIQUE NOT NULL,
                        tenant_id UUID,
                        report_type VARCHAR(100) NOT NULL,
                        time_period JSONB NOT NULL,
                        metrics JSONB NOT NULL,
                        insights JSONB DEFAULT '[]',
                        recommendations JSONB DEFAULT '[]',
                        generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        metadata JSONB DEFAULT '{}'
                    )
                """)
                
                await conn.execute("""
                    INSERT INTO analytics_reports (
                        report_id, tenant_id, report_type, time_period,
                        metrics, insights, recommendations, generated_at, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                report.report_id,
                report.tenant_id,
                report.report_type,
                json.dumps({
                    "start": report.time_period["start"].isoformat(),
                    "end": report.time_period["end"].isoformat()
                }),
                json.dumps(report.metrics, default=str),
                json.dumps(report.insights),
                json.dumps(report.recommendations),
                report.generated_at,
                json.dumps(report.metadata or {})
                )
                
        except Exception as e:
            logger.error(f"Failed to store analytics report: {e}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check search analytics service health"""
        
        try:
            start_time = datetime.now()
            
            # Test database connectivity
            async with self.db_adapter.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            
            # Check metrics cache status
            cache_status = {
                "metrics_in_cache": len(self._recent_metrics),
                "active_alerts": len(self._active_alerts),
                "cache_full": len(self._recent_metrics) >= self._max_recent_metrics
            }
            
            health_check_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return {
                "status": "healthy",
                "search_analytics_service": "operational",
                "database_connectivity": "operational",
                "cache_status": cache_status,
                "stats": self._stats.copy(),
                "health_check_time_ms": health_check_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
