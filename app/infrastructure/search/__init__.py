"""
Infrastructure Search Services

Database-level search adapters and query processing services:
- VectorSearchService: pgvector similarity search adapter
- QueryProcessor: Query analysis, expansion, and optimization
- SearchAnalyticsService: Search metrics tracking and analytics

These are infrastructure-level adapters and services that provide
search capabilities without business logic orchestration.
"""

from app.infrastructure.search.vector_search import (
    VectorSearchService,
    VectorSearchResult,
    VectorSearchResponse,
    SearchFilter
)
from app.infrastructure.search.query_processor import (
    QueryProcessor,
    ProcessedQuery,
    QueryExpansion
)
from app.infrastructure.search.search_analytics import (
    SearchAnalyticsService,
    SearchMetric,
    AnalyticsReport
)

__all__ = [
    # Vector Search
    "VectorSearchService",
    "VectorSearchResult",
    "VectorSearchResponse",
    "SearchFilter",

    # Query Processing
    "QueryProcessor",
    "ProcessedQuery",
    "QueryExpansion",

    # Search Analytics
    "SearchAnalyticsService",
    "SearchMetric",
    "AnalyticsReport",
]
