"""
Search Services Package

Advanced semantic search capabilities for CV matching:
- Query understanding and expansion with AI
- pgvector semantic similarity search
- Hybrid text + vector search optimization
- AI-powered result reranking
- Search performance analytics

This package provides intelligent search capabilities that understand
user intent and deliver highly relevant results through semantic matching.
"""

from .query_processor import QueryProcessor
from .vector_search import VectorSearchService as VectorSearch
from .hybrid_search import HybridSearchService as HybridSearch
from .result_reranker import ResultRerankerService as ResultReranker
from .search_analytics import SearchAnalyticsService as SearchAnalytics

__all__ = [
    "QueryProcessor",
    "VectorSearch",
    "HybridSearch", 
    "ResultReranker",
    "SearchAnalytics",
]