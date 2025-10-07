"""
Search Application Services

This module contains application-layer services for search operations,
orchestrating complex search workflows across multiple infrastructure services.

Components:
- SearchApplicationService: Multi-stage CV search engine with advanced ranking (Wave 7)
- HybridSearchService: Multi-modal search combining text + vector (Wave 9)
- ResultRerankerService: AI-powered result reranking and optimization (Wave 9)
"""

from app.application.search.search_application_service import SearchApplicationService
from app.application.search.hybrid_search import (
    HybridSearchService,
    HybridSearchResult,
    HybridSearchResponse,
    HybridSearchConfig,
    SearchMode,
    FusionMethod
)
from app.application.search.result_reranker import (
    ResultRerankerService,
    RerankingResult,
    RerankingResponse,
    RerankingConfig,
    RankingStrategy
)

__all__ = [
    # Main Search Application Service (Wave 7)
    "SearchApplicationService",

    # Hybrid Search (Wave 9)
    "HybridSearchService",
    "HybridSearchResult",
    "HybridSearchResponse",
    "HybridSearchConfig",
    "SearchMode",
    "FusionMethod",

    # Result Reranking (Wave 9)
    "ResultRerankerService",
    "RerankingResult",
    "RerankingResponse",
    "RerankingConfig",
    "RankingStrategy",
]
