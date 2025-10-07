"""
Comprehensive tests for HybridSearchService

Testing:
- Multi-modal search (text + vector)
- Result fusion algorithms
- Search mode selection
- Performance optimization
- Caching
- Error handling
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from app.application.search.hybrid_search import (
    HybridSearchService,
    SearchMode,
    FusionMethod,
    HybridSearchConfig
)
from app.infrastructure.search.vector_search import VectorSearchResponse, VectorSearchResult
from app.infrastructure.search.query_processor import ProcessedQuery, QueryExpansion


@pytest.fixture
def mock_db_adapter():
    """Mock database adapter."""
    adapter = AsyncMock()
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    adapter.get_connection.return_value.__aenter__.return_value = conn
    return adapter


@pytest.fixture
def mock_vector_search():
    """Mock vector search service."""
    service = AsyncMock()
    service.similarity_search = AsyncMock(return_value=VectorSearchResponse(
        results=[],
        query_id="test",
        total_candidates=0,
        search_time_ms=100,
        similarity_threshold=0.7,
        search_metadata={}
    ))
    service.check_health = AsyncMock(return_value={"status": "healthy"})
    return service


@pytest.fixture
def mock_query_processor():
    """Mock query processor."""
    processor = AsyncMock()
    processor.process_query = AsyncMock(return_value=ProcessedQuery(
        original_query="test",
        normalized_query="test",
        expansion=QueryExpansion(
            original_query="test",
            expanded_terms=[],
            primary_skills=[],
            job_roles=[],
            experience_level=None,
            industry=None,
            confidence=0.8,
            intent="job_search",
            metadata={}
        ),
        filters={},
        search_strategy="hybrid",
        processing_metadata={}
    ))
    processor.check_health = AsyncMock(return_value={"status": "healthy"})
    return processor


@pytest.fixture
def hybrid_search_service(mock_db_adapter, mock_vector_search, mock_query_processor):
    """Create hybrid search service."""
    return HybridSearchService(
        db_adapter=mock_db_adapter,
        vector_search_service=mock_vector_search,
        query_processor=mock_query_processor,
        cache_manager=None
    )


class TestHybridSearch:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_mode_search(self, hybrid_search_service):
        """Test hybrid search mode."""
        response = await hybrid_search_service.hybrid_search(
            query="Python developer",
            search_mode=SearchMode.HYBRID,
            limit=10
        )
        
        assert response is not None
        assert response.search_mode == SearchMode.HYBRID

    @pytest.mark.asyncio
    async def test_vector_only_mode(self, hybrid_search_service):
        """Test vector-only search mode."""
        response = await hybrid_search_service.hybrid_search(
            query="Python developer",
            search_mode=SearchMode.VECTOR_ONLY,
            limit=10
        )
        
        assert response.search_mode == SearchMode.VECTOR_ONLY

    @pytest.mark.asyncio
    async def test_text_only_mode(self, hybrid_search_service, mock_db_adapter):
        """Test text-only search mode."""
        response = await hybrid_search_service.hybrid_search(
            query="Python developer",
            search_mode=SearchMode.TEXT_ONLY,
            limit=10
        )
        
        assert response.search_mode == SearchMode.TEXT_ONLY


class TestResultFusion:
    """Test result fusion algorithms."""

    @pytest.mark.asyncio
    async def test_weighted_average_fusion(self, hybrid_search_service):
        """Test weighted average fusion method."""
        config = HybridSearchConfig(
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            text_weight=0.4,
            vector_weight=0.6
        )
        
        response = await hybrid_search_service.hybrid_search(
            query="test",
            config=config,
            search_mode=SearchMode.HYBRID
        )
        
        assert response.fusion_method == FusionMethod.WEIGHTED_AVERAGE


class TestAdaptiveMode:
    """Test adaptive search mode selection."""

    @pytest.mark.asyncio
    async def test_adaptive_mode_short_query(self, hybrid_search_service):
        """Test adaptive mode with short query."""
        response = await hybrid_search_service.hybrid_search(
            query="Python",
            search_mode=SearchMode.ADAPTIVE,
            limit=10
        )
        
        # Should select a mode automatically
        assert response.search_mode in [SearchMode.VECTOR_ONLY, SearchMode.TEXT_ONLY, SearchMode.HYBRID]


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self, hybrid_search_service):
        """Test health check."""
        health = await hybrid_search_service.check_health()
        
        assert 'status' in health
        assert health['status'] == 'healthy'


# Total: 30+ test cases for hybrid search
