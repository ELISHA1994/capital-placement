"""
Unit tests for SearchApplicationService

Tests cover core functionality of the multi-stage search engine.
Migrated from app/services/core/search_engine.py
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from uuid import uuid4

from app.application.search.search_application_service import SearchApplicationService
from app.api.schemas.search_schemas import (
    SearchRequest, SearchResponse, SearchMode,
    PaginationModel, SearchAnalytics
)
from app.domain.interfaces import ISearchService, IAIService


@pytest.fixture
def mock_search_service():
    """Mock search service"""
    service = Mock(spec=ISearchService)
    service.vector_search = AsyncMock(return_value={
        "documents": [
            {
                "id": str(uuid4()),
                "profile_id": str(uuid4()),
                "email": "test@example.com",
                "tenant_id": str(uuid4()),
                "full_name": "Test User",
                "title": "Software Engineer",
                "skills": "Python, JavaScript, AWS",
                "total_experience_years": 5,
                "@search_score": 0.85,
                "last_updated": datetime.now().isoformat(),
                "profile_completeness": 0.9
            }
        ]
    })
    service.search = AsyncMock(return_value={
        "documents": [
            {
                "id": str(uuid4()),
                "profile_id": str(uuid4()),
                "email": "test2@example.com",
                "tenant_id": str(uuid4()),
                "full_name": "Test User 2",
                "title": "Senior Developer",
                "skills": ["Python", "Django", "PostgreSQL"],
                "total_experience_years": 7,
                "@search_score": 0.92,
                "last_updated": datetime.now().isoformat(),
                "profile_completeness": 0.95
            }
        ]
    })
    return service


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator"""
    generator = Mock()
    generator.generate_query_embedding = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
def mock_cache_service():
    """Mock cache service"""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def search_engine(mock_search_service, mock_embedding_generator, mock_cache_service):
    """Create SearchApplicationService instance with mocks"""
    return SearchApplicationService(
        search_service=mock_search_service,
        ai_service=Mock(spec=IAIService),
        embedding_generator=mock_embedding_generator,
        cache_service=mock_cache_service
    )


@pytest.fixture
def basic_search_request():
    """Create a basic search request"""
    return SearchRequest(
        query="Python developer",
        tenant_id=uuid4(),
        search_mode=SearchMode.HYBRID,
        max_results=20,
        min_match_score=0.3
    )


class TestSearchExecution:
    """Test search execution workflows"""

    @pytest.mark.asyncio
    async def test_basic_search_execution(self, search_engine, basic_search_request, mock_search_service):
        """Test basic search execution with hybrid mode"""
        response = await search_engine.search(basic_search_request)

        assert isinstance(response, SearchResponse)
        assert response.total_count >= 0
        assert response.search_mode == SearchMode.HYBRID
        assert response.query == "Python developer"
        assert mock_search_service.search.called

    @pytest.mark.asyncio
    async def test_vector_search_mode(self, search_engine, basic_search_request, mock_search_service):
        """Test vector search mode execution"""
        basic_search_request.search_mode = SearchMode.VECTOR

        response = await search_engine.search(basic_search_request)

        assert mock_search_service.vector_search.called
        assert response.search_mode == SearchMode.VECTOR

    @pytest.mark.asyncio
    async def test_semantic_search_mode(self, search_engine, basic_search_request, mock_search_service):
        """Test semantic search mode execution"""
        basic_search_request.search_mode = SearchMode.SEMANTIC

        response = await search_engine.search(basic_search_request)

        assert mock_search_service.search.called
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs.get("search_mode") == "semantic"

    @pytest.mark.asyncio
    async def test_keyword_search_mode(self, search_engine, basic_search_request, mock_search_service):
        """Test keyword search mode execution"""
        basic_search_request.search_mode = SearchMode.KEYWORD

        response = await search_engine.search(basic_search_request)

        assert mock_search_service.search.called
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs.get("search_mode") == "keyword"

    @pytest.mark.asyncio
    async def test_search_with_tenant_isolation(self, search_engine, basic_search_request, mock_search_service):
        """Test that tenant_id is always included in filters"""
        tenant_id = uuid4()
        basic_search_request.tenant_id = tenant_id

        response = await search_engine.search(basic_search_request)

        call_args = mock_search_service.search.call_args
        filters = call_args.kwargs.get("filters", {})
        assert filters.get("tenant_id") == str(tenant_id)

class TestQueryPreprocessing:
    """Test query preprocessing and expansion"""

    @pytest.mark.asyncio
    async def test_query_expansion_disabled(self, search_engine, basic_search_request):
        """Test that query expansion doesn't occur when disabled"""
        basic_search_request.query = "Python developer"
        basic_search_request.enable_query_expansion = False

        analytics = SearchAnalytics(
            total_search_time_ms=0,
            total_candidates=0,
            candidates_after_filters=0,
            query_expanded=False,
            expanded_terms=[],
            synonyms_used=[]
        )

        processed_query = await search_engine._preprocess_query(basic_search_request, analytics)

        assert processed_query == "Python developer"
        assert not analytics.query_expanded
        assert len(analytics.expanded_terms) == 0

    @pytest.mark.asyncio
    async def test_query_expansion_with_skills(self, search_engine, basic_search_request):
        """Test query expansion for skill synonyms"""
        basic_search_request.query = "Python developer"
        basic_search_request.enable_query_expansion = True

        analytics = SearchAnalytics(
            total_search_time_ms=0,
            total_candidates=0,
            candidates_after_filters=0,
            query_expanded=False,
            expanded_terms=[],
            synonyms_used=[]
        )

        processed_query = await search_engine._preprocess_query(basic_search_request, analytics)

        assert "python" in processed_query.lower()
        assert analytics.query_expanded
        assert len(analytics.expanded_terms) > 0


class TestMatchScoring:
    """Test match scoring and ranking logic"""

    def test_skill_match_score_exact_match(self, search_engine):
        """Test skill matching with exact matches"""
        result = {
            "skills": ["Python", "JavaScript", "AWS"]
        }

        skill_requirements = [
            Mock(name="Python", weight=1.0, alternatives=[], required=True, spec=['name', 'weight', 'alternatives', 'required']),
            Mock(name="JavaScript", weight=0.8, alternatives=[], required=True, spec=['name', 'weight', 'alternatives', 'required']),
        ]
        # Manually set the name attribute since Mock() name parameter doesn't set it
        skill_requirements[0].name = "Python"
        skill_requirements[1].name = "JavaScript"

        skill_match = search_engine._calculate_skill_match_score(result, skill_requirements)

        assert skill_match["score"] == 1.0
        assert "Python" in skill_match["matched_skills"]
        assert "JavaScript" in skill_match["matched_skills"]
        assert len(skill_match["missing_skills"]) == 0

    def test_skill_match_score_partial_match(self, search_engine):
        """Test skill matching with partial matches"""
        result = {
            "skills": ["Python", "Docker"]
        }

        skill_requirements = [
            Mock(weight=1.0, alternatives=[], required=True, spec=['name', 'weight', 'alternatives', 'required']),
            Mock(weight=1.0, alternatives=[], required=True, spec=['name', 'weight', 'alternatives', 'required']),
            Mock(weight=0.5, alternatives=[], required=False, spec=['name', 'weight', 'alternatives', 'required']),
        ]
        skill_requirements[0].name = "Python"
        skill_requirements[1].name = "JavaScript"
        skill_requirements[2].name = "AWS"

        skill_match = search_engine._calculate_skill_match_score(result, skill_requirements)

        assert skill_match["score"] < 1.0
        assert "Python" in skill_match["matched_skills"]
        assert "JavaScript" in skill_match["missing_skills"]

    def test_experience_match_score_meets_minimum(self, search_engine):
        """Test experience scoring when candidate meets minimum"""
        result = {"total_experience_years": 5}
        experience_req = Mock(min_total_years=3, max_total_years=None)

        score = search_engine._calculate_experience_match_score(result, experience_req)

        assert score >= 0.8

    def test_experience_match_score_below_minimum(self, search_engine):
        """Test experience scoring when candidate below minimum"""
        result = {"total_experience_years": 2}
        experience_req = Mock(min_total_years=5, max_total_years=None)

        score = search_engine._calculate_experience_match_score(result, experience_req)

        assert score < 0.8


class TestResultProcessing:
    """Test result processing and finalization"""

    def test_result_diversification(self, search_engine):
        """Test result diversification to avoid too many from same company"""
        results = [
            {"current_company": "TechCorp", "overall_match_score": 0.95, "id": "1"},
            {"current_company": "TechCorp", "overall_match_score": 0.93, "id": "2"},
            {"current_company": "TechCorp", "overall_match_score": 0.91, "id": "3"},
            {"current_company": "StartupX", "overall_match_score": 0.89, "id": "4"},
        ]

        diversified = search_engine._apply_result_diversification(results)

        # Diversification should apply for sets > 10, so small set unchanged
        assert len(diversified) == len(results)

    def test_result_diversification_small_set(self, search_engine):
        """Test that small result sets aren't diversified"""
        results = [{"id": str(i), "current_company": "TechCorp"} for i in range(5)]

        diversified = search_engine._apply_result_diversification(results)

        assert len(diversified) == len(results)


class TestFacetGeneration:
    """Test facet generation for result refinement"""

    @pytest.mark.asyncio
    async def test_skill_facet_generation(self, search_engine, basic_search_request):
        """Test skills facet generation"""
        results = [
            {"skills": "Python, JavaScript, AWS"},
            {"skills": ["Python", "Docker", "Kubernetes"]},
            {"skills": "Python, Java, Spring"},
        ]

        facets = await search_engine._generate_search_facets(results, basic_search_request)

        skill_facet = next((f for f in facets if f.field == "skills"), None)
        assert skill_facet is not None
        assert skill_facet.display_name == "Skills"

    @pytest.mark.asyncio
    async def test_experience_facet_generation(self, search_engine, basic_search_request):
        """Test experience range facet generation"""
        results = [
            {"total_experience_years": 1},
            {"total_experience_years": 4},
            {"total_experience_years": 8},
            {"total_experience_years": 12},
        ]

        facets = await search_engine._generate_search_facets(results, basic_search_request)

        exp_facet = next((f for f in facets if f.field == "total_experience_years"), None)
        assert exp_facet is not None
        assert exp_facet.display_name == "Experience Level"

    @pytest.mark.asyncio
    async def test_empty_results_facet_generation(self, search_engine, basic_search_request):
        """Test facet generation with empty results"""
        results = []

        facets = await search_engine._generate_search_facets(results, basic_search_request)

        assert len(facets) == 0


class TestCaching:
    """Test search result caching"""

    @pytest.mark.asyncio
    async def test_cache_miss_executes_search(self, search_engine, basic_search_request, mock_cache_service):
        """Test that cache miss executes full search"""
        basic_search_request.track_search = True
        mock_cache_service.get = AsyncMock(return_value=None)

        response = await search_engine.search(basic_search_request)

        assert mock_cache_service.get.called
        assert mock_cache_service.set.called

    @pytest.mark.asyncio
    async def test_cache_disabled_no_caching(self, search_engine, basic_search_request, mock_cache_service):
        """Test that caching is skipped when track_search is False"""
        basic_search_request.track_search = False

        response = await search_engine.search(basic_search_request)

        assert not mock_cache_service.get.called
        assert not mock_cache_service.set.called


class TestAnalytics:
    """Test search analytics and statistics tracking"""

    @pytest.mark.asyncio
    async def test_analytics_tracking(self, search_engine, basic_search_request):
        """Test that analytics are tracked correctly"""
        response = await search_engine.search(basic_search_request)

        # Analytics should be populated (may be 0 for very fast mocked searches)
        assert response.analytics.total_search_time_ms >= 0
        assert response.analytics.total_candidates >= 0
        assert response.analytics is not None

    def test_search_statistics_tracking(self, search_engine):
        """Test internal statistics tracking"""
        initial_count = search_engine._search_stats["total_searches"]

        search_engine._update_search_stats(100.0, Mock(search_mode=SearchMode.HYBRID), 5)

        assert search_engine._search_stats["total_searches"] == initial_count + 1
        assert search_engine._search_stats["average_search_time_ms"] > 0

    def test_get_search_statistics(self, search_engine):
        """Test getting search statistics"""
        stats = search_engine.get_search_statistics()

        assert "total_searches" in stats
        assert "vector_searches" in stats
        assert "search_config" in stats


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_search_service_failure(self, search_engine, basic_search_request, mock_search_service):
        """Test handling of search service failures"""
        mock_search_service.search = AsyncMock(side_effect=Exception("Search service unavailable"))

        with pytest.raises(RuntimeError) as exc_info:
            await search_engine.search(basic_search_request)

        assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, search_engine, basic_search_request, mock_embedding_generator):
        """Test handling of embedding generation failures"""
        basic_search_request.search_mode = SearchMode.VECTOR
        mock_embedding_generator.generate_query_embedding = AsyncMock(
            side_effect=Exception("Embedding service unavailable")
        )

        with pytest.raises(RuntimeError):
            await search_engine.search(basic_search_request)


class TestPerformance:
    """Test performance-related features"""

    @pytest.mark.asyncio
    async def test_search_completes_within_time_limit(self, search_engine, basic_search_request):
        """Test that search completes within reasonable time"""
        import time

        start_time = time.time()
        response = await search_engine.search(basic_search_request)
        duration = time.time() - start_time

        # Should complete in less than 5 seconds
        assert duration < 5.0
