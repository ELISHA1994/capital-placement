"""
Comprehensive tests for VectorSearchService

Testing:
- Similarity search with pgvector
- Batch queries
- Performance (response times)
- Tenant isolation
- Cache hit/miss scenarios
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from app.infrastructure.search.vector_search import (
    VectorSearchService,
    VectorSearchResult,
    VectorSearchResponse,
    SearchFilter
)


@pytest.fixture
def mock_postgres_adapter():
    """Mock PostgreSQL adapter."""
    adapter = AsyncMock()
    adapter.get_connection = AsyncMock()
    return adapter


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = AsyncMock()
    service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    service.model_name = "test-model"
    return service


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.check_health = AsyncMock(return_value={"status": "healthy"})
    return cache


@pytest.fixture
def vector_search_service(mock_postgres_adapter, mock_embedding_service, mock_cache_manager):
    """Create VectorSearchService instance."""
    return VectorSearchService(
        db_adapter=mock_postgres_adapter,
        embedding_service=mock_embedding_service,
        cache_manager=mock_cache_manager
    )


class TestVectorSearchBasic:
    """Test basic vector search functionality."""

    @pytest.mark.asyncio
    async def test_similarity_search_with_query_text(self, vector_search_service, mock_postgres_adapter, mock_embedding_service):
        """Test similarity search with text query."""
        # Setup mock database response
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {
                'entity_id': str(uuid4()),
                'entity_type': 'profile',
                'distance': 0.2,
                'tenant_id': str(uuid4()),
                'embedding_model': 'text-embedding-3-large',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'name': 'Senior Python Developer',
                'email': 'candidate@example.com',
                'phone': None,
                'location_city': 'San Francisco',
                'location_state': 'CA',
                'location_country': 'USA',
                'normalized_skills': ['Python', 'FastAPI'],
                'searchable_text': 'Senior Python Developer with FastAPI experience',
                'status': 'active',
                'experience_level': 'senior',
                'profile_data': {'summary': 'Experienced engineer'},
            }
        ])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        # Execute search
        response = await vector_search_service.similarity_search(
            query_text="Python developer",
            tenant_id=str(uuid4()),
            limit=10,
            similarity_threshold=0.7
        )

        # Assertions
        assert isinstance(response, VectorSearchResponse)
        assert len(response.results) > 0
        assert response.total_candidates > 0
        assert mock_embedding_service.generate_embedding.called

    @pytest.mark.asyncio
    async def test_similarity_search_with_embedding(self, vector_search_service, mock_postgres_adapter):
        """Test similarity search with pre-computed embedding."""
        # Setup
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        query_embedding = [0.1] * 1536

        # Execute
        response = await vector_search_service.similarity_search(
            query_embedding=query_embedding,
            limit=10
        )

        # Assertions
        assert isinstance(response, VectorSearchResponse)
        assert response.total_candidates == 0

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, vector_search_service, mock_postgres_adapter):
        """Test that similarity threshold filters results correctly."""
        # Setup - return results with varying similarities
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {'entity_id': '1', 'entity_type': 'profile', 'distance': 0.1,  # similarity 0.9
             'tenant_id': None, 'embedding_model': 'test', 'created_at': datetime.now(),
             'updated_at': datetime.now(), 'name': 'Candidate One', 'email': 'one@example.com',
             'searchable_text': '', 'normalized_skills': [], 'status': 'active', 'experience_level': 'senior',
             'location_city': None, 'location_state': None, 'location_country': None, 'profile_data': {}},
            {'entity_id': '2', 'entity_type': 'profile', 'distance': 0.5,  # similarity 0.5
             'tenant_id': None, 'embedding_model': 'test', 'created_at': datetime.now(),
             'updated_at': datetime.now(), 'name': 'Candidate Two', 'email': 'two@example.com',
             'searchable_text': '', 'normalized_skills': [], 'status': 'active', 'experience_level': 'senior',
             'location_city': None, 'location_state': None, 'location_country': None, 'profile_data': {}},
        ])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        # Execute with high threshold
        response = await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536,
            similarity_threshold=0.8,
            limit=10
        )

        # Should only get high similarity results
        assert len(response.results) == 1
        assert response.results[0].similarity_score >= 0.8

    @pytest.mark.asyncio
    async def test_similarity_search_with_multiple_embedding_fields(self, vector_search_service, mock_postgres_adapter):
        """Vector search should combine results across multiple profile embedding sections."""
        now = datetime.now()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=[
            [
                {
                    'entity_id': '1',
                    'distance': 0.3,
                    'tenant_id': None,
                    'created_at': now,
                    'updated_at': now,
                    'name': 'Candidate One',
                    'email': 'one@example.com',
                    'phone': None,
                    'location_city': None,
                    'location_state': None,
                    'location_country': None,
                    'normalized_skills': [],
                    'searchable_text': 'Experienced engineer',
                    'status': 'active',
                    'experience_level': 'senior',
                    'profile_data': {}
                }
            ],
            [
                {
                    'entity_id': '1',
                    'distance': 0.1,
                    'tenant_id': None,
                    'created_at': now,
                    'updated_at': now,
                    'name': 'Candidate One',
                    'email': 'one@example.com',
                    'phone': None,
                    'location_city': None,
                    'location_state': None,
                    'location_country': None,
                    'normalized_skills': [],
                    'searchable_text': 'Experienced engineer',
                    'status': 'active',
                    'experience_level': 'senior',
                    'profile_data': {}
                }
            ]
        ])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        search_filter = SearchFilter(embedding_fields=["skills", "experience"])

        response = await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536,
            search_filter=search_filter,
            limit=5
        )

        assert len(response.results) == 1
        assert response.search_metadata.get("embedding_fields") == ["skills", "experience"]
        assert response.results[0].metadata.get("embedding_field") == "experience"


class TestVectorSearchTenantIsolation:
    """Test multi-tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_filtering(self, vector_search_service, mock_postgres_adapter):
        """Test that tenant ID properly filters results."""
        tenant_id = str(uuid4())
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536,
            tenant_id=tenant_id,
            limit=10
        )

        # Verify tenant ID was included in query
        call_args = mock_conn.fetch.call_args
        assert tenant_id in call_args[0]


class TestVectorSearchCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, vector_search_service, mock_cache_manager):
        """Test that cached results are returned."""
        # Setup cached response
        cached_response = {
            'results': [],
            'query_id': str(uuid4()),
            'total_candidates': 0,
            'search_time_ms': 100,
            'similarity_threshold': 0.7,
            'search_metadata': {},
            'cache_hit': False
        }
        mock_cache_manager.get = AsyncMock(return_value=cached_response)

        # Execute
        response = await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536,
            use_cache=True
        )

        # Should return cached result
        assert response.cache_hit is True
        assert mock_cache_manager.get.called

    @pytest.mark.asyncio
    async def test_cache_miss_and_set(self, vector_search_service, mock_cache_manager, mock_postgres_adapter):
        """Test that results are cached on cache miss."""
        # Setup
        mock_cache_manager.get = AsyncMock(return_value=None)
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        # Execute
        await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536,
            use_cache=True
        )

        # Should set cache after search
        assert mock_cache_manager.set.called


class TestVectorSearchBatch:
    """Test batch search operations."""

    @pytest.mark.asyncio
    async def test_batch_similarity_search(self, vector_search_service, mock_postgres_adapter):
        """Test batch search with multiple queries."""
        # Setup
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        # Create multiple queries
        queries = [
            {"query_text": "Python developer", "limit": 10},
            {"query_text": "Java engineer", "limit": 5},
        ]

        # Execute
        results = await vector_search_service.batch_similarity_search(queries)

        # Assertions
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResponse) for r in results)


class TestVectorSearchPerformance:
    """Test search performance metrics."""

    @pytest.mark.asyncio
    async def test_search_time_tracking(self, vector_search_service, mock_postgres_adapter):
        """Test that search time is tracked."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        response = await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536
        )

        # Should have search time
        assert response.search_time_ms >= 0

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, vector_search_service, mock_postgres_adapter):
        """Test that service tracks statistics."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        # Perform search
        await vector_search_service.similarity_search(query_embedding=[0.1] * 1536)

        # Check stats
        analytics = await vector_search_service.get_search_analytics()
        assert 'service_stats' in analytics
        assert analytics['service_stats']['searches_performed'] > 0


class TestVectorSearchErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_invalid_query_parameters(self, vector_search_service):
        """Test handling of invalid parameters."""
        # Should return empty response, not raise
        response = await vector_search_service.similarity_search(
            query_text=None,
            query_embedding=None  # Both None - invalid
        )
        assert response.total_candidates == 0

    @pytest.mark.asyncio
    async def test_database_error_handling(self, vector_search_service, mock_postgres_adapter):
        """Test graceful handling of database errors."""
        # Setup database error
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=Exception("Database error"))
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        # Should not raise, return empty response
        response = await vector_search_service.similarity_search(
            query_embedding=[0.1] * 1536
        )
        assert response.total_candidates == 0


class TestVectorSearchHealth:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, vector_search_service, mock_postgres_adapter, mock_cache_manager):
        """Test health check when all services are healthy."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[1, 42])
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        health = await vector_search_service.check_health()

        assert health['status'] == 'healthy'
        assert 'stats' in health

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, vector_search_service, mock_postgres_adapter):
        """Test health check when database is unhealthy."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("DB error"))
        mock_postgres_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        health = await vector_search_service.check_health()

        assert health['status'] == 'unhealthy'
        assert 'error' in health
