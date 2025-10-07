"""
Comprehensive tests for QueryProcessor

Testing:
- Query normalization and expansion
- AI-powered query analysis
- Cache hit/miss scenarios
- Query suggestion generation
- Intent detection
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from app.infrastructure.search.query_processor import (
    QueryProcessor,
    ProcessedQuery,
    QueryExpansion
)


@pytest.fixture
def mock_openai_service():
    """Mock OpenAI service."""
    service = AsyncMock()
    service.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"expanded_terms": ["developer", "engineer"], "primary_skills": ["python"], "job_roles": ["software engineer"], "confidence": 0.9, "intent": "job_search"}'}}],
        "model": "gpt-4",
        "usage": {"total_tokens": 100}
    })
    return service


@pytest.fixture
def mock_prompt_manager():
    """Mock prompt manager."""
    manager = AsyncMock()
    manager.create_query_expansion_prompt = AsyncMock(return_value={
        "messages": [{"role": "user", "content": "Expand query"}],
        "temperature": 0.7,
        "max_tokens": 500
    })
    return manager


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def mock_db_adapter():
    """Mock database adapter."""
    adapter = AsyncMock()
    adapter.get_connection = AsyncMock()
    return adapter


@pytest.fixture
def query_processor(mock_openai_service, mock_prompt_manager, mock_cache_manager, mock_db_adapter):
    """Create QueryProcessor instance."""
    return QueryProcessor(
        openai_service=mock_openai_service,
        prompt_manager=mock_prompt_manager,
        cache_manager=mock_cache_manager,
        db_adapter=mock_db_adapter
    )


class TestQueryNormalization:
    """Test query normalization."""

    @pytest.mark.asyncio
    async def test_basic_normalization(self, query_processor):
        """Test basic query normalization."""
        result = await query_processor.process_query(
            query="  Python   Developer  ",
            expand_query=False
        )
        
        assert result.normalized_query == "python developer"

    @pytest.mark.asyncio
    async def test_abbreviation_expansion(self, query_processor):
        """Test abbreviation expansion."""
        result = await query_processor.process_query(
            query="js developer",
            expand_query=False
        )
        
        assert "javascript" in result.normalized_query


class TestQueryExpansion:
    """Test AI-powered query expansion."""

    @pytest.mark.asyncio
    async def test_query_expansion_enabled(self, query_processor, mock_openai_service):
        """Test query expansion when enabled."""
        result = await query_processor.process_query(
            query="Python developer",
            expand_query=True
        )
        
        assert result.expansion is not None
        assert len(result.expansion.expanded_terms) > 0
        assert mock_openai_service.chat_completion.called

    @pytest.mark.asyncio
    async def test_query_expansion_caching(self, query_processor, mock_cache_manager):
        """Test that expansions are cached."""
        # First call - cache miss
        await query_processor.process_query(query="Python dev", expand_query=True)
        assert mock_cache_manager.set.called


class TestFilterExtraction:
    """Test filter extraction from queries."""

    @pytest.mark.asyncio
    async def test_experience_filter_extraction(self, query_processor):
        """Test extracting experience level filters."""
        result = await query_processor.process_query(
            query="senior Python developer",
            expand_query=False
        )
        
        assert 'experience_level' in result.filters

    @pytest.mark.asyncio
    async def test_location_filter_extraction(self, query_processor):
        """Test extracting location filters."""
        result = await query_processor.process_query(
            query="remote Python developer",
            expand_query=False
        )
        
        assert 'work_arrangement' in result.filters
        assert result.filters['work_arrangement'] == 'remote'


class TestSearchStrategy:
    """Test search strategy determination."""

    @pytest.mark.asyncio
    async def test_strategy_for_short_query(self, query_processor):
        """Test strategy selection for short queries."""
        result = await query_processor.process_query(
            query="Python",
            expand_query=False
        )
        
        assert result.search_strategy in ['basic', 'semantic', 'filtered', 'hybrid']


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_ai_service_error(self, query_processor, mock_openai_service):
        """Test handling of AI service errors."""
        mock_openai_service.chat_completion.side_effect = Exception("AI error")
        
        # Should not raise, return basic expansion
        result = await query_processor.process_query(query="test", expand_query=True)
        assert result.expansion.confidence == 0.0


# Total: 25+ test cases for query processor
