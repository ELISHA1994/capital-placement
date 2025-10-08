"""Tests for Embedding Service infrastructure implementation."""

import pytest

pytest.importorskip("numpy")
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from app.infrastructure.ai.embedding_service import EmbeddingService


@pytest.fixture
def mock_openai_service():
    """Mock OpenAI service."""
    service = AsyncMock()
    service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    service.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1] * 1536, [0.2] * 1536]
    )
    return service


@pytest.fixture
def mock_db_adapter():
    """Mock database adapter."""
    adapter = AsyncMock()
    adapter.get_connection = MagicMock()
    return adapter


@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    return AsyncMock()


@pytest.fixture
def embedding_service(mock_openai_service, mock_db_adapter, mock_cache_service):
    """Create embedding service with mocks."""
    return EmbeddingService(
        openai_service=mock_openai_service,
        db_adapter=mock_db_adapter,
        cache_service=mock_cache_service
    )


class TestEmbeddingGeneration:
    """Test embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedding_service):
        """Test generating a single embedding."""
        result = await embedding_service.generate_embedding("test text")

        assert len(result) == 1536
        assert embedding_service._metrics["embeddings_generated"] == 1

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_fails(self, embedding_service):
        """Test empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.generate_embedding("")

    @pytest.mark.asyncio
    async def test_generate_and_store_embedding(self, embedding_service, mock_db_adapter):
        """Test generating and storing embedding."""
        mock_conn = AsyncMock()
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await embedding_service.generate_and_store_embedding(
            entity_id="test-id",
            entity_type="profile",
            content="test content",
            tenant_id="tenant-1"
        )

        assert len(result) == 1536
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_and_store_batch(self, embedding_service, mock_db_adapter):
        """Test batch generation and storage."""
        entities = [
            {"entity_id": "1", "entity_type": "profile", "content": "text1"},
            {"entity_id": "2", "entity_type": "profile", "content": "text2"}
        ]

        mock_conn = AsyncMock()
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        results = await embedding_service.generate_and_store_batch(entities)

        assert len(results) == 2
        assert mock_conn.execute.call_count >= 1


class TestSimilaritySearch:
    """Test similarity search functionality."""

    @pytest.mark.asyncio
    async def test_similarity_search(self, embedding_service, mock_db_adapter):
        """Test similarity search."""
        query_embedding = [0.1] * 1536

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(entity_id="1", entity_type="profile", similarity=0.95)
        ]

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        results = await embedding_service.similarity_search(
            query_embedding=query_embedding,
            entity_type="profile",
            limit=10
        )

        assert len(results) >= 0
        assert embedding_service._metrics["similarity_searches"] == 1

    @pytest.mark.asyncio
    async def test_similarity_search_empty_embedding_fails(self, embedding_service):
        """Test empty embedding raises ValueError."""
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            await embedding_service.similarity_search(query_embedding=[])

    @pytest.mark.asyncio
    async def test_find_similar_text(self, embedding_service, mock_db_adapter):
        """Test finding similar entities by text."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        results = await embedding_service.find_similar_text(
            query_text="test query",
            entity_type="profile"
        )

        assert isinstance(results, list)


class TestEmbeddingCRUD:
    """Test CRUD operations for embeddings."""

    @pytest.mark.asyncio
    async def test_get_entity_embedding(self, embedding_service, mock_db_adapter):
        """Test retrieving entity embedding."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(embedding_vector=[0.1] * 1536)

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await embedding_service.get_entity_embedding(
            entity_id="test-id",
            entity_type="profile"
        )

        assert result is not None
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_update_entity_embedding(self, embedding_service, mock_db_adapter):
        """Test updating entity embedding."""
        mock_result = MagicMock()
        mock_result.rowcount = 1

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await embedding_service.update_entity_embedding(
            entity_id="test-id",
            entity_type="profile",
            new_content="updated content"
        )

        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_delete_entity_embedding(self, embedding_service, mock_db_adapter):
        """Test deleting entity embedding."""
        mock_result = MagicMock()
        mock_result.rowcount = 1

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await embedding_service.delete_entity_embedding(
            entity_id="test-id",
            entity_type="profile"
        )

        assert result is True


class TestSimilarityMatrix:
    """Test similarity matrix calculation."""

    @pytest.mark.asyncio
    async def test_calculate_similarity_matrix(self, embedding_service):
        """Test calculating similarity matrix."""
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536
        ]

        matrix = await embedding_service.calculate_similarity_matrix(embeddings)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)

    @pytest.mark.asyncio
    async def test_calculate_similarity_matrix_empty(self, embedding_service):
        """Test empty list returns empty array."""
        matrix = await embedding_service.calculate_similarity_matrix([])
        assert matrix.size == 0


class TestHealthAndMetrics:
    """Test health check and metrics."""

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, embedding_service, mock_db_adapter):
        """Test health check when healthy."""
        mock_conn = AsyncMock()
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        result = await embedding_service.check_health()

        assert result["status"] == "healthy"
        assert "embedding_dimension" in result

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self, embedding_service, mock_openai_service):
        """Test health check when unhealthy."""
        mock_openai_service.generate_embedding.side_effect = Exception("API Error")

        result = await embedding_service.check_health()

        assert result["status"] == "unhealthy"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_metrics(self, embedding_service, mock_db_adapter):
        """Test getting service metrics."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(total_embeddings=100)

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        mock_db_adapter.get_connection.return_value.__aenter__.return_value = mock_conn

        metrics = await embedding_service.get_metrics()

        assert "operations" in metrics
        assert "database" in metrics
        assert "configuration" in metrics
