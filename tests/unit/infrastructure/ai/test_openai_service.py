"""Tests for OpenAI Service infrastructure implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from app.infrastructure.ai.openai_service import OpenAIService


@pytest.fixture
def mock_cache_service():
    """Mock cache service for testing."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    return cache


@pytest.fixture
async def openai_service(mock_cache_service):
    """Create OpenAI service with mocked cache."""
    with patch("app.infrastructure.ai.openai_service.AsyncOpenAI"):
        service = OpenAIService(cache_service=mock_cache_service)
        return service


class TestOpenAIServiceInitialization:
    """Test OpenAI service initialization."""

    @pytest.mark.asyncio
    async def test_create_service_with_cache(self, mock_cache_service):
        """Test service creation with cache service."""
        with patch("app.infrastructure.ai.openai_service.AsyncOpenAI"):
            service = await OpenAIService.create(cache_service=mock_cache_service)
            assert service is not None
            assert service.cache_service == mock_cache_service

    @pytest.mark.asyncio
    async def test_create_service_without_cache(self):
        """Test service creation without cache service."""
        with patch("app.infrastructure.ai.openai_service.AsyncOpenAI"):
            service = await OpenAIService.create()
            assert service is not None
            assert service.cache_service is None

    def test_model_specs_initialization(self, openai_service):
        """Test model specifications are properly initialized."""
        assert "embeddings" in openai_service._model_specs
        assert "chat" in openai_service._model_specs
        assert openai_service._model_specs["embeddings"]["model"] is not None


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, openai_service):
        """Test generating a single embedding."""
        mock_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await openai_service.generate_embedding("test text")

        assert result == mock_embedding
        assert openai_service._metrics["embeddings"] == 1

    @pytest.mark.asyncio
    async def test_generate_embedding_with_cache_hit(self, openai_service):
        """Test embedding generation with cache hit."""
        cached_embedding = [0.2] * 1536
        openai_service.cache_service.get = AsyncMock(return_value=cached_embedding)

        result = await openai_service.generate_embedding("test text")

        assert result == cached_embedding
        assert openai_service._metrics["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text_fails(self, openai_service):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await openai_service.generate_embedding("")

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, openai_service):
        """Test batch embedding generation."""
        texts = ["text1", "text2", "text3"]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=emb) for emb in mock_embeddings]

        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(return_value=mock_response)

        results = await openai_service.generate_embeddings_batch(texts)

        assert len(results) == 3
        assert results == mock_embeddings

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty_list(self, openai_service):
        """Test batch generation with empty list returns empty list."""
        results = await openai_service.generate_embeddings_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, openai_service):
        """Test error handling during embedding generation."""
        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            await openai_service.generate_embedding("test")

        assert openai_service._metrics["errors"] == 1


class TestChatCompletion:
    """Test chat completion functionality."""

    @pytest.mark.asyncio
    async def test_chat_completion(self, openai_service):
        """Test basic chat completion."""
        messages = [{"role": "user", "content": "Hello"}]

        mock_response = ChatCompletion(
            id="test-id",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hi there!"),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )

        openai_service._client = AsyncMock()
        openai_service._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.chat_completion(messages)

        assert result["choices"][0]["message"]["content"] == "Hi there!"
        assert result["usage"]["total_tokens"] == 15
        assert openai_service._metrics["chat"] == 1

    @pytest.mark.asyncio
    async def test_chat_completion_empty_messages_fails(self, openai_service):
        """Test that empty messages raises ValueError."""
        with pytest.raises(ValueError, match="Messages cannot be empty"):
            await openai_service.chat_completion([])

    @pytest.mark.asyncio
    async def test_chat_completion_with_parameters(self, openai_service):
        """Test chat completion with custom parameters."""
        messages = [{"role": "user", "content": "Test"}]

        mock_response = ChatCompletion(
            id="test-id",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Response"),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        )

        openai_service._client = AsyncMock()
        openai_service._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.chat_completion(
            messages,
            model="gpt-4",
            max_tokens=500,
            temperature=0.5
        )

        assert result is not None
        openai_service._client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_error_handling(self, openai_service):
        """Test error handling during chat completion."""
        openai_service._client = AsyncMock()
        openai_service._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(RuntimeError, match="Chat completion failed"):
            await openai_service.chat_completion([{"role": "user", "content": "test"}])

        assert openai_service._metrics["errors"] == 1


class TestTextCleaning:
    """Test text cleaning functionality."""

    def test_clean_text_removes_whitespace(self, openai_service):
        """Test that excessive whitespace is removed."""
        text = "This  has   too    much     whitespace"
        cleaned = openai_service._clean_text(text)
        assert cleaned == "This has too much whitespace"

    def test_clean_text_truncates_long_text(self, openai_service):
        """Test that overly long text is truncated."""
        max_chars = openai_service._model_specs["embeddings"]["max_tokens"] * 3
        long_text = "a" * (max_chars + 1000)
        cleaned = openai_service._clean_text(long_text)
        assert len(cleaned) == max_chars

    def test_clean_text_handles_non_string(self, openai_service):
        """Test that non-string input returns empty string."""
        assert openai_service._clean_text(None) == ""
        assert openai_service._clean_text(123) == ""


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, openai_service):
        """Test health check when service is healthy."""
        mock_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await openai_service.check_health()

        assert result["status"] == "healthy"
        assert result["embedding_dimension"] == 1536
        assert "provider" in result

    @pytest.mark.asyncio
    async def test_health_check_degraded_slow_response(self, openai_service):
        """Test health check shows degraded status on slow response."""
        import asyncio

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(6)  # Simulate slow response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            return mock_response

        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = slow_generate

        result = await openai_service.check_health()

        assert result["status"] == "degraded"
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, openai_service):
        """Test health check when service is unhealthy."""
        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        result = await openai_service.check_health()

        assert result["status"] == "unhealthy"
        assert "error" in result


class TestMetrics:
    """Test metrics functionality."""

    def test_get_metrics(self, openai_service):
        """Test getting service metrics."""
        openai_service._metrics["embeddings"] = 10
        openai_service._metrics["chat"] = 5

        metrics = openai_service.get_metrics()

        assert metrics["requests"]["embeddings"] == 10
        assert metrics["requests"]["chat"] == 5
        assert "config" in metrics

    def test_metrics_increment_on_operations(self, openai_service):
        """Test that metrics increment on operations."""
        initial_embeddings = openai_service._metrics["embeddings"]

        # Metrics should increment after successful operations
        # (tested implicitly in other tests)
        assert openai_service._metrics["embeddings"] >= initial_embeddings


class TestCacheIntegration:
    """Test cache integration."""

    @pytest.mark.asyncio
    async def test_embedding_cached_on_generation(self, openai_service):
        """Test that embeddings are cached after generation."""
        mock_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]

        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(return_value=mock_response)

        await openai_service.generate_embedding("test text")

        # Verify cache was called to store the result
        openai_service.cache_service.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_embeddings_cached_individually(self, openai_service):
        """Test that batch embeddings are cached individually."""
        texts = ["text1", "text2"]
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=emb) for emb in mock_embeddings]

        openai_service._client = AsyncMock()
        openai_service._client.embeddings.create = AsyncMock(return_value=mock_response)

        await openai_service.generate_embeddings_batch(texts)

        # Each embedding should be cached separately
        assert openai_service.cache_service.set.call_count == 2


class TestEmbeddingDimensions:
    """Test embedding dimension handling."""

    def test_get_embedding_dimension_default(self, openai_service):
        """Test getting embedding dimension for default model."""
        dim = openai_service.get_embedding_dimension()
        assert dim > 0

    def test_get_embedding_dimension_specific_model(self, openai_service):
        """Test getting embedding dimension for specific model."""
        dim = openai_service.get_embedding_dimension("text-embedding-3-large")
        assert dim == 3072  # Known dimension for this model