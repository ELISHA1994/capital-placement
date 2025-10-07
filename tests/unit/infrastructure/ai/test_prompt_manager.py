"""Tests for Prompt Manager infrastructure implementation."""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from app.infrastructure.ai.prompt_manager import PromptManager, PromptType


@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def prompt_manager(mock_cache_service):
    """Create prompt manager with mocked cache."""
    return PromptManager(cache_service=mock_cache_service)


class TestPromptGeneration:
    """Test prompt generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_query_expansion_prompt(self, prompt_manager):
        """Test generating query expansion prompt."""
        context = {"query": "Python developer"}

        prompt = await prompt_manager.generate_prompt(
            PromptType.QUERY_EXPANSION,
            context
        )

        assert "messages" in prompt
        assert len(prompt["messages"]) == 2
        assert prompt["messages"][0]["role"] == "system"
        assert prompt["messages"][1]["role"] == "user"
        assert "Python developer" in prompt["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_generate_cv_analysis_prompt(self, prompt_manager):
        """Test generating CV analysis prompt."""
        context = {"cv_content": "Software engineer with 5 years experience"}

        prompt = await prompt_manager.generate_prompt(
            PromptType.CV_ANALYSIS,
            context
        )

        assert "messages" in prompt
        assert "Software engineer" in prompt["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_generate_prompt_with_custom_instructions(self, prompt_manager):
        """Test prompt generation with custom instructions."""
        context = {"query": "test"}
        custom = "Focus on technical skills"

        prompt = await prompt_manager.generate_prompt(
            PromptType.QUERY_EXPANSION,
            context,
            custom_instructions=custom
        )

        assert custom in prompt["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_unknown_prompt_type_fails(self, prompt_manager):
        """Test unknown prompt type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown prompt type"):
            await prompt_manager.generate_prompt("invalid_type", {})


class TestPromptTemplates:
    """Test prompt template functionality."""

    def test_get_available_prompt_types(self, prompt_manager):
        """Test getting list of available prompt types."""
        types = prompt_manager.get_available_prompt_types()

        assert isinstance(types, list)
        assert len(types) > 0
        assert PromptType.QUERY_EXPANSION.value in types
        assert PromptType.CV_ANALYSIS.value in types

    def test_get_template_info(self, prompt_manager):
        """Test getting template information."""
        info = prompt_manager.get_template_info(PromptType.QUERY_EXPANSION)

        assert "prompt_type" in info
        assert "system_message_length" in info
        assert "required_context_vars" in info
        assert "query" in info["required_context_vars"]

    def test_get_template_info_invalid_type_fails(self, prompt_manager):
        """Test getting info for invalid type fails."""
        with pytest.raises(ValueError):
            prompt_manager.get_template_info("invalid")


class TestHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_create_query_expansion_prompt(self, prompt_manager):
        """Test convenience method for query expansion."""
        prompt = await prompt_manager.create_query_expansion_prompt("test query")

        assert "messages" in prompt
        assert "test query" in prompt["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_create_cv_analysis_prompt(self, prompt_manager):
        """Test convenience method for CV analysis."""
        prompt = await prompt_manager.create_cv_analysis_prompt("CV content here")

        assert "messages" in prompt
        assert "CV content here" in prompt["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_create_job_matching_prompt(self, prompt_manager):
        """Test convenience method for job matching."""
        prompt = await prompt_manager.create_job_matching_prompt(
            candidate_profile="Profile data",
            job_description="Job requirements"
        )

        assert "messages" in prompt
        assert "Profile data" in prompt["messages"][1]["content"]
        assert "Job requirements" in prompt["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_create_result_ranking_prompt(self, prompt_manager):
        """Test convenience method for result ranking."""
        candidates = [
            {"id": "1", "summary": "Candidate 1"},
            {"id": "2", "summary": "Candidate 2"}
        ]

        prompt = await prompt_manager.create_result_ranking_prompt(
            job_requirements="Requirements",
            candidate_results=candidates
        )

        assert "messages" in prompt
        assert "Candidate 1" in prompt["messages"][1]["content"]


class TestCacheIntegration:
    """Test cache integration."""

    @pytest.mark.asyncio
    async def test_prompt_cached_after_generation(self, prompt_manager):
        """Test prompts are cached after generation."""
        context = {"query": "test"}

        await prompt_manager.generate_prompt(PromptType.QUERY_EXPANSION, context)

        prompt_manager.cache_service.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_prompt_returned(self, prompt_manager):
        """Test cached prompt is returned."""
        cached_prompt = {"messages": [], "cached": True}
        prompt_manager.cache_service.get = AsyncMock(return_value=cached_prompt)

        result = await prompt_manager.generate_prompt(
            PromptType.QUERY_EXPANSION,
            {"query": "test"}
        )

        assert result == cached_prompt
        assert prompt_manager._metrics["cache_hits"] == 1


class TestMetrics:
    """Test metrics functionality."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, prompt_manager):
        """Test getting prompt manager metrics."""
        # Generate some prompts to populate metrics
        await prompt_manager.generate_prompt(
            PromptType.QUERY_EXPANSION,
            {"query": "test1"}
        )
        await prompt_manager.generate_prompt(
            PromptType.CV_ANALYSIS,
            {"cv_content": "test2"}
        )

        metrics = await prompt_manager.get_metrics()

        assert "generation_stats" in metrics
        assert "available_templates" in metrics
        assert metrics["generation_stats"]["prompts_generated"] >= 2


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, prompt_manager):
        """Test health check when healthy."""
        result = await prompt_manager.check_health()

        assert result["status"] == "healthy"
        assert result["test_generation"] == "successful"
        assert result["templates_loaded"] > 0

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self, prompt_manager):
        """Test health check when unhealthy."""
        # Break the template system
        prompt_manager._prompt_templates = {}

        result = await prompt_manager.check_health()

        assert result["status"] == "unhealthy"
        assert "error" in result