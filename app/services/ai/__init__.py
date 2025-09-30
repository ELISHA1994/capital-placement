"""
AI Services Package

Cloud-agnostic AI services for the CV matching platform:
- OpenAI SDK integration (direct or Azure compatible)
- Embedding generation and vector search
- Document processing with LangChain
- Semantic caching and search optimization
- Prompt management and AI operations

This package replaces the Azure-specific services with cloud-agnostic
implementations that support both OpenAI direct and Azure OpenAI deployments.
"""

from .openai_service import OpenAIService
from .embedding_service import EmbeddingService
from .prompt_manager import PromptManager
from .cache_manager import CacheManager

__all__ = [
    "OpenAIService",
    "EmbeddingService", 
    "PromptManager",
    "CacheManager",
]