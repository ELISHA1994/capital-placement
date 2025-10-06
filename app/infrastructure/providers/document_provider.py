"""Document processing provider utilities."""

from __future__ import annotations

import asyncio
from typing import Optional

from app.services.document.content_extractor import ContentExtractor
from app.services.document.pdf_processor import PDFProcessor
from app.services.document.quality_analyzer import QualityAnalyzer
from app.infrastructure.providers.ai_provider import get_openai_service, get_prompt_manager

_content_extractor: Optional[ContentExtractor] = None
_pdf_processor: Optional[PDFProcessor] = None
_quality_analyzer: Optional[QualityAnalyzer] = None

_content_extractor_lock = asyncio.Lock()
_pdf_processor_lock = asyncio.Lock()
_quality_analyzer_lock = asyncio.Lock()


async def get_content_extractor() -> ContentExtractor:
    """Return singleton content extractor service."""
    global _content_extractor

    if _content_extractor is not None:
        return _content_extractor

    async with _content_extractor_lock:
        if _content_extractor is not None:
            return _content_extractor

        # Content extractor requires AI services
        openai_service = await get_openai_service()
        prompt_manager = await get_prompt_manager()

        _content_extractor = ContentExtractor(
            openai_service=openai_service,
            prompt_manager=prompt_manager,
        )
        return _content_extractor


async def reset_content_extractor() -> None:
    """Reset content extractor singleton."""
    global _content_extractor
    async with _content_extractor_lock:
        _content_extractor = None


async def get_pdf_processor() -> PDFProcessor:
    """Return singleton PDF processor service."""
    global _pdf_processor

    if _pdf_processor is not None:
        return _pdf_processor

    async with _pdf_processor_lock:
        if _pdf_processor is not None:
            return _pdf_processor

        _pdf_processor = PDFProcessor()
        return _pdf_processor


async def reset_pdf_processor() -> None:
    """Reset PDF processor singleton."""
    global _pdf_processor
    async with _pdf_processor_lock:
        _pdf_processor = None


async def get_quality_analyzer() -> QualityAnalyzer:
    """Return singleton quality analyzer service."""
    global _quality_analyzer

    if _quality_analyzer is not None:
        return _quality_analyzer

    async with _quality_analyzer_lock:
        if _quality_analyzer is not None:
            return _quality_analyzer

        # Quality analyzer requires AI services
        openai_service = await get_openai_service()
        prompt_manager = await get_prompt_manager()

        _quality_analyzer = QualityAnalyzer(
            openai_service=openai_service,
            prompt_manager=prompt_manager,
        )
        return _quality_analyzer


async def reset_quality_analyzer() -> None:
    """Reset quality analyzer singleton."""
    global _quality_analyzer
    async with _quality_analyzer_lock:
        _quality_analyzer = None


__all__ = [
    "get_content_extractor",
    "reset_content_extractor",
    "get_pdf_processor",
    "reset_pdf_processor",
    "get_quality_analyzer",
    "reset_quality_analyzer",
]