"""Adapter that composes infrastructure document processors into the upload contract."""

from __future__ import annotations

from typing import Any, Dict


class DocumentProcessorAdapter:
    """Bridge PDF processor output to the IDocumentProcessor protocol."""

    def __init__(self, pdf_processor: Any, content_extractor: Any):
        self._pdf_processor = pdf_processor
        self._content_extractor = content_extractor

    async def process_document(self, file_content: bytes, filename: str, **kwargs: Any) -> Dict[str, Any]:
        """Process the incoming document and return normalized text plus metadata."""
        file_extension = filename.lower().split(".")[-1] if "." in filename else "unknown"

        if file_extension == "pdf":
            pdf_document = await self._pdf_processor.process_pdf(
                pdf_content=file_content,
                filename=filename,
            )
            return {
                "text": pdf_document.full_text,
                "metadata": {
                    **pdf_document.metadata,
                    **pdf_document.processing_info,
                },
            }

        try:
            text_content = file_content.decode("utf-8")
            metadata = {"file_type": file_extension}
            return {
                "text": text_content,
                "metadata": metadata,
            }
        except UnicodeDecodeError:
            text_content = file_content.decode("utf-8", errors="ignore")
            metadata = {"file_type": file_extension, "encoding_issues": True}
            return {
                "text": text_content,
                "metadata": metadata,
            }


__all__ = ["DocumentProcessorAdapter"]
