"""Tests for PDF processor infrastructure service."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from io import BytesIO

from app.infrastructure.document.pdf_processor import (
    PDFProcessor,
    PDFDocument,
    PDFPage,
    PDFProcessingError
)


class TestPDFProcessor:
    """Test suite for PDFProcessor."""

    @pytest.fixture
    def pdf_processor(self):
        """Create PDF processor instance."""
        return PDFProcessor()

    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        # Minimal PDF content (not a real PDF, but good enough for tests)
        return b"%PDF-1.4\nTest content\n%%EOF"

    @pytest.mark.asyncio
    async def test_process_pdf_basic(self, pdf_processor, sample_pdf_content):
        """Test basic PDF processing."""
        with patch('app.infrastructure.document.pdf_processor.PdfReader') as mock_reader:
            # Setup mock
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test page content"
            mock_reader_instance = Mock()
            mock_reader_instance.pages = [mock_page]
            mock_reader_instance.is_encrypted = False
            mock_reader_instance.metadata = {}
            mock_reader.return_value = mock_reader_instance

            # Process PDF
            result = await pdf_processor.process_pdf(
                pdf_content=sample_pdf_content,
                filename="test.pdf"
            )

            # Verify result
            assert isinstance(result, PDFDocument)
            assert result.total_pages == 1
            assert "Test page content" in result.full_text

    @pytest.mark.asyncio
    async def test_process_pdf_empty_content(self, pdf_processor):
        """Test processing empty PDF content."""
        with pytest.raises(PDFProcessingError, match="PDF content is empty"):
            await pdf_processor.process_pdf(pdf_content=b"", filename="empty.pdf")

    @pytest.mark.asyncio
    async def test_process_pdf_size_exceeded(self, pdf_processor):
        """Test processing PDF that exceeds size limit."""
        large_content = b"x" * (pdf_processor.settings.MAX_DOCUMENT_SIZE + 1)

        with pytest.raises(PDFProcessingError, match="exceeds maximum allowed"):
            await pdf_processor.process_pdf(
                pdf_content=large_content,
                filename="large.pdf"
            )

    @pytest.mark.asyncio
    async def test_process_pdf_encrypted(self, pdf_processor, sample_pdf_content):
        """Test processing encrypted PDF."""
        with patch('app.infrastructure.document.pdf_processor.PdfReader') as mock_reader:
            mock_reader_instance = Mock()
            mock_reader_instance.is_encrypted = True
            mock_reader_instance.decrypt.side_effect = Exception("Cannot decrypt")
            mock_reader.return_value = mock_reader_instance

            with pytest.raises(PDFProcessingError, match="password protected"):
                await pdf_processor.process_pdf(
                    pdf_content=sample_pdf_content,
                    filename="encrypted.pdf"
                )

    @pytest.mark.asyncio
    async def test_clean_text(self, pdf_processor):
        """Test text cleaning functionality."""
        dirty_text = "  Text  with   excessive   spaces\n\n\n\n  "
        clean_text = pdf_processor._clean_text(dirty_text)

        assert "excessive spaces" in clean_text
        assert "   " not in clean_text  # No excessive spaces
        assert clean_text.strip() == clean_text  # No leading/trailing whitespace

    @pytest.mark.asyncio
    async def test_pdf_document_get_page(self):
        """Test PDFDocument get_page method."""
        page1 = PDFPage(1, "Page 1 content")
        page2 = PDFPage(2, "Page 2 content")
        doc = PDFDocument(pages=[page1, page2])

        result = doc.get_page(2)
        assert result == page2
        assert result.text == "Page 2 content"

    @pytest.mark.asyncio
    async def test_pdf_document_get_pages_range(self):
        """Test PDFDocument get_pages_range method."""
        pages = [PDFPage(i, f"Page {i}") for i in range(1, 6)]
        doc = PDFDocument(pages=pages)

        result = doc.get_pages_range(2, 4)
        assert len(result) == 3
        assert result[0].page_number == 2
        assert result[2].page_number == 4

    @pytest.mark.asyncio
    async def test_check_health_success(self, pdf_processor):
        """Test health check succeeds."""
        with patch.object(pdf_processor, 'process_pdf', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = PDFDocument(pages=[])

            health = await pdf_processor.check_health()

            assert health["status"] == "healthy"
            assert health["processor"] == "operational"
