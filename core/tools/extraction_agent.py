"""
Document extraction agent used for headless document processing.

This agent handles document extraction without UI dependencies,
working directly with document content and chunks.
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List

import pdf2image
from PIL import Image

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """Agent for extracting structured data from documents."""

    def __init__(self, document_service, document_id: str, auth_ctx):
        self.document_service = document_service
        self.document_id = document_id
        self.auth = auth_ctx
        self.current_page = 0
        self.pages = []
        self.page_contents = []
        self._initialized = False

    async def initialize(self):
        """Initialize the agent with document content."""
        if self._initialized:
            return

        # Get document
        doc = await self.document_service.db.get_document(self.document_id, self.auth)
        if not doc:
            raise ValueError(f"Document {self.document_id} not found or access denied")

        # Handle different document types
        if doc.content_type == "application/pdf":
            await self._load_pdf(doc)
        elif doc.content_type in ["image/png", "image/jpeg", "image/jpg"]:
            await self._load_image(doc)
        else:
            # Text documents or other types
            await self._load_text(doc)

        self._initialized = True

    async def _load_pdf(self, doc):
        """Load PDF document pages."""
        if not doc.storage_info:
            raise ValueError("PDF document missing storage_info")

        bucket = doc.storage_info["bucket"]
        key = doc.storage_info["key"]
        pdf_bytes = await self.document_service.storage.download_file(bucket, key)

        if hasattr(pdf_bytes, "read"):
            pdf_bytes = pdf_bytes.read()

        # Convert PDF to images
        try:
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=150)
            self.pages = images

            # For PDFs, we'll rely on the image content rather than text chunks
            # This simplifies the extraction and avoids the missing get_chunks_for_document method
            self.page_contents = [""] * len(images)

        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            # Fallback to single page with content
            self.pages = [self._create_placeholder_image()]
            self.page_contents = [doc.system_metadata.get("content", "")]

    async def _load_image(self, doc):
        """Load image document."""
        if doc.storage_info:
            bucket = doc.storage_info["bucket"]
            key = doc.storage_info["key"]
            image_bytes = await self.document_service.storage.download_file(bucket, key)

            if hasattr(image_bytes, "read"):
                image_bytes = image_bytes.read()

            image = Image.open(BytesIO(image_bytes))
            self.pages = [image]
        else:
            self.pages = [self._create_placeholder_image()]

        # Get content from metadata (chunks not yet implemented)
        # TODO: Add chunk support when get_chunks_for_document is available
        self.page_contents = [doc.system_metadata.get("content", "")]

    async def _load_text(self, doc):
        """Load text document."""
        # For text documents, create a single "page" with all content
        self.pages = [self._create_placeholder_image()]

        # Get content from various sources
        content = ""

        # Try system metadata first
        if doc.system_metadata.get("content"):
            content = doc.system_metadata["content"]
        elif doc.storage_info:
            # Download and decode if possible
            try:
                bucket = doc.storage_info["bucket"]
                key = doc.storage_info["key"]
                file_bytes = await self.document_service.storage.download_file(bucket, key)

                if hasattr(file_bytes, "read"):
                    file_bytes = file_bytes.read()

                content = file_bytes.decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to load text content: {e}")

        self.page_contents = [content]

    def _create_placeholder_image(self):
        """Create a placeholder image for non-visual documents."""
        img = Image.new("RGB", (800, 600), color="white")
        return img

    def get_total_pages(self) -> int:
        """Get the total number of pages."""
        return len(self.pages)

    def get_current_page_number(self) -> int:
        """Get current page number (1-indexed for display)."""
        return self.current_page + 1

    def go_to_page(self, page_number: int) -> str:
        """Navigate to a specific page (0-indexed)."""
        if page_number < 0 or page_number >= len(self.pages):
            return f"Invalid page number. Must be between 1 and {len(self.pages)}"

        self.current_page = page_number
        return f"Successfully navigated to page {page_number + 1} of {len(self.pages)}"

    def get_next_page(self) -> str:
        """Navigate to the next page."""
        if self.current_page + 1 >= len(self.pages):
            return f"Already at last page ({self.current_page + 1} of {len(self.pages)})"

        self.current_page += 1
        return f"Successfully navigated to page {self.current_page + 1} of {len(self.pages)}"

    def get_previous_page(self) -> str:
        """Navigate to the previous page."""
        if self.current_page <= 0:
            return f"Already at first page (1 of {len(self.pages)})"

        self.current_page -= 1
        return f"Successfully navigated to page {self.current_page + 1} of {len(self.pages)}"

    def get_current_page_content(self) -> str:
        """Get the text content of the current page."""
        if self.current_page < len(self.page_contents):
            return self.page_contents[self.current_page]
        return ""

    def get_page_content(self, page_number: int) -> str:
        """Get the text content of a specific page (0-indexed)."""
        if 0 <= page_number < len(self.page_contents):
            return self.page_contents[page_number]
        return ""

    def get_current_page_image(self) -> str:
        """Get the current page as a base64 image."""
        if self.current_page < len(self.pages):
            image = self.pages[self.current_page]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{image_base64}"
        return ""

    def search_content(self, query: str) -> List[Dict[str, Any]]:
        """Search for content across all pages."""
        results = []
        query_lower = query.lower()

        for i, content in enumerate(self.page_contents):
            if query_lower in content.lower():
                # Find snippet around the match
                idx = content.lower().find(query_lower)
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 50)
                snippet = content[start:end]

                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."

                results.append({"page": i + 1, "snippet": snippet, "match_count": content.lower().count(query_lower)})

        return results

    async def find_most_relevant_page(self, query: str) -> str:
        """Find the most relevant page for a query."""
        results = self.search_content(query)

        if not results:
            return f"No pages found containing '{query}'"

        # Sort by match count
        results.sort(key=lambda x: x["match_count"], reverse=True)

        # Go to the most relevant page
        best_page = results[0]["page"] - 1
        self.current_page = best_page

        return f"Found {len(results)} pages containing '{query}'. Navigated to page {results[0]['page']} with {results[0]['match_count']} matches."
