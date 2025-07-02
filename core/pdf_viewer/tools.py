import base64
import uuid
from io import BytesIO
from typing import List

import fitz  # PyMuPDF
import httpx
from PIL import Image

SUMMARY_PROMPT = "Please provide a concise summary of the provided image of a page from a PDF."
SUMMARY_PROMPT += "Focus on the main topics, key points, and any important information."
SUMMARY_PROMPT += "Your summaries will be used as an *index* to allow an agent to navigate the PDF."


class PDFViewer:
    """A state machine for navigating and viewing PDF pages with lazy loading."""

    def __init__(
        self,
        pdf_document: fitz.Document = None,
        images: List = None,  # Keep for backward compatibility
        api_base_url: str = None,
        session_id: str = None,
        user_id: str = None,
        document_id: str = None,
        document_service=None,
        auth=None,
    ):
        # Support both new PyMuPDF approach and legacy images approach
        if pdf_document is not None:
            self.pdf_document = pdf_document
            self.total_pages = len(pdf_document)
            self.use_lazy_loading = True
        elif images is not None:
            # Backward compatibility with old approach
            self.images = images
            self.total_pages = len(images)
            self.use_lazy_loading = False
            self.pdf_document = None
        else:
            raise ValueError("Either pdf_document or images must be provided")

        self.current_page: int = 0
        self.current_frame: str = self._create_page_url(self.current_page)

        # Use provided api_base_url or fall back to localhost for development
        self.api_base_url: str = api_base_url or "http://localhost:3000/api/pdf"
        # Generate session and user IDs if not provided
        self.session_id: str = session_id or str(uuid.uuid4())
        self.user_id: str = user_id or "anonymous"
        self.client = httpx.Client(base_url=self.api_base_url, follow_redirects=True)

        # Initialize empty summaries - will be generated on demand
        self.summaries: List[str] = [""] * self.total_pages

        # For document retrieval functionality
        self.document_id: str = document_id
        self.document_service = document_service
        self.auth = auth

    def _get_page_image(self, page_number: int) -> Image.Image:
        """Get PIL Image for a specific page, using lazy loading if available."""
        if self.use_lazy_loading:
            # Lazy loading with PyMuPDF - render page on demand
            page = self.pdf_document[page_number]
            # Use high DPI for better quality (150 DPI is a good balance of quality/speed)
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            return Image.open(BytesIO(img_data))
        else:
            # Legacy approach with pre-loaded images
            return self.images[page_number]

    def _create_page_url(self, page_number: int) -> str:
        """Convert a page to base64 data URL."""
        image = self._get_page_image(page_number)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + image_base64

    def _make_api_call(self, method: str, endpoint: str, json_data: dict = None) -> httpx.Response:
        """Make API call to PDF viewer for UI side effects with session and user scoping."""
        # Add session and user info to the request
        if json_data is None:
            json_data = {}

        # Add scoping information
        json_data.update({"sessionId": self.session_id, "userId": self.user_id})

        # Also add as headers for redundancy
        headers = {"x-session-id": self.session_id, "x-user-id": self.user_id, "Content-Type": "application/json"}

        if method.upper() == "POST":
            return self.client.post(endpoint, json=json_data, headers=headers)
        elif method.upper() == "GET":
            return self.client.get(endpoint, headers=headers)

    def get_current_frame(self) -> str:
        """Get the current frame as a base64 data URL."""
        return self.current_frame

    def get_session_info(self) -> dict:
        """Get session and user information."""
        return {"session_id": self.session_id, "user_id": self.user_id, "api_base_url": self.api_base_url}

    def get_next_page(self) -> str:
        """Navigate to the next page and update state."""
        if self.current_page + 1 >= self.total_pages:
            return f"Already at last page ({self.current_page + 1} of {self.total_pages})"

        self.current_page += 1
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate page change to UI
        try:
            response = self._make_api_call("POST", f"/change-page/{self.current_page + 1}")
            if response.status_code != 200:
                print(f"Warning: API call failed with status {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to sync with UI: {e}")

        return f"Successfully navigated to page {self.current_page + 1} of {self.total_pages}"

    def get_previous_page(self) -> str:
        """Navigate to the previous page and update state."""
        if self.current_page <= 0:
            return f"Already at first page (1 of {self.total_pages})"

        self.current_page -= 1
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate page change to UI
        try:
            response = self._make_api_call("POST", f"/change-page/{self.current_page + 1}")
            if response.status_code != 200:
                print(f"Warning: API call failed with status {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to sync with UI: {e}")

        return f"Successfully navigated to page {self.current_page + 1} of {self.total_pages}"

    def go_to_page(self, page_number: int) -> str:
        """Navigate to a specific page number (0-indexed) and update state."""
        if page_number < 0 or page_number >= self.total_pages:
            return f"Invalid page number. Must be between 1 and {self.total_pages}"

        self.current_page = page_number
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate page change to UI (API uses 1-indexed)
        try:
            response = self._make_api_call("POST", f"/change-page/{self.current_page + 1}")
            if response.status_code != 200:
                print(f"Warning: API call failed with status {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to sync with UI: {e}")

        return f"Successfully navigated to page {self.current_page + 1} of {self.total_pages}"

    def get_total_pages(self) -> int:
        """Get the total number of pages."""
        return self.total_pages

    def zoom_in(self, box_2d: List[int]) -> str:
        """Zoom into a specific region and update state."""
        if len(box_2d) != 4:
            return "Error: box_2d must contain exactly 4 coordinates [x1, y1, x2, y2]"

        x1, y1, x2, y2 = box_2d

        # Validate coordinates are within 0-1000 range
        for coord in box_2d:
            if not (0 <= coord <= 1000):
                return "Error: All coordinates must be between 0 and 1000"

        # Validate box coordinates
        if x1 >= x2 or y1 >= y2:
            return "Error: Invalid box coordinates. x1 must be < x2 and y1 must be < y2"

        # Get current frame image by decoding the base64 data
        # if self.current_frame.startswith("data:image/png;base64,"):
        #     base64_data = self.current_frame.split(",", 1)[1]
        #     image_data = base64.b64decode(base64_data)
        #     buffer = BytesIO(image_data)
        #     from PIL import Image

        #     image = Image.open(buffer)
        # else:
        #     # Fallback to original page if current_frame is not properly formatted
        #     image = self.images[self.current_page]

        image = self._get_page_image(self.current_page)
        width, height = image.size

        # Convert normalized coordinates (0-1000) to actual pixel coordinates
        actual_x1 = int((x1 / 1000) * width)
        actual_y1 = int((y1 / 1000) * height)
        actual_x2 = int((x2 / 1000) * width)
        actual_y2 = int((y2 / 1000) * height)

        # Crop the image to the specified region
        cropped_image = image.crop((actual_x1, actual_y1, actual_x2, actual_y2))

        # Convert cropped image to base64
        buffer = BytesIO()
        cropped_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        self.current_frame = "data:image/png;base64," + image_base64

        # Propagate zoom to UI
        try:
            y_response = self._make_api_call("POST", "/zoom/y", {"top": y1, "bottom": y2})
            x_response = self._make_api_call("POST", "/zoom/x", {"left": x1, "right": x2})
            if y_response.status_code != 200 or x_response.status_code != 200:
                print(f"Warning: Zoom API calls failed - Y: {y_response.status_code}, X: {x_response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to sync zoom with UI: {e}")

        return f"Successfully zoomed into region [{x1}, {y1}, {x2}, {y2}]"

    def zoom_out(self) -> str:
        """Reset zoom to show full page and update state."""
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate full page zoom to UI (reset to full bounds)
        try:
            x_response = self._make_api_call("POST", "/zoom/x", {"left": 0, "right": 1000})
            y_response = self._make_api_call("POST", "/zoom/y", {"top": 0, "bottom": 1000})
            if x_response.status_code != 200 or y_response.status_code != 200:
                print(f"Warning: Zoom out API calls failed - X: {x_response.status_code}, Y: {y_response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to sync zoom out with UI: {e}")

        return "Successfully zoomed out to full page view"

    def get_page_summary(self, page_number: int) -> str:
        """Get the summary for a specific page."""
        if self.summaries[page_number] == "":
            self.summaries[page_number] = self._summarize_page(page_number)
        return self.summaries[page_number]
        # if 0 <= page_number < self.total_pages:
        #     return self.summaries[page_number]
        # return f"Invalid page number. Must be between 0 and {self.total_pages - 1}"

    async def find_most_relevant_page(self, query: str) -> str:
        """Find and navigate to the most relevant page based on a search query."""
        if not self.document_service or not self.auth or not self.document_id:
            return "Error: Document search functionality not available. Missing document service, authentication, or document ID."

        try:
            # Search for relevant chunks in this specific document
            chunks = await self.document_service.retrieve_chunks(
                query=query,
                auth=self.auth,
                filters={"external_id": self.document_id},  # Filter to only this document
                k=5,  # Get top 5 chunks to find the best page
                min_score=0.0,
                use_colpali=True,  # Use multimodal search for better PDF results
                use_reranking=False,
            )

            if not chunks:
                return f"No relevant content found for query: '{query}'"

            # Find the chunk with the highest score
            best_chunk = max(chunks, key=lambda x: x.score)

            # Extract page information from chunk metadata
            # Chunk numbers typically correspond to pages, but we need to be careful about 0-indexing
            target_page = best_chunk.chunk_number

            # Ensure the page number is valid (0-indexed)
            if target_page < 0 or target_page >= self.total_pages:
                return f"Error: Found relevant content on page {target_page + 1}, but this page is out of range (1-{self.total_pages})"

            # Navigate to the most relevant page
            old_page = self.current_page + 1  # Convert to 1-indexed for display
            self.current_page = target_page
            self.current_frame = self._create_page_url(self.current_page)

            # Propagate page change to UI (API uses 1-indexed)
            try:
                response = self._make_api_call("POST", f"/change-page/{self.current_page + 1}")
                if response.status_code != 200:
                    print(f"Warning: API call failed with status {response.status_code}")
            except Exception as e:
                print(f"Warning: Failed to sync with UI: {e}")

            # Create a detailed response with search results
            result_summary = f"Found most relevant content on page {target_page + 1} (score: {best_chunk.score:.3f})"
            if old_page != target_page + 1:
                result_summary += f". Navigated from page {old_page} to page {target_page + 1}."
            else:
                result_summary += ". Already on the most relevant page."

            # Add preview of the found content
            content_preview = best_chunk.content[:200] + "..." if len(best_chunk.content) > 200 else best_chunk.content
            result_summary += f"\n\nRelevant content preview: {content_preview}"

            # Add information about other relevant pages if found
            if len(chunks) > 1:
                other_pages = [str(chunk.chunk_number + 1) for chunk in chunks[1:] if chunk.chunk_number != target_page]
                if other_pages:
                    result_summary += f"\n\nOther relevant pages found: {', '.join(other_pages[:3])}"

            return result_summary

        except Exception as e:
            return f"Error searching document: {str(e)}"

    def _summarize_page(self, page_number: int) -> str:
        """Summarize a page using Gemini 2.5 Flash model."""
        import litellm

        # Get the page image URL
        page_url = self._create_page_url(page_number)

        # Create the message with the image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SUMMARY_PROMPT},
                    {"type": "image_url", "image_url": {"url": page_url}},
                ],
            }
        ]

        # Call Gemini 2.5 Flash using litellm
        response = litellm.completion(model="gemini/gemini-2.5-flash-preview-05-20", messages=messages, max_tokens=500)

        return response.choices[0].message.content

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def close(self):
        """Clean up resources."""
        if self.pdf_document is not None:
            self.pdf_document.close()
        if hasattr(self, "client"):
            self.client.close()


# LiteLLM Tools Description for PDF Viewer
PDF_VIEWER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_next_page",
            "description": "Navigate to the next page in the PDF. Returns success message or error if already at last page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_page",
            "description": "Navigate to the previous page in the PDF. Returns success message or error if already at first page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_to_page",
            "description": "Navigate to a specific page number in the PDF. Page numbers are 0-indexed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "The page number to navigate to (0-indexed)",
                    }
                },
                "required": ["page_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": "Zoom into a specific rectangular region of the current PDF page. Coordinates use a 0-1000 scale where (0,0) is top-left and (1000,1000) is bottom-right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Bounding box coordinates [x1, y1, x2, y2] where x1 < x2 and y1 < y2. Coordinates are on 0-1000 scale.",
                    }
                },
                "required": ["box_2d"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_out",
            "description": "Reset the zoom to show the full current page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_summary",
            "description": "Get a summary of a specific page in the PDF. Useful for understanding page content before navigating to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "The page number to get summary for (0-indexed)",
                    }
                },
                "required": ["page_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_pages",
            "description": "Get the total number of pages in the PDF document.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_most_relevant_page",
            "description": "Search the entire PDF document for content most relevant to a query and automatically navigate to that page. This tool performs semantic search across all pages and jumps to the most relevant one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant content in the PDF. Can be a question, topic, or keywords.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def get_pdf_viewer_tools_for_litellm():
    """Returns the tools description that can be passed to LiteLLM completion calls."""
    return PDF_VIEWER_TOOLS
