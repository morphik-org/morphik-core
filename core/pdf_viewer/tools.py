import base64
from io import BytesIO
from typing import List

import httpx
from PIL.Image import Image as ImageType

SUMMARY_PROMPT = "Please provide a concise summary of the provided image of a page from a PDF."
SUMMARY_PROMPT += "Focus on the main topics, key points, and any important information."
SUMMARY_PROMPT += "Your summaries will be used as an *index* to allow an agent to navigate the PDF."


class PDFViewer:
    """A state machine for navigating and viewing PDF pages."""

    def __init__(self, images: List[ImageType]):
        self.current_page: int = 0
        self.total_pages: int = len(images)
        self.images: List[ImageType] = images
        self.current_frame: str = self._create_page_url(self.current_page)
        self.api_base_url: str = "http://localhost:3000/api/pdf"
        self.client = httpx.Client(base_url=self.api_base_url)
        # Execute summarization in parallel batches of 10
        self.summaries: List[str] = [self._summarize_page(i) for i in range(self.total_pages)]

    def _make_api_call(self, method: str, endpoint: str, json_data: dict = None) -> httpx.Response:
        """Make API call to PDF viewer for UI side effects."""
        if method.upper() == "POST":
            return self.client.post(endpoint, json=json_data)
        elif method.upper() == "GET":
            return self.client.get(endpoint)

    def _create_page_url(self, page_number: int) -> str:
        """Convert a PIL image to base64 data URL."""
        image = self.images[page_number]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + image_base64

    def get_current_frame(self) -> str:
        """Get the current frame as a base64 data URL."""
        return self.current_frame

    def get_next_page(self) -> str:
        """Navigate to the next page and update state."""
        if self.current_page + 1 >= self.total_pages:
            return f"Already at last page ({self.current_page + 1} of {self.total_pages})"

        self.current_page += 1
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate page change to UI
        self._make_api_call("POST", f"/change-page/{self.current_page + 1}")

        return f"Successfully navigated to page {self.current_page + 1} of {self.total_pages}"

    def get_previous_page(self) -> str:
        """Navigate to the previous page and update state."""
        if self.current_page <= 0:
            return f"Already at first page (1 of {self.total_pages})"

        self.current_page -= 1
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate page change to UI
        self._make_api_call("POST", f"/change-page/{self.current_page + 1}")

        return f"Successfully navigated to page {self.current_page + 1} of {self.total_pages}"

    def go_to_page(self, page_number: int) -> str:
        """Navigate to a specific page number (0-indexed) and update state."""
        if page_number < 0 or page_number >= self.total_pages:
            return f"Invalid page number. Must be between 1 and {self.total_pages}"

        self.current_page = page_number
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate page change to UI (API uses 1-indexed)
        self._make_api_call("POST", f"/change-page/{self.current_page + 1}")

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
        if self.current_frame.startswith("data:image/png;base64,"):
            base64_data = self.current_frame.split(",", 1)[1]
            image_data = base64.b64decode(base64_data)
            buffer = BytesIO(image_data)
            from PIL import Image

            image = Image.open(buffer)
        else:
            # Fallback to original page if current_frame is not properly formatted
            image = self.images[self.current_page]

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
        self._make_api_call("POST", "/zoom/y", {"top": y1, "bottom": y2})
        self._make_api_call("POST", "/zoom/x", {"left": x1, "right": x2})

        return f"Successfully zoomed into region [{x1}, {y1}, {x2}, {y2}]"

    def zoom_out(self) -> str:
        """Reset zoom to show full page and update state."""
        self.current_frame = self._create_page_url(self.current_page)

        # Propagate full page zoom to UI (reset to full bounds)
        self._make_api_call("POST", "/zoom/x", {"left": 0, "right": 1000})
        self._make_api_call("POST", "/zoom/y", {"top": 0, "bottom": 1000})

        return "Successfully zoomed out to full page view"

    def get_page_summary(self, page_number: int) -> str:
        """Get the summary for a specific page."""
        if 0 <= page_number < self.total_pages:
            return self.summaries[page_number]
        return f"Invalid page number. Must be between 0 and {self.total_pages - 1}"

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
]


def get_pdf_viewer_tools_for_litellm():
    """Returns the tools description that can be passed to LiteLLM completion calls."""
    return PDF_VIEWER_TOOLS
