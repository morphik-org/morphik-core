import base64
import io
from logging import getLogger

import pdf2image
import pdf2image.exceptions
from google.genai import types
from PIL.Image import Image

from core.database.base_database import BaseDatabase
from core.models.auth import AuthContext
from core.storage.base_storage import BaseStorage

logger = getLogger(__name__)


def scale_and_clamp(val1, val2, current_scale, desired_scale, padding_percent):
    padding_multiplier1, padding_multiplier2 = 1 - padding_percent / 200, 1 + padding_percent / 200
    true_val1 = int((val1 / current_scale) * desired_scale * padding_multiplier1)
    true_val2 = int((val2 / current_scale) * desired_scale * padding_multiplier2)
    return max(true_val1, 0), min(true_val2, desired_scale)


def crop_image(image: Image, ymin: int, ymax: int, xmin: int, xmax: int, padding_percent: int = 20) -> Image:
    width, height = image.size
    abs_y1, abs_y2 = scale_and_clamp(ymin, ymax, 1000, height, padding_percent)
    abs_x1, abs_x2 = scale_and_clamp(xmin, xmax, 1000, width, padding_percent)
    cropped_image = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
    return cropped_image


async def get_pdf(document_id: str, storage: BaseStorage, db: BaseDatabase, auth: AuthContext) -> bytes:
    document = await db.get_document(document_id, auth)
    if not document:
        raise ValueError(f"Document {document_id} not found")
    storage_info = document.storage_files[-1]
    main_file = await storage.download_file(bucket=storage_info.bucket, key=storage_info.key)
    return main_file


async def get_pdf_images(document: bytes):
    try:
        images = pdf2image.convert_from_bytes(document)
    except pdf2image.exceptions.PDFPageCountError:
        logger.error("Unable to read the document, it might not be a valid PDF")
        raise ValueError("Unable to read the document, it might not be a valid PDF")
    return images


def image_to_base64(image: Image) -> str:
    """Convert PIL Image to base64 string for Gemini."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class DocumentNavigator:
    def __init__(self, document_id: str, storage: BaseStorage, db: BaseDatabase, auth: AuthContext):
        # metadata, morphik
        self.document_id = document_id
        self.storage = storage
        self.db = db
        self.auth = auth

        # navigation state - will be initialized async
        self.pdf_bytes = None
        self.pdf_images = None
        self.current_page_number = 0
        self.length = 0
        self.is_initialized = False

        # zoom
        self.padding = 20

    async def initialize(self):
        """Initialize the navigator with PDF data."""
        if not self.is_initialized:
            self.pdf_bytes = await get_pdf(self.document_id, self.storage, self.db, self.auth)
            self.pdf_images = await get_pdf_images(self.pdf_bytes)
            self.length = len(self.pdf_images)
            self.is_initialized = True

    @property
    def current_page(self) -> Image:
        if not self.is_initialized:
            raise ValueError("Navigator not initialized. Call initialize() first.")
        return self.pdf_images[self.current_page_number]

    def get_current_page_info(self) -> dict:
        """Get information about the current page."""
        return {
            "current_page": self.current_page_number + 1,  # 1-indexed for user
            "total_pages": self.length,
            "has_next": self.current_page_number < self.length - 1,
            "has_previous": self.current_page_number > 0,
        }

    def next_page(self) -> dict:
        """Navigate to the next page."""
        if not self.is_initialized:
            raise ValueError("Navigator not initialized")

        if self.current_page_number >= self.length - 1:
            return {"success": False, "message": "Already on the last page", "page_info": self.get_current_page_info()}

        self.current_page_number += 1
        return {
            "success": True,
            "message": f"Moved to page {self.current_page_number + 1}",
            "page_info": self.get_current_page_info(),
        }

    def previous_page(self) -> dict:
        """Navigate to the previous page."""
        if not self.is_initialized:
            raise ValueError("Navigator not initialized")

        if self.current_page_number <= 0:
            return {"success": False, "message": "Already on the first page", "page_info": self.get_current_page_info()}

        self.current_page_number -= 1
        return {
            "success": True,
            "message": f"Moved to page {self.current_page_number + 1}",
            "page_info": self.get_current_page_info(),
        }

    def jump_to_page(self, page: int) -> dict:
        """Jump to a specific page (1-indexed)."""
        if not self.is_initialized:
            raise ValueError("Navigator not initialized")

        # Convert to 0-indexed
        page_index = page - 1

        if page_index < 0 or page_index >= self.length:
            return {
                "success": False,
                "message": f"Invalid page number. Must be between 1 and {self.length}",
                "page_info": self.get_current_page_info(),
            }

        self.current_page_number = page_index
        return {"success": True, "message": f"Jumped to page {page}", "page_info": self.get_current_page_info()}

    def zoom_in(self, ymin: int, ymax: int, xmin: int, xmax: int) -> dict:
        """Zoom into a specific region of the current page."""
        if not self.is_initialized:
            raise ValueError("Navigator not initialized")

        try:
            cropped_image = crop_image(self.current_page, ymin, ymax, xmin, xmax, self.padding)
            return {
                "success": True,
                "message": f"Zoomed into region ({xmin}, {ymin}) to ({xmax}, {ymax})",
                "cropped_image": cropped_image,
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to zoom: {str(e)}", "page_info": self.get_current_page_info()}

    def get_navigation_tools(self):
        """Get the navigation tools for Gemini function calling."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="next_page",
                        description="Navigate to the next page in the PDF document",
                        parameters=types.Schema(type=types.Type.OBJECT, properties={}, required=[]),
                    ),
                    types.FunctionDeclaration(
                        name="previous_page",
                        description="Navigate to the previous page in the PDF document",
                        parameters=types.Schema(type=types.Type.OBJECT, properties={}, required=[]),
                    ),
                    types.FunctionDeclaration(
                        name="jump_to_page",
                        description="Jump to a specific page number in the PDF document",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "page": types.Schema(
                                    type=types.Type.INTEGER, description="The page number to jump to (1-indexed)"
                                )
                            },
                            required=["page"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="zoom_in",
                        description="Zoom into a specific rectangular region of the current page. Coordinates are normalized to 1000x1000 scale.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "ymin": types.Schema(type=types.Type.INTEGER, description="Top Y coordinate (0-1000)"),
                                "ymax": types.Schema(
                                    type=types.Type.INTEGER, description="Bottom Y coordinate (0-1000)"
                                ),
                                "xmin": types.Schema(type=types.Type.INTEGER, description="Left X coordinate (0-1000)"),
                                "xmax": types.Schema(
                                    type=types.Type.INTEGER, description="Right X coordinate (0-1000)"
                                ),
                            },
                            required=["ymin", "ymax", "xmin", "xmax"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="get_current_page_info",
                        description="Get information about the current page and navigation state",
                        parameters=types.Schema(type=types.Type.OBJECT, properties={}, required=[]),
                    ),
                ]
            )
        ]

    def execute_function(self, function_name: str, arguments: dict):
        """Execute a navigation function by name."""
        if function_name == "next_page":
            return self.next_page()
        elif function_name == "previous_page":
            return self.previous_page()
        elif function_name == "jump_to_page":
            return self.jump_to_page(arguments.get("page"))
        elif function_name == "zoom_in":
            return self.zoom_in(
                arguments.get("ymin"), arguments.get("ymax"), arguments.get("xmin"), arguments.get("xmax")
            )
        elif function_name == "get_current_page_info":
            return self.get_current_page_info()
        else:
            raise ValueError(f"Unknown function: {function_name}")
