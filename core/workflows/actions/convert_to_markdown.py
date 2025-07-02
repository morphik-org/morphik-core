from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

from google import genai
from google.genai import types

from core.models.workflows import ActionDefinition

"""Convert documents to markdown format using Google Gemini.

This action converts various file formats (PDF, images, documents) to markdown,
preserving structure and formatting as much as possible.
"""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static action metadata (exported for registry)
# ---------------------------------------------------------------------------

action_id = "morphik.actions.convert_to_markdown"

definition = ActionDefinition(
    id=action_id,
    name="Convert to Markdown",
    description="Convert documents to markdown format using Google Gemini",
    parameters_schema={
        "type": "object",
        "properties": {
            "api_key_env": {
                "type": "string",
                "description": "Environment variable name containing the Gemini API key",
                "default": "GEMINI_API_KEY",
            },
            "model": {
                "type": "string",
                "description": "Gemini model to use for conversion",
                "default": "gemini-2.5-pro",
            },
            "temperature": {
                "type": "number",
                "description": "Temperature for generation (0-1)",
                "default": 0,
                "minimum": 0,
                "maximum": 1,
            },
            "custom_prompt": {
                "type": "string",
                "description": "Optional custom prompt to append to the conversion request",
                "default": "",
            },
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "markdown": {"type": "string", "description": "The converted markdown content"},
            "original_filename": {"type": "string"},
            "mime_type": {"type": "string"},
            "model_used": {"type": "string"},
        },
    },
)


# ---------------------------------------------------------------------------
# Runtime implementation
# ---------------------------------------------------------------------------

# Map file extensions to MIME types
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


async def run(document_service, document_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert document to markdown format.

    Parameters
    ----------
    document_service : DocumentService
        Service to fetch document content.
    document_id : str
        Target document.
    params : dict
        Action parameters containing:
        - api_key_env: Environment variable name for Gemini API key
        - model: Gemini model to use
        - temperature: Generation temperature
        - custom_prompt: Optional custom prompt
        - auth: AuthContext (provided by WorkflowService)

    Returns
    -------
    dict
        JSON serializable dict with markdown content and metadata.
    """
    api_key_env = params.get("api_key_env", "GEMINI_API_KEY")
    model = params.get("model", "gemini-2.5-pro")
    temperature = params.get("temperature", 0)
    custom_prompt = params.get("custom_prompt", "")
    auth_ctx = params.get("auth")

    # Get API key from environment
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} not set")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Fetch document
    doc = await document_service.db.get_document(document_id, auth_ctx)
    if not doc:
        raise ValueError(f"Document {document_id} not found or access denied")

    # Get document content from storage
    if not doc.storage_info:
        raise ValueError(f"Document {document_id} has no storage information")

    bucket = doc.storage_info["bucket"]
    key = doc.storage_info["key"]

    # Download file content
    file_bytes = await document_service.storage.download_file(bucket, key)
    if hasattr(file_bytes, "read"):
        file_bytes = file_bytes.read()

    # Determine MIME type from filename
    filename = doc.filename or "document"
    file_ext = Path(filename).suffix.lower()
    mime_type = MIME_TYPES.get(file_ext, doc.content_type or "application/octet-stream")

    # Prepare the prompt
    base_prompt = "Convert this file to markdown format. Preserve the structure and formatting as much as possible."
    if custom_prompt:
        full_prompt = f"{base_prompt}\n\nAdditional instructions: {custom_prompt}"
    else:
        full_prompt = base_prompt

    # Create request for Gemini
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type=mime_type,
                    data=file_bytes,
                ),
                types.Part.from_text(text=full_prompt),
            ],
        ),
    ]

    # Generate response - run in thread pool to avoid blocking
    try:
        # Run the blocking Gemini call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                ),
            ),
        )

        markdown_content = response.text

        logger.info(
            "Converted document %s to markdown using model %s",
            document_id,
            model,
        )

        return {
            "markdown": markdown_content,
            "original_filename": filename,
            "mime_type": mime_type,
            "model_used": model,
        }

    except Exception as e:
        logger.error(f"Failed to convert document to markdown: {e}")
        raise RuntimeError(f"Failed to convert document to markdown: {str(e)}")
