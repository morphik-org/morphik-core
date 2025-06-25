from __future__ import annotations

import logging
from typing import Any, Dict

import litellm

from core.models.workflows import ActionDefinition

"""Apply custom LLM instructions to transform documents.

This action allows users to apply arbitrary instructions to document content
using an LLM, with support for prompt templates and variable substitution.
"""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static action metadata (exported for registry)
# ---------------------------------------------------------------------------

action_id = "morphik.actions.apply_instruction"

definition = ActionDefinition(
    id=action_id,
    name="Apply Custom Instruction",
    description="Apply a custom AI instruction to transform the document",
    parameters_schema={
        "type": "object",
        "properties": {
            "prompt_template": {
                "type": "string",
                "description": "Instruction template with {input_text} placeholder for document content",
            },
            "model": {
                "type": "string",
                "description": "Model to use for instruction (defaults to gpt-4o-mini)",
                "default": "gpt-4o-mini",
            },
            "temperature": {
                "type": "number",
                "description": "Temperature for generation (0-2)",
                "default": 0.7,
                "minimum": 0,
                "maximum": 2,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens to generate",
                "default": 4096,
            },
        },
        "required": ["prompt_template"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "result": {"type": "string", "description": "The transformed/generated text"},
            "model_used": {"type": "string"},
            "tokens_used": {"type": "integer"},
        },
    },
)


# ---------------------------------------------------------------------------
# Runtime implementation
# ---------------------------------------------------------------------------


async def run(document_service, document_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply custom instruction to document content.

    Parameters
    ----------
    document_service : DocumentService
        Service to fetch document content.
    document_id : str
        Target document.
    params : dict
        Action parameters containing:
        - prompt_template: Template with {input_text} placeholder
        - model: LLM model to use
        - temperature: Generation temperature
        - max_tokens: Max tokens to generate
        - auth: AuthContext (provided by WorkflowService)

    Returns
    -------
    dict
        JSON serializable dict with transformed text and metadata.
    """

    prompt_template = params["prompt_template"]
    model = params.get("model", "gpt-4o-mini")
    temperature = params.get("temperature", 0.7)
    max_tokens = params.get("max_tokens", 4096)
    auth_ctx = params.get("auth")

    # Fetch document
    doc = await document_service.db.get_document(document_id, auth_ctx)
    if not doc:
        raise ValueError(f"Document {document_id} not found or access denied")

    # Get document content
    # For text documents, use the content from system_metadata
    # For other types, try to get chunks and concatenate
    if doc.content_type == "text/plain" and doc.system_metadata.get("content"):
        input_text = doc.system_metadata["content"]
    else:
        # Try to get document chunks
        chunks = await document_service.db.get_chunks_for_document(document_id, auth_ctx)
        if chunks:
            input_text = "\n\n".join(chunk.content for chunk in chunks)
        else:
            # Fallback: try to download and parse the document
            if doc.storage_info:
                bucket = doc.storage_info["bucket"]
                key = doc.storage_info["key"]
                file_bytes = await document_service.storage.download_file(bucket, key)
                if hasattr(file_bytes, "read"):
                    file_bytes = file_bytes.read()

                # For now, convert to string if possible
                try:
                    input_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"Cannot decode document {document_id} content as text")
            else:
                raise ValueError(f"No content available for document {document_id}")

    # Apply the prompt template
    if "{input_text}" not in prompt_template:
        logger.warning("Prompt template does not contain {input_text} placeholder, appending document content")
        prompt = f"{prompt_template}\n\nDocument content:\n{input_text}"
    else:
        prompt = prompt_template.replace("{input_text}", input_text)

    # Get model configuration
    from core.config import get_settings

    settings = get_settings()

    # Check if requested model is available
    model_name = model
    if hasattr(settings, "REGISTERED_MODELS"):
        # Try to find the model in registered models
        if model in settings.REGISTERED_MODELS:
            model_config = settings.REGISTERED_MODELS[model]
            model_name = model_config.get("model_name", model)
        else:
            # Try to find a similar model
            for key, config in settings.REGISTERED_MODELS.items():
                if model.lower() in key.lower() or key.lower() in model.lower():
                    model_name = config.get("model_name", key)
                    logger.info(f"Using registered model {key} for requested model {model}")
                    break

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that follows instructions precisely.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Call LLM
    try:
        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.info(
            "Applied instruction to document %s using model %s (%d tokens)",
            document_id,
            model_name,
            tokens_used,
        )

        return {
            "result": result_text,
            "model_used": model_name,
            "tokens_used": tokens_used,
        }

    except Exception as e:
        logger.error(f"Failed to apply instruction: {e}")
        raise RuntimeError(f"Failed to apply instruction: {str(e)}")
