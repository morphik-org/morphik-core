from __future__ import annotations

import logging
from typing import Any, Dict

from core.models.workflows import ActionDefinition

"""Ingest workflow output as a new document.

This action takes output from previous steps (typically markdown) and ingests it
as a new document in the system.
"""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static action metadata (exported for registry)
# ---------------------------------------------------------------------------

action_id = "morphik.actions.ingest_output"

definition = ActionDefinition(
    id=action_id,
    name="Ingest Output",
    description="Ingest workflow output as a new document",
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename for the ingested document (e.g., 'output.md')",
                "default": "workflow_output.md",
            },
            "source": {
                "type": "string",
                "description": "Source of content to ingest",
                "enum": ["previous_step", "all_steps"],
                "default": "previous_step",
            },
            "content_field": {
                "type": "string",
                "description": "Field name containing the content to ingest (e.g., 'markdown', 'result')",
                "default": "markdown",
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata to attach to the document",
                "default": {},
            },
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "ID of the ingested document"},
            "filename": {"type": "string"},
            "status": {"type": "string"},
        },
    },
)


# ---------------------------------------------------------------------------
# Runtime implementation
# ---------------------------------------------------------------------------


async def run(document_service, document_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest workflow output as a new document.

    Parameters
    ----------
    document_service : DocumentService
        Service to ingest documents.
    document_id : str
        Source document ID (for context).
    params : dict
        Action parameters containing:
        - filename: Name for the ingested document
        - source: Whether to use previous_step or all_steps output
        - content_field: Field name containing content to ingest
        - metadata: Additional metadata
        - auth: AuthContext (provided by WorkflowService)
        - _previous_outputs: Results from previous steps (provided by WorkflowService)
        - _workflow_context: Workflow context (provided by WorkflowService)

    Returns
    -------
    dict
        JSON serializable dict with ingested document info.
    """
    filename = params.get("filename", "workflow_output.md")
    source = params.get("source", "previous_step")
    content_field = params.get("content_field", "markdown")
    metadata = params.get("metadata", {})
    auth_ctx = params.get("auth")
    workflow_context = params.get("_workflow_context", {})

    # Get previous outputs from workflow
    previous_outputs = params.get("_previous_outputs", [])

    if not previous_outputs:
        raise ValueError("No previous outputs available to ingest")

    # Extract content based on source
    if source == "previous_step":
        # Get the result from the previous step
        prev_result = previous_outputs[-1]

        # Extract content from the specified field
        if isinstance(prev_result, dict) and content_field in prev_result:
            content = prev_result[content_field]
        elif isinstance(prev_result, str):
            content = prev_result
        else:
            raise ValueError(f"Could not find field '{content_field}' in previous step output")

    else:  # all_steps
        # Combine all results
        content_parts = []
        for i, result in enumerate(previous_outputs):
            if isinstance(result, dict) and content_field in result:
                content_parts.append(f"## Step {i + 1} Output\n\n{result[content_field]}")
            elif isinstance(result, str):
                content_parts.append(f"## Step {i + 1} Output\n\n{result}")

        if not content_parts:
            raise ValueError(f"No '{content_field}' field found in any previous step outputs")

        content = "\n\n---\n\n".join(content_parts)

    # Convert content to string if needed
    if not isinstance(content, str):
        content = str(content)

    # Get source document info for better filename
    source_doc = await document_service.db.get_document(document_id, auth_ctx)
    if source_doc and source_doc.filename:
        # Extract base filename without extension
        from pathlib import Path

        base_name = Path(source_doc.filename).stem
        # Get workflow name from context if available
        workflow_name = workflow_context.get("workflow_name", "workflow")
        # Create descriptive filename
        generated_filename = f"{base_name}_{workflow_name}.md"
    else:
        generated_filename = filename

    # Prepare metadata
    doc_metadata = {
        "workflow_generated": True,
        "source_document_id": document_id,
        "workflow_name": workflow_context.get("workflow_name", "unknown"),
        **metadata,
    }

    # Ingest the document as text
    try:
        doc = await document_service.ingest_text(
            content=content, filename=generated_filename, metadata=doc_metadata, auth=auth_ctx
        )

        logger.info("Ingested workflow output as document %s with filename %s", doc.external_id, generated_filename)

        return {
            "document_id": doc.external_id,
            "filename": generated_filename,
            "status": doc.system_metadata.get("status", "processing"),
        }

    except Exception as e:
        logger.error(f"Failed to ingest output: {e}")
        raise RuntimeError(f"Failed to ingest output: {str(e)}")
