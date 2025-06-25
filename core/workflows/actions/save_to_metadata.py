from __future__ import annotations

import logging
from typing import Any, Dict

from core.models.workflows import ActionDefinition

"""Save workflow output to document metadata.

This action takes the output from the previous workflow step and saves it
to the document's user_metadata under a specified key.
"""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static action metadata (exported for registry)
# ---------------------------------------------------------------------------

action_id = "morphik.actions.save_to_metadata"

definition = ActionDefinition(
    id=action_id,
    name="Save to Metadata",
    description="Save the output from previous step to document metadata",
    parameters_schema={
        "type": "object",
        "properties": {
            "metadata_key": {
                "type": "string",
                "description": "Key to store the data under in document metadata (optional - if not provided, fields are merged at top level)",
            },
            "source": {
                "type": "string",
                "enum": ["previous_step", "all_steps"],
                "description": "Whether to save output from previous step or all steps",
                "default": "previous_step",
            },
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "metadata_key": {"type": "string"},
            "data_saved": {"type": "object"},
        },
    },
)


# ---------------------------------------------------------------------------
# Runtime implementation
# ---------------------------------------------------------------------------


async def run(document_service, document_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Save workflow output to document metadata.

    Parameters
    ----------
    document_service : DocumentService
        Service to fetch and update document.
    document_id : str
        Target document.
    params : dict
        Action parameters containing:
        - metadata_key: Key to store data under
        - source: Where to get data from (previous_step or all_steps)
        - auth: AuthContext (provided by WorkflowService)
        - _previous_outputs: List of outputs from previous steps (injected by WorkflowService)

    Returns
    -------
    dict
        JSON serializable dict with success status and saved data info.
    """

    metadata_key = params.get("metadata_key", "")
    source = params.get("source", "previous_step")
    auth_ctx = params.get("auth")
    previous_outputs = params.get("_previous_outputs", [])

    # Fetch document
    doc = await document_service.db.get_document(document_id, auth_ctx)
    if not doc:
        raise ValueError(f"Document {document_id} not found or access denied")

    # Determine what data to save
    if source == "previous_step":
        if not previous_outputs:
            raise ValueError("No previous step output to save")
        data_to_save = previous_outputs[-1]
    else:  # all_steps
        data_to_save = {f"step_{i+1}": output for i, output in enumerate(previous_outputs)}

    # Update document metadata
    current_metadata = doc.metadata.copy() if doc.metadata else {}

    if metadata_key:
        # Store under specified key
        current_metadata[metadata_key] = data_to_save
    else:
        # No key specified - merge at top level if data is a dict
        if isinstance(data_to_save, dict):
            current_metadata.update(data_to_save)
        else:
            # If not a dict, we need to store it under a default key
            current_metadata["data"] = data_to_save

    # Save to database
    updates = {"metadata": current_metadata}
    success = await document_service.db.update_document(document_id, updates, auth_ctx)

    if not success:
        raise RuntimeError(f"Failed to update document {document_id} metadata")

    if metadata_key:
        logger.info(
            "Saved workflow output to document %s metadata under key '%s'",
            document_id,
            metadata_key,
        )
    else:
        logger.info(
            "Saved workflow output to document %s metadata at top level",
            document_id,
        )

    return {
        "success": True,
        "metadata_key": metadata_key or "__top_level__",
        "data_saved": data_to_save,
    }
