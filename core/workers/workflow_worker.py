"""
Background worker for executing workflows.

This worker processes workflow execution jobs queued via arq.
"""

import logging
from typing import Any, Dict

from core.models.auth import AuthContext, EntityType
from core.services_init import workflow_service

logger = logging.getLogger(__name__)


async def execute_workflow_run(ctx: Dict[str, Any], run_id: str, auth_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a workflow run in the background.

    Args:
        ctx: The ARQ context dictionary
        run_id: The workflow run ID to execute
        auth_dict: Dictionary representation of AuthContext

    Returns:
        A dictionary with execution results
    """
    try:
        logger.info(f"Starting workflow execution for run {run_id}")

        # Reconstruct auth context
        auth = AuthContext(
            entity_type=EntityType(auth_dict.get("entity_type", "unknown")),
            entity_id=auth_dict.get("entity_id", ""),
            app_id=auth_dict.get("app_id"),
            permissions=set(auth_dict.get("permissions", ["read"])),
            user_id=auth_dict.get("user_id", auth_dict.get("entity_id", "")),
        )

        # Execute the workflow
        result = await workflow_service.execute_workflow_run(run_id, auth)

        logger.info(f"Completed workflow execution for run {run_id}")
        return {"run_id": run_id, "status": "completed", "result": result}

    except Exception as e:
        logger.error(f"Failed to execute workflow run {run_id}: {e}")
        return {"run_id": run_id, "status": "failed", "error": str(e)}
