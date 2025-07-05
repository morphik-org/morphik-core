from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from core.database.base_database import BaseDatabase
from core.models.auth import AuthContext
from core.models.workflows import Workflow, WorkflowRun, WorkflowRunStatus
from core.workflows.registry import ACTION_REGISTRY

"""WorkflowService – foundational CRUD & run orchestrator for document workflows.

This is *step-2* of the Workflows roadmap.  The service purposely keeps minimal
logic – complex execution happens later in ARQ jobs (not implemented yet).
"""

logger = logging.getLogger(__name__)


class WorkflowService:
    """High-level façade to manage `Workflow` objects."""

    # ---------------------------------------------------------------------
    # Construction & helpers
    # ---------------------------------------------------------------------

    def __init__(self, database: BaseDatabase, document_service_ref=None):
        self.db = database
        # Weak ref to document_service to reuse models/storage, injected from services_init
        self._document_service_ref = document_service_ref

    # ------------------------------------------------------------------
    # Public CRUD operations
    # ------------------------------------------------------------------

    async def create_workflow(self, workflow: Workflow, auth: AuthContext) -> Workflow:
        """Persist *workflow* and return saved copy."""
        await self._enforce_write(auth)

        # Persist to database
        success = await self.db.store_workflow(workflow, auth)
        if not success:
            raise RuntimeError("Failed to store workflow in database")

        logger.info("Workflow %s created by %s", workflow.id, auth.entity_id)
        return workflow

    async def list_workflows(self, auth: AuthContext) -> List[Workflow]:
        """Return workflows visible to current *auth* scope."""
        # TODO – refine visibility rules (owner/app/user scoping)
        return await self.db.list_workflows(auth)

    async def get_workflow(self, workflow_id: str, auth: AuthContext) -> Optional[Workflow]:
        return await self.db.get_workflow(workflow_id, auth)

    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any], auth: AuthContext) -> Workflow:
        await self._enforce_write(auth)

        wf = await self.db.update_workflow(workflow_id, updates, auth)
        if wf is None:
            raise RuntimeError("Failed to update workflow in database")
        return wf

    async def delete_workflow(self, workflow_id: str, auth: AuthContext) -> bool:
        await self._enforce_write(auth)
        return await self.db.delete_workflow(workflow_id, auth)

    # ------------------------------------------------------------------
    # Background-friendly execution helpers
    # ------------------------------------------------------------------

    async def queue_workflow_run(self, workflow_id: str, document_id: str, auth: AuthContext) -> WorkflowRun:
        """Create a *queued* WorkflowRun and mark the document as *processing*.

        The heavy execution is expected to be scheduled by the caller (using
        FastAPI BackgroundTasks or *asyncio.create_task*).  This method is
        intentionally lightweight so it can be awaited directly inside an
        API handler before the HTTP response is sent back.
        """

        # Verify workflow exists & is accessible
        wf = await self.get_workflow(workflow_id, auth)
        if not wf:
            raise ValueError(f"Workflow {workflow_id} not found or not accessible")

        # Check if workflow has already been run for this document
        existing_runs = await self.list_workflow_runs(workflow_id, auth)
        for run in existing_runs:
            if run.document_id == document_id and run.status in [
                WorkflowRunStatus.completed,
                WorkflowRunStatus.running,
            ]:
                logger.info(f"Workflow {workflow_id} already {run.status} for document {document_id}, skipping")
                return run

        run = WorkflowRun(
            workflow_id=workflow_id,
            document_id=document_id,
            user_id=auth.user_id,
            app_id=auth.app_id,
            status=WorkflowRunStatus.queued,
            started_at=None,
        )

        # Persist the queued run
        await self.db.store_workflow_run(run)

        # Optimistically mark the document so the UI can show "processing"
        try:
            if hasattr(self.db, "update_document"):
                await self.db.update_document(
                    document_id=document_id,
                    updates={
                        "system_metadata": {
                            "workflow_status": "processing",
                            "updated_at": datetime.now(UTC),
                        }
                    },
                    auth=auth,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update document %s workflow_status: %s", document_id, exc)

        logger.info("Queued workflow %s for doc %s (run=%s)", workflow_id, document_id, run.id)
        return run

    # Public helper for background task
    async def execute_workflow_run(self, run_id: str, auth: AuthContext) -> None:
        """Execute a previously queued run (spawn inside BackgroundTasks)."""

        # Retrieve run & workflow
        run = await self.get_run(run_id, auth)
        if not run:
            logger.error("WorkflowRun %s not found – cannot execute", run_id)
            return

        wf = await self.get_workflow(run.workflow_id, auth)
        if not wf:
            logger.error("Workflow %s not found for run %s", run.workflow_id, run_id)
            return

        # Check if document is fully processed before executing workflow
        document = await self._document_service_ref.db.get_document(run.document_id, auth)
        if not document:
            logger.error("Document %s not found for workflow run %s", run.document_id, run_id)
            run.status = WorkflowRunStatus.failed
            run.error = "Document not found"
            run.completed_at = datetime.now(UTC)
            await self._persist_run(run)
            return

        # Check document processing status
        doc_status = document.system_metadata.get("status")
        logger.info(f"Document {run.document_id} has status: {doc_status}")

        # Check if document is ready for workflow execution
        if doc_status == "failed":
            logger.error("Cannot run workflow on failed document %s", run.document_id)
            run.status = WorkflowRunStatus.failed
            run.error = "Document processing previously failed"
            run.completed_at = datetime.now(UTC)
            await self._persist_run(run)
            await self._safe_update_document_status(run.document_id, "failed", auth)
            return
        elif doc_status == "processing":
            # This should not happen with our new architecture where workflows execute after processing
            logger.error(
                "Document %s is still processing - workflows should only execute after completion", run.document_id
            )
            run.status = WorkflowRunStatus.failed
            run.error = "Document is still processing"
            run.completed_at = datetime.now(UTC)
            await self._persist_run(run)
            return

        # Document is ready (status is "completed" or None for text documents)
        logger.info(f"Document {run.document_id} is ready for workflow execution")

        # Mark as running
        run.status = WorkflowRunStatus.running
        run.started_at = datetime.now(UTC)
        await self._persist_run(run)

        outputs = []
        try:
            # Add timeout for overall workflow execution (5 minutes by default)
            import asyncio

            workflow_timeout = 300  # 5 minutes

            async def run_with_timeout():
                # Create a shared context for this workflow execution
                workflow_context = {
                    "document_content": None,  # Cache document content
                    "document_chunks": None,  # Cache document chunks
                    "metadata": {},  # Shared metadata
                    "workflow_name": wf.name,  # Include workflow name
                    "workflow_id": wf.id,  # Include workflow ID
                }

                for step in wf.steps:
                    runner = ACTION_REGISTRY.get_runner(step.action_id)
                    if not runner:
                        raise ValueError(f"Action {step.action_id} not found in registry")

                    # Pass previous outputs and context to actions
                    action_params = step.parameters | {
                        "auth": auth,
                        "_previous_outputs": outputs.copy(),  # Pass copy of outputs so far
                        "_workflow_context": workflow_context,  # Pass shared context
                    }
                    result = await runner(
                        document_service=self._document_service_ref,
                        document_id=run.document_id,
                        params=action_params,  # type: ignore[arg-type]
                    )
                    outputs.append(result)

                    # Store intermediate results
                    run.results_per_step.append(result)

            await asyncio.wait_for(run_with_timeout(), timeout=workflow_timeout)

            run.status = WorkflowRunStatus.completed
            run.completed_at = datetime.now(UTC)
            run.final_output = outputs[-1] if outputs else {}

            logger.info("Workflow run %s finished", run.id)

            # Mark document as completed
            await self._safe_update_document_status(run.document_id, "completed", auth)

        except asyncio.TimeoutError:
            logger.error("Workflow run %s timed out after %s seconds", run.id, workflow_timeout)
            run.status = WorkflowRunStatus.failed
            run.error = f"Workflow execution timed out after {workflow_timeout} seconds"
            run.completed_at = datetime.now(UTC)
            await self._safe_update_document_status(run.document_id, "failed", auth)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Workflow run %s failed: %s", run.id, exc)
            run.status = WorkflowRunStatus.failed
            run.error = str(exc)
            run.completed_at = datetime.now(UTC)
            await self._safe_update_document_status(run.document_id, "failed", auth)

        await self._persist_run(run)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _persist_run(self, run: WorkflowRun):
        """Persist *run* to the database."""
        await self.db.store_workflow_run(run)

    async def _safe_update_document_status(self, document_id: str, status: str, auth: AuthContext):
        """Best-effort update of document.workflow_status."""
        try:
            if hasattr(self.db, "update_document"):
                await self.db.update_document(
                    document_id=document_id,
                    updates={
                        "system_metadata": {
                            "workflow_status": status,
                            "updated_at": datetime.now(UTC),
                        }
                    },
                    auth=auth,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update doc %s workflow_status to %s: %s", document_id, status, exc)

    async def get_run(self, run_id: str, auth: AuthContext) -> Optional[WorkflowRun]:
        """Get a specific workflow run by ID."""
        return await self.db.get_workflow_run(run_id, auth)

    async def list_workflow_runs(self, workflow_id: str, auth: AuthContext) -> List[WorkflowRun]:
        """List all runs for a specific workflow."""
        return await self.db.list_workflow_runs(workflow_id, auth)

    async def delete_run(self, run_id: str, auth: AuthContext) -> bool:
        """Delete a workflow run."""
        await self._enforce_write(auth)
        return await self.db.delete_workflow_run(run_id, auth)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _enforce_write(auth: AuthContext):
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission for workflows")
