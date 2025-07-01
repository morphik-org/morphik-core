from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.models.workflows import Workflow, WorkflowRun
from core.services_init import workflow_service

"""Workflow API routes (Step-2 foundation).

Minimal CRUD + run endpoints that delegate to WorkflowService.  This surfaces
an early public API while database migrations & action-runner are still WIP.
"""

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("", response_model=Workflow)
async def create_workflow(workflow: Workflow, auth: AuthContext = Depends(verify_token)) -> Workflow:  # noqa: D401
    """Create a new workflow."""
    try:
        return await workflow_service.create_workflow(workflow, auth)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.get("", response_model=List[Workflow])
async def list_workflows(auth: AuthContext = Depends(verify_token)) -> List[Workflow]:
    """List workflows visible to the caller."""
    return await workflow_service.list_workflows(auth)


@router.get("/{workflow_id}", response_model=Workflow)
async def get_workflow(workflow_id: str, auth: AuthContext = Depends(verify_token)) -> Workflow:
    wf = await workflow_service.get_workflow(workflow_id, auth)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf


@router.put("/{workflow_id}", response_model=Workflow)
async def update_workflow(
    workflow_id: str,
    updates: Dict[str, Any],
    auth: AuthContext = Depends(verify_token),
) -> Workflow:
    try:
        return await workflow_service.update_workflow(workflow_id, updates, auth)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, str]:
    try:
        success = await workflow_service.delete_workflow(workflow_id, auth)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"success": "Workflow deleted successfully"}


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


@router.post("/{workflow_id}/run/{document_id}", response_model=WorkflowRun)
async def run_workflow(
    workflow_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(verify_token),
) -> WorkflowRun:
    """Queue a workflow run and return immediately.

    The heavy execution is scheduled in a *BackgroundTask* so the request
    finishes quickly, enabling the UI to update the document row to
    "processing" without waiting for LLM calls.
    """
    try:
        run = await workflow_service.queue_workflow_run(workflow_id, document_id, auth)
        background_tasks.add_task(workflow_service.execute_workflow_run, run.id, auth)
        return run
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/runs/{run_id}", response_model=WorkflowRun)
async def get_workflow_run(run_id: str, auth: AuthContext = Depends(verify_token)) -> WorkflowRun:
    run = await workflow_service.get_run(run_id, auth)
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return run


@router.get("/{workflow_id}/runs")
async def get_workflow_runs(workflow_id: str, auth: AuthContext = Depends(verify_token)) -> List[WorkflowRun]:
    """Get all runs for a specific workflow."""
    return await workflow_service.list_workflow_runs(workflow_id, auth)


@router.delete("/runs/{run_id}")
async def delete_workflow_run(run_id: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, str]:
    """Delete a workflow run."""
    try:
        success = await workflow_service.delete_run(run_id, auth)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if not success:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return {"success": "Workflow run deleted successfully"}
