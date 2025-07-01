from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Workflow & Action domain models (Pydantic)
# ---------------------------------------------------------------------------


class WorkflowRunStatus(str, Enum):
    """Allowed status values for *WorkflowRun* objects."""

    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ConfiguredAction(BaseModel):
    """A single step inside a *Workflow* configuration.

    The concrete implementation is looked-up at runtime via the *action_id* in
    the Action Registry (see *core.workflows.registry*).
    """

    action_id: str = Field(..., description="Registry identifier of the action to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON-serialisable parameters for the action")


class Workflow(BaseModel):
    """High-level definition of a multi-step document processing workflow."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

    # Ownership / scoping fields (similar to documents & folders)
    owner_id: str = Field(..., description="Organization / developer ID owning this workflow")
    user_id: Optional[str] = Field(None, description="End-user ID when created in a narrowed scope")
    app_id: Optional[str] = Field(None, description="App ID when created inside an application context")

    steps: List[ConfiguredAction] = Field(..., min_items=1, description="Ordered list of actions to execute")

    system_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
    )

    @field_validator("steps")
    @classmethod
    def _validate_steps_not_empty(cls, v):  # noqa: D401, N805
        if not v:
            raise ValueError("Workflow must contain at least one action step")
        return v


class WorkflowRun(BaseModel):
    """Represents a concrete execution of a *Workflow* against a document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    document_id: str

    user_id: Optional[str] = None
    app_id: Optional[str] = None

    status: WorkflowRunStatus = WorkflowRunStatus.queued

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Each step can write arbitrary JSON output (kept small)
    results_per_step: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: Optional[Dict[str, Any]] = None

    error: Optional[str] = None

    system_metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionDefinition(BaseModel):
    """Metadata describing an Action implementation loaded into the registry."""

    id: str = Field(..., description="Registry identifier, e.g. 'morphik.actions.extract_structured'")
    name: str
    description: Optional[str] = None
    # JSON Schema fragments describing expected parameters / output for UI & validation.
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
