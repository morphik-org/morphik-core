import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Folder(BaseModel):
    """Represents a folder that contains documents"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    owner: Dict[str, str]
    document_ids: List[str] | None = Field(default_factory=list)
    system_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
    )
    access_control: Dict[str, List[str]] = Field(default_factory=lambda: {"readers": [], "writers": [], "admins": []})
    rules: List[Dict[str, Any]] = Field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Folder):
            return False
        return self.id == other.id


class FolderCreate(BaseModel):
    """Request model for folder creation"""

    name: str
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Lightweight projection used by the `/folders/summary` endpoint.  This keeps
# payloads tiny by excluding the expansive *document_ids* array while still
# providing enough metadata for the UI.
# ---------------------------------------------------------------------------


class FolderSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    doc_count: int = 0
    updated_at: Optional[str] = None
