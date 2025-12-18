from typing import Optional

from pydantic import BaseModel, Field


class SummaryUpsertRequest(BaseModel):
    """Request payload for writing or updating an entity summary."""

    content: str = Field(..., description="Summary content to persist (markdown/text)")
    versioning: bool = Field(
        default=True,
        description="When true, automatically increments the summary version instead of overwriting the latest version",
    )
    overwrite_latest: bool = Field(
        default=False,
        description="Allow overwriting the latest summary when versioning is disabled",
    )


class SummaryResponse(BaseModel):
    """Response payload returned when reading or writing summaries."""

    content: str
    storage_key: str
    bucket: Optional[str] = None
    version: int
    updated_at: Optional[str] = None
