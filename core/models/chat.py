from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Simple chat message model for chat persistence."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
