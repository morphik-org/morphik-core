from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Simple chat message model for chat persistence."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    agent_data: Optional[Dict[str, Any]] = None  # For agent-specific data like display_objects, tool_history, sources


class ChatConversation(BaseModel):
    """Conversation container persisted in the database."""

    conversation_id: str
    user_id: Optional[str] = None
    app_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
