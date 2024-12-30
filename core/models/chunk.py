from typing import Dict, Any, List
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a chunk containing content and metadata"""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
