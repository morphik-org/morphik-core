from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .documents import QueryReturnType


class IngestTextRequest(BaseModel):
    """Request model for text ingestion"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Query request model with validation"""
    query: str = Field(..., min_length=1)
    return_type: QueryReturnType = QueryReturnType.CHUNKS
    filters: Optional[Dict[str, Any]] = None
    k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.0)
