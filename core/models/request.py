from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    """Request model for text ingestion"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    """Base retrieve request model"""
    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.0)


class CompletionQueryRequest(RetrieveRequest):
    """Request model for completion generation"""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
