from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class V2IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: Optional[str] = None


class V2RetrieveFilters(BaseModel):
    document_ids: Optional[List[str]] = Field(default=None, description="Limit to specific document IDs")
    folder_paths: Optional[List[str]] = Field(default=None, description="Limit to specific folder paths")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Chunk-level metadata filters")


class V2RetrieveRequest(BaseModel):
    query: str
    filters: Optional[V2RetrieveFilters] = None
    top_k: int = Field(default=5, ge=1, le=100)
    end_user_id: Optional[str] = Field(default=None, description="Optional end-user scope")

    @model_validator(mode="after")
    def validate_query(self):
        if not self.query or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        return self


class V2ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    page_number: Optional[int] = None
    chunk_number: Optional[int] = None
    score: float
    content: str


class V2RetrieveResponse(BaseModel):
    query: str
    chunks: List[V2ChunkResult]
