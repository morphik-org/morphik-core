from typing import Dict, Any, List
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunk stored in VectorStore"""

    document_id: str  # external_id of parent document
    content: str
    embedding: List[float]
    chunk_number: int
    # chunk-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class Chunk(BaseModel):
    """Represents a chunk containing content and metadata"""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_document_chunk(
        self, document_id: str, chunk_number: int, embedding: List[float]
    ) -> DocumentChunk:
        return DocumentChunk(
            document_id=document_id,
            content=self.content,
            embedding=embedding,
            chunk_number=chunk_number,
            metadata=self.metadata,
        )
