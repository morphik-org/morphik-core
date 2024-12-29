from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from core.models.documents import DocumentChunk


class BaseVectorStore(ABC):
    @abstractmethod
    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks and their embeddings"""
        pass

    @abstractmethod
    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks"""
        pass
