from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from core.models.chunk import DocumentChunk


class BaseVectorStore(ABC):
    @abstractmethod
    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Store document chunks and their embeddings"""
        pass

    @abstractmethod
    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
        app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks"""
        pass

    @abstractmethod
    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
        app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve specific chunks by document ID and chunk number.

        Args:
            chunk_identifiers: List of (document_id, chunk_number) tuples
            app_id: Optional app ID for filtering chunks

        Returns:
            List of DocumentChunk objects
        """
        pass

    @abstractmethod
    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: ID of the document whose chunks should be deleted
            app_id: Optional app ID for filtering chunks

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        pass
