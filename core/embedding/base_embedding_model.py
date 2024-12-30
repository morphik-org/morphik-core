from abc import ABC, abstractmethod
from typing import List, Union

from core.models.chunk import Chunk


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """Generate embeddings for input text"""
        pass

    @abstractmethod
    async def embed_for_query(self, text: str) -> List[float]:
        """Generate embeddings for input text"""
        pass
