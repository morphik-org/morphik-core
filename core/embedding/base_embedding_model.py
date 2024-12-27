from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_for_ingestion(self, text: Union[str, List[str]]) -> List[float]:
        """Generate embeddings for input text"""
        pass

    @abstractmethod
    async def embed_for_query(self, text: str) -> List[float]:
        """Generate embeddings for input text"""
        pass
