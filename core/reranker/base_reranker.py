from abc import ABC, abstractmethod
from typing import List, Union

from core.models.chunk import DocumentChunk


class BaseReranker(ABC):
    """Base class for reranking search results"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        """Rerank chunks based on their relevance to the query"""
        pass

    @abstractmethod
    async def compute_score(
        self,
        query: str,
        text: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute relevance scores between query and text"""
        pass
