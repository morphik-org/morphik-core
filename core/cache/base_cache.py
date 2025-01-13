from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
from core.models.completion import CompletionRequest, CompletionResponse


class BaseCache(ABC):
    """Base class for cache implementations.

    This class defines the interface for cache implementations that support
    document ingestion, updates, and cache-augmented text generation.
    """

    @abstractmethod
    async def ingest(self, docs: List[str]) -> bool:
        """Ingest documents into the cache.

        Args:
            docs: List of documents to ingest

        Returns:
            bool: True if ingestion was successful
        """
        pass

    @abstractmethod
    async def update(self, new_doc: str) -> bool:
        """Update the cache with a new document.

        Args:
            new_doc: Document to add to the cache

        Returns:
            bool: True if update was successful
        """
        pass

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using the cached context.

        Args:
            request: Completion request containing the prompt

        Returns:
            CompletionResponse: Generated completion
        """
        pass

    @abstractmethod
    def save_cache(self) -> Path:
        """Save the cache state to disk.

        Returns:
            Path: Path where the cache was saved
        """
        pass

    @abstractmethod
    def load_cache(self, cache_path: Union[str, Path]) -> None:
        """Load a previously saved cache state.

        Args:
            cache_path: Path to the saved cache file
        """
        pass
