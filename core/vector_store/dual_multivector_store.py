"""
Dual MultiVector Store for migration scenarios.

This wrapper allows simultaneous ingestion to both FastMultiVectorStore and MultiVectorStore
while maintaining search operations on only the slow store during migration.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore
from .fast_multivector_store import FastMultiVectorStore
from .multi_vector_store import MultiVectorStore

logger = logging.getLogger(__name__)


class DualMultiVectorStore(BaseVectorStore):
    """
    A wrapper that manages both FastMultiVectorStore and MultiVectorStore for migration scenarios.

    During migration:
    - store_embeddings: Writes to both stores simultaneously
    - query_similar: Reads only from the slow store (MultiVectorStore)
    - get_chunks_by_id: Reads only from the slow store (MultiVectorStore)
    - delete_chunks_by_document_id: Deletes from both stores
    """

    def __init__(
        self,
        fast_store: FastMultiVectorStore,
        slow_store: MultiVectorStore,
        enable_dual_ingestion: bool = True,
    ):
        """
        Initialize the dual vector store.

        Args:
            fast_store: The FastMultiVectorStore instance (Turbopuffer)
            slow_store: The MultiVectorStore instance (PostgreSQL)
            enable_dual_ingestion: Whether to write to both stores (True) or just slow store (False)
        """
        self.fast_store = fast_store
        self.slow_store = slow_store
        self.enable_dual_ingestion = enable_dual_ingestion

    def initialize(self):
        """Initialize both stores."""
        fast_result = self.fast_store.initialize()
        slow_result = self.slow_store.initialize()
        return fast_result and slow_result

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Store embeddings in both stores simultaneously during migration.

        Args:
            chunks: List of DocumentChunk objects to store

        Returns:
            Tuple of (success, stored_ids) - uses slow store's result as primary
        """
        if not self.enable_dual_ingestion:
            # If dual ingestion is disabled, only use slow store
            return await self.slow_store.store_embeddings(chunks, app_id)

        logger.info(f"Dual ingestion: storing {len(chunks)} chunks in both fast and slow stores")

        # Store in both stores concurrently
        try:
            fast_task = asyncio.create_task(self.fast_store.store_embeddings(chunks, app_id))
            slow_task = asyncio.create_task(self.slow_store.store_embeddings(chunks, app_id))

            # Wait for both to complete
            fast_result, slow_result = await asyncio.gather(fast_task, slow_task, return_exceptions=True)

            # Handle results
            fast_success = False
            fast_ids: list[str] = []
            if isinstance(fast_result, BaseException):
                logger.error(f"Fast store ingestion failed: {fast_result}")
            else:
                fast_success, fast_ids = fast_result

            slow_success = False
            slow_ids: list[str] = []
            if isinstance(slow_result, BaseException):
                logger.error(f"Slow store ingestion failed: {slow_result}")
                # If slow store fails, this is critical since we search from it
                raise slow_result
            else:
                slow_success, slow_ids = slow_result

            # Log results
            if fast_success:
                logger.info(f"Fast store: successfully stored {len(fast_ids)} chunks")
            else:
                logger.warning("Fast store: ingestion failed")

            if slow_success:
                logger.info(f"Slow store: successfully stored {len(slow_ids)} chunks")
            else:
                logger.error("Slow store: ingestion failed")

            # Return slow store result as primary (since we search from it)
            return slow_success, slow_ids

        except Exception as e:
            logger.error(f"Error during dual ingestion: {e}")
            # If dual ingestion fails, fall back to slow store only
            logger.warning("Falling back to slow store only due to dual ingestion error")
            return await self.slow_store.store_embeddings(chunks, app_id)

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
        app_id: Optional[str] = None,
        skip_image_content: bool = False,
    ) -> List[DocumentChunk]:
        """
        Query similar chunks from the slow store only during migration.

        This ensures consistent search results during migration period.
        """
        logger.debug("Querying from slow store only during migration")
        return await self.slow_store.query_similar(query_embedding, k, doc_ids, app_id, skip_image_content)

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
        app_id: Optional[str] = None,
        skip_image_content: bool = False,
    ) -> List[DocumentChunk]:
        """
        Get chunks by ID from the slow store only during migration.
        """
        logger.debug("Getting chunks from slow store only during migration")
        return await self.slow_store.get_chunks_by_id(chunk_identifiers, app_id, skip_image_content)

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        """
        Delete chunks from both stores to maintain consistency.

        Args:
            document_id: ID of the document whose chunks should be deleted
            app_id: Optional app ID for filtering chunks

        Returns:
            bool: True if deletion succeeded in at least the slow store
        """
        logger.info(f"Dual deletion: removing chunks for document {document_id} from both stores")

        try:
            # Delete from both stores concurrently
            fast_task = asyncio.create_task(self.fast_store.delete_chunks_by_document_id(document_id, app_id))
            slow_task = asyncio.create_task(self.slow_store.delete_chunks_by_document_id(document_id, app_id))

            fast_result, slow_result = await asyncio.gather(fast_task, slow_task, return_exceptions=True)

            # Handle results
            fast_success = False
            if isinstance(fast_result, Exception):
                logger.error(f"Fast store deletion failed for document {document_id}: {fast_result}")
            else:
                fast_success = fast_result

            slow_success = False
            if isinstance(slow_result, Exception):
                logger.error(f"Slow store deletion failed for document {document_id}: {slow_result}")
            else:
                slow_success = slow_result

            # Log results
            if fast_success:
                logger.info(f"Fast store: successfully deleted chunks for document {document_id}")
            else:
                logger.warning(f"Fast store: deletion failed for document {document_id}")

            if slow_success:
                logger.info(f"Slow store: successfully deleted chunks for document {document_id}")
            else:
                logger.error(f"Slow store: deletion failed for document {document_id}")

            # Return success if at least slow store succeeded (since we search from it)
            return slow_success

        except Exception as e:
            logger.error(f"Error during dual deletion for document {document_id}: {e}")
            # Fall back to slow store only
            return await self.slow_store.delete_chunks_by_document_id(document_id, app_id)

    def close(self):
        """Close both stores."""
        try:
            self.fast_store.close() if hasattr(self.fast_store, "close") else None
        except Exception as e:
            logger.error(f"Error closing fast store: {e}")

        try:
            self.slow_store.close()
        except Exception as e:
            logger.error(f"Error closing slow store: {e}")

    # Expose properties for compatibility
    @property
    def uri(self):
        """Return slow store URI for compatibility."""
        return self.slow_store.uri

    @property
    def storage(self):
        """Return slow store storage for compatibility."""
        return self.slow_store.storage

    def latest_store_metrics(self) -> dict:
        metrics: dict = {}
        fast_getter = getattr(self.fast_store, "latest_store_metrics", None)
        slow_getter = getattr(self.slow_store, "latest_store_metrics", None)
        fast_metrics = fast_getter() if callable(fast_getter) else {}
        slow_metrics = slow_getter() if callable(slow_getter) else {}
        if fast_metrics:
            metrics["fast"] = fast_metrics
        if slow_metrics:
            metrics["slow"] = slow_metrics
        if metrics:
            metrics["mode"] = "dual"
        return metrics
