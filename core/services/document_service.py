import asyncio
import base64
import json
import logging
import os
import tempfile
import time  # Add time import for profiling
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import arq
import filetype
import fitz  # PyMuPDF - faster alternative to pdf2image
import pdf2image

# from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from fastapi import HTTPException, UploadFile
from filetype.types import IMAGE  # , DOCUMENT, document
from PIL import Image as PILImage
from pydantic import BaseModel

from core.cache.base_cache import BaseCache
from core.cache.base_cache_factory import BaseCacheFactory
from core.completion.base_completion import BaseCompletionModel
from core.config import get_settings
from core.database.base_database import BaseDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel

# from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.chat import ChatMessage
from core.models.chunk import Chunk, DocumentChunk
from core.models.completion import ChunkSource, CompletionRequest, CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentContent, DocumentResult, StorageFileInfo
from core.models.prompts import GraphPromptOverrides, QueryPromptOverrides
from core.parser.base_parser import BaseParser
from core.reranker.base_reranker import BaseReranker
from core.services.graph_service import GraphService
from core.services.morphik_graph_service import MorphikGraphService
from core.services.rules_processor import RulesProcessor
from core.storage.base_storage import BaseStorage
from core.storage.utils_file_extensions import detect_file_type
from core.vector_store.base_vector_store import BaseVectorStore

from ..models.auth import AuthContext
from ..models.folders import Folder
from ..models.graph import Graph

logger = logging.getLogger(__name__)
IMAGE = {im.mime for im in IMAGE}

CHARS_PER_TOKEN = 4
TOKENS_PER_PAGE = 630

settings = get_settings()


class PdfConversionError(Exception):
    """Raised when the service cannot rasterize a PDF into images."""


class DocumentService:
    _SYSTEM_METADATA_SCOPE_KEYS = {"folder_name", "end_user_id", "app_id"}

    @classmethod
    def _clean_system_metadata(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Remove scope fields that are persisted in dedicated columns."""
        if not metadata:
            return {}

        cleaned_metadata = dict(metadata)
        for key in cls._SYSTEM_METADATA_SCOPE_KEYS:
            cleaned_metadata.pop(key, None)
        return cleaned_metadata

    async def _ensure_folder_exists(
        self, folder_name: Union[str, List[str]], document_id: str, auth: AuthContext
    ) -> Optional[Folder]:
        """
        Check if a folder exists, if not create it. Also adds the document to the folder.

        Args:
            folder_name: Name of the folder
            document_id: ID of the document to add to the folder
            auth: Authentication context

        Returns:
            Folder object if found or created, None on error
        """
        try:
            # If multiple folders provided, ensure each exists and contains the document
            if isinstance(folder_name, list):
                last_folder = None
                for fname in folder_name:
                    last_folder = await self._ensure_folder_exists(fname, document_id, auth)
                return last_folder

            # Validate folder name - no slashes allowed (nested folders not supported)
            if "/" in folder_name:
                error_msg = (
                    f"Invalid folder name '{folder_name}'. Folder names cannot contain '/'. "
                    f"Nested folders are not supported. Use '_' instead to denote subfolders "
                    f"(e.g., 'folder_subfolder_subsubfolder')."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # First check if the folder already exists
            folder = await self.db.get_folder_by_name(folder_name, auth)
            if folder:
                # Add document to existing folder
                if document_id not in folder.document_ids:
                    success = await self.db.add_document_to_folder(folder.id, document_id, auth)
                    if not success:
                        logger.warning(
                            f"Failed to add document {document_id} to existing folder {folder.name}. This may be due to a race condition during ingestion - the document should be accessible shortly."
                        )
                        # Return the folder anyway since it exists, even if document addition failed
                        # The retry mechanism in add_document_to_folder should handle transient issues
                    else:
                        logger.info(f"Successfully added document {document_id} to existing folder {folder.name}")
                else:
                    logger.info(f"Document {document_id} is already in folder {folder.name}")
                return folder  # Folder already exists

            # Create a new folder
            folder = Folder(
                name=folder_name,
                document_ids=[document_id],
                app_id=auth.app_id,  # Add document_id to the new folder
            )

            await self.db.create_folder(folder, auth)

            return folder

        except Exception as e:
            # Log error but don't raise - we want document ingestion to continue even if folder creation fails
            logger.error(f"Error ensuring folder exists: {e}")
            return None

    def __init__(
        self,
        database: BaseDatabase,
        vector_store: BaseVectorStore,
        storage: BaseStorage,
        parser: BaseParser,
        embedding_model: BaseEmbeddingModel,
        completion_model: Optional[BaseCompletionModel] = None,
        cache_factory: Optional[BaseCacheFactory] = None,
        reranker: Optional[BaseReranker] = None,
        enable_colpali: bool = False,
        colpali_embedding_model=None,  # Optional[ColpaliEmbeddingModel] = None,
        colpali_vector_store: Optional[BaseVectorStore] = None,
    ):
        self.db = database
        self.vector_store = vector_store
        self.storage = storage
        self.parser = parser
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.reranker = reranker
        self.cache_factory = cache_factory
        self.rules_processor = RulesProcessor()
        self.colpali_embedding_model = colpali_embedding_model
        self.colpali_vector_store = colpali_vector_store

        # Initialize the graph service only if completion_model is provided
        # (e.g., not needed for ingestion worker)
        if completion_model is not None:
            self.graph_service = (
                GraphService(
                    db=database,
                    embedding_model=embedding_model,
                    completion_model=completion_model,
                )
                if settings.GRAPH_MODE == "local"
                else MorphikGraphService(
                    db=database,
                    embedding_model=embedding_model,
                    completion_model=completion_model,
                    base_url=settings.MORPHIK_GRAPH_BASE_URL,
                    graph_api_key=settings.MORPHIK_GRAPH_API_KEY,
                )
            )
        else:
            self.graph_service = None

        # MultiVectorStore initialization is now handled in the FastAPI startup event
        # so we don't need to initialize it here again

        # Cache-related data structures
        # Maps cache name to active cache object
        self.active_caches: Dict[str, BaseCache] = {}

        # Store for aggregated metadata from chunk rules
        self._last_aggregated_metadata: Dict[str, Any] = {}

    async def retrieve_chunks(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        perf_tracker: Optional[Any] = None,  # Performance tracker from API layer
        padding: int = 0,  # Number of additional chunks to retrieve before and after matched chunks
    ) -> List[ChunkResult]:
        """Retrieve relevant chunks."""

        phase_times: Dict[str, float] = {}

        # Use provided performance tracker or create a local one
        if perf_tracker:
            local_perf = False
            retrieve_start_time = None
        else:
            # For standalone calls, create local performance tracking
            local_perf = True
            retrieve_start_time = time.time()

        # 4 configurations:
        # 1. No reranking, no colpali -> just return regular chunks
        # 2. No reranking, colpali  -> return colpali chunks + regular chunks - no need to run smaller colpali model
        # 3. Reranking, no colpali -> sort regular chunks by re-ranker score
        # 4. Reranking, colpali -> return merged chunks sorted by smaller colpali model score

        # Setup phase
        if perf_tracker:
            perf_tracker.start_phase("retrieve_setup")
        else:
            setup_start = time.time()

        settings = get_settings()
        should_rerank = use_reranking if use_reranking is not None else settings.USE_RERANKING
        multivector_available = bool(self.colpali_embedding_model and self.colpali_vector_store)
        requested_multivector = use_colpali if use_colpali is not None else False
        using_multivector = bool(requested_multivector and settings.ENABLE_COLPALI and multivector_available)
        if requested_multivector and not using_multivector:
            logger.warning(
                "Multivector retrieval requested but required components are unavailable. Falling back to regular search."
            )

        # Build system filters for folder_name and end_user_id
        system_filters = {}
        if folder_name:
            # Allow folder_name to be a single string or list[str]
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        # Note: Don't add auth.app_id here - it's already handled in _build_access_filter_optimized

        async def measure_phase(coro, phase_key: Optional[str], perf_parent: Optional[str] = None):
            """
            Measure coroutine execution time and accumulate into phase_times.
            """
            if not phase_key:
                return await coro
            start = time.time()
            try:
                return await coro
            finally:
                duration = time.time() - start
                phase_times[phase_key] = phase_times.get(phase_key, 0.0) + duration
                if perf_tracker:
                    perf_tracker.add_suboperation(phase_key, duration, perf_parent)

        # Launch embedding queries concurrently
        embedding_tasks = []
        if using_multivector:
            embedding_tasks.append(
                measure_phase(
                    self.colpali_embedding_model.embed_for_query(query),
                    "multivector_query_embedding",
                    "retrieve_embeddings_and_auth",
                )
            )
            multivector_pipeline_start = time.time()
        else:
            multivector_pipeline_start = None
            phase_times["multivector_query_embedding"] = 0.0
            embedding_tasks.append(
                measure_phase(
                    self.embedding_model.embed_for_query(query),
                    "query_embedding",
                    "retrieve_embeddings_and_auth",
                )
            )
        if using_multivector:
            phase_times["query_embedding"] = 0.0

        if not perf_tracker:
            phase_times["setup"] = time.time() - setup_start

        # Run embeddings and document authorization in parallel
        if perf_tracker:
            perf_tracker.start_phase("retrieve_embeddings_and_auth")
        else:
            parallel_start = time.time()

        # Create tasks with individual timing to measure embeddings vs auth separately
        async def timed_embeddings():
            embedding_start = time.time()
            result = await asyncio.gather(*embedding_tasks)
            embedding_duration = time.time() - embedding_start
            if perf_tracker:
                perf_tracker.add_suboperation("retrieve_embeddings", embedding_duration, "retrieve_embeddings_and_auth")
            else:
                phase_times["retrieve_embeddings"] = embedding_duration
            return result

        async def timed_auth():
            auth_start = time.time()
            result = await self.db.find_authorized_and_filtered_documents(
                auth,
                filters,
                system_filters,
                status_filter=["completed"],
            )
            auth_duration = time.time() - auth_start
            if perf_tracker:
                perf_tracker.add_suboperation("retrieve_auth", auth_duration, "retrieve_embeddings_and_auth")
            else:
                phase_times["retrieve_auth"] = auth_duration
            return result

        results = await asyncio.gather(
            timed_embeddings(),
            timed_auth(),
        )

        embedding_results, doc_ids = results
        query_embedding_regular = None
        query_embedding_multivector = None
        if using_multivector:
            query_embedding_multivector = embedding_results[0]
        else:
            query_embedding_regular = embedding_results[0]

        if not perf_tracker:
            phase_times["retrieve_embeddings_and_auth"] = time.time() - parallel_start

        logger.info("Generated query embedding")

        if not doc_ids:
            logger.info("No authorized documents found")
            return []
        logger.info(f"Found {len(doc_ids)} authorized documents")

        # Vector search phase
        if perf_tracker:
            perf_tracker.start_phase("retrieve_vector_search")
        else:
            search_setup_start = time.time()

        # Check if we're using colpali multivector search
        search_multi = using_multivector and self.colpali_vector_store and query_embedding_multivector is not None

        # For regular reranking (without colpali), we'll use the existing reranker if available
        # When ColPali is enabled we rely on the ColPali vector store scoring directly.
        use_standard_reranker = should_rerank and (not using_multivector) and self.reranker is not None

        # Search chunks with vector similarity in parallel
        # When using standard reranker, we get more chunks initially to improve reranking quality
        search_tasks = []
        if not using_multivector:
            oversample_k = k
            if use_standard_reranker:
                oversample_k = max(k, min(3 * k, 20))
            search_tasks.append(
                measure_phase(
                    self.vector_store.query_similar(
                        query_embedding_regular,
                        k=oversample_k,
                        doc_ids=doc_ids,
                        app_id=auth.app_id,
                    ),
                    "vector_search_regular",
                    "retrieve_vector_search",
                )
            )
        else:
            phase_times["vector_search_regular"] = 0.0

        if search_multi:
            search_tasks.append(
                measure_phase(
                    self.colpali_vector_store.query_similar(
                        query_embedding_multivector, k=k, doc_ids=doc_ids, app_id=auth.app_id
                    ),
                    "multivector_vector_search",
                    "retrieve_vector_search",
                )
            )
        elif not using_multivector:
            phase_times["multivector_vector_search"] = 0.0

        if not perf_tracker:
            phase_times["search_setup"] = time.time() - search_setup_start

        # Execute vector searches
        if not perf_tracker:
            vector_search_start = time.time()

        search_results = await asyncio.gather(*search_tasks)
        chunks: List[DocumentChunk] = []
        chunks_multivector: List[DocumentChunk] = []
        idx = 0
        if not using_multivector:
            chunks = search_results[idx]
            idx += 1
        if search_multi:
            chunks_multivector = search_results[idx]

        if not perf_tracker:
            phase_times["vector_search"] = time.time() - vector_search_start

        if not using_multivector:
            logger.debug(f"Found {len(chunks)} similar chunks via regular embedding")
        if search_multi:
            logger.debug(
                f"Found {len(chunks_multivector)} similar chunks via multivector embedding "
                f"since we are also using colpali"
            )

        # Rerank chunks using the standard reranker if enabled and available
        # This handles configuration 3: Reranking without colpali
        if perf_tracker:
            perf_tracker.start_phase("retrieve_reranking")
        else:
            reranking_start = time.time()

        if chunks and use_standard_reranker:
            chunks = await self.reranker.rerank(query, chunks)
            chunks.sort(key=lambda x: x.score, reverse=True)
            chunks = chunks[:k]
            logger.debug(f"Reranked {k*10} chunks and selected the top {k}")

        if not perf_tracker:
            phase_times["reranking"] = time.time() - reranking_start

        # Combine multiple chunk sources if needed
        if perf_tracker:
            perf_tracker.start_phase("retrieve_chunk_combination")

        combination_start = time.time()
        if using_multivector:
            chunks = chunks_multivector
        combination_duration = time.time() - combination_start
        if not perf_tracker:
            phase_times["chunk_combination"] = combination_duration
        if using_multivector:
            phase_times["multivector_chunk_combination"] = combination_duration
            if perf_tracker:
                perf_tracker.add_suboperation(
                    "multivector_chunk_combination",
                    combination_duration,
                    "retrieve_chunk_combination",
                )
        else:
            phase_times["multivector_chunk_combination"] = phase_times.get("multivector_chunk_combination", 0.0)

        # Apply padding if requested and using colpali
        if padding > 0 and using_multivector:
            if perf_tracker:
                perf_tracker.start_phase("retrieve_padding")

            padding_start = time.time()
            chunks = await self._apply_padding_to_chunks(chunks, padding, auth)
            padding_duration = time.time() - padding_start

            if not perf_tracker:
                phase_times["padding"] = padding_duration
            phase_times["multivector_padding"] = padding_duration
            if perf_tracker:
                perf_tracker.add_suboperation("multivector_padding", padding_duration, "retrieve_padding")
        else:
            phase_times["multivector_padding"] = phase_times.get("multivector_padding", 0.0)

        # Create and return chunk results
        if perf_tracker:
            perf_tracker.start_phase("retrieve_result_creation")
        else:
            result_creation_start = time.time()

        results = await self._create_chunk_results(auth, chunks)

        if not perf_tracker:
            phase_times["result_creation"] = time.time() - result_creation_start

        if using_multivector and multivector_pipeline_start is not None:
            phase_times["multivector_pipeline_total"] = time.time() - multivector_pipeline_start
            if perf_tracker:
                perf_tracker.add_suboperation(
                    "multivector_pipeline_total",
                    phase_times["multivector_pipeline_total"],
                    "retrieve_embeddings_and_auth",
                )
        else:
            phase_times["multivector_pipeline_total"] = phase_times.get("multivector_pipeline_total", 0.0)

        # Log performance summary only for standalone calls
        if local_perf:
            total_time = time.time() - retrieve_start_time
            logger.info("=== DocumentService.retrieve_chunks Performance Summary ===")
            logger.info(f"Total retrieve_chunks time: {total_time:.2f}s")
            for phase, duration in sorted(phase_times.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")
            logger.info(f"Returning {len(results)} chunk results")
            logger.info("==========================================================")

        return results

    def _count_tokens_simple(self, text: str) -> int:
        """Simple token counting using whitespace splitting.

        This is a conservative estimate that works well for batching purposes.
        """
        return len(text.split())

    def _batch_chunks_by_tokens(self, chunks: List[DocumentChunk], max_tokens: int = 6000) -> List[List[DocumentChunk]]:
        """Batch chunks to ensure total token count doesn't exceed max_tokens.

        Args:
            chunks: List of chunks to batch
            max_tokens: Maximum tokens per batch (conservative limit under 8192)

        Returns:
            List of chunk batches
        """
        if not chunks:
            return []

        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self._count_tokens_simple(chunk.content)

            # If a single chunk exceeds the limit, put it in its own batch
            if chunk_tokens > max_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([chunk])
                logger.warning(f"Chunk with {chunk_tokens} tokens exceeds limit of {max_tokens}")
                continue

            # If adding this chunk would exceed the limit, start a new batch
            if current_tokens + chunk_tokens > max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_tokens += chunk_tokens

        # Add the last batch if it has chunks
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Created {len(batches)} batches from {len(chunks)} chunks")
        return batches

    async def _apply_padding_to_chunks(
        self,
        chunks: List[DocumentChunk],
        padding: int,
        auth: AuthContext,
    ) -> List[DocumentChunk]:
        """
        Apply padding to chunks by retrieving additional chunks before and after each matched chunk.
        This is only relevant for ColPali retrieval path where chunks correspond to pages.
        Only applies to image chunks - non-image chunks are filtered out when padding is enabled.

        Args:
            chunks: Original matched chunks
            padding: Number of chunks to retrieve before and after each matched chunk
            auth: Authentication context for access control

        Returns:
            List of image chunks with padding applied (deduplicated)
        """
        if not chunks or padding <= 0:
            return chunks
        logger.info(f"chunks: {[chunk.content[:100] for chunk in chunks]}")

        # Filter to only image chunks when padding is enabled
        image_chunks = [chunk for chunk in chunks if chunk.content.startswith("data")]

        if not image_chunks:
            # No image chunks to pad, return empty list since padding is only for images
            logger.info("No image chunks found for padding, returning empty list")
            return []

        logger.info(
            f"Applying padding of {padding} to {len(image_chunks)} image chunks (filtered from {len(chunks)} total chunks)"
        )

        # Group image chunks by document to apply padding efficiently
        chunks_by_doc = {}
        for chunk in image_chunks:
            if chunk.document_id not in chunks_by_doc:
                chunks_by_doc[chunk.document_id] = []
            chunks_by_doc[chunk.document_id].append(chunk)

        # Collect all chunk identifiers we need to retrieve (including padding)
        chunk_identifiers_to_retrieve = set()
        original_keys = {(chunk.document_id, chunk.chunk_number) for chunk in image_chunks}

        for doc_id, doc_chunks in chunks_by_doc.items():
            for chunk in doc_chunks:
                # Add padding chunks before and after
                for i in range(1, padding + 1):
                    # Add chunks before (if chunk_number > i)
                    if chunk.chunk_number >= i:
                        candidate = (doc_id, chunk.chunk_number - i)
                        if candidate not in original_keys:
                            chunk_identifiers_to_retrieve.add(candidate)

                    # Add chunks after
                    candidate = (doc_id, chunk.chunk_number + i)
                    if candidate not in original_keys:
                        chunk_identifiers_to_retrieve.add(candidate)

        logger.debug(f"Need to retrieve {len(chunk_identifiers_to_retrieve)} additional padding chunks")

        # Convert to list for batch retrieval
        chunk_identifiers = list(chunk_identifiers_to_retrieve)

        # Use colpali vector store for retrieval since padding is only for colpali path
        padding_chunks: List[DocumentChunk] = []
        if self.colpali_vector_store and chunk_identifiers:
            try:
                retrieval_start = time.time()
                padded_chunks = await self.colpali_vector_store.get_chunks_by_id(chunk_identifiers, auth.app_id)
                logger.debug(
                    "Multivector padding retrieval took %.2fs for %d chunks",
                    time.time() - retrieval_start,
                    len(padded_chunks),
                )
                logger.debug(f"Retrieved {len(padded_chunks)} chunks from colpali vector store")
                padding_chunks = padded_chunks
            except Exception as e:
                logger.error(f"Error retrieving padded chunks from colpali vector store: {e}")
                # Fallback to original image chunks if padding fails
                return image_chunks
        else:
            if not self.colpali_vector_store:
                logger.warning("ColPali vector store not available for padding, returning original image chunks")
            else:
                logger.debug("No additional padding chunks required")
            padding_chunks = []

        # Filter retrieved chunks to only image chunks (padding chunks should also be images)
        padded_image_chunks = [chunk for chunk in padding_chunks if chunk.content.startswith("data")]
        logger.debug(f"Filtered to {len(padded_image_chunks)} image chunks from {len(padding_chunks)} retrieved chunks")

        # Preserve original scores for matched chunks; padding gets 0.0
        original_scores = {(c.document_id, c.chunk_number): c.score for c in image_chunks}
        combined_chunks = list(image_chunks) + padded_image_chunks

        deduped: List[DocumentChunk] = []
        seen = set()
        for chunk in combined_chunks:
            key = (chunk.document_id, chunk.chunk_number)
            if key in seen:
                continue
            chunk.score = original_scores.get(key, 0.0)
            deduped.append(chunk)
            seen.add(key)

        # Sort: matched chunks (higher score) first, then by document and page order
        deduped.sort(key=lambda x: (-float(x.score or 0.0), x.document_id, x.chunk_number))

        logger.info(
            f"Applied padding: returning {len(deduped)} image chunks (was {len(image_chunks)} image chunks, "
            f"added {len(padded_image_chunks)} padding chunks)"
        )
        return deduped

    async def _create_grouped_chunk_response_from_results(
        self,
        original_chunk_results: List[ChunkResult],
        final_chunk_results: List[ChunkResult],
        padding: int,
    ):  # -> "GroupedChunkResponse"
        """
        Create a grouped response directly from ChunkResult objects.

        Args:
            original_chunk_results: The original matched chunks (before padding)
            final_chunk_results: All chunks including padding
            padding: The padding value used

        Returns:
            GroupedChunkResponse with both flat and grouped results
        """
        from core.models.documents import ChunkGroup, GroupedChunkResponse

        # Create mapping of original chunks for easy lookup
        original_chunk_keys = {(chunk.document_id, chunk.chunk_number) for chunk in original_chunk_results}

        # Mark chunks as padding or not
        for result in final_chunk_results:
            result.is_padding = (result.document_id, result.chunk_number) not in original_chunk_keys

        # If no padding was applied, return simple response
        if padding == 0:
            return GroupedChunkResponse(
                chunks=final_chunk_results,
                groups=[
                    ChunkGroup(main_chunk=result, padding_chunks=[], total_chunks=1) for result in final_chunk_results
                ],
                total_results=len(final_chunk_results),
                has_padding=False,
            )

        # Group chunks by main chunks
        groups = []
        processed_chunks = set()

        # First, identify all main (non-padding) chunks
        main_chunks = [result for result in final_chunk_results if not result.is_padding]

        for main_chunk in main_chunks:
            if (main_chunk.document_id, main_chunk.chunk_number) in processed_chunks:
                continue

            # Find all padding chunks for this main chunk
            padding_chunks = []

            # Look for chunks in the padding range
            for i in range(1, padding + 1):
                # Check chunks before
                before_key = (main_chunk.document_id, main_chunk.chunk_number - i)
                after_key = (main_chunk.document_id, main_chunk.chunk_number + i)

                for result in final_chunk_results:
                    result_key = (result.document_id, result.chunk_number)
                    if result.is_padding and (result_key == before_key or result_key == after_key):
                        padding_chunks.append(result)
                        processed_chunks.add(result_key)

            # Create group
            group = ChunkGroup(
                main_chunk=main_chunk, padding_chunks=padding_chunks, total_chunks=1 + len(padding_chunks)
            )
            groups.append(group)
            processed_chunks.add((main_chunk.document_id, main_chunk.chunk_number))

        return GroupedChunkResponse(
            chunks=final_chunk_results, groups=groups, total_results=len(final_chunk_results), has_padding=padding > 0
        )

    async def retrieve_chunks_grouped(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        perf_tracker: Optional[Any] = None,
        padding: int = 0,
    ):  # -> "GroupedChunkResponse"
        """
        Retrieve chunks with grouped response format that differentiates main chunks from padding.

        Returns both flat results (for backward compatibility) and grouped results (for UI).
        """
        # Get original chunks before padding (as ChunkResult objects)
        original_chunk_results = await self.retrieve_chunks(
            query,
            auth,
            filters,
            k,
            min_score,
            use_reranking,
            use_colpali,
            folder_name,
            end_user_id,
            perf_tracker,
            padding=0,  # No padding for original
        )

        # Get final chunks with padding (as ChunkResult objects)
        if padding > 0 and use_colpali:
            final_chunk_results = await self.retrieve_chunks(
                query,
                auth,
                filters,
                k,
                min_score,
                use_reranking,
                use_colpali,
                folder_name,
                end_user_id,
                perf_tracker,
                padding,
            )
        else:
            final_chunk_results = original_chunk_results

        # Create grouped response directly from ChunkResult objects
        return await self._create_grouped_chunk_response_from_results(
            original_chunk_results, final_chunk_results, padding
        )

    async def retrieve_docs(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> List[DocumentResult]:
        """Retrieve relevant documents."""
        # Get chunks first
        chunks = await self.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali, folder_name, end_user_id
        )
        # Convert to document results
        results = await self._create_document_results(auth, chunks)
        documents = list(results.values())
        logger.info(f"Returning {len(documents)} document results")
        return documents

    async def batch_retrieve_documents(
        self,
        document_ids: List[str],
        auth: AuthContext,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.

        Args:
            document_ids: List of document IDs to retrieve
            auth: Authentication context

        Returns:
            List of Document objects that user has access to
        """
        if not document_ids:
            return []

        # Build system filters for folder_name and end_user_id
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        # Note: Don't add auth.app_id here - it's already handled in _build_access_filter_optimized

        # Use the database's batch retrieval method
        documents = await self.db.get_documents_by_id(document_ids, auth, system_filters)
        logger.info(f"Batch retrieved {len(documents)} documents out of {len(document_ids)} requested")
        return documents

    async def batch_retrieve_chunks(
        self,
        chunk_ids: List[ChunkSource],
        auth: AuthContext,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        use_colpali: Optional[bool] = None,
    ) -> List[ChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation.

        Args:
            chunk_ids: List of ChunkSource objects with document_id and chunk_number
            auth: Authentication context
            folder_name: Optional folder to scope the operation to
            end_user_id: Optional end-user ID to scope the operation to
            use_colpali: Whether to use colpali multimodal features for image chunks

        Returns:
            List of ChunkResult objects
        """
        if not chunk_ids:
            return []

        # Collect unique document IDs to check authorization in a single query
        doc_ids = list({source.document_id for source in chunk_ids})

        # Find authorized documents in a single query
        authorized_docs = await self.batch_retrieve_documents(doc_ids, auth, folder_name, end_user_id)
        authorized_doc_ids = {doc.external_id for doc in authorized_docs}

        # Filter sources to only include authorized documents
        authorized_sources = [source for source in chunk_ids if source.document_id in authorized_doc_ids]

        if not authorized_sources:
            return []

        # Create list of (document_id, chunk_number) tuples for vector store query
        chunk_identifiers = [(source.document_id, source.chunk_number) for source in authorized_sources]

        # Set up vector store retrieval tasks
        retrieval_tasks = [self.vector_store.get_chunks_by_id(chunk_identifiers, auth.app_id)]

        # Add colpali vector store task if needed
        settings = get_settings()
        if use_colpali and settings.ENABLE_COLPALI and self.colpali_vector_store:
            logger.info("Preparing to retrieve chunks from both regular and colpali vector stores")
            retrieval_tasks.append(self.colpali_vector_store.get_chunks_by_id(chunk_identifiers, auth.app_id))

        # Execute vector store retrievals in parallel
        try:
            vector_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process regular chunks
            chunks = vector_results[0] if not isinstance(vector_results[0], Exception) else []

            # Process colpali chunks if available
            if len(vector_results) > 1 and not isinstance(vector_results[1], Exception):
                colpali_chunks = vector_results[1]

                if colpali_chunks:
                    # Create a dictionary of (doc_id, chunk_number) -> chunk for fast lookup
                    chunk_dict = {(c.document_id, c.chunk_number): c for c in chunks}

                    logger.debug(f"Found {len(colpali_chunks)} chunks in colpali store")
                    for colpali_chunk in colpali_chunks:
                        key = (colpali_chunk.document_id, colpali_chunk.chunk_number)
                        # Replace chunks with colpali chunks when available
                        chunk_dict[key] = colpali_chunk

                    # Update chunks list with the combined/replaced chunks
                    chunks = list(chunk_dict.values())
                    logger.info(f"Enhanced {len(colpali_chunks)} chunks with colpali/multimodal data")

            # Handle any exceptions that occurred during retrieval
            for i, result in enumerate(vector_results):
                if isinstance(result, Exception):
                    store_type = "regular" if i == 0 else "colpali"
                    logger.error(f"Error retrieving chunks from {store_type} vector store: {result}", exc_info=True)
                    if i == 0:  # If regular store failed, we can't proceed
                        return []

        except Exception as e:
            logger.error(f"Error during parallel chunk retrieval: {e}", exc_info=True)
            return []

        # Create a mapping of original scores from ChunkSource objects (O(n) time)
        score_map = {
            (source.document_id, source.chunk_number): source.score
            for source in authorized_sources
            if source.score is not None
        }

        # Apply original scores to the retrieved chunks (O(m) time with O(1) lookups)
        for chunk in chunks:
            key = (chunk.document_id, chunk.chunk_number)
            if key in score_map:
                chunk.score = score_map[key]
                logger.debug(f"Restored score {chunk.score} for chunk {key}")

        # Sort chunks by score in descending order (highest score first)
        chunks.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Sorted {len(chunks)} chunks by score")

        # Convert to chunk results
        results = await self._create_chunk_results(auth, chunks)
        logger.info(f"Batch retrieved {len(results)} chunks out of {len(chunk_ids)} requested")
        return results

    async def query(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,  # from contextual embedding paper
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        graph_name: Optional[str] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional["QueryPromptOverrides"] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        perf_tracker: Optional[Any] = None,  # Performance tracker from API layer
        stream_response: Optional[bool] = False,
        llm_config: Optional[Dict[str, Any]] = None,
        padding: int = 0,  # Number of additional chunks to retrieve before and after matched chunks
        inline_citations: bool = False,  # Whether to include inline citations with filename and page number
    ) -> Union[CompletionResponse, tuple[AsyncGenerator[str, None], List[ChunkSource]]]:
        """Generate completion using relevant chunks as context.

        When graph_name is provided, the query will leverage the knowledge graph
        to enhance retrieval by finding relevant entities and their connected documents.

        Args:
            query: The query text
            auth: Authentication context
            filters: Optional metadata filters for documents
            k: Number of chunks to retrieve
            min_score: Minimum similarity score
            max_tokens: Maximum tokens for completion
            temperature: Temperature for completion
            use_reranking: Whether to use reranking
            use_colpali: Whether to use colpali embedding
            graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            hop_depth: Number of relationship hops to traverse in the graph (1-3)
            include_paths: Whether to include relationship paths in the response
            prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
            folder_name: Optional folder to scope the operation to
            end_user_id: Optional end-user ID to scope the operation to
            schema: Optional schema for structured output
        """
        # Use provided performance tracker or create a local one for standalone calls
        if perf_tracker:
            local_perf = False
        else:
            local_perf = True
            query_start_time = time.time()
            phase_times = {}

        # Graph routing check
        if perf_tracker:
            perf_tracker.start_phase("graph_routing_check")
        else:
            graph_check_start = time.time()

        if graph_name:
            # Use knowledge graph enhanced retrieval via GraphService
            return await self.graph_service.query_with_graph(
                query=query,
                graph_name=graph_name,
                auth=auth,
                document_service=self,
                filters=filters,
                k=k,
                min_score=min_score,
                max_tokens=max_tokens,
                temperature=temperature,
                use_reranking=use_reranking,
                use_colpali=use_colpali,
                hop_depth=hop_depth,
                include_paths=include_paths,
                prompt_overrides=prompt_overrides,
                folder_name=folder_name,
                end_user_id=end_user_id,
                stream_response=stream_response,
            )

        if not perf_tracker:
            phase_times["graph_routing_check"] = time.time() - graph_check_start

        # Standard retrieval without graph
        if perf_tracker:
            perf_tracker.start_phase("chunk_retrieval")
        else:
            chunk_retrieval_start = time.time()

        chunks = await self.retrieve_chunks(
            query,
            auth,
            filters,
            k,
            min_score,
            use_reranking,
            use_colpali,
            folder_name,
            end_user_id,
            perf_tracker,
            padding,
        )

        if not perf_tracker:
            phase_times["chunk_retrieval"] = time.time() - chunk_retrieval_start

        # Create document results
        if perf_tracker:
            perf_tracker.start_phase("document_results_creation")
        else:
            doc_results_start = time.time()

        documents = await self._create_document_results(auth, chunks)

        if not perf_tracker:
            phase_times["document_results_creation"] = time.time() - doc_results_start

        # Create augmented chunk contents
        if perf_tracker:
            perf_tracker.start_phase("content_augmentation")
        else:
            augmentation_start = time.time()

        chunk_contents = [chunk.augmented_content(documents[chunk.document_id]) for chunk in chunks]

        # Collect chunk metadata for inline citations if enabled
        chunk_metadata = None
        if inline_citations:
            chunk_metadata = []
            for chunk in chunks:
                # Get the document for this chunk
                doc = documents.get(chunk.document_id, {})
                filename = (
                    chunk.filename or doc.metadata.get("filename", "unknown") if hasattr(doc, "metadata") else "unknown"
                )

                # Check if this is a ColPali/image chunk
                is_colpali = chunk.metadata.get("is_image", False)

                metadata = {
                    "filename": filename,
                    "chunk_number": chunk.chunk_number,
                    "document_id": chunk.document_id,
                    "is_colpali": is_colpali,
                }

                # For ColPali chunks, chunk_number corresponds to page number (0-indexed)
                # Add 1 to make it 1-indexed for user display
                if is_colpali:
                    metadata["page_number"] = chunk.chunk_number + 1
                else:
                    # For regular text chunks, check if page_number is stored in metadata
                    metadata["page_number"] = chunk.metadata.get("page_number")

                chunk_metadata.append(metadata)

        if not perf_tracker:
            phase_times["content_augmentation"] = time.time() - augmentation_start

        # Collect sources information
        if perf_tracker:
            perf_tracker.start_phase("sources_collection")
        else:
            sources_start = time.time()

        sources = [
            ChunkSource(document_id=chunk.document_id, chunk_number=chunk.chunk_number, score=chunk.score)
            for chunk in chunks
        ]

        if not perf_tracker:
            phase_times["sources_collection"] = time.time() - sources_start

        # Generate completion with prompt override if provided
        if perf_tracker:
            perf_tracker.start_phase("completion_generation")
        else:
            completion_start = time.time()

        custom_prompt_template = None
        custom_system_prompt = None
        if prompt_overrides and prompt_overrides.query:
            if hasattr(prompt_overrides.query, "prompt_template"):
                custom_prompt_template = prompt_overrides.query.prompt_template
            if hasattr(prompt_overrides.query, "system_prompt"):
                custom_system_prompt = prompt_overrides.query.system_prompt

        request = CompletionRequest(
            query=query,
            context_chunks=chunk_contents,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_template=custom_prompt_template,
            system_prompt=custom_system_prompt,
            schema=schema,
            chat_history=chat_history,
            stream_response=stream_response,
            llm_config=llm_config,
            inline_citations=inline_citations,
            chunk_metadata=chunk_metadata,
        )

        response = await self.completion_model.complete(request)

        if not perf_tracker:
            phase_times["completion_generation"] = time.time() - completion_start

        # Handle streaming vs non-streaming responses
        if stream_response:
            # For streaming responses, return the async generator and sources separately

            # Log performance summary for streaming calls
            if local_perf:
                total_time = time.time() - query_start_time
                logger.info("=== DocumentService.query Performance Summary (Streaming) ===")
                logger.info(f"Total setup time: {total_time:.2f}s")
                for phase, duration in sorted(phase_times.items(), key=lambda x: x[1], reverse=True):
                    percentage = (duration / total_time) * 100 if total_time > 0 else 0
                    logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")
                logger.info(f"Starting streaming with {len(sources)} sources")
                logger.info("=" * 59)

            return response, sources
        else:
            # Add sources information at the document service level for non-streaming
            response.sources = sources

            # Log performance summary only for standalone calls
            if local_perf:
                total_time = time.time() - query_start_time
                logger.info("=== DocumentService.query Performance Summary ===")
                logger.info(f"Total query time: {total_time:.2f}s")
                for phase, duration in sorted(phase_times.items(), key=lambda x: x[1], reverse=True):
                    percentage = (duration / total_time) * 100 if total_time > 0 else 0
                    logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")
                logger.info(f"Generated completion with {len(sources)} sources")
                logger.info("================================================")

            return response

    async def ingest_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auth: AuthContext = None,
        rules: Optional[List[str]] = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[str] = None,
        end_user_id: Optional[str] = None,
    ) -> Document:
        """Ingest a text document."""
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission")
            raise PermissionError("User does not have write permission")

        # First check ingest limits if in cloud mode
        from core.config import get_settings

        settings = get_settings()

        doc = Document(
            content_type="text/plain",
            filename=filename,
            metadata=metadata or {},
            folder_name=folder_name,
            end_user_id=end_user_id,
            app_id=auth.app_id,
        )

        logger.debug(f"Created text document record with ID {doc.external_id}")

        if settings.MODE == "cloud" and auth.user_id:
            # Verify limits before heavy processing
            num_pages = estimate_pages_by_chars(len(content))
            await check_and_increment_limits(
                auth,
                "ingest",
                num_pages,
                doc.external_id,
                verify_only=True,
            )

        # === Apply post_parsing rules ===
        document_rule_metadata = {}
        if rules:
            logger.info("Applying post-parsing rules...")
            document_rule_metadata, content = await self.rules_processor.process_document_rules(content, rules)
            # Update document metadata with extracted metadata from rules
            metadata.update(document_rule_metadata)
            doc.metadata = metadata  # Update doc metadata after rules
            logger.info(f"Document metadata after post-parsing rules: {metadata}")
            logger.info(f"Content length after post-parsing rules: {len(content)}")

        # Store full content before chunking
        doc.system_metadata["content"] = content

        # Split text into chunks
        parsed_chunks = await self.parser.split_text(content)
        if not parsed_chunks:
            raise ValueError("No content chunks extracted after rules processing")
        logger.debug(f"Split processed text into {len(parsed_chunks)} chunks")

        # === Apply post_chunking rules and aggregate metadata ===
        processed_chunks = []
        aggregated_chunk_metadata: Dict[str, Any] = {}  # Initialize dict for aggregated metadata
        chunk_contents = []  # Initialize list to collect chunk contents efficiently

        if rules:
            logger.info("Applying post-chunking rules...")

            for chunk_obj in parsed_chunks:
                # Get metadata *and* the potentially modified chunk
                chunk_rule_metadata, processed_chunk = await self.rules_processor.process_chunk_rules(chunk_obj, rules)
                processed_chunks.append(processed_chunk)
                chunk_contents.append(processed_chunk.content)  # Collect content as we process
                # Aggregate the metadata extracted from this chunk
                aggregated_chunk_metadata.update(chunk_rule_metadata)
            logger.info(f"Finished applying post-chunking rules to {len(processed_chunks)} chunks.")
            logger.info(f"Aggregated metadata from all chunks: {aggregated_chunk_metadata}")

            # Update the document content with the stitched content from processed chunks
            if processed_chunks:
                logger.info("Updating document content with processed chunks...")
                stitched_content = "\n".join(chunk_contents)
                doc.system_metadata["content"] = stitched_content
                logger.info(f"Updated document content with stitched chunks (length: {len(stitched_content)})")
        else:
            processed_chunks = parsed_chunks  # No rules, use original chunks

        # Generate embeddings for processed chunks
        embeddings = await self.embedding_model.embed_for_ingestion(processed_chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")

        # Create chunk objects with processed chunk content
        chunk_objects = self._create_chunk_objects(doc.external_id, processed_chunks, embeddings)
        logger.debug(f"Created {len(chunk_objects)} chunk objects")

        chunk_objects_multivector = []

        # Check both use_colpali parameter AND global enable_colpali setting
        settings = get_settings()
        if use_colpali and settings.ENABLE_COLPALI and self.colpali_embedding_model:
            embeddings_multivector = await self.colpali_embedding_model.embed_for_ingestion(processed_chunks)
            logger.info(f"Generated {len(embeddings_multivector)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(
                doc.external_id, processed_chunks, embeddings_multivector
            )
            logger.info(f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding")

        # Create and store chunk objects

        # === Merge aggregated chunk metadata into document metadata ===
        if aggregated_chunk_metadata:
            logger.info("Merging aggregated chunk metadata into document metadata...")
            # Make sure doc.metadata exists
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(aggregated_chunk_metadata)
            logger.info(f"Final document metadata after merge: {doc.metadata}")
        # ===========================================================

        # Store everything
        await self._store_chunks_and_doc(
            chunk_objects,
            doc,
            use_colpali and settings.ENABLE_COLPALI,
            chunk_objects_multivector,
            auth=auth,
        )
        logger.debug(f"Successfully stored text document {doc.external_id}")

        # Ensure folder membership now that the document is persisted.
        if folder_name:
            try:
                await self._ensure_folder_exists(folder_name, doc.external_id, auth)
            except Exception as folder_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to ensure folder %s contains text document %s: %s",
                    folder_name,
                    doc.external_id,
                    folder_exc,
                )

        # Update the document status to completed after successful storage
        # This matches the behavior in ingestion_worker.py
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        doc.system_metadata = self._clean_system_metadata(doc.system_metadata)
        await self.db.update_document(
            document_id=doc.external_id, updates={"system_metadata": doc.system_metadata}, auth=auth
        )
        logger.debug(f"Updated document status to 'completed' for {doc.external_id}")

        # Determine the final page count for usage recording
        colpali_count_for_limit_fn = (
            len(chunk_objects_multivector)
            if use_colpali and settings.ENABLE_COLPALI and chunk_objects_multivector
            else None
        )
        final_page_count = estimate_pages_by_chars(len(content))
        if use_colpali and settings.ENABLE_COLPALI and colpali_count_for_limit_fn is not None:
            final_page_count = colpali_count_for_limit_fn
        final_page_count = max(1, final_page_count)  # Ensure minimum of 1 page
        logger.info(f"Determined final page count for ingest_text usage: {final_page_count}")

        # Record ingest usage after successful completion
        if settings.MODE == "cloud" and auth.user_id:
            try:
                await check_and_increment_limits(
                    auth,
                    "ingest",
                    final_page_count,  # Use the determined final count
                    doc.external_id,
                    use_colpali=use_colpali and settings.ENABLE_COLPALI,  # Pass colpali status
                    colpali_chunks_count=colpali_count_for_limit_fn,  # Pass actual colpali count
                )
            except Exception as rec_exc:
                # Log error but don't fail the synchronous request at this point
                logger.error("Failed to record ingest usage in ingest_text: %s", rec_exc)

        return doc

    async def ingest_file_content(
        self,
        file_content_bytes: bytes,
        filename: str,
        content_type: Optional[str],
        metadata: Optional[Dict[str, Any]],
        auth: AuthContext,
        redis: arq.ArqRedis,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        rules: Optional[List[str]] = None,
        use_colpali: Optional[bool] = False,
    ) -> Document:
        """
        Ingests file content from bytes. Saves to storage, creates document record,
        and then enqueues a background job for chunking and embedding.
        """
        settings = get_settings()

        logger.info(
            f"Starting ingestion for filename: {filename}, content_type: {content_type}, "
            f"user: {auth.user_id or auth.entity_id}"
        )

        # Ensure user has write permission
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission for ingest_file_content")
            raise PermissionError("User does not have write permission for ingest_file_content")

        doc = Document(
            filename=filename,
            content_type=content_type,
            metadata=metadata or {},
            system_metadata={"status": "processing"},  # Initial status
            content_info={"type": "file", "mime_type": content_type},
            app_id=auth.app_id,
            end_user_id=end_user_id,
            folder_name=folder_name,
        )

        # --------------------------------------------------------
        # Verify quotas before incurring heavy compute or storage
        # --------------------------------------------------------
        if settings.MODE == "cloud" and auth.user_id:
            num_pages = estimate_pages_by_chars(len(file_content_bytes))

            # Dry-run checks; nothing is recorded yet
            await check_and_increment_limits(
                auth,
                "ingest",
                num_pages,
                doc.external_id,
                verify_only=True,
            )
            await check_and_increment_limits(auth, "storage_file", 1, verify_only=True)
            await check_and_increment_limits(
                auth,
                "storage_size",
                len(file_content_bytes),
                verify_only=True,
            )
            logger.info(
                "Quota verification passed for user %s – pages=%s, file=%s bytes",
                auth.user_id,
                num_pages,
                len(file_content_bytes),
            )

        # 1. Create initial document record in DB
        # The app_db concept from core/api.py implies self.db is already app-specific if needed
        await self.db.store_document(doc, auth)
        logger.info(f"Initial document record created for {filename} (doc_id: {doc.external_id})")

        # 2. Save raw file to Storage
        # Using a unique key structure similar to /ingest/file to avoid collisions if worker needs it
        file_key_suffix = str(uuid.uuid4())
        safe_filename = Path(filename or "").name or "uploaded_file"
        storage_key = f"ingest_uploads/{file_key_suffix}/{safe_filename}"
        if not Path(storage_key).suffix:
            detected_ext = detect_file_type(file_content_bytes)
            if detected_ext:
                storage_key = f"{storage_key}{detected_ext}"

        try:
            bucket_name, full_storage_path = await self._upload_to_app_bucket(
                auth=auth, content_bytes=file_content_bytes, key=storage_key, content_type=content_type
            )
            # Create StorageFileInfo with version as INT
            sfi = StorageFileInfo(
                bucket=bucket_name,
                key=full_storage_path,
                content_type=content_type,
                size=len(file_content_bytes),
                last_modified=datetime.now(UTC),
                version=1,  # INT, as per StorageFileInfo model
                filename=safe_filename,
            )
            # Populate legacy doc.storage_info (Dict[str, str]) with stringified values
            doc.storage_info = {k: str(v) if v is not None else "" for k, v in sfi.model_dump().items()}

            # Initialize storage_files list with the StorageFileInfo object (version remains int)
            doc.storage_files = [sfi]

            doc.system_metadata = self._clean_system_metadata(doc.system_metadata)
            await self.db.update_document(
                document_id=doc.external_id,
                updates={
                    "storage_info": doc.storage_info,  # This is now Dict[str, str]
                    "storage_files": [sf.model_dump() for sf in doc.storage_files],  # Dumps SFI, version is int
                    "system_metadata": doc.system_metadata,  # system_metadata already has status processing
                },
                auth=auth,
            )
            logger.info(
                "File %s (doc_id: %s) uploaded to storage at %s/%s and DB updated.",
                filename,
                doc.external_id,
                bucket_name,
                full_storage_path,
            )

            # -----------------------------------
            # Record usage now that upload passed
            # -----------------------------------
            if settings.MODE == "cloud" and auth.user_id:
                try:
                    await check_and_increment_limits(auth, "storage_file", 1)
                    await check_and_increment_limits(auth, "storage_size", len(file_content_bytes))
                except Exception as rec_err:
                    logger.error("Failed recording usage for doc %s: %s", doc.external_id, rec_err)

        except Exception as e:
            logger.error(f"Failed to upload file {filename} (doc_id: {doc.external_id}) to storage or update DB: {e}")
            # Update document status to failed if initial storage fails
            doc.system_metadata["status"] = "failed"
            doc.system_metadata["error"] = f"Storage upload/DB update failed: {str(e)}"
            try:
                await self.db.update_document(
                    doc.external_id,
                    {"system_metadata": self._clean_system_metadata(doc.system_metadata)},
                    auth=auth,
                )
            except Exception as db_update_err:
                logger.error(f"Additionally failed to mark doc {doc.external_id} as failed in DB: {db_update_err}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file to storage: {str(e)}")

        # 3. Ensure folder exists if folder_name is provided (after doc is created)
        if folder_name:
            try:
                await self._ensure_folder_exists(folder_name, doc.external_id, auth)
                logger.debug(f"Ensured folder '{folder_name}' exists " f"and contains document {doc.external_id}")
            except Exception as e:
                logger.error(
                    f"Error during _ensure_folder_exists for doc {doc.external_id}"
                    f"in folder {folder_name}: {e}. Continuing."
                )

        # 4. Enqueue background job for processing
        auth_dict = {
            "entity_type": auth.entity_type.value,
            "entity_id": auth.entity_id,
            "app_id": auth.app_id,
            "permissions": list(auth.permissions),
            "user_id": auth.user_id,
        }

        metadata_json_str = json.dumps(metadata or {})
        rules_list_for_job = rules or []

        try:
            job = await redis.enqueue_job(
                "process_ingestion_job",
                _job_id=f"ingest:{doc.external_id}",
                document_id=doc.external_id,
                file_key=full_storage_path,  # This is the key in storage
                bucket=bucket_name,
                original_filename=filename,
                content_type=content_type,
                metadata_json=metadata_json_str,
                auth_dict=auth_dict,
                rules_list=rules_list_for_job,
                use_colpali=use_colpali,
                folder_name=str(folder_name) if folder_name else None,  # Ensure folder_name is str or None
                end_user_id=end_user_id,
            )
            if job is None:
                logger.info("Connector file ingestion job already queued (doc_id=%s)", doc.external_id)
            else:
                logger.info(
                    "Connector file ingestion job queued with ID: %s for document: %s", job.job_id, doc.external_id
                )
        except Exception as e:
            logger.error(f"Failed to enqueue ingestion job for doc {doc.external_id} ({filename}): {e}")
            # Update document status to failed if enqueuing fails
            doc.system_metadata["status"] = "failed"
            doc.system_metadata["error"] = f"Failed to enqueue processing job: {str(e)}"
            try:
                await self.db.update_document(
                    doc.external_id,
                    {"system_metadata": self._clean_system_metadata(doc.system_metadata)},
                    auth=auth,
                )
            except Exception as db_update_err:
                logger.error(f"Additionally failed to mark doc {doc.external_id} as failed in DB: {db_update_err}")
            raise HTTPException(status_code=500, detail=f"Failed to enqueue document processing job: {str(e)}")

        return doc

    def _image_bytes_to_chunk(
        self,
        image_bytes: bytes,
        mime_type: str,
        base64_override: Optional[str] = None,
    ) -> Chunk:
        """
        Build a Chunk that preserves raw image bytes alongside the data URI used for storage.
        """
        content = base64_override
        if content is None:
            content = f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode()
        return Chunk(
            content=content,
            metadata={"is_image": True, "_image_bytes": image_bytes, "mime_type": mime_type},
        )

    def img_to_base64_with_bytes(
        self,
        img: PILImage.Image,
        format: str = "PNG",
        mime_type: Optional[str] = None,
    ) -> Tuple[str, bytes]:
        buffered = BytesIO()
        img.save(buffered, format=format)
        buffered.seek(0)
        img_bytes = buffered.getvalue()
        mime = mime_type or f"image/{format.lower()}"
        img_str = f"data:{mime};base64," + base64.b64encode(img_bytes).decode()
        return img_str, img_bytes

    def img_to_base64_str(self, img: PILImage.Image):
        img_str, _ = self.img_to_base64_with_bytes(img)
        return img_str

    def _render_pdf_with_pymupdf(
        self, file_content: bytes, dpi: int, include_bytes: bool = False
    ) -> List[Union[str, Tuple[str, bytes]]]:
        """
        Render a PDF into base64-encoded PNG images using PyMuPDF.
        Raises the underlying PyMuPDF exception so callers can decide on fallbacks.
        """
        pdf_document = fitz.open("pdf", file_content)
        try:
            images: List[Union[str, Tuple[str, bytes]]] = []
            for page in pdf_document:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")
                b64 = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
                if include_bytes:
                    images.append((b64, png_bytes))
                else:
                    images.append(b64)
            return images
        finally:
            pdf_document.close()

    def _create_chunks_multivector(
        self,
        file_type,
        file_content_base64: Optional[str],
        file_content: bytes,
        chunks: List[Chunk],
    ):
        # Derive a safe MIME type string regardless of input shape
        if isinstance(file_type, str):
            mime_type = file_type
        elif file_type is not None and hasattr(file_type, "mime"):
            mime_type = file_type.mime
        else:
            mime_type = "text/plain"
        logger.info(f"Creating chunks for multivector embedding for file type {mime_type}")

        # If file_type is None, attempt a light-weight heuristic to detect images
        # Some JPGs with uncommon EXIF markers fail `filetype.guess`, leading to
        # false "text" classification and, eventually, empty chunk lists. Try to
        # open the bytes with Pillow; if that succeeds, treat it as an image.
        if file_type is None:
            try:
                # PILImage is already imported at the top of the file
                PILImage.open(BytesIO(file_content)).verify()
                logger.info("Heuristic image detection succeeded (Pillow). Treating as image.")
                if file_content_base64 is None:
                    file_content_base64 = base64.b64encode(file_content).decode()
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type="image/unknown",
                        base64_override=file_content_base64,
                    )
                ]
            except Exception:
                logger.info("File type is None and not an image – treating as text")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

        # Treat any direct image MIME (e.g. "image/jpeg") as an image regardless of
        # the more specialised pattern matching below. This is more robust for files
        # where `filetype.guess` fails but we still know from the upload metadata that
        # it is an image.
        if mime_type.startswith("image/"):
            try:
                img = PILImage.open(BytesIO(file_content))
                # Resize and compress aggressively to minimize context window footprint
                max_width = 256  # reduce width to shrink payload dramatically
                if img.width > max_width:
                    ratio = max_width / float(img.width)
                    new_height = int(float(img.height) * ratio)
                    img = img.resize((max_width, new_height))

                buffered = BytesIO()
                # Save as JPEG with moderate quality instead of PNG to reduce size further
                img.convert("RGB").save(buffered, format="JPEG", quality=70, optimize=True)
                jpeg_bytes = buffered.getvalue()
                img_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
                return [
                    self._image_bytes_to_chunk(
                        jpeg_bytes,
                        mime_type="image/jpeg",
                        base64_override=img_b64,
                    )
                ]
            except Exception as e:
                logger.error(f"Error resizing image for base64 encoding: {e}. Falling back to original size.")
                if file_content_base64 is None:
                    file_content_base64 = base64.b64encode(file_content).decode()
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type=mime_type,
                        base64_override=file_content_base64,
                    )
                ]

        match mime_type:
            case file_type if file_type in IMAGE:
                if file_content_base64 is None:
                    file_content_base64 = base64.b64encode(file_content).decode()
                detected_mime = mime_type if isinstance(mime_type, str) else "image/unknown"
                return [
                    self._image_bytes_to_chunk(
                        file_content,
                        mime_type=detected_mime,
                        base64_override=file_content_base64,
                    )
                ]
            case "application/pdf":
                logger.info("Working with PDF file - using PyMuPDF for faster processing!")

                if not file_content:
                    logger.error("PDF file content is empty")
                    raise PdfConversionError("PDF file content is empty")

                try:
                    dpi = int(os.getenv("COLPALI_PDF_DPI", "150"))
                except Exception:
                    dpi = 150

                try:
                    images_with_bytes = self._render_pdf_with_pymupdf(file_content, dpi, include_bytes=True)
                    logger.info(f"PyMuPDF processed {len(images_with_bytes)} pages")
                    return [
                        self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                        for image_b64, raw_bytes in images_with_bytes
                    ]

                except Exception as e:
                    logger.warning(f"PyMuPDF failed ({e}), falling back to pdf2image")

                    try:
                        images = pdf2image.convert_from_bytes(file_content, dpi=dpi)
                        image_payloads = [self.img_to_base64_with_bytes(image) for image in images]
                        logger.info(f"pdf2image fallback processed {len(image_payloads)} pages")
                        return [
                            self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                            for image_b64, raw_bytes in image_payloads
                        ]
                    except Exception as fallback_error:
                        logger.error(f"pdf2image fallback failed: {fallback_error}")
                        raise PdfConversionError(
                            f"Unable to convert PDF to images: {fallback_error}"
                        ) from fallback_error
            case "application/vnd.openxmlformats-officedocument.wordprocessingml.document" | "application/msword":
                logger.info("Working with Word document!")

                # Check if file content is empty
                if not file_content or len(file_content) == 0:
                    logger.error("Word document content is empty")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]

                # Convert Word document to PDF first
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_docx:
                    temp_docx.write(file_content)
                    temp_docx_path = temp_docx.name

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                    temp_pdf_path = temp_pdf.name

                try:
                    # Convert Word to PDF
                    import subprocess

                    # Get the base filename without extension
                    base_filename = os.path.splitext(os.path.basename(temp_docx_path))[0]
                    output_dir = os.path.dirname(temp_pdf_path)
                    expected_pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")

                    result = subprocess.run(
                        [
                            "soffice",
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            output_dir,
                            temp_docx_path,
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode != 0:
                        logger.error(f"Failed to convert Word to PDF: {result.stderr}")
                        return [
                            Chunk(
                                content=chunk.content,
                                metadata=(chunk.metadata | {"is_image": False}),
                            )
                            for chunk in chunks
                        ]

                    # LibreOffice creates the PDF with the same base name in the output directory
                    # Check if the expected PDF file exists
                    if not os.path.exists(expected_pdf_path) or os.path.getsize(expected_pdf_path) == 0:
                        logger.error(f"Generated PDF is empty or doesn't exist at expected path: {expected_pdf_path}")
                        return [
                            Chunk(
                                content=chunk.content,
                                metadata=(chunk.metadata | {"is_image": False}),
                            )
                            for chunk in chunks
                        ]

                    # Now process the PDF using the correct path
                    with open(expected_pdf_path, "rb") as pdf_file:
                        pdf_content = pdf_file.read()

                    try:
                        # Use PyMuPDF for PDF processing (faster than pdf2image)
                        try:
                            pdf_document = fitz.open("pdf", pdf_content)
                            images_payload: List[Tuple[str, bytes]] = []

                            # Process each page individually
                            for page_num in range(len(pdf_document)):
                                page = pdf_document[page_num]
                                try:
                                    dpi = int(os.getenv("COLPALI_PDF_DPI", "150"))
                                except Exception:
                                    dpi = 150
                                mat = fitz.Matrix(dpi / 72, dpi / 72)
                                pix = page.get_pixmap(matrix=mat)
                                img_data = pix.tobytes("png")

                                # Convert to PIL Image and then to base64
                                img = PILImage.open(BytesIO(img_data))
                                img_str, img_bytes = self.img_to_base64_with_bytes(img)
                                images_payload.append((img_str, img_bytes))

                            pdf_document.close()  # Clean up resources

                        except Exception as pymupdf_error:
                            # Fallback to pdf2image if PyMuPDF fails
                            logger.warning(
                                f"PyMuPDF failed for Word document ({pymupdf_error}), falling back to pdf2image"
                            )
                            images = pdf2image.convert_from_bytes(pdf_content)
                            if not images:
                                logger.warning("No images extracted from PDF")
                                return [
                                    Chunk(
                                        content=chunk.content,
                                        metadata=(chunk.metadata | {"is_image": False}),
                                    )
                                    for chunk in chunks
                                ]
                            images_payload = [self.img_to_base64_with_bytes(image) for image in images]

                        if not images_payload:
                            logger.warning("No images extracted from Word document PDF")
                            return [
                                Chunk(
                                    content=chunk.content,
                                    metadata=(chunk.metadata | {"is_image": False}),
                                )
                                for chunk in chunks
                            ]

                        logger.info(f"Word document processed {len(images_payload)} pages")
                        return [
                            self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                            for image_b64, raw_bytes in images_payload
                        ]
                    except Exception as pdf_error:
                        logger.error(f"Error converting PDF to images: {str(pdf_error)}")
                        return [
                            Chunk(
                                content=chunk.content,
                                metadata=(chunk.metadata | {"is_image": False}),
                            )
                            for chunk in chunks
                        ]
                except Exception as e:
                    logger.error(f"Error processing Word document: {str(e)}")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]
                finally:
                    # Clean up temporary files
                    if os.path.exists(temp_docx_path):
                        os.unlink(temp_docx_path)
                    if os.path.exists(temp_pdf_path):
                        os.unlink(temp_pdf_path)
                    # Also clean up the expected PDF path if it exists and is different from temp_pdf_path
                    if (
                        "expected_pdf_path" in locals()
                        and os.path.exists(expected_pdf_path)
                        and expected_pdf_path != temp_pdf_path
                    ):
                        os.unlink(expected_pdf_path)

            # PowerPoint presentations
            case (
                "application/vnd.ms-powerpoint"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
            ):
                logger.info("Working with PowerPoint presentation!")

                # Check if file content is empty
                if not file_content or len(file_content) == 0:
                    logger.error("PowerPoint presentation content is empty")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]

                # Try to convert to images, but fall back to text if LibreOffice is not available
                try:
                    # Check if LibreOffice is available
                    import shutil
                    import subprocess

                    if not shutil.which("soffice"):
                        logger.warning(
                            "LibreOffice (soffice) not found in PATH. Falling back to text extraction for PowerPoint."
                        )
                        logger.info(
                            "To enable visual PowerPoint processing, install LibreOffice: apt-get install libreoffice"
                        )
                        return [
                            Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                            for chunk in chunks
                        ]

                    # Determine file extension based on MIME type
                    if mime_type == "application/vnd.ms-powerpoint":
                        suffix = ".ppt"
                    else:
                        suffix = ".pptx"

                    # Convert PowerPoint to PDF first using LibreOffice
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_ppt:
                        temp_ppt.write(file_content)
                        temp_ppt_path = temp_ppt.name

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                        temp_pdf_path = temp_pdf.name

                    try:
                        # Get the base filename without extension
                        base_filename = os.path.splitext(os.path.basename(temp_ppt_path))[0]
                        output_dir = os.path.dirname(temp_pdf_path)
                        expected_pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")

                        # Convert PowerPoint to PDF with timeout
                        result = subprocess.run(
                            [
                                "soffice",
                                "--headless",
                                "--convert-to",
                                "pdf",
                                "--outdir",
                                output_dir,
                                temp_ppt_path,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,  # 30 second timeout
                        )

                        if result.returncode != 0:
                            logger.warning(f"LibreOffice conversion failed for PowerPoint: {result.stderr}")
                            logger.info("Falling back to text extraction for PowerPoint")
                            return [
                                Chunk(
                                    content=chunk.content,
                                    metadata=(chunk.metadata | {"is_image": False}),
                                )
                                for chunk in chunks
                            ]

                        # Check if the expected PDF file exists
                        if not os.path.exists(expected_pdf_path) or os.path.getsize(expected_pdf_path) == 0:
                            logger.warning(f"Generated PDF is empty or doesn't exist at: {expected_pdf_path}")
                            logger.info("Falling back to text extraction for PowerPoint")
                            return [
                                Chunk(
                                    content=chunk.content,
                                    metadata=(chunk.metadata | {"is_image": False}),
                                )
                                for chunk in chunks
                            ]

                        # Now process the PDF
                        with open(expected_pdf_path, "rb") as pdf_file:
                            pdf_content = pdf_file.read()

                        try:
                            # Use PyMuPDF for PDF processing
                            pdf_document = fitz.open("pdf", pdf_content)
                            images_payload: List[Tuple[str, bytes]] = []

                            # Process each slide as an image
                            for page_num in range(len(pdf_document)):
                                page = pdf_document[page_num]
                                try:
                                    dpi = int(os.getenv("COLPALI_PDF_DPI", "150"))
                                except Exception:
                                    dpi = 150
                                mat = fitz.Matrix(dpi / 72, dpi / 72)
                                pix = page.get_pixmap(matrix=mat)
                                img_data = pix.tobytes("png")

                                # Convert to PIL Image and then to base64
                                img = PILImage.open(BytesIO(img_data))
                                img_str, img_bytes = self.img_to_base64_with_bytes(img)
                                images_payload.append((img_str, img_bytes))

                            pdf_document.close()

                            logger.info(
                                f"PowerPoint presentation successfully processed {len(images_payload)} slides as images"
                            )
                            return [
                                self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                                for image_b64, raw_bytes in images_payload
                            ]

                        except Exception as pymupdf_error:
                            # Fallback to pdf2image if PyMuPDF fails
                            logger.warning(f"PyMuPDF failed for PowerPoint ({pymupdf_error}), trying pdf2image")
                            try:
                                images = pdf2image.convert_from_bytes(pdf_content)
                                images_payload = [self.img_to_base64_with_bytes(image) for image in images]

                                logger.info(
                                    f"PowerPoint presentation processed {len(images_payload)} slides with pdf2image"
                                )
                                return [
                                    self._image_bytes_to_chunk(
                                        raw_bytes, mime_type="image/png", base64_override=image_b64
                                    )
                                    for image_b64, raw_bytes in images_payload
                                ]
                            except Exception as pdf2image_error:
                                logger.warning(f"pdf2image also failed: {pdf2image_error}")
                                logger.info("Falling back to text extraction for PowerPoint")
                                return [
                                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                                    for chunk in chunks
                                ]

                    except subprocess.TimeoutExpired:
                        logger.warning("LibreOffice conversion timed out for PowerPoint")
                        logger.info("Falling back to text extraction")
                        return [
                            Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                            for chunk in chunks
                        ]
                    except Exception as conversion_error:
                        logger.warning(f"Error during PowerPoint conversion: {str(conversion_error)}")
                        logger.info("Falling back to text extraction for PowerPoint")
                        return [
                            Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                            for chunk in chunks
                        ]
                    finally:
                        # Clean up temporary files
                        try:
                            if "temp_ppt_path" in locals() and os.path.exists(temp_ppt_path):
                                os.unlink(temp_ppt_path)
                            if "temp_pdf_path" in locals() and os.path.exists(temp_pdf_path):
                                os.unlink(temp_pdf_path)
                            if (
                                "expected_pdf_path" in locals()
                                and os.path.exists(expected_pdf_path)
                                and expected_pdf_path != temp_pdf_path
                            ):
                                os.unlink(expected_pdf_path)
                        except Exception as cleanup_error:
                            logger.debug(f"Error cleaning up temporary files: {cleanup_error}")

                except Exception as e:
                    logger.warning(f"Unexpected error processing PowerPoint presentation: {str(e)}")
                    logger.info("Falling back to text extraction for PowerPoint")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]

            # Excel spreadsheets
            case (
                "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.ms-excel.sheet.macroEnabled.12"
            ):
                logger.info("Working with Excel spreadsheet!")

                # Check if file content is empty
                if not file_content or len(file_content) == 0:
                    logger.error("Excel spreadsheet content is empty")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]

                # Try to convert to images, but fall back to text if LibreOffice is not available
                try:
                    # Check if LibreOffice is available
                    import shutil
                    import subprocess

                    if not shutil.which("soffice"):
                        logger.warning(
                            "LibreOffice (soffice) not found in PATH. Falling back to text extraction for Excel."
                        )
                        logger.info(
                            "To enable visual Excel processing, install LibreOffice: apt-get install libreoffice"
                        )
                        return [
                            Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                            for chunk in chunks
                        ]

                    # Determine file extension based on MIME type
                    if mime_type == "application/vnd.ms-excel":
                        suffix = ".xls"
                    else:
                        suffix = ".xlsx"

                    # Convert Excel to PDF first using LibreOffice
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_excel:
                        temp_excel.write(file_content)
                        temp_excel_path = temp_excel.name

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                        temp_pdf_path = temp_pdf.name

                    try:
                        # Get the base filename without extension
                        base_filename = os.path.splitext(os.path.basename(temp_excel_path))[0]
                        output_dir = os.path.dirname(temp_pdf_path)
                        expected_pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")

                        # Convert Excel to PDF with timeout
                        result = subprocess.run(
                            [
                                "soffice",
                                "--headless",
                                "--convert-to",
                                "pdf",
                                "--outdir",
                                output_dir,
                                temp_excel_path,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,  # 30 second timeout
                        )

                        if result.returncode != 0:
                            logger.warning(f"LibreOffice conversion failed for Excel: {result.stderr}")
                            logger.info("Falling back to text extraction for Excel")
                            return [
                                Chunk(
                                    content=chunk.content,
                                    metadata=(chunk.metadata | {"is_image": False}),
                                )
                                for chunk in chunks
                            ]

                        # Check if the expected PDF file exists
                        if not os.path.exists(expected_pdf_path) or os.path.getsize(expected_pdf_path) == 0:
                            logger.warning(f"Generated PDF is empty or doesn't exist at: {expected_pdf_path}")
                            logger.info("Falling back to text extraction for Excel")
                            return [
                                Chunk(
                                    content=chunk.content,
                                    metadata=(chunk.metadata | {"is_image": False}),
                                )
                                for chunk in chunks
                            ]

                        # Now process the PDF
                        with open(expected_pdf_path, "rb") as pdf_file:
                            pdf_content = pdf_file.read()

                        try:
                            # Use PyMuPDF for PDF processing
                            pdf_document = fitz.open("pdf", pdf_content)
                            images_payload: List[Tuple[str, bytes]] = []

                            # Process each page/sheet as an image
                            for page_num in range(len(pdf_document)):
                                page = pdf_document[page_num]
                                try:
                                    dpi = int(os.getenv("COLPALI_PDF_DPI", "150"))
                                except Exception:
                                    dpi = 150
                                mat = fitz.Matrix(dpi / 72, dpi / 72)
                                pix = page.get_pixmap(matrix=mat)
                                img_data = pix.tobytes("png")

                                # Convert to PIL Image and then to base64
                                img = PILImage.open(BytesIO(img_data))
                                img_str, img_bytes = self.img_to_base64_with_bytes(img)
                                images_payload.append((img_str, img_bytes))

                            pdf_document.close()

                            logger.info(
                                f"Excel spreadsheet successfully processed {len(images_payload)} pages as images"
                            )
                            return [
                                self._image_bytes_to_chunk(raw_bytes, mime_type="image/png", base64_override=image_b64)
                                for image_b64, raw_bytes in images_payload
                            ]

                        except Exception as pymupdf_error:
                            # Fallback to pdf2image if PyMuPDF fails
                            logger.warning(f"PyMuPDF failed for Excel ({pymupdf_error}), trying pdf2image")
                            try:
                                images = pdf2image.convert_from_bytes(pdf_content)
                                images_payload = [self.img_to_base64_with_bytes(image) for image in images]

                                logger.info(f"Excel spreadsheet processed {len(images_payload)} pages with pdf2image")
                                return [
                                    self._image_bytes_to_chunk(
                                        raw_bytes, mime_type="image/png", base64_override=image_b64
                                    )
                                    for image_b64, raw_bytes in images_payload
                                ]
                            except Exception as pdf2image_error:
                                logger.warning(f"pdf2image also failed: {pdf2image_error}")
                                logger.info("Falling back to text extraction for Excel")
                                return [
                                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                                    for chunk in chunks
                                ]

                    except subprocess.TimeoutExpired:
                        logger.warning("LibreOffice conversion timed out for Excel")
                        logger.info("Falling back to text extraction")
                        return [
                            Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                            for chunk in chunks
                        ]
                    except Exception as conversion_error:
                        logger.warning(f"Error during Excel conversion: {str(conversion_error)}")
                        logger.info("Falling back to text extraction for Excel")
                        return [
                            Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                            for chunk in chunks
                        ]
                    finally:
                        # Clean up temporary files
                        try:
                            if "temp_excel_path" in locals() and os.path.exists(temp_excel_path):
                                os.unlink(temp_excel_path)
                            if "temp_pdf_path" in locals() and os.path.exists(temp_pdf_path):
                                os.unlink(temp_pdf_path)
                            if (
                                "expected_pdf_path" in locals()
                                and os.path.exists(expected_pdf_path)
                                and expected_pdf_path != temp_pdf_path
                            ):
                                os.unlink(expected_pdf_path)
                        except Exception as cleanup_error:
                            logger.debug(f"Error cleaning up temporary files: {cleanup_error}")

                except Exception as e:
                    logger.warning(f"Unexpected error processing Excel spreadsheet: {str(e)}")
                    logger.info("Falling back to text extraction for Excel")
                    return [
                        Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                        for chunk in chunks
                    ]
            # case file_type if file_type in DOCUMENT:
            #     pass
            case _:
                logger.warning(f"Colpali is not supported for file type {mime_type} - skipping")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

    def _create_chunk_objects(
        self,
        doc_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        start_index: int = 0,
    ) -> List[DocumentChunk]:
        """Helper to create chunk objects

        Note: folder_name and end_user_id are not needed in chunk metadata because:
        1. Filtering by these values happens at the document level in find_authorized_and_filtered_documents
        2. Vector search is only performed on already authorized and filtered documents
        3. This approach is more efficient as it reduces the size of chunk metadata
        """
        chunk_objects: List[DocumentChunk] = []
        for index, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            original_metadata = chunk.metadata or {}
            sanitized_metadata: Dict[str, Any] = {}
            for key, value in original_metadata.items():
                if key == "_image_bytes":
                    continue
                if isinstance(value, (bytes, bytearray, memoryview)):
                    sanitized_metadata[key] = base64.b64encode(bytes(value)).decode()
                else:
                    sanitized_metadata[key] = value
            sanitized_chunk = Chunk(content=chunk.content, metadata=sanitized_metadata)
            chunk_objects.append(
                sanitized_chunk.to_document_chunk(
                    chunk_number=start_index + index,
                    embedding=embedding,
                    document_id=doc_id,
                )
            )
        return chunk_objects

    async def _store_chunks_and_doc(
        self,
        chunk_objects: List[DocumentChunk],
        doc: Document,
        use_colpali: bool = False,
        chunk_objects_multivector: Optional[List[DocumentChunk]] = None,
        is_update: bool = False,
        auth: Optional[AuthContext] = None,
    ) -> List[str]:
        """Helper to store chunks and document"""
        # Add retry logic for vector store operations
        max_retries = 3
        retry_delay = 1.0

        # Helper function to store embeddings with retry
        async def store_with_retry(store, objects, store_name="regular"):
            attempt = 0
            success = False
            result = None
            current_retry_delay = retry_delay

            while attempt < max_retries and not success:
                try:
                    success, result = await store.store_embeddings(objects, auth.app_id if auth else None)
                    if not success:
                        raise Exception(f"Failed to store {store_name} chunk embeddings")
                    return result
                except Exception as e:
                    attempt += 1
                    error_msg = str(e)
                    if "connection was closed" in error_msg or "ConnectionDoesNotExistError" in error_msg:
                        if attempt < max_retries:
                            logger.warning(
                                f"Database connection error during {store_name} embeddings storage "
                                f"(attempt {attempt}/{max_retries}): {error_msg}. "
                                f"Retrying in {current_retry_delay}s..."
                            )
                            await asyncio.sleep(current_retry_delay)
                            # Increase delay for next retry (exponential backoff)
                            current_retry_delay *= 2
                        else:
                            logger.error(
                                f"All {store_name} database connection attempts failed "
                                f"after {max_retries} retries: {error_msg}"
                            )
                            raise Exception(f"Failed to store {store_name} chunk embeddings after multiple retries")
                    else:
                        # For other exceptions, don't retry
                        logger.error(f"Error storing {store_name} embeddings: {error_msg}")
                        raise

        # Store document metadata with retry
        async def store_document_with_retry():
            attempt = 0
            success = False
            current_retry_delay = retry_delay

            while attempt < max_retries and not success:
                try:
                    doc.system_metadata = self._clean_system_metadata(doc.system_metadata)

                    if is_update and auth:
                        # For updates, use update_document, serialize StorageFileInfo into plain dicts
                        updates = {
                            "chunk_ids": doc.chunk_ids,
                            "metadata": doc.metadata,
                            "system_metadata": doc.system_metadata,
                            "filename": doc.filename,
                            "content_type": doc.content_type,
                            "storage_info": doc.storage_info,
                            "storage_files": (
                                [
                                    (
                                        file.model_dump()
                                        if hasattr(file, "model_dump")
                                        else (file.dict() if hasattr(file, "dict") else file)
                                    )
                                    for file in doc.storage_files
                                ]
                                if doc.storage_files
                                else []
                            ),
                        }
                        success = await self.db.update_document(doc.external_id, updates, auth)
                        if not success:
                            raise Exception("Failed to update document metadata")
                    else:
                        # For new documents, use store_document
                        success = await self.db.store_document(doc, auth)
                        if not success:
                            raise Exception("Failed to store document metadata")
                    return success
                except Exception as e:
                    attempt += 1
                    error_msg = str(e)
                    if "connection was closed" in error_msg or "ConnectionDoesNotExistError" in error_msg:
                        if attempt < max_retries:
                            logger.warning(
                                f"Database connection error during document metadata storage "
                                f"(attempt {attempt}/{max_retries}): {error_msg}. "
                                f"Retrying in {current_retry_delay}s..."
                            )
                            await asyncio.sleep(current_retry_delay)
                            # Increase delay for next retry (exponential backoff)
                            current_retry_delay *= 2
                        else:
                            logger.error(
                                f"All database connection attempts failed " f"after {max_retries} retries: {error_msg}"
                            )
                            raise Exception("Failed to store document metadata after multiple retries")
                    else:
                        # For other exceptions, don't retry
                        logger.error(f"Error storing document metadata: {error_msg}")
                        raise

        # Store in the appropriate vector store based on use_colpali
        if use_colpali and self.colpali_vector_store and chunk_objects_multivector:
            # Store only in ColPali vector store when ColPali is enabled
            chunk_ids = await store_with_retry(self.colpali_vector_store, chunk_objects_multivector, "colpali")
        else:
            # Store in regular vector store when ColPali is not enabled
            chunk_ids = await store_with_retry(self.vector_store, chunk_objects, "regular")

        doc.chunk_ids = chunk_ids

        logger.debug(f"Stored chunk embeddings in vector stores: {len(doc.chunk_ids)} chunks total")

        # Store document metadata (this must be done after chunk storage)
        await store_document_with_retry()

        logger.debug("Stored document metadata in database")
        logger.debug(f"Chunk IDs stored: {doc.chunk_ids}")
        return doc.chunk_ids

    async def _create_chunk_results(self, auth: AuthContext, chunks: List[DocumentChunk]) -> List[ChunkResult]:
        """Create ChunkResult objects with document metadata."""
        results = []
        if not chunks:
            logger.info("No chunks provided, returning empty results")
            return results

        # Collect all unique document IDs from chunks
        unique_doc_ids = list({chunk.document_id for chunk in chunks})

        # Fetch all required documents in a single batch query
        docs = await self.batch_retrieve_documents(unique_doc_ids, auth)

        # Create a lookup dictionary of documents by ID
        doc_map = {doc.external_id: doc for doc in docs}
        logger.debug(f"Retrieved metadata for {len(doc_map)} unique documents in a single batch")

        # Create chunk results using the lookup dictionaries
        for chunk in chunks:
            doc = doc_map.get(chunk.document_id)
            if not doc:
                logger.warning(f"Document {chunk.document_id} not found")
                continue

            # Start with document metadata, then merge in chunk-specific metadata
            metadata = doc.metadata.copy()
            # Add all chunk metadata (this includes our XML metadata like unit, xml_id, breadcrumbs, etc.)
            metadata.update(chunk.metadata)
            # Ensure is_image is set (fallback to False if not present)
            metadata["is_image"] = chunk.metadata.get("is_image", False)
            results.append(
                ChunkResult(
                    content=chunk.content,
                    score=chunk.score,
                    document_id=chunk.document_id,
                    chunk_number=chunk.chunk_number,
                    metadata=metadata,
                    content_type=doc.content_type,
                    filename=doc.filename,
                    download_url=None,
                )
            )

        logger.info(f"Created {len(results)} chunk results")
        return results

    async def _create_document_results(self, auth: AuthContext, chunks: List[ChunkResult]) -> Dict[str, DocumentResult]:
        """Group chunks by document and create DocumentResult objects."""
        if not chunks:
            logger.info("No chunks provided, returning empty results")
            return {}

        # Group chunks by document and get highest scoring chunk per doc
        doc_chunks: Dict[str, ChunkResult] = {}
        for chunk in chunks:
            if chunk.document_id not in doc_chunks or chunk.score > doc_chunks[chunk.document_id].score:
                doc_chunks[chunk.document_id] = chunk
        logger.info(f"Grouped chunks into {len(doc_chunks)} documents")

        # Get unique document IDs
        unique_doc_ids = list(doc_chunks.keys())

        # Fetch all documents in a single batch query
        docs = await self.batch_retrieve_documents(unique_doc_ids, auth)

        # Create a lookup dictionary of documents by ID
        doc_map = {doc.external_id: doc for doc in docs}
        logger.debug(f"Retrieved metadata for {len(doc_map)} unique documents in a single batch")

        # Create document results using the lookup dictionaries
        results = {}
        for doc_id, chunk in doc_chunks.items():
            doc = doc_map.get(doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found")
                continue

            # Use chunk content directly; callers can request download URLs explicitly when needed.
            content = DocumentContent(type="string", value=chunk.content, filename=doc.filename)

            results[doc_id] = DocumentResult(
                score=chunk.score,
                document_id=doc_id,
                metadata=doc.metadata,
                content=content,
                additional_metadata=doc.additional_metadata,
            )

        logger.info(f"Created {len(results)} document results")
        return results

    async def create_cache(
        self,
        name: str,
        model: str,
        gguf_file: str,
        docs: List[Document | None],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Create a new cache with specified configuration.

        Args:
            name: Name of the cache to create
            model: Name of the model to use
            gguf_file: Name of the GGUF file to use
            filters: Optional metadata filters for documents to include
            docs: Optional list of specific document IDs to include
        """
        # Create cache metadata
        metadata = {
            "model": model,
            "model_file": gguf_file,
            "filters": filters,
            "docs": [doc.model_dump_json() for doc in docs],
            "storage_info": {
                "bucket": "caches",
                "key": f"{name}_state.pkl",
            },
        }

        # Store metadata in database
        success = await self.db.store_cache_metadata(name, metadata)
        if not success:
            logger.error(f"Failed to store cache metadata for cache {name}")
            return {"success": False, "message": f"Failed to store cache metadata for cache {name}"}

        # Create cache instance
        cache = self.cache_factory.create_new_cache(
            name=name, model=model, model_file=gguf_file, filters=filters, docs=docs
        )
        cache_bytes = cache.saveable_state
        base64_cache_bytes = base64.b64encode(cache_bytes).decode()
        bucket, key = await self.storage.upload_from_base64(
            base64_cache_bytes,
            key=metadata["storage_info"]["key"],
            bucket=metadata["storage_info"]["bucket"],
        )
        return {
            "success": True,
            "message": f"Cache created successfully, state stored in bucket `{bucket}` with key `{key}`",
        }

    async def load_cache(self, name: str) -> bool:
        """Load a cache into memory.

        Args:
            name: Name of the cache to load

        Returns:
            bool: Whether the cache exists and was loaded successfully
        """
        try:
            # Get cache metadata from database
            metadata = await self.db.get_cache_metadata(name)
            if not metadata:
                logger.error(f"No metadata found for cache {name}")
                return False

            # Get cache bytes from storage
            cache_bytes = await self.storage.download_file(
                metadata["storage_info"]["bucket"], "caches/" + metadata["storage_info"]["key"]
            )
            cache_bytes = cache_bytes.read()
            cache = self.cache_factory.load_cache_from_bytes(name=name, cache_bytes=cache_bytes, metadata=metadata)
            self.active_caches[name] = cache
            return {"success": True, "message": "Cache loaded successfully"}
        except Exception as e:
            logger.error(f"Failed to load cache {name}: {e}")
            # raise e
            return {"success": False, "message": f"Failed to load cache {name}: {e}"}

    async def update_document(
        self,
        document_id: str,
        auth: AuthContext,
        content: Optional[str] = None,
        file: Optional[UploadFile] = None,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> Optional[Document]:
        """
        Update a document with new content and/or metadata using the specified strategy.

        Args:
            document_id: ID of the document to update
            auth: Authentication context
            content: The new text content to add (either content or file must be provided)
            file: File to add (either content or file must be provided)
            filename: Optional new filename for the document
            metadata: Additional metadata to update
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document ('add' to append content)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Updated document if successful, None if failed
        """
        # Validate permissions and get document
        doc = await self._validate_update_access(document_id, auth)
        if not doc:
            return None

        # Get current content and determine update type
        raw_current_content = doc.system_metadata.get("content", "")
        current_content = self._normalize_document_content(raw_current_content)
        if current_content != raw_current_content:
            logger.warning(
                "Normalized non-textual stored content for document %s before applying update strategy",
                doc.external_id,
            )
            doc.system_metadata["content"] = current_content
        metadata_only_update = content is None and file is None and metadata is not None

        # Process content based on update type
        update_content = None
        file_content = None
        file_type = None
        file_content_base64 = None
        if content is not None:
            update_content = await self._process_text_update(content, doc, filename, metadata, rules)
            update_content = self._normalize_document_content(update_content)
        elif file is not None:
            update_content, file_content, file_type, file_content_base64 = await self._process_file_update(
                file, doc, metadata, rules
            )
            update_content = self._normalize_document_content(update_content)
            await self._update_storage_info(doc, file, file_content_base64)

            # ------------------------------------------------------------------
            # Record storage usage for the newly uploaded file (cloud mode)
            # ------------------------------------------------------------------
            settings = get_settings()
            if settings.MODE == "cloud" and auth.user_id:
                try:
                    await check_and_increment_limits(auth, "storage_file", 1)
                    await check_and_increment_limits(auth, "storage_size", len(file_content))
                except Exception as rec_err:  # noqa: BLE001
                    # Do not fail the update on metering issues – just log
                    logger.error("Failed to record storage usage in update_document: %s", rec_err)
        elif not metadata_only_update:
            logger.error("Neither content nor file provided for document update")
            return None

        # Apply content update strategy if we have new content
        if update_content:
            # Fix for initial file upload - if current_content is empty, just use the update_content
            # without trying to use the update strategy (since there's nothing to update)
            if not current_content:
                logger.info(f"No current content found, using only new content of length {len(update_content)}")
                updated_content = update_content
            else:
                updated_content = self._apply_update_strategy(current_content, update_content, update_strategy)
                logger.info(
                    f"Applied update strategy '{update_strategy}': original length={len(current_content)}, "
                    f"new length={len(updated_content)}"
                )

            # Always update the content in system_metadata
            doc.system_metadata["content"] = updated_content
            logger.info(f"Updated system_metadata['content'] with content of length {len(updated_content)}")
        else:
            updated_content = current_content
            logger.info(f"No content update - keeping current content of length {len(current_content)}")

        # Update metadata and version information
        self._update_metadata_and_version(doc, metadata, update_strategy, file)

        # For metadata-only updates, we don't need to re-process chunks
        if metadata_only_update:
            return await self._update_document_metadata_only(doc, auth)

        # Process content into chunks and generate embeddings
        chunks, chunk_objects = await self._process_chunks_and_embeddings(doc.external_id, updated_content, rules)
        if not chunks:
            return None

        # If we have rules processing, the chunks may have modified content
        # Update document content with stitched content from processed chunks
        if rules and chunks:
            chunk_contents = [chunk.content for chunk in chunks]
            stitched_content = "\n".join(chunk_contents)
            # Check if content actually changed
            if stitched_content != updated_content:
                logger.info("Updating document content with stitched content from processed chunks...")
                doc.system_metadata["content"] = stitched_content
                logger.info(f"Updated document content with stitched chunks (length: {len(stitched_content)})")

        # Merge any aggregated metadata from chunk rules
        if hasattr(self, "_last_aggregated_metadata") and self._last_aggregated_metadata:
            logger.info("Merging aggregated chunk metadata into document metadata...")
            # Make sure doc.metadata exists
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(self._last_aggregated_metadata)
            logger.info(f"Final document metadata after merge: {doc.metadata}")
            # Clear the temporary metadata
            self._last_aggregated_metadata = {}

        # Handle colpali (multi-vector) embeddings if needed
        chunk_objects_multivector = await self._process_colpali_embeddings(
            use_colpali, doc.external_id, chunks, file, file_type, file_content, file_content_base64
        )

        # Store everything - this will replace existing chunks with new ones
        await self._store_chunks_and_doc(
            chunk_objects,
            doc,
            use_colpali,
            chunk_objects_multivector,
            is_update=True,
            auth=auth,
        )
        logger.info(f"Successfully updated document {doc.external_id}")

        return doc

    async def _validate_update_access(self, document_id: str, auth: AuthContext) -> Optional[Document]:
        """Validate user permissions and document access."""
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission")
            raise PermissionError("User does not have write permission")

        # Check if document exists and user has write access
        doc = await self.db.get_document(document_id, auth)
        if not doc:
            logger.error(f"Document {document_id} not found or not accessible")
            return None

        if not await self.db.check_access(document_id, auth, "write"):
            logger.error(f"User {auth.entity_id} does not have write permission for document {document_id}")
            raise PermissionError(f"User does not have write permission for document {document_id}")

        return doc

    async def _process_text_update(
        self,
        content: str,
        doc: Document,
        filename: Optional[str],
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List],
    ) -> str:
        """Process text content updates."""
        update_content = content

        # Update filename if provided
        if filename:
            doc.filename = filename

        # Apply post_parsing rules if provided
        if rules:
            logger.info("Applying post-parsing rules to text update...")
            rule_metadata, modified_content = await self.rules_processor.process_document_rules(content, rules)
            # Update metadata with extracted metadata from rules
            if metadata is not None:
                metadata.update(rule_metadata)

            update_content = modified_content
            logger.info(f"Content length after post-parsing rules: {len(update_content)}")

        return update_content

    async def _process_file_update(
        self,
        file: UploadFile,
        doc: Document,
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List],
    ) -> tuple[str, bytes, Any, str]:
        """Process file content updates."""
        # Read file content
        file_content = await file.read()

        # Parse the file content
        additional_file_metadata, file_text = await self.parser.parse_file_to_text(file_content, file.filename)
        logger.info(f"Parsed file into text of length {len(file_text)}")

        # Apply post_parsing rules if provided for file content
        if rules:
            logger.info("Applying post-parsing rules to file update...")
            rule_metadata, modified_text = await self.rules_processor.process_document_rules(file_text, rules)
            # Update metadata with extracted metadata from rules
            if metadata is not None:
                metadata.update(rule_metadata)

            file_text = modified_text
            logger.info(f"File content length after post-parsing rules: {len(file_text)}")

        # Add additional metadata from file if available
        if additional_file_metadata:
            if not doc.additional_metadata:
                doc.additional_metadata = {}
            doc.additional_metadata.update(additional_file_metadata)

        # Store file in storage if needed
        file_content_base64 = base64.b64encode(file_content).decode()

        # Store file in storage and update storage info
        await self._update_storage_info(doc, file, file_content_base64)

        # Store file type
        file_type = filetype.guess(file_content)
        if file_type:
            doc.content_type = file_type.mime
        else:
            # If filetype.guess failed, try to determine from filename
            import mimetypes

            guessed_type = mimetypes.guess_type(file.filename)[0]
            if guessed_type:
                doc.content_type = guessed_type
            else:
                # Default fallback
                doc.content_type = "text/plain" if file.filename.endswith(".txt") else "application/octet-stream"

        # Update filename
        doc.filename = file.filename

        return file_text, file_content, file_type, file_content_base64

    async def _update_storage_info(self, doc: Document, file: UploadFile, file_content_base64: str):
        """Update document storage information for file content."""
        # Initialize storage_files array if needed - using the passed doc object directly
        # No need to refetch from the database as we already have the full document state
        if not hasattr(doc, "storage_files") or not doc.storage_files:
            # Initialize empty list
            doc.storage_files = []

            # If storage_files is empty but we have storage_info, migrate legacy data
            if doc.storage_info and doc.storage_info.get("bucket") and doc.storage_info.get("key"):
                # Create StorageFileInfo from storage_info
                legacy_file_info = StorageFileInfo(
                    bucket=doc.storage_info.get("bucket", ""),
                    key=doc.storage_info.get("key", ""),
                    version=1,
                    filename=doc.filename,
                    content_type=doc.content_type,
                    timestamp=doc.system_metadata.get("updated_at", datetime.now(UTC)),
                )
                doc.storage_files.append(legacy_file_info)
                logger.info(f"Migrated legacy storage_info to storage_files: {doc.storage_files}")

        # Upload the new file with a unique key including version number
        # The version is based on the current length of storage_files to ensure correct versioning
        version = len(doc.storage_files) + 1
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""

        # Route file uploads to the dedicated app bucket when available
        bucket_override = await self._get_bucket_for_app(doc.app_id)

        storage_info_tuple = await self.storage.upload_from_base64(
            file_content_base64,
            f"{doc.external_id}_{version}{file_extension}",
            file.content_type,
            bucket=bucket_override or "",
        )

        # Add the new file to storage_files, version is INT
        new_sfi = StorageFileInfo(
            bucket=storage_info_tuple[0],
            key=storage_info_tuple[1],
            version=version,  # version variable is already an int
            filename=file.filename,
            content_type=file.content_type,
            timestamp=datetime.now(UTC),
        )
        doc.storage_files.append(new_sfi)

        # Still update legacy storage_info (Dict[str, str]) with the latest file, stringifying values
        doc.storage_info = {k: str(v) if v is not None else "" for k, v in new_sfi.model_dump().items()}
        logger.info(f"Stored file in bucket `{storage_info_tuple[0]}` with key `{storage_info_tuple[1]}`")

    @staticmethod
    def _normalize_document_content(content: Any) -> str:
        """Ensure stored document content is always handled as text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("Failed to decode bytes document content using UTF-8; returning base64 encoded fallback")
                return base64.b64encode(content).decode("utf-8")
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                logger.warning(
                    "Failed to serialize %s content to JSON; falling back to string conversion",
                    type(content).__name__,
                )
                return str(content)
        logger.warning("Coercing unexpected content type %s to string", type(content).__name__)
        return str(content)

    def _apply_update_strategy(self, current_content: str, update_content: str, update_strategy: str) -> str:
        """Apply the update strategy to combine current and new content."""
        if update_strategy == "add":
            # Append the new content
            return current_content + "\n\n" + update_content
        else:
            # For now, just use 'add' as default strategy
            logger.warning(f"Unknown update strategy '{update_strategy}', defaulting to 'add'")
            return current_content + "\n\n" + update_content

    async def _update_document_metadata_only(self, doc: Document, auth: AuthContext) -> Optional[Document]:
        """Update document metadata without reprocessing chunks."""
        doc.system_metadata = self._clean_system_metadata(doc.system_metadata)

        updates = {
            "metadata": doc.metadata,
            "system_metadata": doc.system_metadata,
            "filename": doc.filename,
            "storage_files": doc.storage_files if hasattr(doc, "storage_files") else None,
            "storage_info": doc.storage_info if hasattr(doc, "storage_info") else None,
        }
        # Remove None values
        updates = {k: v for k, v in updates.items() if v is not None}

        success = await self.db.update_document(doc.external_id, updates, auth)
        if not success:
            logger.error(f"Failed to update document {doc.external_id} metadata")
            return None

        logger.info(f"Successfully updated document metadata for {doc.external_id}")
        return doc

    async def _process_chunks_and_embeddings(
        self, doc_id: str, content: str, rules: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[List[Chunk], List[DocumentChunk]]:
        """Process content into chunks and generate embeddings."""
        # Split content into chunks
        parsed_chunks = await self.parser.split_text(content)
        if not parsed_chunks:
            logger.error("No content chunks extracted after update")
            return None, None

        logger.info(f"Split updated text into {len(parsed_chunks)} chunks")

        # Apply post_chunking rules and aggregate metadata if provided
        processed_chunks = []
        aggregated_chunk_metadata: Dict[str, Any] = {}  # Initialize dict for aggregated metadata
        chunk_contents = []  # Initialize list to collect chunk contents efficiently

        if rules:
            logger.info("Applying post-chunking rules...")

            for chunk_obj in parsed_chunks:
                # Get metadata *and* the potentially modified chunk
                chunk_rule_metadata, processed_chunk = await self.rules_processor.process_chunk_rules(chunk_obj, rules)
                processed_chunks.append(processed_chunk)
                chunk_contents.append(processed_chunk.content)  # Collect content as we process
                # Aggregate the metadata extracted from this chunk
                aggregated_chunk_metadata.update(chunk_rule_metadata)
            logger.info(f"Finished applying post-chunking rules to {len(processed_chunks)} chunks.")
            logger.info(f"Aggregated metadata from all chunks: {aggregated_chunk_metadata}")

            # Return this metadata so the calling method can update the document metadata
            self._last_aggregated_metadata = aggregated_chunk_metadata
        else:
            processed_chunks = parsed_chunks  # No rules, use original chunks
            self._last_aggregated_metadata = {}

        # Generate embeddings for processed chunks
        embeddings = await self.embedding_model.embed_for_ingestion(processed_chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Create new chunk objects
        chunk_objects = self._create_chunk_objects(doc_id, processed_chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")

        return processed_chunks, chunk_objects

    async def _process_colpali_embeddings(
        self,
        use_colpali: bool,
        doc_id: str,
        chunks: List[Chunk],
        file: Optional[UploadFile],
        file_type: Any,
        file_content: Optional[bytes],
        file_content_base64: Optional[str],
    ) -> List[DocumentChunk]:
        """Process colpali multi-vector embeddings if enabled."""
        chunk_objects_multivector = []

        settings = get_settings()
        if not (use_colpali and settings.ENABLE_COLPALI and self.colpali_embedding_model and self.colpali_vector_store):
            return chunk_objects_multivector

        # For file updates, we need special handling for images and PDFs
        # Safely resolve MIME regardless of whether file_type is a Kind object or str
        file_type_mime = (
            file_type if isinstance(file_type, str) else (file_type.mime if file_type is not None else None)
        )
        if file and file_type_mime and (file_type_mime in IMAGE or file_type_mime == "application/pdf"):
            # Rewind the file and read it again if needed
            if hasattr(file, "seek") and callable(file.seek) and not file_content:
                await file.seek(0)
                file_content = await file.read()
                file_content_base64 = base64.b64encode(file_content).decode()

            chunks_multivector = self._create_chunks_multivector(file_type, file_content_base64, file_content, chunks)
            logger.info(f"Created {len(chunks_multivector)} chunks for multivector embedding")
            colpali_embeddings = await self.colpali_embedding_model.embed_for_ingestion(chunks_multivector)
            logger.info(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(doc_id, chunks_multivector, colpali_embeddings)
        else:
            # For text updates or non-image/PDF files
            embeddings_multivector = await self.colpali_embedding_model.embed_for_ingestion(chunks)
            logger.info(f"Generated {len(embeddings_multivector)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(doc_id, chunks, embeddings_multivector)

        logger.info(f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding")
        return chunk_objects_multivector

    async def create_graph(
        self,
        name: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """Create a graph from documents.

        This function processes documents matching filters or specific document IDs,
        extracts entities and relationships from document chunks, and saves them as a graph.

        Args:
            name: Name of the graph to create
            auth: Authentication context
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts
            system_filters: Optional system filters like folder_name and end_user_id for scoping

        Returns:
            Graph: The created graph
        """
        # Delegate to the GraphService
        return await self.graph_service.create_graph(
            name=name,
            auth=auth,
            document_service=self,
            filters=filters,
            documents=documents,
            prompt_overrides=prompt_overrides,
            system_filters=system_filters,
        )

    async def update_graph(
        self,
        name: str,
        auth: AuthContext,
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,
        is_initial_build: bool = False,  # New parameter
    ) -> Graph:
        """Update an existing graph with new documents.

        This function processes additional documents matching the original or new filters,
        extracts entities and relationships, and updates the graph with new information.

        Args:
            name: Name of the graph to update
            auth: Authentication context
            additional_filters: Optional additional metadata filters to determine which new documents to include
            additional_documents: Optional list of additional document IDs to include
            prompt_overrides: Optional customizations for entity extraction and resolution prompts
            system_filters: Optional system filters like folder_name and end_user_id for scoping
            is_initial_build: Whether this is the initial build of the graph

        Returns:
            Graph: The updated graph
        """
        # Delegate to the GraphService
        return await self.graph_service.update_graph(
            name=name,
            auth=auth,
            document_service=self,
            additional_filters=additional_filters,
            additional_documents=additional_documents,
            prompt_overrides=prompt_overrides,
            system_filters=system_filters,
            is_initial_build=is_initial_build,  # Pass through
        )

    async def delete_document(self, document_id: str, auth: AuthContext) -> bool:
        """
        Delete a document and all its associated data.

        This method:
        1. Checks if the user has write access to the document
        2. Gets the document to retrieve its chunk IDs
        3. Deletes the document from the database
        4. Deletes all associated chunks from the vector store (if possible)
        5. Deletes the original file from storage if present

        Args:
            document_id: ID of the document to delete
            auth: Authentication context

        Returns:
            bool: True if deletion was successful, False otherwise

        Raises:
            PermissionError: If the user doesn't have write access
        """
        # First get the document to retrieve its chunk IDs
        document = await self.db.get_document(document_id, auth)

        if not document:
            logger.error(f"Document {document_id} not found")
            return False

        # Verify write access - the database layer also checks this, but we check here too
        # to avoid unnecessary operations if the user doesn't have permission
        if not await self.db.check_access(document_id, auth, "write"):
            logger.error(f"User {auth.entity_id} doesn't have write access to document {document_id}")
            raise PermissionError(f"User doesn't have write access to document {document_id}")

        # Delete document from database
        db_success = await self.db.delete_document(document_id, auth)
        if not db_success:
            logger.error(f"Failed to delete document {document_id} from database")
            return False

        logger.info(f"Deleted document {document_id} from database")

        # Collect storage deletion tasks
        storage_deletion_tasks = []

        # Collect vector store deletion tasks
        vector_deletion_tasks = []

        # Add vector store deletion tasks if chunks exist
        if hasattr(document, "chunk_ids") and document.chunk_ids:
            # Try to delete chunks by document ID
            # Note: Some vector stores may not implement this method
            if hasattr(self.vector_store, "delete_chunks_by_document_id"):
                vector_deletion_tasks.append(self.vector_store.delete_chunks_by_document_id(document_id, auth.app_id))

            # Try to delete from colpali vector store as well
            if self.colpali_vector_store and hasattr(self.colpali_vector_store, "delete_chunks_by_document_id"):
                vector_deletion_tasks.append(
                    self.colpali_vector_store.delete_chunks_by_document_id(document_id, auth.app_id)
                )

        # Collect storage file deletion tasks
        if hasattr(document, "storage_info") and document.storage_info:
            bucket = document.storage_info.get("bucket")
            key = document.storage_info.get("key")
            if bucket and key and hasattr(self.storage, "delete_file"):
                storage_deletion_tasks.append(self.storage.delete_file(bucket, key))

        # Also handle the case of multiple file versions in storage_files
        if hasattr(document, "storage_files") and document.storage_files:
            for file_info in document.storage_files:
                bucket = file_info.bucket
                key = file_info.key
                if bucket and key and hasattr(self.storage, "delete_file"):
                    storage_deletion_tasks.append(self.storage.delete_file(bucket, key))

        # Execute deletion tasks in parallel
        if vector_deletion_tasks or storage_deletion_tasks:
            try:
                # Run all deletion tasks concurrently
                all_deletion_results = await asyncio.gather(
                    *vector_deletion_tasks, *storage_deletion_tasks, return_exceptions=True
                )

                # Log any errors but continue with deletion
                for i, result in enumerate(all_deletion_results):
                    if isinstance(result, Exception):
                        # Determine if this was a vector store or storage deletion
                        task_type = "vector store" if i < len(vector_deletion_tasks) else "storage"
                        logger.error(f"Error during {task_type} deletion for document {document_id}: {result}")

            except Exception as e:
                logger.error(f"Error during parallel deletion operations for document {document_id}: {e}")
                # We continue even if deletions fail - document is already deleted from DB

        logger.info(f"Successfully deleted document {document_id} and all associated data")
        return True

    async def extract_pdf_pages(
        self,
        bucket: str,
        key: str,
        start_page: int,
        end_page: int,
    ) -> Dict[str, Any]:
        """
        Extract specific pages from a PDF document as base64-encoded images.

        Args:
            bucket: Storage bucket containing the PDF
            key: Storage key for the PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)

        Returns:
            Dict containing:
                - pages: List of base64-encoded images
                - total_pages: Total number of pages in the PDF
        """
        try:
            # Download the PDF file from storage
            file_content = await self.storage.download_file(bucket, key)

            # Open PDF directly from bytes using BytesIO
            pdf_stream = BytesIO(file_content)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

            total_pages = len(pdf_document)

            # Always clamp the page numbers to the total number of pages
            start_page = max(1, start_page)
            end_page = min(end_page, total_pages)

            # # Validate page numbers
            # if start_page < 1 or end_page > total_pages:
            #     raise ValueError(f"Page range {start_page}-{end_page} is invalid for PDF with {total_pages} pages")

            # Extract pages as images
            pages_base64 = []
            for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
                page = pdf_document[page_num]

                # Render page as image with high DPI for quality
                matrix = fitz.Matrix(2.0, 2.0)  # 2x scaling for better quality
                pix = page.get_pixmap(matrix=matrix)

                # Convert to PIL Image and save as JPEG for smaller size
                img_data = pix.tobytes("jpeg", jpg_quality=85)  # Use JPEG with good quality
                img = PILImage.open(BytesIO(img_data))

                # Convert to base64
                base64_str = self.img_to_base64_str(img)
                pages_base64.append(base64_str)

            pdf_document.close()

            return {"pages": pages_base64, "total_pages": total_pages}

        except Exception as e:
            logger.error(f"Error extracting PDF pages from {bucket}/{key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF pages: {str(e)}")

    async def extract_presentation_pages(
        self,
        bucket: str,
        key: str,
        filename: str,
        start_page: int,
        end_page: int,
    ) -> Dict[str, Any]:
        """
        Extract specific slides from a PowerPoint presentation as base64-encoded images.

        Converts the presentation to PDF using LibreOffice (soffice) and then renders the
        requested slide range to images via PyMuPDF.

        Args:
            bucket: Storage bucket containing the presentation
            key: Storage key for the presentation file
            filename: Original filename (used to determine extension)
            start_page: Starting slide number (1-indexed)
            end_page: Ending slide number (1-indexed)

        Returns:
            Dict containing:
                - pages: List of base64-encoded images
                - total_pages: Total number of slides
        """
        import shutil
        import subprocess

        try:
            # Ensure LibreOffice is available for conversion
            if not shutil.which("soffice"):
                raise HTTPException(
                    status_code=400,
                    detail="PowerPoint extraction requires LibreOffice (soffice) to be installed on the server",
                )

            # Download the presentation file
            file_content = await self.storage.download_file(bucket, key)
            if not file_content:
                raise HTTPException(status_code=404, detail="Presentation file is empty or missing")

            # Determine suffix from filename
            _, ext = os.path.splitext((filename or "").lower())
            suffix = ".ppt" if ext == ".ppt" else ".pptx"

            temp_ppt_path = None
            temp_pdf_path = None
            expected_pdf_path = None

            # Write the presentation to a temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_ppt:
                temp_ppt.write(file_content)
                temp_ppt_path = temp_ppt.name

            # Create a temporary target path to locate the output directory
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name

            # Compute expected output PDF path
            base_filename = os.path.splitext(os.path.basename(temp_ppt_path))[0]
            output_dir = os.path.dirname(temp_pdf_path)
            expected_pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")

            # Convert presentation to PDF
            result = subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    output_dir,
                    temp_ppt_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"LibreOffice conversion failed: {result.stderr}")
                raise HTTPException(status_code=500, detail="Failed to convert presentation to PDF")

            if not os.path.exists(expected_pdf_path) or os.path.getsize(expected_pdf_path) == 0:
                logger.error(f"Converted PDF missing or empty at: {expected_pdf_path}")
                raise HTTPException(status_code=500, detail="Converted PDF is missing or empty")

            # Read the converted PDF
            with open(expected_pdf_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()

            # Open with PyMuPDF
            pdf_document = fitz.open("pdf", pdf_content)
            total_pages = len(pdf_document)

            # Clamp requested range
            start_page = max(1, start_page)
            end_page = min(end_page, total_pages)

            pages_base64: List[str] = []
            for page_num in range(start_page - 1, end_page):
                page = pdf_document[page_num]
                matrix = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=matrix)
                img_data = pix.tobytes("jpeg", jpg_quality=85)
                img = PILImage.open(BytesIO(img_data))
                pages_base64.append(self.img_to_base64_str(img))

            pdf_document.close()

            return {"pages": pages_base64, "total_pages": total_pages}

        except HTTPException:
            raise
        except subprocess.TimeoutExpired:
            logger.error("LibreOffice conversion timed out for presentation")
            raise HTTPException(status_code=500, detail="Presentation to PDF conversion timed out")
        except Exception as e:
            logger.error(f"Error extracting presentation pages from {bucket}/{key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract presentation pages: {str(e)}")
        finally:
            try:
                if "temp_ppt_path" in locals() and temp_ppt_path and os.path.exists(temp_ppt_path):
                    os.unlink(temp_ppt_path)
                if "temp_pdf_path" in locals() and temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
                if "expected_pdf_path" in locals() and expected_pdf_path and os.path.exists(expected_pdf_path):
                    os.unlink(expected_pdf_path)
            except Exception as cleanup_error:
                logger.debug(f"Cleanup error: {cleanup_error}")

    def close(self):
        """Close all resources."""
        # Close any active caches
        self.active_caches.clear()

    def _update_metadata_and_version(
        self,
        doc: Document,
        metadata: Optional[Dict[str, Any]],
        update_strategy: str,
        file: Optional[UploadFile],
    ):
        """Update document metadata and version tracking."""

        # Merge/replace metadata
        if metadata:
            doc.metadata.update(metadata)

        # Ensure external_id is preserved
        doc.metadata["external_id"] = doc.external_id

        # Increment version counter
        current_version = doc.system_metadata.get("version", 1)
        doc.system_metadata["version"] = current_version + 1
        doc.system_metadata["updated_at"] = datetime.now(UTC)

        # Maintain simple history list
        history = doc.system_metadata.setdefault("update_history", [])
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": current_version + 1,
            "strategy": update_strategy,
        }
        if file:
            entry["filename"] = file.filename
        if metadata:
            entry["metadata_updated"] = True

        history.append(entry)

    # ------------------------------------------------------------------
    # Helper – choose bucket per app (isolation)
    # ------------------------------------------------------------------

    async def _get_bucket_for_app(self, app_id: str | None) -> str | None:
        """Return dedicated bucket for *app_id* if catalog entry exists."""
        if not app_id:
            return None

        try:
            from sqlalchemy import select
            from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
            from sqlalchemy.orm import sessionmaker

            from core.models.app_metadata import AppMetadataModel

            settings = get_settings()

            engine = create_async_engine(settings.POSTGRES_URI)
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session() as sess:
                result = await sess.execute(select(AppMetadataModel).where(AppMetadataModel.id == app_id))
                meta = result.scalars().first()
                if meta and meta.extra and meta.extra.get("s3_bucket"):
                    return meta.extra["s3_bucket"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch bucket for app %s: %s", app_id, exc)
        return None

    async def _upload_to_app_bucket(
        self,
        auth: AuthContext,
        content_bytes: bytes,
        key: str,
        content_type: Optional[str] = None,
    ) -> tuple[str, str]:
        bucket_override = await self._get_bucket_for_app(auth.app_id)
        return await self.storage.upload_file(
            content_bytes,
            key,
            content_type,
            bucket=bucket_override or "",
        )

    async def get_graph_visualization_data(
        self,
        name: str,
        auth: AuthContext,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get graph visualization data.

        Args:
            name: Name of the graph to visualize
            auth: Authentication context
            folder_name: Optional folder name for scoping
            end_user_id: Optional end user ID for scoping

        Returns:
            Dict containing nodes and links for visualization
        """
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Delegate to the GraphService
        return await self.graph_service.get_graph_visualization_data(
            graph_name=name,
            auth=auth,
            system_filters=system_filters,
        )

    async def search_documents_by_name(
        self,
        query: str,
        auth: AuthContext,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> List[Document]:
        """Search documents by filename using full-text search.

        Args:
            query: Search query for document names/filenames
            auth: Authentication context
            limit: Maximum number of documents to return (1-100)
            filters: Optional metadata filters
            folder_name: Optional folder to scope search
            end_user_id: Optional end-user ID to scope search

        Returns:
            List of documents matching the search query, ordered by relevance
        """
        # Build system filters
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Clamp limit to reasonable range
        limit = max(1, min(100, limit))

        # Delegate to database layer
        return await self.db.search_documents_by_name(
            query=query,
            auth=auth,
            limit=limit,
            filters=filters,
            system_filters=system_filters,
        )
