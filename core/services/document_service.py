import asyncio
import base64
import logging
import os
import tempfile
import time
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Type, Union

import fitz  # PyMuPDF for PDF/presentation processing
from fastapi import HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

from core.completion.base_completion import BaseCompletionModel
from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.models.chat import ChatMessage
from core.models.chunk import DocumentChunk
from core.models.completion import ChunkSource, CompletionRequest, CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentContent, DocumentResult
from core.models.prompts import GraphPromptOverrides, QueryPromptOverrides
from core.parser.base_parser import BaseParser
from core.reranker.base_reranker import BaseReranker
from core.services.graph_service import GraphService
from core.services.morphik_graph_service import MorphikGraphService
from core.storage.base_storage import BaseStorage
from core.vector_store.base_vector_store import BaseVectorStore

from ..models.auth import AuthContext
from ..models.graph import Graph
from ..utils.folder_utils import normalize_folder_selector

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
TOKENS_PER_PAGE = 630

settings = get_settings()


class PdfConversionError(Exception):
    """Raised when the service cannot rasterize a PDF into images."""


class DocumentService:
    """Service for document retrieval and query operations.

    Note: Ingestion operations have been moved to IngestionService.
    """

    def __init__(
        self,
        database: PostgresDatabase,
        vector_store: BaseVectorStore,
        storage: BaseStorage,
        parser: BaseParser,
        embedding_model: BaseEmbeddingModel,
        completion_model: Optional[BaseCompletionModel] = None,
        reranker: Optional[BaseReranker] = None,
        enable_colpali: bool = False,
        colpali_embedding_model: Optional[ColpaliEmbeddingModel] = None,
        colpali_vector_store: Optional[BaseVectorStore] = None,
    ):
        self.db = database
        self.vector_store = vector_store
        self.storage = storage
        self.parser = parser
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.reranker = reranker
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

    @staticmethod
    def _normalize_folder_filter(folder_name: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
        """Normalize folder selector to canonical paths."""
        if not folder_name:
            return None
        return normalize_folder_selector(folder_name)

    @staticmethod
    def _build_folder_scope_filters(
        folder_name: Optional[Union[str, List[str]]], folder_depth: Optional[int]
    ) -> Dict[str, Any]:
        """
        Build system_filters entries for folder scoping with optional nesting depth.

        folder_depth semantics:
        - None or 0: exact match only.
        - -1: include all descendants.
        - n > 0: include descendants up to n levels deeper than the base path.
        """
        if folder_name is None:
            return {}

        def _depth(path: str) -> int:
            if path == "/":
                return 0
            return len([p for p in path.strip("/").split("/") if p])

        normalized = normalize_folder_selector(folder_name)
        paths = normalized if isinstance(normalized, list) else [normalized]

        exact_paths: List[Optional[str]] = []
        prefix_paths: List[str] = []
        prefix_depth: List[Dict[str, Any]] = []

        for path in paths:
            if path is None:
                exact_paths.append(None)
                continue

            if folder_depth is None or folder_depth == 0:
                exact_paths.append(path)
                continue

            base_depth = _depth(path)
            if folder_depth < 0:
                prefix_paths.append(path)
                continue

            max_depth = base_depth + folder_depth
            prefix_depth.append({"prefix": path, "max_depth": max_depth})

        filters: Dict[str, Any] = {}
        if prefix_depth:
            filters["folder_path_prefix_depth"] = prefix_depth
        if prefix_paths:
            filters["folder_path_prefix"] = prefix_paths if len(prefix_paths) > 1 else prefix_paths[0]
        if exact_paths:
            filters["folder_path"] = exact_paths if len(exact_paths) > 1 else exact_paths[0]
        return filters

    async def retrieve_chunks(
        self,
        query: Optional[str],
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        folder_depth: Optional[int] = None,
        end_user_id: Optional[str] = None,
        perf_tracker: Optional[Any] = None,  # Performance tracker from API layer
        padding: int = 0,  # Number of additional chunks to retrieve before and after matched chunks
        output_format: str = "base64",
        query_image: Optional[bytes] = None,  # Base64-decoded image bytes for visual search
    ) -> List[ChunkResult]:
        """Retrieve relevant chunks.

        Either query (text) or query_image (image bytes) must be provided.
        Image queries require use_colpali=True for Morphik multimodal retrieval.
        """

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

        # Image queries require Morphik multimodal retrieval (use_colpali=True)
        if query_image and not using_multivector:
            raise HTTPException(
                status_code=400,
                detail="Image queries require use_colpali=True for Morphik multimodal retrieval",
            )

        # Validate image size (max 10MB to prevent memory issues)
        MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
        if query_image and len(query_image) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image size exceeds maximum allowed size of 10MB (got {len(query_image) / (1024 * 1024):.1f}MB)",
            )

        if requested_multivector and not using_multivector:
            logger.warning(
                "Multivector retrieval requested but required components are unavailable. Falling back to regular search."
            )

        # Build system filters for folder_name and end_user_id
        system_filters = self._build_folder_scope_filters(folder_name, folder_depth)
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
            # For image queries, use generate_embeddings with a PIL Image
            if query_image:
                try:
                    query_pil_image = PILImage.open(BytesIO(query_image))
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid or unsupported image format: {e}",
                    )
                embedding_coro = self.colpali_embedding_model.generate_embeddings(query_pil_image)
            else:
                # Text query path - query is guaranteed non-None by request validation
                assert query is not None, "Either query or query_image must be provided"
                embedding_coro = self.colpali_embedding_model.embed_for_query(query)
            embedding_tasks.append(
                measure_phase(
                    embedding_coro,
                    "multivector_query_embedding",
                    "retrieve_embeddings_and_auth",
                )
            )
            multivector_pipeline_start = time.time()
        else:
            # Non-multivector path only supports text queries (image queries require use_colpali=True)
            assert query is not None, "Text query required for non-ColPali retrieval"
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

        results = await self._create_chunk_results(auth, chunks, output_format=output_format)

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
        if local_perf and retrieve_start_time is not None:
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
        query: Optional[str],
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        folder_depth: Optional[int] = None,
        end_user_id: Optional[str] = None,
        perf_tracker: Optional[Any] = None,
        padding: int = 0,
        output_format: str = "base64",
        query_image: Optional[bytes] = None,
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
            folder_depth,
            end_user_id,
            perf_tracker,
            padding=0,  # No padding for original
            output_format=output_format,
            query_image=query_image,
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
                folder_depth,
                end_user_id,
                perf_tracker,
                padding,
                output_format=output_format,
                query_image=query_image,
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
        folder_depth: Optional[int] = None,
        end_user_id: Optional[str] = None,
    ) -> List[DocumentResult]:
        """Retrieve relevant documents."""
        # Get chunks first
        chunks = await self.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali, folder_name, folder_depth, end_user_id
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
        folder_depth: Optional[int] = None,
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
        system_filters = self._build_folder_scope_filters(folder_name, folder_depth)
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
        folder_depth: Optional[int] = None,
        end_user_id: Optional[str] = None,
        use_colpali: Optional[bool] = None,
        output_format: str = "base64",
    ) -> List[ChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation.

        Args:
            chunk_ids: List of ChunkSource objects with document_id and chunk_number
            auth: Authentication context
            folder_name: Optional folder to scope the operation to
            end_user_id: Optional end-user ID to scope the operation to
            use_colpali: Whether to use colpali multimodal features for image chunks
            output_format: How to return image chunks (base64 data or presigned URLs)

        Returns:
            List of ChunkResult objects
        """
        if not chunk_ids:
            return []

        # Collect unique document IDs to check authorization in a single query
        doc_ids = list({source.document_id for source in chunk_ids})

        # Find authorized documents in a single query
        authorized_docs = await self.batch_retrieve_documents(doc_ids, auth, folder_name, folder_depth, end_user_id)
        authorized_doc_map = {doc.external_id: doc for doc in authorized_docs}
        authorized_doc_ids = set(authorized_doc_map.keys())

        # Filter sources to only include authorized documents
        authorized_sources = [source for source in chunk_ids if source.document_id in authorized_doc_ids]

        if not authorized_sources:
            return []

        # Create list of (document_id, chunk_number) tuples for vector store query
        chunk_identifiers: List[Tuple[str, int]] = []
        seen_identifiers: Set[Tuple[str, int]] = set()
        for source in authorized_sources:
            identifier = (source.document_id, source.chunk_number)
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)
            chunk_identifiers.append(identifier)

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
            chunks = vector_results[0] if not isinstance(vector_results[0], BaseException) else []

            # Process colpali chunks if available
            if len(vector_results) > 1 and not isinstance(vector_results[1], BaseException):
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
                if isinstance(result, BaseException):
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
        results = await self._create_chunk_results(
            auth,
            chunks,
            preloaded_docs=authorized_doc_map,
            output_format=output_format,
        )
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
        folder_depth: Optional[int] = None,
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
                folder_depth=folder_depth,
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
            folder_depth,
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

    async def _create_chunk_results(
        self,
        auth: AuthContext,
        chunks: List[DocumentChunk],
        preloaded_docs: Optional[Dict[str, Document]] = None,
        output_format: str = "base64",
    ) -> List[ChunkResult]:
        """Create ChunkResult objects with document metadata."""
        results = []
        if not chunks:
            logger.info("No chunks provided, returning empty results")
            return results

        # Collect all unique document IDs from chunks
        unique_doc_ids = list({chunk.document_id for chunk in chunks})

        # Start with any preloaded documents if provided
        doc_map: Dict[str, Document] = dict(preloaded_docs) if preloaded_docs else {}

        # Fetch any documents that weren't preloaded
        missing_doc_ids = [doc_id for doc_id in unique_doc_ids if doc_id not in doc_map]
        if missing_doc_ids:
            docs = await self.batch_retrieve_documents(missing_doc_ids, auth)
            doc_map.update({doc.external_id: doc for doc in docs})
            logger.debug(f"Retrieved metadata for {len(docs)} additional documents in a single batch")
        else:
            logger.debug(f"Using preloaded metadata for {len(doc_map)} unique documents")

        if not doc_map:
            logger.info("No document metadata available for provided chunks")

        # Lazy import to avoid circular dependency
        try:
            from core.vector_store.multi_vector_store import MULTIVECTOR_CHUNKS_BUCKET
        except Exception:
            MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"

        mime_to_ext = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }

        def _infer_image_mime_from_content(content_str: str) -> Optional[str]:
            """Try to infer an image MIME type from base64 or data URI content.

            Returns a MIME string (e.g., 'image/png') if detection succeeds, otherwise None.
            """
            if not isinstance(content_str, str):
                return None
            # Data URI path
            if content_str.startswith("data:"):
                try:
                    header = content_str.split(",", 1)[0]
                    return header.split(":", 1)[1].split(";", 1)[0]
                except Exception:
                    return None
            # Raw base64 path – attempt to decode and inspect magic bytes
            try:
                raw = base64.b64decode(content_str, validate=False)
            except Exception:
                return None
            if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                return "image/png"
            if raw.startswith(b"\xff\xd8"):
                return "image/jpeg"
            if raw.startswith(b"GIF8"):
                return "image/gif"
            if raw.startswith(b"BM"):
                return "image/bmp"
            # TIFF little/big endian
            if raw.startswith(b"II*\x00") or raw.startswith(b"MM\x00*"):
                return "image/tiff"
            # WEBP: RIFF....WEBP
            if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
                return "image/webp"
            return None

        def _convert_image_to_text(content_str: str) -> str:
            """Convert an image chunk (base64 or data URI) to markdown text using Docling OCR.

            Returns the extracted markdown text, or empty string on failure.
            """
            try:
                # Decode the image to bytes
                if content_str.startswith("data:"):
                    # Data URI format: data:image/png;base64,<data>
                    raw_b64 = content_str.split(",", 1)[1]
                else:
                    raw_b64 = content_str
                image_bytes = base64.b64decode(raw_b64)

                # Save to temp file and process with Docling
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name

                try:
                    from docling.datamodel.base_models import InputFormat
                    from docling.datamodel.pipeline_options import PdfPipelineOptions
                    from docling.document_converter import DocumentConverter, ImageFormatOption

                    # Create converter with OCR enabled for images
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = True
                    converter = DocumentConverter(
                        format_options={
                            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                        }
                    )
                    result = converter.convert(tmp_path)
                    return result.document.export_to_markdown()
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception as e:
                logger.warning(f"Failed to convert image to text: {e}")
                return ""

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
            # Default values
            content_value = chunk.content
            download_url: Optional[str] = None

            # If requested, convert image chunks to presigned URLs or text
            is_img = bool(metadata.get("is_image"))
            mime = chunk.metadata.get("mime_type") if isinstance(chunk.metadata, dict) else None
            # Try to infer from content if metadata was not properly populated
            if not is_img and (output_format or "base64") in ("url", "text"):
                inferred_mime = _infer_image_mime_from_content(chunk.content)
                if inferred_mime:
                    is_img = True
                    if not mime:
                        mime = inferred_mime

            # Handle "text" output format: convert image to markdown text via Docling OCR
            if (output_format or "base64") == "text" and is_img:
                extracted_text = _convert_image_to_text(chunk.content)
                if extracted_text:
                    content_value = extracted_text
                    metadata["is_image"] = False  # Content is now text
                else:
                    # Fallback: keep original base64 if OCR fails
                    logger.warning(f"OCR failed for chunk {chunk.document_id}-{chunk.chunk_number}, keeping base64")

            elif (output_format or "base64") == "url" and is_img:
                try:
                    app_part = doc.app_id or auth.app_id or "default"
                    ext = mime_to_ext.get(mime)
                    if not ext and isinstance(chunk.content, str) and chunk.content.startswith("data:"):
                        try:
                            mime_from_data = chunk.content.split(",", 1)[0].split(":", 1)[1].split(";", 1)[0]
                            ext = mime_to_ext.get(mime_from_data)
                        except Exception:
                            ext = None
                    if not ext:
                        ext = ".png"

                    storage_key = f"{app_part}/{doc.external_id}/{chunk.chunk_number}{ext}"

                    # Choose storage and bucket
                    storage = None
                    bucket_name = MULTIVECTOR_CHUNKS_BUCKET
                    # Prefer the ColPali vector store's storage if available
                    if getattr(self, "colpali_vector_store", None) is not None:
                        if hasattr(self.colpali_vector_store, "storage"):
                            storage = self.colpali_vector_store.storage
                        # Some stores expose a chunk_bucket
                        if hasattr(self.colpali_vector_store, "chunk_bucket") and getattr(
                            self.colpali_vector_store, "chunk_bucket"
                        ):
                            bucket_name = getattr(self.colpali_vector_store, "chunk_bucket")
                    if storage is None:
                        storage = self.storage
                    # If storage has a default_bucket and no explicit chunk_bucket was set, use it
                    if hasattr(storage, "default_bucket") and bucket_name == MULTIVECTOR_CHUNKS_BUCKET:
                        try:
                            dbucket = getattr(storage, "default_bucket")
                            if dbucket:
                                bucket_name = dbucket
                        except Exception:
                            pass

                    # Hotswap: ensure object exists; if missing, convert from base64/data URI and upload
                    if storage is not None:
                        # try:
                        # Check existing object: if missing or not binary image, upload raw bytes
                        existing_bytes: Optional[bytes] = None
                        try:
                            existing_bytes = await storage.download_file(bucket=bucket_name, key=storage_key)
                        except Exception:
                            existing_bytes = None

                        def _is_binary_image(b: bytes) -> bool:
                            return (
                                b.startswith(b"\x89PNG\r\n\x1a\n")
                                or b.startswith(b"\xff\xd8")
                                or b.startswith(b"GIF8")
                                or b.startswith(b"BM")
                                or b.startswith(b"II*\x00")
                                or b.startswith(b"MM\x00*")
                                or (b.startswith(b"RIFF") and b"WEBP" in b[:16])
                            )

                        needs_upload = existing_bytes is None
                        # If a file exists but is not a recognized image binary, we will attempt to convert
                        if existing_bytes is not None and not _is_binary_image(existing_bytes):
                            needs_upload = True

                        if needs_upload:
                            try:
                                # Prepare raw bytes from base64 or data URI and upload as binary
                                payload = chunk.content
                                raw_bytes: Optional[bytes] = None
                                if isinstance(payload, str) and payload.startswith("data:"):
                                    try:
                                        header, base64_part = payload.split(",", 1)
                                        raw_bytes = base64.b64decode(base64_part)
                                    except Exception:
                                        raw_bytes = None
                                if raw_bytes is None and isinstance(payload, str):
                                    try:
                                        raw = base64.b64decode(payload)
                                        # If decoding yields a data URI string, unwrap one more time
                                        try:
                                            as_text = raw.decode("utf-8")
                                            if as_text.strip().startswith("data:") and "," in as_text:
                                                inner_b64 = as_text.split(",", 1)[1]
                                                raw = base64.b64decode(inner_b64)
                                        except Exception:
                                            pass
                                        raw_bytes = raw
                                    except Exception:
                                        raw_bytes = None
                                if raw_bytes is None and existing_bytes is not None:
                                    # Last resort: the existing file might be a data URI string
                                    try:
                                        s = existing_bytes.decode("utf-8", errors="ignore")
                                        if s.startswith("data:"):
                                            raw_bytes = base64.b64decode(s.split(",", 1)[1])
                                        else:
                                            # Attempt plain base64 decode
                                            raw_bytes = base64.b64decode(s)
                                    except Exception:
                                        raw_bytes = None

                                if raw_bytes is None:
                                    raise ValueError("Unable to decode image payload for hotswap upload")

                                effective_mime = mime or _infer_image_mime_from_content(chunk.content) or "image/png"
                                await storage.upload_file(
                                    file=raw_bytes,
                                    key=storage_key,
                                    content_type=effective_mime,
                                    bucket=bucket_name,
                                )
                            except Exception as up_e:
                                logger.warning(
                                    f"Failed to hotswap-upload image for {chunk.document_id}-{chunk.chunk_number}: {up_e}"
                                )

                    if storage is not None and hasattr(storage, "get_download_url"):
                        download_url = await storage.get_download_url(bucket=bucket_name, key=storage_key)
                        if download_url:
                            content_value = download_url
                except Exception as e:
                    logger.warning(
                        f"Failed to create presigned URL for image chunk {chunk.document_id}-{chunk.chunk_number}: {e}"
                    )

            results.append(
                ChunkResult(
                    content=content_value,
                    score=chunk.score,
                    document_id=chunk.document_id,
                    chunk_number=chunk.chunk_number,
                    metadata=metadata,
                    content_type=doc.content_type,
                    filename=doc.filename,
                    download_url=download_url,
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
            logger.error(f"User {auth.user_id} doesn't have write access to document {document_id}")
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

        # Collect storage file deletion task
        if hasattr(document, "storage_info") and document.storage_info:
            bucket = document.storage_info.get("bucket")
            key = document.storage_info.get("key")
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

    # -------------------------------------------------------------------------
    # Image conversion helpers (for page extraction)
    # -------------------------------------------------------------------------

    def img_to_base64_str(self, img: PILImage.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_bytes = buffered.getvalue()
        return "data:image/png;base64," + base64.b64encode(img_bytes).decode()

    # -------------------------------------------------------------------------
    # Page extraction (for document viewing)
    # -------------------------------------------------------------------------

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
        pass

    async def get_graph_visualization_data(
        self,
        name: str,
        auth: AuthContext,
        folder_name: Optional[Union[str, List[str]]] = None,
        folder_depth: Optional[int] = None,
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
        system_filters.update(self._build_folder_scope_filters(folder_name, folder_depth))
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
        folder_depth: Optional[int] = None,
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
        system_filters = self._build_folder_scope_filters(folder_name, folder_depth)
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
