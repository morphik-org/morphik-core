import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Type, Union

import arq
import filetype
import pdf2image
import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from fastapi import HTTPException, UploadFile
from filetype.types import IMAGE  # , DOCUMENT, document
from PIL.Image import Image
from pydantic import BaseModel

from core.cache.base_cache import BaseCache
from core.cache.base_cache_factory import BaseCacheFactory
from core.completion.base_completion import BaseCompletionModel
from core.config import get_settings
from core.database.base_database import BaseDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.models.chunk import Chunk, DocumentChunk
from core.models.completion import ChunkSource, CompletionRequest, CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentContent, DocumentResult, StorageFileInfo
from core.models.prompts import GraphPromptOverrides, QueryPromptOverrides
from core.parser.base_parser import BaseParser
from core.reranker.base_reranker import BaseReranker
from core.services.graph_service import GraphService
from core.services.rules_processor import RulesProcessor
from core.storage.base_storage import BaseStorage
from core.vector_store.base_vector_store import BaseVectorStore
from core.vector_store.multi_vector_store import MultiVectorStore

from ..models.auth import AuthContext
from ..models.folders import Folder
from ..models.graph import Graph

logger = logging.getLogger(__name__)
IMAGE = {im.mime for im in IMAGE}

CHARS_PER_TOKEN = 4
TOKENS_PER_PAGE = 630


class DocumentService:
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

            # First check if the folder already exists
            folder = await self.db.get_folder_by_name(folder_name, auth)
            if folder:
                # Add document to existing folder
                if document_id not in folder.document_ids:
                    success = await self.db.add_document_to_folder(folder.id, document_id, auth)
                    if not success:
                        logger.warning(f"Failed to add document {document_id} to existing folder {folder.name}")
                return folder  # Folder already exists

            # Create a new folder
            folder = Folder(
                name=folder_name,
                owner={
                    "type": auth.entity_type.value,
                    "id": auth.entity_id,
                },
                document_ids=[document_id],  # Add document_id to the new folder
            )

            # Scope folder to the application ID for developer tokens
            if auth.app_id:
                folder.system_metadata["app_id"] = auth.app_id

            await self.db.create_folder(folder)
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
        colpali_embedding_model: Optional[ColpaliEmbeddingModel] = None,
        colpali_vector_store: Optional[MultiVectorStore] = None,
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
            self.graph_service = GraphService(
                db=database,
                embedding_model=embedding_model,
                completion_model=completion_model,
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
    ) -> List[ChunkResult]:
        """Retrieve relevant chunks."""

        # 4 configurations:
        # 1. No reranking, no colpali -> just return regular chunks
        # 2. No reranking, colpali  -> return colpali chunks + regular chunks - no need to run smaller colpali model
        # 3. Reranking, no colpali -> sort regular chunks by re-ranker score
        # 4. Reranking, colpali -> return merged chunks sorted by smaller colpali model score

        settings = get_settings()
        should_rerank = use_reranking if use_reranking is not None else settings.USE_RERANKING
        using_colpali = use_colpali if use_colpali is not None else False

        # Build system filters for folder_name and end_user_id
        system_filters = {}
        if folder_name:
            # Allow folder_name to be a single string or list[str]
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        # Launch embedding queries concurrently
        embedding_tasks = [self.embedding_model.embed_for_query(query)]
        if using_colpali and self.colpali_embedding_model:
            embedding_tasks.append(self.colpali_embedding_model.embed_for_query(query))

        # Run embeddings and document authorization in parallel
        results = await asyncio.gather(
            asyncio.gather(*embedding_tasks),
            self.db.find_authorized_and_filtered_documents(auth, filters, system_filters),
        )

        embedding_results, doc_ids = results
        query_embedding_regular = embedding_results[0]
        query_embedding_multivector = embedding_results[1] if len(embedding_results) > 1 else None

        logger.info("Generated query embedding")

        if not doc_ids:
            logger.info("No authorized documents found")
            return []
        logger.info(f"Found {len(doc_ids)} authorized documents")

        # Check if we're using colpali multivector search
        search_multi = using_colpali and self.colpali_vector_store and query_embedding_multivector is not None

        # For regular reranking (without colpali), we'll use the existing reranker if available
        # For colpali reranking, we'll handle it in _combine_multi_and_regular_chunks
        use_standard_reranker = should_rerank and (not search_multi) and self.reranker is not None

        # Search chunks with vector similarity in parallel
        # When using standard reranker, we get more chunks initially to improve reranking quality
        search_tasks = [
            self.vector_store.query_similar(
                query_embedding_regular, k=10 * k if use_standard_reranker else k, doc_ids=doc_ids
            )
        ]

        if search_multi:
            search_tasks.append(
                self.colpali_vector_store.query_similar(query_embedding_multivector, k=k, doc_ids=doc_ids)
            )

        search_results = await asyncio.gather(*search_tasks)
        chunks = search_results[0]
        chunks_multivector = search_results[1] if len(search_results) > 1 else []

        logger.debug(f"Found {len(chunks)} similar chunks via regular embedding")
        if using_colpali:
            logger.debug(
                f"Found {len(chunks_multivector)} similar chunks via multivector embedding "
                f"since we are also using colpali"
            )

        # Rerank chunks using the standard reranker if enabled and available
        # This handles configuration 3: Reranking without colpali
        if chunks and use_standard_reranker:
            chunks = await self.reranker.rerank(query, chunks)
            chunks.sort(key=lambda x: x.score, reverse=True)
            chunks = chunks[:k]
            logger.debug(f"Reranked {k*10} chunks and selected the top {k}")

        # Combine multiple chunk sources if needed
        chunks = await self._combine_multi_and_regular_chunks(
            query, chunks, chunks_multivector, should_rerank=should_rerank
        )

        # Create and return chunk results
        results = await self._create_chunk_results(auth, chunks)
        logger.info(f"Returning {len(results)} chunk results")
        return results

    async def _combine_multi_and_regular_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        chunks_multivector: List[DocumentChunk],
        should_rerank: bool = None,
    ):
        """Combine and potentially rerank regular and colpali chunks based on configuration.

        # 4 configurations:
        # 1. No reranking, no colpali -> just return regular chunks - this already happens upstream, correctly
        # 2. No reranking, colpali  -> return colpali chunks + regular chunks - no need to run smaller colpali model
        # 3. Reranking, no colpali -> sort regular chunks by re-ranker score - this already happens upstream, correctly
        # 4. Reranking, colpali -> return merged chunks sorted by smaller colpali model score

        Args:
            query: The user query
            chunks: Regular chunks with embeddings
            chunks_multivector: Colpali multi-vector chunks
            should_rerank: Whether reranking is enabled
        """
        # Handle simple cases first
        if len(chunks_multivector) == 0:
            return chunks
        if len(chunks) == 0:
            return chunks_multivector

        # Use global setting if not provided
        if should_rerank is None:
            settings = get_settings()
            should_rerank = settings.USE_RERANKING

        # Check if we need to run the reranking - if reranking is disabled, we just combine the chunks
        # This is Configuration 2: No reranking, with colpali
        if not should_rerank:
            # For configuration 2, simply combine the chunks with multivector chunks first
            # since they are generally higher quality
            logger.debug("Using configuration 2: No reranking, with colpali - combining chunks without rescoring")
            combined_chunks = chunks_multivector + chunks
            return combined_chunks

        # Configuration 4: Reranking with colpali
        # Use colpali as a reranker to get consistent similarity scores for both types of chunks
        logger.debug("Using configuration 4: Reranking with colpali - rescoring chunks with colpali model")

        model_name = "vidore/colSmol-256M"
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,  # "cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="eager",  # "flash_attention_2" if is_flash_attn_2_available() else None,
            # or "eager" if "mps"
        ).eval()
        processor = ColIdefics3Processor.from_pretrained(model_name)

        # Score regular chunks with colpali model for consistent comparison
        batch_chunks = processor.process_queries([chunk.content for chunk in chunks]).to(device)
        query_rep = processor.process_queries([query]).to(device)
        multi_vec_representations = model(**batch_chunks)
        query_rep = model(**query_rep)
        scores = processor.score_multi_vector(query_rep, multi_vec_representations)
        for chunk, score in zip(chunks, scores[0]):
            chunk.score = score

        # Combine and sort all chunks
        full_chunks = chunks + chunks_multivector
        full_chunks.sort(key=lambda x: x.score, reverse=True)
        return full_chunks

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
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

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
        retrieval_tasks = [self.vector_store.get_chunks_by_id(chunk_identifiers)]

        # Add colpali vector store task if needed
        if use_colpali and self.colpali_vector_store:
            logger.info("Preparing to retrieve chunks from both regular and colpali vector stores")
            retrieval_tasks.append(self.colpali_vector_store.get_chunks_by_id(chunk_identifiers))

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
    ) -> CompletionResponse:
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
            )

        # Standard retrieval without graph
        chunks = await self.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali, folder_name, end_user_id
        )
        documents = await self._create_document_results(auth, chunks)

        # Create augmented chunk contents
        chunk_contents = [chunk.augmented_content(documents[chunk.document_id]) for chunk in chunks]

        # Collect sources information
        sources = [
            ChunkSource(document_id=chunk.document_id, chunk_number=chunk.chunk_number, score=chunk.score)
            for chunk in chunks
        ]

        # Generate completion with prompt override if provided
        custom_prompt_template = None
        if prompt_overrides and prompt_overrides.query:
            custom_prompt_template = prompt_overrides.query.prompt_template

        request = CompletionRequest(
            query=query,
            context_chunks=chunk_contents,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_template=custom_prompt_template,
            schema=schema,
        )

        response = await self.completion_model.complete(request)

        # Add sources information at the document service level
        response.sources = sources

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
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
                "user_id": [auth.user_id if auth.user_id else []],  # user scoping
            },
        )

        # Add folder_name and end_user_id to system_metadata if provided
        if folder_name:
            doc.system_metadata["folder_name"] = folder_name

            # Check if the folder exists, if not create it
            await self._ensure_folder_exists(folder_name, doc.external_id, auth)

        if end_user_id:
            doc.system_metadata["end_user_id"] = end_user_id

        # Tag document with app_id for segmentation
        if auth.app_id:
            doc.system_metadata["app_id"] = auth.app_id

        logger.debug(f"Created text document record with ID {doc.external_id}")

        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            from core.api import check_and_increment_limits

            num_pages = int(len(content) / (CHARS_PER_TOKEN * TOKENS_PER_PAGE))  #
            await check_and_increment_limits(auth, "ingest", num_pages, doc.external_id)

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

        if use_colpali and self.colpali_embedding_model:
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
        await self._store_chunks_and_doc(chunk_objects, doc, use_colpali, chunk_objects_multivector)
        logger.debug(f"Successfully stored text document {doc.external_id}")

        # Update the document status to completed after successful storage
        # This matches the behavior in ingestion_worker.py
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        await self.db.update_document(
            document_id=doc.external_id, updates={"system_metadata": doc.system_metadata}, auth=auth
        )
        logger.debug(f"Updated document status to 'completed' for {doc.external_id}")

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
            owner={"type": auth.entity_type.value, "id": auth.entity_id},
            metadata=metadata or {},
            system_metadata={"status": "processing"},  # Initial status
            content_info={"type": "file", "mime_type": content_type},
            # Ensure access_control is set similar to /ingest/file
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
                "user_id": [auth.user_id] if auth.user_id else [],
                "app_access": ([auth.app_id] if auth.app_id else []),
            },
        )

        if auth.app_id:
            doc.system_metadata["app_id"] = auth.app_id
        if end_user_id:
            doc.system_metadata["end_user_id"] = end_user_id
        # folder_name is handled later by _ensure_folder_exists if needed by background worker

        # Check limits before proceeding with DB operations or storage
        if settings.MODE == "cloud" and auth.user_id:
            from core.api import check_and_increment_limits

            # Estimate num_pages based on byte length.
            # This is an approximation; CHARS_PER_TOKEN is an average.
            # For binary files, this might not perfectly represent "pages"
            # but aligns with ingest_text's content length based approach.
            if CHARS_PER_TOKEN > 0 and TOKENS_PER_PAGE > 0:
                num_pages = int(len(file_content_bytes) / (CHARS_PER_TOKEN * TOKENS_PER_PAGE))
                if num_pages == 0 and len(file_content_bytes) > 0:
                    num_pages = 1
            else:
                num_pages = 1

            await check_and_increment_limits(auth, "ingest", num_pages, doc.external_id)
            logger.info(f"Limit check passed for user {auth.user_id}, {num_pages} estimated pages for {filename}.")

        # 1. Create initial document record in DB
        # The app_db concept from core/api.py implies self.db is already app-specific if needed
        await self.db.store_document(doc)
        logger.info(f"Initial document record created for {filename} (doc_id: {doc.external_id})")

        # 2. Save raw file to Storage
        # Using a unique key structure similar to /ingest/file to avoid collisions if worker needs it
        file_key_suffix = str(uuid.uuid4())
        storage_key = f"ingest_uploads/{file_key_suffix}/{filename}"
        content_base64 = base64.b64encode(file_content_bytes).decode("utf-8")

        try:
            bucket_name, full_storage_path = await self._upload_to_app_bucket(
                auth=auth, content_base64=content_base64, key=storage_key, content_type=content_type
            )
            # Create StorageFileInfo with version as INT
            sfi = StorageFileInfo(
                bucket=bucket_name,
                key=full_storage_path,
                content_type=content_type,
                size=len(file_content_bytes),
                last_modified=datetime.now(UTC),
                version=1,  # INT, as per StorageFileInfo model
                filename=filename,
            )
            # Populate legacy doc.storage_info (Dict[str, str]) with stringified values
            doc.storage_info = {k: str(v) if v is not None else "" for k, v in sfi.model_dump().items()}

            # Initialize storage_files list with the StorageFileInfo object (version remains int)
            doc.storage_files = [sfi]

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
                f"File {filename} (doc_id: {doc.external_id}) uploaded to storage: "
                f"{bucket_name}/{full_storage_path} and DB record updated."
            )

        except Exception as e:
            logger.error(f"Failed to upload file {filename} (doc_id: {doc.external_id}) to storage or update DB: {e}")
            # Update document status to failed if initial storage fails
            doc.system_metadata["status"] = "failed"
            doc.system_metadata["error"] = f"Storage upload/DB update failed: {str(e)}"
            try:
                await self.db.update_document(doc.external_id, {"system_metadata": doc.system_metadata}, auth=auth)
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
            logger.info(f"Connector file ingestion job queued with ID: {job.job_id} for document: {doc.external_id}")
        except Exception as e:
            logger.error(f"Failed to enqueue ingestion job for doc {doc.external_id} ({filename}): {e}")
            # Update document status to failed if enqueuing fails
            doc.system_metadata["status"] = "failed"
            doc.system_metadata["error"] = f"Failed to enqueue processing job: {str(e)}"
            try:
                await self.db.update_document(doc.external_id, {"system_metadata": doc.system_metadata}, auth=auth)
            except Exception as db_update_err:
                logger.error(f"Additionally failed to mark doc {doc.external_id} as failed in DB: {db_update_err}")
            raise HTTPException(status_code=500, detail=f"Failed to enqueue document processing job: {str(e)}")

        return doc

    def img_to_base64_str(self, img: Image):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_byte = buffered.getvalue()
        img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
        return img_str

    def _create_chunks_multivector(self, file_type, file_content_base64: str, file_content: bytes, chunks: List[Chunk]):
        # Handle the case where file_type is None
        mime_type = file_type.mime if file_type is not None else "text/plain"
        logger.info(f"Creating chunks for multivector embedding for file type {mime_type}")

        # If file_type is None, attempt a light-weight heuristic to detect images
        # Some JPGs with uncommon EXIF markers fail `filetype.guess`, leading to
        # false "text" classification and, eventually, empty chunk lists. Try to
        # open the bytes with Pillow; if that succeeds, treat it as an image.
        if file_type is None:
            try:
                from PIL import Image as PILImage

                PILImage.open(BytesIO(file_content)).verify()
                logger.info("Heuristic image detection succeeded (Pillow). Treating as image.")
                return [Chunk(content=file_content_base64, metadata={"is_image": True})]
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
                from PIL import Image as PILImage

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
                img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
                return [Chunk(content=img_b64, metadata={"is_image": True})]
            except Exception as e:
                logger.error(f"Error resizing image for base64 encoding: {e}. Falling back to original size.")
                return [Chunk(content=file_content_base64, metadata={"is_image": True})]

        match mime_type:
            case file_type if file_type in IMAGE:
                return [Chunk(content=file_content_base64, metadata={"is_image": True})]
            case "application/pdf":
                logger.info("Working with PDF file!")
                images = pdf2image.convert_from_bytes(file_content)
                images_b64 = [self.img_to_base64_str(image) for image in images]
                return [Chunk(content=image_b64, metadata={"is_image": True}) for image_b64 in images_b64]
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

                        images_b64 = [self.img_to_base64_str(image) for image in images]
                        return [Chunk(content=image_b64, metadata={"is_image": True}) for image_b64 in images_b64]
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

            # case filetype.get_type(ext="txt"):
            #     logger.info(f"Found text input: chunks for multivector embedding")
            #     return chunks.copy()
            # TODO: Add support for office documents
            # case document.Xls | document.Xlsx | document.Ods |document.Odp:
            #     logger.warning(f"Colpali is not supported for file type {file_type.mime} - skipping")
            # case file_type if file_type in DOCUMENT:
            #     pass
            case _:
                logger.warning(f"Colpali is not supported for file type {file_type.mime} - skipping")
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False})) for chunk in chunks
                ]

    def _create_chunk_objects(
        self,
        doc_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[DocumentChunk]:
        """Helper to create chunk objects

        Note: folder_name and end_user_id are not needed in chunk metadata because:
        1. Filtering by these values happens at the document level in find_authorized_and_filtered_documents
        2. Vector search is only performed on already authorized and filtered documents
        3. This approach is more efficient as it reduces the size of chunk metadata
        """
        return [
            c.to_document_chunk(chunk_number=i, embedding=embedding, document_id=doc_id)
            for i, (embedding, c) in enumerate(zip(embeddings, chunks))
        ]

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
                    success, result = await store.store_embeddings(objects)
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
                        success = await self.db.store_document(doc)
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

        # Run storage operations in parallel when possible
        storage_tasks = [store_with_retry(self.vector_store, chunk_objects, "regular")]

        # Add colpali storage task if needed
        if use_colpali and self.colpali_vector_store and chunk_objects_multivector:
            storage_tasks.append(store_with_retry(self.colpali_vector_store, chunk_objects_multivector, "colpali"))

        # Execute storage tasks concurrently
        storage_results = await asyncio.gather(*storage_tasks)

        # Combine chunk IDs
        regular_chunk_ids = storage_results[0]
        colpali_chunk_ids = storage_results[1] if len(storage_results) > 1 else []
        doc.chunk_ids = regular_chunk_ids + colpali_chunk_ids

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

        # Generate download URLs for all documents that have storage info
        download_urls = {}
        for doc_id, doc in doc_map.items():
            if doc.storage_info:
                download_urls[doc_id] = await self.storage.get_download_url(
                    doc.storage_info["bucket"], doc.storage_info["key"]
                )
                logger.debug(f"Generated download URL for document {doc_id}")

        # Create chunk results using the lookup dictionaries
        for chunk in chunks:
            doc = doc_map.get(chunk.document_id)
            if not doc:
                logger.warning(f"Document {chunk.document_id} not found")
                continue

            metadata = doc.metadata.copy()
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
                    download_url=download_urls.get(chunk.document_id),
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

        # Generate download URLs for non-text documents in a single loop
        download_urls = {}
        for doc_id, doc in doc_map.items():
            if doc.content_type != "text/plain" and doc.storage_info:
                download_urls[doc_id] = await self.storage.get_download_url(
                    doc.storage_info["bucket"], doc.storage_info["key"]
                )
                logger.debug(f"Generated download URL for document {doc_id}")

        # Create document results using the lookup dictionaries
        results = {}
        for doc_id, chunk in doc_chunks.items():
            doc = doc_map.get(doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found")
                continue

            # Create DocumentContent based on content type
            if doc.content_type == "text/plain":
                content = DocumentContent(type="string", value=chunk.content, filename=None)
                logger.debug(f"Created text content for document {doc_id}")
            else:
                # Use pre-generated download URL for file types
                content = DocumentContent(type="url", value=download_urls.get(doc_id), filename=doc.filename)
                logger.debug(f"Created URL content for document {doc_id}")

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
        current_content = doc.system_metadata.get("content", "")
        metadata_only_update = content is None and file is None and metadata is not None

        # Process content based on update type
        update_content = None
        file_content = None
        file_type = None
        file_content_base64 = None
        if content is not None:
            update_content = await self._process_text_update(content, doc, filename, metadata, rules)
        elif file is not None:
            update_content, file_content, file_type, file_content_base64 = await self._process_file_update(
                file, doc, metadata, rules
            )
            await self._update_storage_info(doc, file, file_content_base64)
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
            chunk_objects, doc, use_colpali, chunk_objects_multivector, is_update=True, auth=auth
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
        bucket_override = await self._get_bucket_for_app(doc.system_metadata.get("app_id"))

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

        if not (use_colpali and self.colpali_embedding_model and self.colpali_vector_store):
            return chunk_objects_multivector

        # For file updates, we need special handling for images and PDFs
        if file and file_type and (file_type.mime in IMAGE or file_type.mime == "application/pdf"):
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
                vector_deletion_tasks.append(self.vector_store.delete_chunks_by_document_id(document_id))

            # Try to delete from colpali vector store as well
            if self.colpali_vector_store and hasattr(self.colpali_vector_store, "delete_chunks_by_document_id"):
                vector_deletion_tasks.append(self.colpali_vector_store.delete_chunks_by_document_id(document_id))

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
        content_base64: str,
        key: str,
        content_type: Optional[str] = None,
    ) -> tuple[str, str]:
        bucket_override = await self._get_bucket_for_app(auth.app_id)
        return await self.storage.upload_from_base64(content_base64, key, content_type, bucket=bucket_override or "")
