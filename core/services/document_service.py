import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
from fastapi import UploadFile

from core.models.chunk import Chunk, DocumentChunk
from core.models.documents import (
    Document,
    ChunkResult,
    DocumentContent,
    DocumentResult,
)
from ..models.auth import AuthContext
from core.database.base_database import BaseDatabase
from core.storage.base_storage import BaseStorage
from core.vector_store.base_vector_store import BaseVectorStore
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.parser.base_parser import BaseParser
from core.completion.base_completion import BaseCompletionModel
from core.completion.base_completion import CompletionRequest, CompletionResponse
import logging
from core.reranker.base_reranker import BaseReranker
from core.config import get_settings
from core.cache.base_cache import BaseCache
from core.cache.base_cache_factory import BaseCacheFactory
from core.services.rules_processor import RulesProcessor
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.vector_store.multi_vector_store import MultiVectorStore
import filetype
from filetype.types import IMAGE  # , DOCUMENT, document
import pdf2image
from PIL.Image import Image

logger = logging.getLogger(__name__)
IMAGE = {im.mime for im in IMAGE}


class DocumentService:
    def __init__(
        self,
        database: BaseDatabase,
        vector_store: BaseVectorStore,
        storage: BaseStorage,
        parser: BaseParser,
        embedding_model: BaseEmbeddingModel,
        completion_model: BaseCompletionModel,
        cache_factory: BaseCacheFactory,
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

        colpali_vector_store.initialize()

        # Cache-related data structures
        # Maps cache name to active cache object
        self.active_caches: Dict[str, BaseCache] = {}

    async def retrieve_chunks(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
    ) -> List[ChunkResult]:
        """Retrieve relevant chunks."""
        settings = get_settings()
        should_rerank = use_reranking if use_reranking is not None else settings.USE_RERANKING

        # Get embedding for query
        query_embedding_regular = await self.embedding_model.embed_for_query(query)
        query_embedding_multivector = await self.colpali_embedding_model.embed_for_query(query) if use_colpali else None
        logger.info("Generated query embedding")

        # Find authorized documents
        doc_ids = await self.db.find_authorized_and_filtered_documents(auth, filters)
        if not doc_ids:
            logger.info("No authorized documents found")
            return []
        logger.info(f"Found {len(doc_ids)} authorized documents")

        # Search chunks with vector similarity
        chunks = await self.vector_store.query_similar(
            query_embedding_regular, k=10 * k if should_rerank else k, doc_ids=doc_ids
        )

        chunks_multivector = (
            await self.colpali_vector_store.query_similar(
                query_embedding_multivector, k=k, doc_ids=doc_ids
            )
            if use_colpali
            else []
        )

        logger.info(f"Found {len(chunks)} similar chunks via regular embedding")
        if use_colpali:
            logger.info(
                f"Found {len(chunks_multivector)} similar chunks via multivector embedding since we are also using colpali"
            )

        # Rerank chunks using the reranker if enabled and available
        if chunks and should_rerank and self.reranker is not None:
            chunks = await self.reranker.rerank(query, chunks)
            chunks.sort(key=lambda x: x.score, reverse=True)
            chunks = chunks[:k]
            logger.info(f"Reranked {k*10} chunks and selected the top {k}")

        chunks = chunks_multivector + chunks

        # Create and return chunk results
        results = await self._create_chunk_results(auth, chunks)
        logger.info(f"Returning {len(results)} chunk results")
        return results

    async def retrieve_docs(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.0,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
    ) -> List[DocumentResult]:
        """Retrieve relevant documents."""
        # Get chunks first
        chunks = await self.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali
        )
        # Convert to document results
        results = await self._create_document_results(auth, chunks)
        documents = list(results.values())
        logger.info(f"Returning {len(documents)} document results")
        return documents

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
    ) -> CompletionResponse:
        """Generate completion using relevant chunks as context."""
        # Get relevant chunks
        chunks = await self.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali
        )
        documents = await self._create_document_results(auth, chunks)

        chunk_contents = [chunk.augmented_content(documents[chunk.document_id]) for chunk in chunks]

        # Generate completion
        request = CompletionRequest(
            query=query,
            context_chunks=chunk_contents,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = await self.completion_model.complete(request)
        return response

    async def ingest_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        auth: AuthContext = None,
        rules: Optional[List[str]] = None,
        use_colpali: Optional[bool] = None,
    ) -> Document:
        """Ingest a text document."""
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission")
            raise PermissionError("User does not have write permission")

        doc = Document(
            content_type="text/plain",
            metadata=metadata or {},
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
            },
        )
        logger.info(f"Created text document record with ID {doc.external_id}")

        # Apply rules if provided
        if rules:
            rule_metadata, modified_text = await self.rules_processor.process_rules(content, rules)
            # Update document metadata with extracted metadata from rules
            metadata.update(rule_metadata)
            doc.metadata = metadata  # Update doc metadata after rules

            if modified_text:
                content = modified_text
                logger.info("Updated content with modified text from rules")

        # Store full content before chunking
        doc.system_metadata["content"] = content

        # Split into chunks after all processing is done
        chunks = await self.parser.split_text(content)
        if not chunks:
            raise ValueError("No content chunks extracted")
        logger.info(f"Split processed text into {len(chunks)} chunks")

        # Generate embeddings for chunks
        embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        chunk_objects = self._create_chunk_objects(doc.external_id, chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")

        chunk_objects_multivector = []

        if use_colpali:
            embeddings_multivector = await self.colpali_embedding_model.embed_for_ingestion(chunks)
            logger.info(
                f"Generated {len(embeddings_multivector)} embeddings for multivector embedding"
            )
            chunk_objects_multivector = self._create_chunk_objects(
                doc.external_id, chunks, embeddings_multivector
            )
            logger.info(
                f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding"
            )

        # Create and store chunk objects

        # Store everything
        await self._store_chunks_and_doc(chunk_objects, doc, use_colpali, chunk_objects_multivector)
        logger.info(f"Successfully stored text document {doc.external_id}")

        return doc

    async def ingest_file(
        self,
        file: UploadFile,
        metadata: Dict[str, Any],
        auth: AuthContext,
        rules: Optional[List[str]] = None,
        use_colpali: Optional[bool] = None,
    ) -> Document:
        """Ingest a file document."""
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Read file content
        file_content = await file.read()
        file_type = filetype.guess(file_content)

        # Parse file to text first
        additional_metadata, text = await self.parser.parse_file_to_text(
            file_content, file.filename
        )
        logger.info(f"Parsed file into text of length {len(text)}")

        # Apply rules if provided
        if rules:
            rule_metadata, modified_text = await self.rules_processor.process_rules(text, rules)
            # Update document metadata with extracted metadata from rules
            metadata.update(rule_metadata)
            if modified_text:
                text = modified_text
                logger.info("Updated text with modified content from rules")

        # Create document record
        doc = Document(
            content_type=file_type.mime or "",
            filename=file.filename,
            metadata=metadata,
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
            },
            additional_metadata=additional_metadata,
        )

        # Store full content
        doc.system_metadata["content"] = text
        logger.info(f"Created file document record with ID {doc.external_id}")

        file_content_base64 = base64.b64encode(file_content).decode()
        # Store the original file
        storage_info = await self.storage.upload_from_base64(
            file_content_base64, doc.external_id, file.content_type
        )
        doc.storage_info = {"bucket": storage_info[0], "key": storage_info[1]}
        logger.info(f"Stored file in bucket `{storage_info[0]}` with key `{storage_info[1]}`")

        # Split into chunks after all processing is done
        chunks = await self.parser.split_text(text)
        if not chunks:
            raise ValueError("No content chunks extracted")
        logger.info(f"Split processed text into {len(chunks)} chunks")

        # Generate embeddings for chunks
        embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Create and store chunk objects
        chunk_objects = self._create_chunk_objects(doc.external_id, chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")

        chunk_objects_multivector = []
        logger.info(f"use_colpali: {use_colpali}")
        if use_colpali:
            chunks_multivector = self._create_chunks_multivector(
                file_type, file_content_base64, file_content, chunks
            )
            logger.info(f"Created {len(chunks_multivector)} chunks for multivector embedding")
            colpali_embeddings = await self.colpali_embedding_model.embed_for_ingestion(
                chunks_multivector
            )
            logger.info(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(
                doc.external_id, chunks_multivector, colpali_embeddings
            )

        # Store everything
        doc.chunk_ids = await self._store_chunks_and_doc(
            chunk_objects, doc, use_colpali, chunk_objects_multivector
        )
        logger.info(f"Successfully stored file document {doc.external_id}")

        return doc

    def img_to_base64_str(self, img: Image):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_byte = buffered.getvalue()
        img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
        return img_str

    def _create_chunks_multivector(
        self, file_type, file_content_base64: str, file_content: bytes, chunks: List[Chunk]
    ):
        logger.info(f"Creating chunks for multivector embedding for file type {file_type.mime}")
        match file_type.mime:
            case file_type if file_type in IMAGE:
                return [Chunk(content=file_content_base64, metadata={"is_image": True})]
            case "application/pdf":
                logger.info("Working with PDF file!")
                images = pdf2image.convert_from_bytes(file_content)
                images_b64 = [self.img_to_base64_str(image) for image in images]
                return [
                    Chunk(content=image_b64, metadata={"is_image": True})
                    for image_b64 in images_b64
                ]
            # case filetype.get_type(ext="txt"):
            #     logger.info(f"Found text input: chunks for multivector embedding")
            #     return chunks.copy()
            # TODO: Add support for office documents
            # case document.Xls | document.Xlsx | document.Ods |document.Odp:
            #     logger.warning(f"Colpali is not supported for file type {file_type.mime} - skipping")
            # case file_type if file_type in DOCUMENT:
            #     pass
            case _:
                logger.warning(
                    f"Colpali is not supported for file type {file_type.mime} - skipping"
                )
                return [
                    Chunk(content=chunk.content, metadata=(chunk.metadata | {"is_image": False}))
                    for chunk in chunks
                ]

    def _create_chunk_objects(
        self,
        doc_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> List[DocumentChunk]:
        """Helper to create chunk objects"""
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
    ) -> List[str]:
        """Helper to store chunks and document"""
        # Store chunks in vector store
        success, result = await self.vector_store.store_embeddings(chunk_objects)
        if not success:
            raise Exception("Failed to store chunk embeddings")
        logger.debug("Stored chunk embeddings in vector store")
        doc.chunk_ids = result

        if use_colpali and chunk_objects_multivector is not None:
            success, result_multivector = await self.colpali_vector_store.store_embeddings(
                chunk_objects_multivector
            )
            if not success:
                raise Exception("Failed to store multivector chunk embeddings")
            logger.debug("Stored multivector chunk embeddings in vector store")
            doc.chunk_ids += result_multivector

        # Store document metadata
        if not await self.db.store_document(doc):
            raise Exception("Failed to store document metadata")
        logger.debug("Stored document metadata in database")
        logger.debug(f"Chunk IDs stored: {doc.chunk_ids}")
        return doc.chunk_ids

    async def _create_chunk_results(
        self, auth: AuthContext, chunks: List[DocumentChunk]
    ) -> List[ChunkResult]:
        """Create ChunkResult objects with document metadata."""
        results = []
        for chunk in chunks:
            # Get document metadata
            doc = await self.db.get_document(chunk.document_id, auth)
            if not doc:
                logger.warning(f"Document {chunk.document_id} not found")
                continue
            logger.debug(f"Retrieved metadata for document {chunk.document_id}")

            # Generate download URL if needed
            download_url = None
            if doc.storage_info:
                download_url = await self.storage.get_download_url(
                    doc.storage_info["bucket"], doc.storage_info["key"]
                )
                logger.debug(f"Generated download URL for document {chunk.document_id}")

            metadata = doc.metadata
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
                    download_url=download_url,
                )
            )

        logger.info(f"Created {len(results)} chunk results")
        return results

    async def _create_document_results(
        self, auth: AuthContext, chunks: List[ChunkResult]
    ) -> Dict[str, DocumentResult]:
        """Group chunks by document and create DocumentResult objects."""
        # Group chunks by document and get highest scoring chunk per doc
        doc_chunks: Dict[str, ChunkResult] = {}
        for chunk in chunks:
            if (
                chunk.document_id not in doc_chunks
                or chunk.score > doc_chunks[chunk.document_id].score
            ):
                doc_chunks[chunk.document_id] = chunk
        logger.info(f"Grouped chunks into {len(doc_chunks)} documents")
        logger.info(f"Document chunks: {doc_chunks}")
        results = {}
        for doc_id, chunk in doc_chunks.items():
            # Get document metadata
            doc = await self.db.get_document(doc_id, auth)
            if not doc:
                logger.warning(f"Document {doc_id} not found")
                continue
            logger.info(f"Retrieved metadata for document {doc_id}")

            # Create DocumentContent based on content type
            if doc.content_type == "text/plain":
                content = DocumentContent(type="string", value=chunk.content, filename=None)
                logger.debug(f"Created text content for document {doc_id}")
            else:
                # Generate download URL for file types
                download_url = await self.storage.get_download_url(
                    doc.storage_info["bucket"], doc.storage_info["key"]
                )
                content = DocumentContent(type="url", value=download_url, filename=doc.filename)
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
            cache = self.cache_factory.load_cache_from_bytes(
                name=name, cache_bytes=cache_bytes, metadata=metadata
            )
            self.active_caches[name] = cache
            return {"success": True, "message": "Cache loaded successfully"}
        except Exception as e:
            logger.error(f"Failed to load cache {name}: {e}")
            # raise e
            return {"success": False, "message": f"Failed to load cache {name}: {e}"}

    def close(self):
        """Close all resources."""
        # Close any active caches
        self.active_caches.clear()
