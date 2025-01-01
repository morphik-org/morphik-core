import base64
from typing import Dict, Any, List, Optional
from fastapi import UploadFile

from core.models.request import IngestTextRequest
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

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        database: BaseDatabase,
        vector_store: BaseVectorStore,
        storage: BaseStorage,
        parser: BaseParser,
        embedding_model: BaseEmbeddingModel,
        completion_model: BaseCompletionModel,
    ):
        self.db = database
        self.vector_store = vector_store
        self.storage = storage
        self.parser = parser
        self.embedding_model = embedding_model
        self.completion_model = completion_model

    async def retrieve_chunks(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
    ) -> List[ChunkResult]:
        """Retrieve relevant chunks."""
        # Get embedding for query
        query_embedding = await self.embedding_model.embed_for_query(query)
        logger.info("Generated query embedding")

        # Find authorized documents
        doc_ids = await self.db.find_authorized_and_filtered_documents(auth, filters)
        if not doc_ids:
            logger.info("No authorized documents found")
            return []
        logger.info(f"Found {len(doc_ids)} authorized documents")

        # Search chunks with vector similarity
        chunks = await self.vector_store.query_similar(query_embedding, k=k, doc_ids=doc_ids)
        logger.info(f"Found {len(chunks)} similar chunks")

        # Create and return chunk results
        results = await self._create_chunk_results(auth, chunks)
        logger.info(f"Returning {len(results)} chunk results")
        return results

    async def retrieve_docs(
        self,
        query: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 4,
        min_score: float = 0.0,
    ) -> List[DocumentResult]:
        """Retrieve relevant documents."""
        # Get chunks first
        chunks = await self.retrieve_chunks(query, auth, filters, k, min_score)
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
    ) -> CompletionResponse:
        """Generate completion using relevant chunks as context."""
        # Get relevant chunks
        chunks = await self.retrieve_chunks(query, auth, filters, k, min_score)
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

    async def ingest_text(self, request: IngestTextRequest, auth: AuthContext) -> Document:
        """Ingest a text document."""
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission")
            raise PermissionError("User does not have write permission")

        # 1. Create document record
        doc = Document(
            content_type="text/plain",
            metadata=request.metadata,
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
            },
        )
        logger.info(f"Created text document record with ID {doc.external_id}")

        # 2. Parse content into chunks
        chunks = await self.parser.split_text(request.content)
        if not chunks:
            raise ValueError("No content chunks extracted from text")
        logger.info(f"Split text into {len(chunks)} chunks")

        # 3. Generate embeddings for chunks
        embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # 4. Create and store chunk objects
        chunk_objects = self._create_chunk_objects(doc.external_id, chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")

        # 5. Store everything
        await self._store_chunks_and_doc(chunk_objects, doc)
        logger.info(f"Successfully stored text document {doc.external_id}")

        return doc

    async def ingest_file(
        self, file: UploadFile, metadata: Dict[str, Any], auth: AuthContext
    ) -> Document:
        """Ingest a file document."""
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        file_content = await file.read()
        additional_metadata, chunks = await self.parser.parse_file(
            file_content, file.content_type or "", file.filename
        )

        doc = Document(
            content_type=file.content_type or "",
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
        logger.info(f"Created file document record with ID {doc.external_id}")

        storage_info = await self.storage.upload_from_base64(
            base64.b64encode(file_content).decode(), doc.external_id, file.content_type
        )
        doc.storage_info = {"bucket": storage_info[0], "key": storage_info[1]}
        logger.info(f"Stored file in bucket `{storage_info[0]}` with key `{storage_info[1]}`")

        if not chunks:
            raise ValueError("No content chunks extracted from file")
        logger.info(f"Parsed file into {len(chunks)} chunks")

        # 4. Generate embeddings for chunks
        embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # 5. Create and store chunk objects
        chunk_objects = self._create_chunk_objects(doc.external_id, chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")

        # 6. Store everything
        doc.chunk_ids = await self._store_chunks_and_doc(chunk_objects, doc)
        logger.info(f"Successfully stored file document {doc.external_id}")

        return doc

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
        self, chunk_objects: List[DocumentChunk], doc: Document
    ) -> List[str]:
        """Helper to store chunks and document"""
        # Store chunks in vector store
        success, result = await self.vector_store.store_embeddings(chunk_objects)
        if not success:
            raise Exception("Failed to store chunk embeddings")
        logger.debug("Stored chunk embeddings in vector store")

        doc.chunk_ids = result
        # Store document metadata
        if not await self.db.store_document(doc):
            raise Exception("Failed to store document metadata")
        logger.debug("Stored document metadata in database")
        logger.debug(f"Chunk IDs stored: {result}")
        return result

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

            results.append(
                ChunkResult(
                    content=chunk.content,
                    score=chunk.score,
                    document_id=chunk.document_id,
                    chunk_number=chunk.chunk_number,
                    metadata=doc.metadata,
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
