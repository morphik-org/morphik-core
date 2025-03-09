import base64
from io import BytesIO
import json
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
from datetime import datetime, UTC

from core.models.chunk import Chunk, DocumentChunk
from core.models.documents import (
    Document,
    ChunkResult,
    DocumentContent,
    DocumentResult,
    StorageFileInfo,
)
from ..models.auth import AuthContext
from ..models.graph import Graph, Entity, Relationship
from core.database.base_database import BaseDatabase
from core.storage.base_storage import BaseStorage
from core.vector_store.base_vector_store import BaseVectorStore
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.parser.base_parser import BaseParser
from core.completion.base_completion import BaseCompletionModel
from core.models.completion import CompletionRequest, CompletionResponse, ChunkSource
import logging
from core.reranker.base_reranker import BaseReranker
from core.config import get_settings
from core.cache.base_cache import BaseCache
from core.cache.base_cache_factory import BaseCacheFactory
from core.services.rules_processor import RulesProcessor
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.vector_store.multi_vector_store import MultiVectorStore
from openai import AsyncOpenAI
import filetype
from filetype.types import IMAGE  # , DOCUMENT, document
import pdf2image
from PIL.Image import Image
import tempfile
import os

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

        if colpali_vector_store:
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
        query_embedding_multivector = await self.colpali_embedding_model.embed_for_query(query) if (use_colpali and self.colpali_embedding_model) else None
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
            if (use_colpali and self.colpali_vector_store and query_embedding_multivector)
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
        
    async def batch_retrieve_documents(
        self,
        document_ids: List[str],
        auth: AuthContext
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
            
        # Use the database's batch retrieval method
        documents = await self.db.get_documents_by_id(document_ids, auth)
        logger.info(f"Batch retrieved {len(documents)} documents out of {len(document_ids)} requested")
        return documents
        
    async def batch_retrieve_chunks(
        self,
        chunk_ids: List[ChunkSource],
        auth: AuthContext
    ) -> List[ChunkResult]:
        """
        Retrieve specific chunks by their document ID and chunk number in a single batch operation.
        
        Args:
            chunk_ids: List of ChunkSource objects with document_id and chunk_number
            auth: Authentication context
            
        Returns:
            List of ChunkResult objects
        """
        if not chunk_ids:
            return []
            
        # Collect unique document IDs to check authorization in a single query
        doc_ids = list({source.document_id for source in chunk_ids})
        
        # Find authorized documents in a single query
        authorized_docs = await self.batch_retrieve_documents(doc_ids, auth)
        authorized_doc_ids = {doc.external_id for doc in authorized_docs}
        
        # Filter sources to only include authorized documents
        authorized_sources = [
            source for source in chunk_ids 
            if source.document_id in authorized_doc_ids
        ]
        
        if not authorized_sources:
            return []
            
        # Create list of (document_id, chunk_number) tuples for vector store query
        chunk_identifiers = [
            (source.document_id, source.chunk_number) 
            for source in authorized_sources
        ]
        
        # Retrieve the chunks from vector store in a single query
        chunks = await self.vector_store.get_chunks_by_id(chunk_identifiers)
        
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
        """
        if graph_name:
            # Use knowledge graph enhanced retrieval
            return await self._query_with_graph(
                query, 
                auth, 
                graph_name,
                filters=filters,
                k=k,
                min_score=min_score,
                max_tokens=max_tokens,
                temperature=temperature,
                use_reranking=use_reranking,
                use_colpali=use_colpali,
                hop_depth=hop_depth,
                include_paths=include_paths,
            )
        else:
            # Use standard retrieval
            chunks = await self.retrieve_chunks(
                query, auth, filters, k, min_score, use_reranking, use_colpali
            )
            documents = await self._create_document_results(auth, chunks)

        chunk_contents = [chunk.augmented_content(documents[chunk.document_id]) for chunk in chunks]
        
        # Collect sources information
        sources = [
            ChunkSource(document_id=chunk.document_id, chunk_number=chunk.chunk_number, score=chunk.score)
            for chunk in chunks
        ]

            # Generate completion
            request = CompletionRequest(
                query=query,
                context_chunks=chunk_contents,
                max_tokens=max_tokens,
                temperature=temperature,
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
    ) -> Document:
        """Ingest a text document."""
        if "write" not in auth.permissions:
            logger.error(f"User {auth.entity_id} does not have write permission")
            raise PermissionError("User does not have write permission")

        doc = Document(
            content_type="text/plain",
            filename=filename,
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

        if use_colpali and self.colpali_embedding_model:
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
        if use_colpali and self.colpali_embedding_model:
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
            case (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/msword"
            ):
                logger.info("Working with Word document!")
                # Check if file content is empty
                if not file_content or len(file_content) == 0:
                    logger.error("Word document content is empty")
                    return [
                        Chunk(
                            content=chunk.content, metadata=(chunk.metadata | {"is_image": False})
                        )
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
                    if (
                        not os.path.exists(expected_pdf_path)
                        or os.path.getsize(expected_pdf_path) == 0
                    ):
                        logger.error(
                            f"Generated PDF is empty or doesn't exist at expected path: {expected_pdf_path}"
                        )
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
                        return [
                            Chunk(content=image_b64, metadata={"is_image": True})
                            for image_b64 in images_b64
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
                        Chunk(
                            content=chunk.content, metadata=(chunk.metadata | {"is_image": False})
                        )
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
        is_update: bool = False,
        auth: Optional[AuthContext] = None,
    ) -> List[str]:
        """Helper to store chunks and document"""
        # Store chunks in vector store
        success, result = await self.vector_store.store_embeddings(chunk_objects)
        if not success:
            raise Exception("Failed to store chunk embeddings")
        logger.debug("Stored chunk embeddings in vector store")
        doc.chunk_ids = result

        if use_colpali and self.colpali_vector_store and chunk_objects_multivector:
            success, result_multivector = await self.colpali_vector_store.store_embeddings(
                chunk_objects_multivector
            )
            if not success:
                raise Exception("Failed to store multivector chunk embeddings")
            logger.debug("Stored multivector chunk embeddings in vector store")
            doc.chunk_ids += result_multivector

        # Store document metadata
        if is_update and auth:
            # For updates, use update_document
            updates = {
                "chunk_ids": doc.chunk_ids,
                "metadata": doc.metadata,
                "system_metadata": doc.system_metadata,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "storage_info": doc.storage_info,
            }
            if not await self.db.update_document(doc.external_id, updates, auth):
                raise Exception("Failed to update document metadata")
            logger.debug("Updated document metadata in database")
        else:
            # For new documents, use store_document
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
        metadata_only_update = (content is None and file is None and metadata is not None)
        
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
        elif not metadata_only_update:
            logger.error("Neither content nor file provided for document update")
            return None
        
        # Apply content update strategy if we have new content
        if update_content:
            updated_content = self._apply_update_strategy(current_content, update_content, update_strategy)
            doc.system_metadata["content"] = updated_content
        else:
            updated_content = current_content
        
        # Update metadata and version information
        self._update_metadata_and_version(doc, metadata, update_strategy, file)
        
        # For metadata-only updates, we don't need to re-process chunks
        if metadata_only_update:
            return await self._update_document_metadata_only(doc, auth)
        
        # Process content into chunks and generate embeddings
        chunks, chunk_objects = await self._process_chunks_and_embeddings(doc.external_id, updated_content)
        if not chunks:
            return None
        
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
        rules: Optional[List]
    ) -> str:
        """Process text content updates."""
        update_content = content
        
        # Update filename if provided
        if filename:
            doc.filename = filename
        
        # Apply rules if provided for text content
        if rules:
            rule_metadata, modified_text = await self.rules_processor.process_rules(content, rules)
            # Update metadata with extracted metadata from rules
            if metadata is not None:
                metadata.update(rule_metadata)
            
            if modified_text:
                update_content = modified_text
                logger.info("Updated content with modified text from rules")
                
        return update_content
        
    async def _process_file_update(
        self,
        file: UploadFile,
        doc: Document,
        metadata: Optional[Dict[str, Any]],
        rules: Optional[List]
    ) -> tuple[str, bytes, Any, str]:
        """Process file content updates."""
        # Read file content
        file_content = await file.read()
        
        # Parse the file content
        additional_file_metadata, file_text = await self.parser.parse_file_to_text(
            file_content, file.filename
        )
        logger.info(f"Parsed file into text of length {len(file_text)}")
        
        # Apply rules if provided for file content
        if rules:
            rule_metadata, modified_text = await self.rules_processor.process_rules(file_text, rules)
            # Update metadata with extracted metadata from rules
            if metadata is not None:
                metadata.update(rule_metadata)
            
            if modified_text:
                file_text = modified_text
                logger.info("Updated file content with modified text from rules")
        
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
        
        # Update filename
        doc.filename = file.filename
        
        return file_text, file_content, file_type, file_content_base64
        
    async def _update_storage_info(self, doc: Document, file: UploadFile, file_content_base64: str):
        """Update document storage information for file content."""
        # Check if we should keep previous file versions
        if hasattr(doc, "storage_files") and len(doc.storage_files) > 0:
            # In "add" strategy, create a new StorageFileInfo and append it
            storage_info = await self.storage.upload_from_base64(
                file_content_base64, f"{doc.external_id}_{len(doc.storage_files)}", file.content_type
            )
            
            # Create a new StorageFileInfo
            if not hasattr(doc, "storage_files"):
                doc.storage_files = []
                
            # If storage_files doesn't exist yet but we have legacy storage_info, migrate it
            if len(doc.storage_files) == 0 and doc.storage_info:
                # Create StorageFileInfo from legacy storage_info
                legacy_file_info = StorageFileInfo(
                    bucket=doc.storage_info.get("bucket", ""),
                    key=doc.storage_info.get("key", ""),
                    version=1,
                    filename=doc.filename,
                    content_type=doc.content_type,
                    timestamp=doc.system_metadata.get("updated_at", datetime.now(UTC))
                )
                doc.storage_files.append(legacy_file_info)
            
            # Add the new file to storage_files
            new_file_info = StorageFileInfo(
                bucket=storage_info[0],
                key=storage_info[1],
                version=len(doc.storage_files) + 1,
                filename=file.filename,
                content_type=file.content_type,
                timestamp=datetime.now(UTC)
            )
            doc.storage_files.append(new_file_info)
            
            # Still update legacy storage_info for backward compatibility
            doc.storage_info = {"bucket": storage_info[0], "key": storage_info[1]}
        else:
            # In replace mode (default), just update the storage_info
            storage_info = await self.storage.upload_from_base64(
                file_content_base64, doc.external_id, file.content_type
            )
            doc.storage_info = {"bucket": storage_info[0], "key": storage_info[1]}
            
            # Update storage_files field as well
            if not hasattr(doc, "storage_files"):
                doc.storage_files = []
            
            # Add or update the primary file info
            new_file_info = StorageFileInfo(
                bucket=storage_info[0],
                key=storage_info[1],
                version=1,
                filename=file.filename,
                content_type=file.content_type,
                timestamp=datetime.now(UTC)
            )
            
            # Replace the current main file (first file) or add if empty
            if len(doc.storage_files) > 0:
                doc.storage_files[0] = new_file_info
            else:
                doc.storage_files.append(new_file_info)
                
        logger.info(f"Stored file in bucket `{storage_info[0]}` with key `{storage_info[1]}`")
        
    def _apply_update_strategy(self, current_content: str, update_content: str, update_strategy: str) -> str:
        """Apply the update strategy to combine current and new content."""
        if update_strategy == "add":
            # Append the new content
            return current_content + "\n\n" + update_content
        else:
            # For now, just use 'add' as default strategy
            logger.warning(f"Unknown update strategy '{update_strategy}', defaulting to 'add'")
            return current_content + "\n\n" + update_content
        
    def _update_metadata_and_version(
        self, 
        doc: Document, 
        metadata: Optional[Dict[str, Any]], 
        update_strategy: str, 
        file: Optional[UploadFile]
    ):
        """Update document metadata and version tracking."""
        # Update metadata if provided - additive but replacing existing keys
        if metadata:
            doc.metadata.update(metadata)
        
        # Increment version
        current_version = doc.system_metadata.get("version", 1)
        doc.system_metadata["version"] = current_version + 1
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        
        # Track update history
        if "update_history" not in doc.system_metadata:
            doc.system_metadata["update_history"] = []
            
        update_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": current_version + 1,
            "strategy": update_strategy,
        }
        
        if file:
            update_entry["filename"] = file.filename
            
        if metadata:
            update_entry["metadata_updated"] = True
            
        doc.system_metadata["update_history"].append(update_entry)
        
    async def _update_document_metadata_only(self, doc: Document, auth: AuthContext) -> Optional[Document]:
        """Update document metadata without reprocessing chunks."""
        updates = {
            "metadata": doc.metadata,
            "system_metadata": doc.system_metadata,
            "filename": doc.filename,
        }
        success = await self.db.update_document(doc.external_id, updates, auth)
        if not success:
            logger.error(f"Failed to update document {doc.external_id} metadata")
            return None
            
        logger.info(f"Successfully updated document metadata for {doc.external_id}")
        return doc
        
    async def _process_chunks_and_embeddings(self, doc_id: str, content: str) -> tuple[List[Chunk], List[DocumentChunk]]:
        """Process content into chunks and generate embeddings."""
        # Split content into chunks
        chunks = await self.parser.split_text(content)
        if not chunks:
            logger.error("No content chunks extracted after update")
            return None, None
            
        logger.info(f"Split updated text into {len(chunks)} chunks")
        
        # Generate embeddings for new chunks
        embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Create new chunk objects
        chunk_objects = self._create_chunk_objects(doc_id, chunks, embeddings)
        logger.info(f"Created {len(chunk_objects)} chunk objects")
        
        return chunks, chunk_objects
        
    async def _process_colpali_embeddings(
        self,
        use_colpali: bool,
        doc_id: str,
        chunks: List[Chunk],
        file: Optional[UploadFile],
        file_type: Any,
        file_content: Optional[bytes],
        file_content_base64: Optional[str]
    ) -> List[DocumentChunk]:
        """Process colpali multi-vector embeddings if enabled."""
        chunk_objects_multivector = []
        
        if not (use_colpali and self.colpali_embedding_model and self.colpali_vector_store):
            return chunk_objects_multivector
            
        # For file updates, we need special handling for images and PDFs
        if file and file_type and (file_type.mime in IMAGE or file_type.mime == "application/pdf"):
            # Rewind the file and read it again if needed
            if hasattr(file, 'seek') and callable(file.seek) and not file_content:
                await file.seek(0)
                file_content = await file.read()
                file_content_base64 = base64.b64encode(file_content).decode()
            
            chunks_multivector = self._create_chunks_multivector(
                file_type, file_content_base64, file_content, chunks
            )
            logger.info(f"Created {len(chunks_multivector)} chunks for multivector embedding")
            colpali_embeddings = await self.colpali_embedding_model.embed_for_ingestion(chunks_multivector)
            logger.info(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(
                doc_id, chunks_multivector, colpali_embeddings
            )
        else:
            # For text updates or non-image/PDF files
            embeddings_multivector = await self.colpali_embedding_model.embed_for_ingestion(chunks)
            logger.info(f"Generated {len(embeddings_multivector)} embeddings for multivector embedding")
            chunk_objects_multivector = self._create_chunk_objects(
                doc_id, chunks, embeddings_multivector
            )
            
        logger.info(f"Created {len(chunk_objects_multivector)} chunk objects for multivector embedding")
        return chunk_objects_multivector
    async def create_graph(
        self,
        name: str,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
    ) -> Graph:
        """Create a graph from documents.

        This function processes documents matching filters or specific document IDs,
        extracts entities and relationships, and saves them as a graph using Apache AGE.

        Args:
            name: Name of the graph to create
            auth: Authentication context
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include

        Returns:
            Graph: The created graph
        """
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Find documents to process based on filters and/or specific document IDs
        document_objects = []

        # If specific document IDs were provided, get those documents
        if documents:
            for doc_id in documents:
                doc = await self.db.get_document(doc_id, auth)
                if doc:
                    document_objects.append(doc)

        # If filters were provided, get matching documents
        if filters:
            filtered_docs = await self.db.get_documents(auth, filters=filters)
            # Add only documents that aren't already in the list
            for doc in filtered_docs:
                if doc not in document_objects:
                    document_objects.append(doc)

        if not document_objects:
            raise ValueError("No documents found matching criteria")

        # Create a new graph
        graph = Graph(
            name=name,
            document_ids=[doc.external_id for doc in document_objects],
            filters=filters,
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
            },
        )

        # Process each document to extract entities and relationships using LLM
        entities = {}
        relationships = []

        for doc in document_objects:
            # Get the text content from document
            content = doc.system_metadata.get("content", "")
            if not content:
                logger.warning(f"No content found for document {doc.external_id}")
                continue

            # Extract entities and relationships using LLM
            doc_entities, doc_relationships = (
                await self._extract_entities_and_relationships_with_llm(content, doc.external_id)
            )

            # Add entities to the graph, avoiding duplicates
            for entity in doc_entities:
                if entity.label not in entities:
                    entities[entity.label] = entity
                else:
                    # If entity already exists, add this document to its list of document IDs
                    existing_entity = entities[entity.label]
                    if doc.external_id not in existing_entity.document_ids:
                        existing_entity.document_ids.append(doc.external_id)

            # Add relationships to the graph
            relationships.extend(doc_relationships)

        # Update the graph with extracted entities and relationships
        graph.entities = list(entities.values())
        graph.relationships = relationships

        # Store the graph in the database
        if not await self.db.store_graph(graph):
            raise Exception("Failed to store graph")

        return graph

    async def _extract_entities_and_relationships_with_llm(
        self, content: str, doc_id: str
    ) -> tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from document content using the LLM.

        Args:
            content: Document content to process
            doc_id: Document ID

        Returns:
            Tuple of (entities, relationships)
        """
        # Define the extraction schema for entities and relationships
        extraction_schema = {
            "entities": [
                {
                    "label": "string - name of the entity",
                    "type": "string - one of PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT, OTHER",
                    "properties": "dictionary of additional properties",
                }
            ],
            "relationships": [
                {
                    "source": "string - label of the source entity",
                    "target": "string - label of the target entity",
                    "type": "string - type of relationship",
                    "properties": "dictionary of additional properties",
                }
            ],
        }

        # Get the LLM model for entity extraction from settings
        settings = get_settings()

        # Prepare the prompt for entity and relationship extraction
        prompt = f"""
        Extract entities and relationships from the following text according to this schema:
        {extraction_schema}

        Text to extract from:
        {content[:5000]}  # Limiting to first 5000 chars to avoid token limits
        
        Return ONLY a JSON object with the extracted entities and relationships.
        """

        # Use the appropriate model based on the provider specified in settings
        try:
            extraction_result = {}

            if settings.GRAPH_PROVIDER == "openai":
                import openai

                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

                response = await client.chat.completions.create(
                    model=settings.GRAPH_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an entity extraction assistant. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                extraction_result = json.loads(response.choices[0].message.content)

            elif settings.GRAPH_PROVIDER == "ollama":
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{settings.EMBEDDING_OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": settings.GRAPH_MODEL,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are an entity extraction assistant. Always respond with valid JSON.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "stream": False,
                            "format": "json",  # Request JSON format from Ollama
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    extraction_result = json.loads(result["message"]["content"])
            else:
                logger.error(f"Unsupported graph provider: {settings.GRAPH_PROVIDER}")
                return [], []

            # Convert the extracted data to our model objects
            entities = []
            for entity_data in extraction_result.get("entities", []):
                entity = Entity(
                    label=entity_data["label"],
                    type=entity_data["type"],
                    properties=entity_data.get("properties", {}),
                    document_ids=[doc_id],
                )
                entities.append(entity)

            # Create a mapping of entity labels to IDs
            entity_mapping = {entity.label: entity.id for entity in entities}

            # Convert relationships
            relationships = []
            for relationship_data in extraction_result.get("relationships", []):
                source_label = relationship_data["source"]
                target_label = relationship_data["target"]

                # Check if both source and target entities exist
                if source_label in entity_mapping and target_label in entity_mapping:
                    relationship = Relationship(
                        source_id=entity_mapping[source_label],
                        target_id=entity_mapping[target_label],
                        type=relationship_data["type"],
                        properties=relationship_data.get("properties", {}),
                        document_ids=[doc_id],
                    )
                    relationships.append(relationship)

            return entities, relationships

        except Exception as e:
            logger.error(f"Error extracting entities from document {doc_id}: {str(e)}")
            return [], []

    async def _query_with_graph(
        self,
        query: str,
        auth: AuthContext,
        graph_name: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
    ) -> CompletionResponse:
        """Generate completion using knowledge graph-enhanced retrieval.
        
        This method enhances retrieval by combining vector search with graph traversal:
        1. Performs standard vector similarity search
        2. In parallel, finds relevant entities and their connected documents through graph traversal
        3. Combines both result sets to provide more comprehensive context
        4. Generates a completion with the enhanced context
        
        Args:
            query: The query text
            auth: Authentication context
            graph_name: Name of the graph to use
            filters: Optional metadata filters
            k: Number of chunks to retrieve
            min_score: Minimum similarity score
            max_tokens: Maximum tokens for completion
            temperature: Temperature for completion
            use_reranking: Whether to use reranking
            use_colpali: Whether to use colpali embedding
            hop_depth: Number of relationship hops to traverse (1-3)
            include_paths: Whether to include relationship paths in response
        """
        logger.info(f"Querying with graph: {graph_name}, hop depth: {hop_depth}")
        
        # Step 1: Get the knowledge graph
        graph = await self.db.get_graph(graph_name, auth)
        if not graph:
            logger.warning(f"Graph '{graph_name}' not found or not accessible")
            # Fall back to standard retrieval if graph not found
            return await self.query(
                query=query,
                auth=auth,
                filters=filters,
                k=k,
                min_score=min_score,
                max_tokens=max_tokens,
                temperature=temperature,
                use_reranking=use_reranking,
                use_colpali=use_colpali,
                graph_name=None,
            )
        
        # Step 2: PARALLEL APPROACH - Run both retrieval methods independently
        
        # 2A: Standard vector search
        vector_chunks = await self.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali
        )
        logger.info(f"Vector search retrieved {len(vector_chunks)} chunks")
        
        # 2B: Graph-based retrieval
        # Find relevant entities based on the query
        query_embedding = await self.embedding_model.embed_for_query(query)
        
        # Calculate entity similarity to query by comparing their properties with the query
        entity_similarities = []
        for entity in graph.entities:
            # Create entity text representation
            entity_text = f"{entity.label} {entity.type} " + " ".join(
                f"{key}: {value}" for key, value in entity.properties.items()
            )
            
            # Get entity embedding and calculate similarity
            entity_embedding = await self.embedding_model.embed_for_query(entity_text)
            similarity = self._calculate_cosine_similarity(query_embedding, entity_embedding)
            
            entity_similarities.append((entity, similarity))
        
        # Sort entities by similarity
        entity_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top entities based on k parameter
        top_k_entities = min(k, len(entity_similarities))
        top_entities = entity_similarities[:top_k_entities]
        logger.info(f"Found {len(top_entities)} initial relevant entities")
        
        # Traverse the graph to find related entities
        if hop_depth > 1:
            expanded_entities = self._expand_entities_with_relationships(
                graph, [e[0] for e in top_entities], hop_depth
            )
            logger.info(f"Expanded to {len(expanded_entities)} entities after traversal")
        else:
            expanded_entities = [e[0] for e in top_entities]
        
        # Get document IDs connected to these entities
        graph_doc_ids = set()
        for entity in expanded_entities:
            graph_doc_ids.update(entity.document_ids)
            
        logger.info(f"Graph search found {len(graph_doc_ids)} documents connected to relevant entities")
        
        # Step 3: Get chunks for graph-based documents
        graph_chunks = []
        for doc_id in graph_doc_ids:
            # Get document if authorized
            doc = await self.db.get_document(doc_id, auth)
            if not doc:
                continue
                
            # Check filters if provided
            if filters and not all(doc.metadata.get(k) == v for k, v in filters.items()):
                continue
                
            # Get chunks for this document
            for chunk_id in doc.chunk_ids:
                chunk = await self.vector_store.get_chunk(chunk_id)
                if chunk:
                    graph_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(graph_chunks)} chunks from graph-connected documents")
        
        # Calculate paths if requested
        paths = []
        if include_paths:
            paths = self._find_relationship_paths(graph, [e[0] for e in top_entities], hop_depth)
            logger.info(f"Found {len(paths)} relationship paths")
        
        # Step 4: Combine and deduplicate chunks from both methods
        all_chunks = {}
        
        # Add vector chunks
        for chunk in vector_chunks:
            chunk_key = f"{chunk.document_id}_{chunk.chunk_number}"
            if chunk_key not in all_chunks or chunk.score > all_chunks[chunk_key].score:
                all_chunks[chunk_key] = chunk
        
        # Add graph chunks - with a slight boost to their score to prefer graph results
        for chunk in graph_chunks:
            chunk_key = f"{chunk.document_id}_{chunk.chunk_number}"
            # Add a small boost to graph chunks (5%) and ensure they have a score
            if not hasattr(chunk, 'score') or chunk.score is None:
                chunk.score = 0.7  # Default score
            chunk.score = min(1.0, chunk.score * 1.05)  # 5% boost, capped at 1.0
            
            if chunk_key not in all_chunks or chunk.score > all_chunks[chunk_key].score:
                all_chunks[chunk_key] = chunk
        
        # Convert to list and sort by score
        combined_chunks = list(all_chunks.values())
        combined_chunks.sort(key=lambda x: x.score if hasattr(x, 'score') and x.score is not None else 0, reverse=True)
        
        # Limit to top k
        combined_chunks = combined_chunks[:k]
        logger.info(f"Combined and sorted to {len(combined_chunks)} chunks")
        
        # If we didn't find any chunks, fall back to just the vector chunks
        if not combined_chunks:
            logger.warning("No graph-enhanced chunks found, using regular retrieval")
            combined_chunks = vector_chunks[:k]
        
        # Step 5: Get chunk results with document metadata
        chunk_results = await self._create_chunk_results(auth, combined_chunks)
        documents = await self._create_document_results(auth, chunk_results)
        
        # Create augmented chunk contents
        chunk_contents = [chunk.augmented_content(documents[chunk.document_id]) for chunk in chunk_results]
        
        # Step 6: Include graph context in prompt if paths are requested
        if include_paths and paths:
            # Convert paths to a text representation
            paths_text = "Knowledge Graph Context:\n"
            for path in paths[:5]:  # Limit to 5 paths to avoid token limits
                paths_text += " -> ".join(path) + "\n"
                
            # Prepend to the first chunk
            if chunk_contents:
                chunk_contents[0] = paths_text + "\n\n" + chunk_contents[0]
        
        # Step 7: Generate completion with enhanced context
        request = CompletionRequest(
            query=query,
            context_chunks=chunk_contents,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        response = await self.completion_model.complete(request)
        
        # Include paths in response metadata if requested
        if include_paths:
            # Check if response has metadata attribute
            if not hasattr(response, 'metadata'):
                # Add metadata attribute if it doesn't exist
                response.metadata = {}
            elif response.metadata is None:
                response.metadata = {}
            
            response.metadata["graph"] = {
                "name": graph_name,
                "relevant_entities": [e.label for e in expanded_entities[:10]],
                "paths": paths[:5] if paths else [],
            }
            
        return response
        
    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate dot product and magnitude
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def _expand_entities_with_relationships(self, graph, seed_entities, hop_depth):
        """Expand seed entities by traversing relationships up to hop_depth.
        
        Args:
            graph: Knowledge graph
            seed_entities: Initial entities to expand from
            hop_depth: Maximum number of hops to traverse
            
        Returns:
            List of all connected entities
        """
        # Create a set of entity IDs we've seen
        seen_entity_ids = set(entity.id for entity in seed_entities)
        all_entities = list(seed_entities)
        
        # Create a map from entity ID to entity object for fast lookup
        entity_map = {entity.id: entity for entity in graph.entities}
        
        # For each hop
        for _ in range(hop_depth - 1):  # -1 because seed entities count as hop 0
            new_entities = []
            
            # For each entity we've found so far
            for entity in all_entities:
                # Find relationships where this entity is the source
                for relationship in graph.relationships:
                    if relationship.source_id == entity.id and relationship.target_id not in seen_entity_ids:
                        # Add the target entity if we haven't seen it before
                        target_entity = entity_map.get(relationship.target_id)
                        if target_entity:
                            new_entities.append(target_entity)
                            seen_entity_ids.add(target_entity.id)
                    
                    # Also find relationships where this entity is the target
                    elif relationship.target_id == entity.id and relationship.source_id not in seen_entity_ids:
                        # Add the source entity if we haven't seen it before
                        source_entity = entity_map.get(relationship.source_id)
                        if source_entity:
                            new_entities.append(source_entity)
                            seen_entity_ids.add(source_entity.id)
            
            # Add new entities to our list
            all_entities.extend(new_entities)
            
            # If we didn't find any new entities, we can stop early
            if not new_entities:
                break
                
        return all_entities
    
    def _find_relationship_paths(self, graph, seed_entities, hop_depth):
        """Find meaningful paths in the graph starting from seed entities.
        
        Args:
            graph: Knowledge graph
            seed_entities: Initial entities to start from
            hop_depth: Maximum length of paths
            
        Returns:
            List of paths, where each path is a list of entity labels
        """
        paths = []
        entity_map = {entity.id: entity for entity in graph.entities}
        
        # For each seed entity
        for start_entity in seed_entities:
            # Start a BFS from this entity
            queue = [(start_entity.id, [start_entity.label])]
            visited = set([start_entity.id])
            
            while queue:
                entity_id, path = queue.pop(0)
                
                # If path is already at max length, record it but don't expand
                if len(path) >= hop_depth * 2:  # *2 because path includes relationship types
                    paths.append(path)
                    continue
                
                # Find outgoing relationships
                for relationship in graph.relationships:
                    target_id = None
                    
                    if relationship.source_id == entity_id:
                        target_id = relationship.target_id
                        direction = "outgoing"
                    elif relationship.target_id == entity_id:
                        target_id = relationship.source_id
                        direction = "incoming"
                        
                    if target_id and target_id not in visited:
                        target_entity = entity_map.get(target_id)
                        if target_entity:
                            # Add to visited and queue
                            visited.add(target_id)
                            new_path = path + [f"({relationship.type})", target_entity.label]
                            queue.append((target_id, new_path))
                            
                            # Record this path
                            paths.append(new_path)
        
        return paths

    def close(self):
        """Close all resources."""
        # Close any active caches
        self.active_caches.clear()
