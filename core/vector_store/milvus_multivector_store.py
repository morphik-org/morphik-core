import asyncio
import concurrent.futures
import json
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pymilvus import DataType, MilvusClient

from core.config import get_settings
from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

settings = get_settings()


class MilvusMultiVectorStore(BaseVectorStore):
    """Milvus implementation for storing and querying multi-vector embeddings with MaxSim scoring."""

    def __init__(
        self,
        collection_name: str = "multi_vector_embeddings",
        embedding_dim: int = 128,
        max_workers: int = 300,
        batch_size: int = 1000,
        auto_initialize: bool = True,
    ):
        """Initialize Milvus connection for multi-vector storage.

        Args:
            collection_name: Name of the collection to store embeddings
            embedding_dim: Dimension of individual embeddings in the multi-vector
            max_workers: Maximum number of workers for parallel processing
            batch_size: Number of vectors to insert per batch (to avoid gRPC limits)
            auto_initialize: Whether to automatically initialize the store
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Initialize Milvus client
        self.client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_API_KEY)

        # Optionally initialize collection
        if auto_initialize:
            try:
                self.initialize()
            except Exception as exc:
                logger.error("Auto-initialization of MilvusMultiVectorStore failed: %s", exc)

    def initialize(self):
        """Initialize the Milvus collection with proper schema for multi-vector storage."""
        try:
            # Check if collection exists and load it
            if self.client.has_collection(collection_name=self.collection_name):
                self.client.load_collection(collection_name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                # Create new collection with schema for multi-vector storage
                self._create_collection()
                logger.info(f"Created new collection: {self.collection_name}")

            # Create indexes for efficient search
            self._create_indexes()

            logger.info("MilvusMultiVectorStore initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MilvusMultiVectorStore: {str(e)}")
            return False

    def _create_collection(self):
        """Create collection with schema optimized for multi-vector storage."""
        # Drop existing collection if it exists
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        # Create schema following the ColPali pattern from the documentation
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )

        # Primary key field
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)

        # Vector field for individual embeddings
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)

        # Sequence ID to track position within multi-vector
        schema.add_field(field_name="seq_id", datatype=DataType.INT32)

        # Document identifier
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=255)

        # Chunk number
        schema.add_field(field_name="chunk_number", datatype=DataType.INT32)

        # Content - increase max length to handle larger chunks
        # Note: Milvus VARCHAR max is ~65535, so we'll use a larger limit and truncate if needed
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

        # Metadata as JSON string
        schema.add_field(field_name="chunk_metadata", datatype=DataType.VARCHAR, max_length=65535)

        # Create the collection
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

    def _create_indexes(self):
        """Create indexes for efficient search operations."""
        try:
            # Release collection before creating indexes
            self.client.release_collection(collection_name=self.collection_name)

            # Create vector index for similarity search
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector_index",
                index_type="IVF_FLAT",
                metric_type="IP",  # Inner Product for MaxSim calculation
                params={
                    "M": 16,
                    "efConstruction": 500,
                },
            )

            self.client.create_index(collection_name=self.collection_name, index_params=index_params, sync=True)

            # Create scalar index for document_id field
            self.client.release_collection(collection_name=self.collection_name)

            scalar_index_params = self.client.prepare_index_params()
            scalar_index_params.add_index(
                field_name="document_id",
                index_name="document_id_index",
                index_type="INVERTED",
            )

            self.client.create_index(collection_name=self.collection_name, index_params=scalar_index_params, sync=True)

            # Load collection after creating indexes
            self.client.load_collection(collection_name=self.collection_name)

        except Exception as e:
            logger.warning(f"Failed to create indexes: {str(e)}")
            # Continue even if index creation fails

    def _normalize_embeddings(self, embeddings: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert embeddings to normalized numpy array format."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(embeddings, list) and not isinstance(embeddings[0], np.ndarray):
            embeddings = np.array(embeddings)

        # Ensure 2D array (num_vectors, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings.astype(np.float32)

    def _estimate_batch_size(self, sample_row: dict) -> int:
        """Estimate safe batch size based on data size to avoid gRPC limits."""
        import sys

        # Estimate size of a single row
        row_size = sys.getsizeof(json.dumps(sample_row))

        # Conservative estimate: aim for batches under 100MB to stay well under 512MB limit
        # Account for gRPC overhead and serialization
        target_batch_size_mb = 50  # 50MB per batch
        target_batch_size_bytes = target_batch_size_mb * 1024 * 1024

        estimated_batch_size = max(1, target_batch_size_bytes // row_size)

        # Cap at configured batch_size and minimum of 10
        safe_batch_size = min(self.batch_size, max(10, estimated_batch_size))

        logger.debug(f"Estimated row size: {row_size} bytes, calculated batch size: {safe_batch_size}")
        return safe_batch_size

    def _truncate_content(self, content: str, max_length: int = 65000) -> str:
        """Truncate content to fit within Milvus VARCHAR limits.

        Args:
            content: The content to truncate
            max_length: Maximum length (default 65000 to leave some buffer)

        Returns:
            Truncated content with truncation indicator if needed
        """
        if len(content) <= max_length:
            return content

        # Truncate and add indicator
        truncated = content[: max_length - 50] + "... [TRUNCATED BY MILVUS STORAGE]"
        logger.warning(f"Content truncated from {len(content)} to {len(truncated)} characters for Milvus storage")
        return truncated

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their multi-vector embeddings."""
        # Prepare data for batch insertion
        data_rows = []

        for chunk in chunks:
            if not hasattr(chunk, "embedding") or chunk.embedding is None:
                logger.error(f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}")
                continue

            # Normalize embeddings to proper format
            normalized_embeddings = self._normalize_embeddings(chunk.embedding)

            # Truncate content if it's too long for Milvus VARCHAR field
            truncated_content = self._truncate_content(chunk.content)

            # Truncate metadata JSON if needed
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
            truncated_metadata = self._truncate_content(metadata_json, max_length=65000)

            # Create a row for each vector in the multi-vector embedding
            for seq_id, vector in enumerate(normalized_embeddings):
                data_rows.append(
                    {
                        "vector": vector.tolist(),
                        "seq_id": seq_id,
                        "document_id": chunk.document_id,
                        "chunk_number": chunk.chunk_number,
                        "content": truncated_content,
                        "chunk_metadata": truncated_metadata,
                    }
                )

        if not data_rows:
            return True, []

        # Log the size of the data being inserted
        logger.info(f"Preparing to insert {len(data_rows)} vectors for {len(chunks)} chunks")

        # Dynamically calculate safe batch size based on actual data
        safe_batch_size = self._estimate_batch_size(data_rows[0]) if data_rows else self.batch_size

        try:
            for i in range(0, len(data_rows), safe_batch_size):
                batch = data_rows[i : i + safe_batch_size]
                batch_num = i // safe_batch_size + 1
                total_batches = (len(data_rows) + safe_batch_size - 1) // safe_batch_size

                logger.info(f"Inserting batch {batch_num}/{total_batches} with {len(batch)} vectors")

                # Use asyncio.to_thread for non-blocking insertion
                await asyncio.to_thread(self._bulk_insert_rows, batch)

                # Small delay between batches to avoid overwhelming the server
                if i + safe_batch_size < len(data_rows):
                    await asyncio.sleep(0.1)

            stored_ids = [f"{chunk.document_id}-{chunk.chunk_number}" for chunk in chunks]
            logger.info(f"{len(stored_ids)} multi-vector embeddings stored successfully in {total_batches} batches")
            return True, stored_ids

        except Exception as e:
            logger.error(f"Error storing multi-vector embeddings: {str(e)}")
            # Re-raise the exception so the caller knows the operation failed
            raise

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using MaxSim scoring for multi-vectors."""
        # Normalize query embeddings
        query_vectors = self._normalize_embeddings(query_embedding)

        # Execute search in thread pool to avoid blocking
        return await asyncio.to_thread(self._execute_maxsim_search, query_vectors, k, doc_ids)

    def _execute_maxsim_search(
        self, query_vectors: np.ndarray, k: int, doc_ids: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """Execute MaxSim search following the ColPali pattern."""

        # Step 1: Perform initial vector search to get candidate documents
        search_params = {"metric_type": "IP", "params": {}}

        # Build filter expression for document filtering
        filter_expr = None
        if doc_ids:
            # Escape document IDs and create filter
            escaped_ids = [f'"{doc_id}"' for doc_id in doc_ids]
            filter_expr = f"document_id in [{', '.join(escaped_ids)}]"

        # Search with multiple query vectors to get broader candidate set
        all_candidate_docs = set()

        for query_vector in query_vectors:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector.tolist()],
                    limit=50,  # Get broader set of candidates
                    output_fields=["document_id", "chunk_number"],
                    search_params=search_params,
                    filter=filter_expr,
                )

                # Extract unique (document_id, chunk_number) pairs
                for result_set in results:
                    for result in result_set:
                        doc_id = result["entity"]["document_id"]
                        chunk_num = result["entity"]["chunk_number"]
                        all_candidate_docs.add((doc_id, chunk_num))

            except Exception as e:
                logger.error(f"Error in initial search: {str(e)}")
                continue

        if not all_candidate_docs:
            return []

        # Step 2: Rerank using MaxSim scoring
        scores = []

        def rerank_single_chunk(doc_chunk_pair, query_vectors, client, collection_name):
            """Calculate MaxSim score for a single chunk."""
            doc_id, chunk_num = doc_chunk_pair

            try:
                # Get all vectors for this specific chunk
                chunk_vectors = client.query(
                    collection_name=collection_name,
                    filter=f'document_id == "{doc_id}" && chunk_number == {chunk_num}',
                    output_fields=["vector", "seq_id", "content", "chunk_metadata"],
                    limit=2000,  # Should be enough for most multi-vector embeddings
                )

                if not chunk_vectors:
                    return (0.0, doc_id, chunk_num, "", {})

                # Extract vectors and sort by seq_id to maintain order
                sorted_vectors = sorted(chunk_vectors, key=lambda x: x["seq_id"])
                doc_vecs = np.vstack([v["vector"] for v in sorted_vectors])

                # Calculate MaxSim score: for each query vector, find max similarity with doc vectors
                # Then sum across all query vectors
                similarity_matrix = np.dot(query_vectors, doc_vecs.T)  # (num_query_vecs, num_doc_vecs)
                max_similarities = np.max(similarity_matrix, axis=1)  # Max for each query vector
                maxsim_score = np.sum(max_similarities)  # Sum across query vectors

                # Get content and metadata from the first vector entry
                content = sorted_vectors[0].get("content", "")
                metadata_str = sorted_vectors[0].get("chunk_metadata", "{}")
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing metadata for {doc_id}-{chunk_num}: {str(e)}")
                    metadata = {}

                return (maxsim_score, doc_id, chunk_num, content, metadata)

            except Exception as e:
                logger.error(f"Error calculating MaxSim for {doc_id}-{chunk_num}: {str(e)}")
                return (0.0, doc_id, chunk_num, "", {})

        # Use ThreadPoolExecutor for parallel reranking
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    rerank_single_chunk, doc_chunk_pair, query_vectors, self.client, self.collection_name
                ): doc_chunk_pair
                for doc_chunk_pair in all_candidate_docs
            }

            for future in concurrent.futures.as_completed(futures):
                score, doc_id, chunk_num, content, metadata = future.result()
                scores.append((score, doc_id, chunk_num, content, metadata))

        # Sort by score and return top-k results
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:k]

        # Convert to DocumentChunk objects
        chunks = []
        for score, doc_id, chunk_num, content, metadata in top_scores:
            chunk = DocumentChunk(
                document_id=doc_id,
                chunk_number=chunk_num,
                content=content,
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=float(score),
            )
            chunks.append(chunk)

        return chunks

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
    ) -> List[DocumentChunk]:
        """Retrieve specific chunks by document ID and chunk number."""
        if not chunk_identifiers:
            return []

        # Build filter expressions for each chunk
        filter_conditions = []
        for doc_id, chunk_num in chunk_identifiers:
            filter_conditions.append(f'(document_id == "{doc_id}" && chunk_number == {chunk_num})')

        filter_expr = " || ".join(filter_conditions)

        try:
            # Query for the specific chunks
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["document_id", "chunk_number", "content", "chunk_metadata"],
                limit=len(chunk_identifiers) * 1000,  # Account for multi-vector nature
            )

            # Group results by (document_id, chunk_number) and take first occurrence
            seen_chunks = set()
            chunks = []

            for result in results:
                doc_id = result["document_id"]
                chunk_num = result["chunk_number"]
                chunk_key = (doc_id, chunk_num)

                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)

                    try:
                        metadata = json.loads(result["chunk_metadata"]) if result["chunk_metadata"] else {}
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing metadata for {doc_id}-{chunk_num}: {str(e)}")
                        metadata = {}

                    chunk = DocumentChunk(
                        document_id=doc_id,
                        chunk_number=chunk_num,
                        content=result["content"],
                        embedding=[],  # Don't send embeddings back
                        metadata=metadata,
                        score=0.0,  # No relevance score for direct retrieval
                    )
                    chunks.append(chunk)

            logger.debug(f"Found {len(chunks)} chunks in batch retrieval from Milvus multi-vector store")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by ID from Milvus multi-vector store: {str(e)}")
            return []

    async def delete_chunks_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks associated with a document."""
        try:
            # Delete all entities for the specified document
            self.client.delete(collection_name=self.collection_name, filter=f'document_id == "{document_id}"')

            logger.info(f"Deleted all chunks for document {document_id} from Milvus multi-vector store")
            return True

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id} from Milvus multi-vector store: {str(e)}")
            return False

    def close(self):
        """Close the Milvus connection."""
        try:
            # Milvus client doesn't have an explicit close method, but we can release the collection
            if hasattr(self.client, "release_collection"):
                self.client.release_collection(collection_name=self.collection_name)
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {e}")

    # Internal helper methods

    def _bulk_insert_rows(self, data_rows: List[dict]):
        """Sync helper executed in a worker thread to avoid blocking."""
        try:
            result = self.client.insert(collection_name=self.collection_name, data=data_rows)
            logger.debug(f"Bulk inserted {len(data_rows)} vectors into Milvus")
            return result
        except Exception as e:
            logger.error(f"Error in bulk insert: {str(e)}")
            raise
