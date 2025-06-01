import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pymilvus import DataType, MilvusClient

from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusMultiVectorStore(BaseVectorStore):
    """Milvus implementation for storing and querying multi-vector embeddings."""

    def __init__(
        self,
        uri: str = "milvus_multi_vector.db",
        collection_name: str = "multi_vector_embeddings",
        dimension: int = 128,
        max_vectors_per_chunk: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_initialize: bool = True,
    ):
        """Initialize Milvus client and collection for multi-vector storage.

        Args:
            uri: Milvus connection URI (for Milvus Lite: local file path, for server: host:port)
            collection_name: Name of the collection to store embeddings
            dimension: Dimension of each individual vector in the multi-vector
            max_vectors_per_chunk: Maximum number of vectors per chunk (for schema definition)
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
            auto_initialize: Whether to automatically initialize the collection
        """
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.max_vectors_per_chunk = max_vectors_per_chunk
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize Milvus client
        self.client = MilvusClient(uri=self.uri)

        if auto_initialize:
            try:
                self.initialize()
            except Exception as exc:
                logger.error("Auto-initialization of MilvusMultiVectorStore failed: %s", exc)

    def initialize(self):
        """Initialize collection if it doesn't exist."""
        try:
            # Check if collection exists
            if not self.client.has_collection(collection_name=self.collection_name):
                # Create collection with schema
                schema = self.client.create_schema(
                    auto_id=True,
                    enable_dynamic_field=True,  # Enable for flexible vector storage
                )

                # Add fields
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="chunk_number", datatype=DataType.INT64)
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="vector_count", datatype=DataType.INT64)

                # Add multiple vector fields for multi-vector support
                for i in range(self.max_vectors_per_chunk):
                    schema.add_field(field_name=f"vector_{i}", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)

                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    metric_type="COSINE",
                    consistency_level="Strong",
                )

                # Create index on first vector field (primary search field)
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="vector_0", index_type="AUTOINDEX", metric_type="COSINE")
                self.client.create_index(collection_name=self.collection_name, index_params=index_params)

                logger.info(f"Created Milvus multi-vector collection: {self.collection_name}")
            else:
                logger.info(f"Milvus multi-vector collection {self.collection_name} already exists")

            return True
        except Exception as e:
            logger.error(f"Error initializing MilvusMultiVectorStore: {str(e)}")
            return False

    def _prepare_multi_vectors(
        self, embeddings: Union[np.ndarray, torch.Tensor, List]
    ) -> Tuple[List[List[float]], int]:
        """Convert multi-vector embeddings to the expected format."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(embeddings, list) and not isinstance(embeddings[0], np.ndarray):
            embeddings = np.array(embeddings)

        # Convert to list of lists
        if embeddings.ndim == 2:
            vectors = [embedding.tolist() for embedding in embeddings]
        else:
            vectors = [embeddings.tolist()]

        return vectors, len(vectors)

    def _calculate_max_similarity(self, doc_vectors: List[List[float]], query_vectors: List[List[float]]) -> float:
        """Calculate maximum similarity between document vectors and query vectors."""
        max_similarities = []

        for query_vec in query_vectors:
            query_array = np.array(query_vec)
            max_sim = 0.0

            for doc_vec in doc_vectors:
                doc_array = np.array(doc_vec)
                # Calculate cosine similarity
                dot_product = np.dot(query_array, doc_array)
                norm_query = np.linalg.norm(query_array)
                norm_doc = np.linalg.norm(doc_array)

                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                    max_sim = max(max_sim, similarity)

            max_similarities.append(max_sim)

        return sum(max_similarities)

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their multi-vector embeddings."""
        try:
            # Prepare data for insertion
            data = []
            stored_ids = []

            for chunk in chunks:
                if not hasattr(chunk, "embedding") or chunk.embedding is None:
                    logger.error(f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}")
                    continue

                # Convert multi-vector embeddings to proper format
                vectors, vector_count = self._prepare_multi_vectors(chunk.embedding)

                if vector_count > self.max_vectors_per_chunk:
                    logger.warning(f"Chunk has {vector_count} vectors, truncating to {self.max_vectors_per_chunk}")
                    vectors = vectors[: self.max_vectors_per_chunk]
                    vector_count = self.max_vectors_per_chunk

                # Prepare metadata as JSON string
                metadata_str = str(chunk.metadata) if chunk.metadata else "{}"

                # Prepare row data
                row_data = {
                    "document_id": chunk.document_id,
                    "chunk_number": chunk.chunk_number,
                    "content": chunk.content,
                    "metadata": metadata_str,
                    "vector_count": vector_count,
                }

                # Add vector fields
                for i in range(self.max_vectors_per_chunk):
                    if i < len(vectors):
                        row_data[f"vector_{i}"] = vectors[i]
                    else:
                        # Pad with zero vectors
                        row_data[f"vector_{i}"] = [0.0] * self.dimension

                data.append(row_data)
                stored_ids.append(f"{chunk.document_id}-{chunk.chunk_number}")

            if not data:
                return True, []

            # Insert data into Milvus
            self.client.insert(collection_name=self.collection_name, data=data)

            logger.debug(f"{len(stored_ids)} multi-vector embeddings stored in Milvus")
            return True, stored_ids

        except Exception as e:
            logger.error(f"Error storing multi-vector embeddings in Milvus: {str(e)}")
            return False, []

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using multi-vector similarity search."""
        try:
            # Convert query embeddings to proper format
            query_vectors, _ = self._prepare_multi_vectors(query_embedding)

            # Use the first query vector for initial search
            # Then re-rank using multi-vector similarity
            search_params = {
                "collection_name": self.collection_name,
                "data": [query_vectors[0]],
                "anns_field": "vector_0",
                "limit": k * 10,  # Get more results for re-ranking
                "output_fields": ["document_id", "chunk_number", "content", "metadata", "vector_count"]
                + [f"vector_{i}" for i in range(self.max_vectors_per_chunk)],
            }

            # Add filter if doc_ids are specified
            if doc_ids:
                filter_expr = f"document_id in {doc_ids}"
                search_params["filter"] = filter_expr

            # Perform initial search
            results = self.client.search(**search_params)

            # Re-rank using multi-vector similarity
            scored_chunks = []
            for result in results[0]:  # results is a list of lists
                try:
                    # Parse metadata
                    metadata = eval(result["entity"]["metadata"]) if result["entity"]["metadata"] else {}
                except Exception:
                    metadata = {}

                # Extract document vectors
                vector_count = result["entity"]["vector_count"]
                doc_vectors = []
                for i in range(vector_count):
                    if f"vector_{i}" in result["entity"]:
                        doc_vectors.append(result["entity"][f"vector_{i}"])

                # Calculate max similarity score
                if doc_vectors and query_vectors:
                    similarity_score = self._calculate_max_similarity(doc_vectors, query_vectors)
                else:
                    similarity_score = float(result["distance"])

                chunk = DocumentChunk(
                    document_id=result["entity"]["document_id"],
                    chunk_number=result["entity"]["chunk_number"],
                    content=result["entity"]["content"],
                    embedding=[],  # Don't return embeddings
                    metadata=metadata,
                    score=similarity_score,
                )
                scored_chunks.append(chunk)

            # Sort by score and return top k
            scored_chunks.sort(key=lambda x: x.score, reverse=True)
            return scored_chunks[:k]

        except Exception as e:
            logger.error(f"Error querying similar multi-vector chunks in Milvus: {str(e)}")
            return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
    ) -> List[DocumentChunk]:
        """Retrieve specific chunks by document ID and chunk number."""
        try:
            if not chunk_identifiers:
                return []

            # Build filter expression for specific chunks
            conditions = []
            for doc_id, chunk_num in chunk_identifiers:
                conditions.append(f"(document_id == '{doc_id}' && chunk_number == {chunk_num})")

            filter_expr = " || ".join(conditions)

            # Query with filter
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["document_id", "chunk_number", "content", "metadata"],
            )

            # Convert to DocumentChunks
            chunks = []
            for result in results:
                try:
                    metadata = eval(result["metadata"]) if result["metadata"] else {}
                except Exception:
                    metadata = {}

                chunk = DocumentChunk(
                    document_id=result["document_id"],
                    chunk_number=result["chunk_number"],
                    content=result["content"],
                    embedding=[],  # Don't return embeddings
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
            # Delete all chunks for the specified document
            self.client.delete(collection_name=self.collection_name, filter=f"document_id == '{document_id}'")

            logger.info(f"Deleted all chunks for document {document_id} from Milvus multi-vector store")
            return True

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id} from Milvus multi-vector store: {str(e)}")
            return False

    def close(self):
        """Close the Milvus client connection."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing Milvus multi-vector client: {e}")
