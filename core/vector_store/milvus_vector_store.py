import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from pymilvus import DataType, MilvusClient

from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """Milvus implementation for storing and querying vector embeddings."""

    def __init__(
        self,
        uri: str = "milvus_lite.db",
        collection_name: str = "document_embeddings",
        dimension: int = 768,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_initialize: bool = True,
    ):
        """Initialize Milvus client and collection.

        Args:
            uri: Milvus connection URI (for Milvus Lite: local file path, for server: host:port)
            collection_name: Name of the collection to store embeddings
            dimension: Dimension of the embeddings
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
            auto_initialize: Whether to automatically initialize the collection
        """
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize Milvus client
        self.client = MilvusClient(uri=self.uri)

        if auto_initialize:
            try:
                self.initialize()
            except Exception as exc:
                logger.error("Auto-initialization of MilvusVectorStore failed: %s", exc)

    def initialize(self):
        """Initialize collection if it doesn't exist."""
        try:
            # Check if collection exists
            if not self.client.has_collection(collection_name=self.collection_name):
                # Create collection with schema
                schema = self.client.create_schema(
                    auto_id=True,
                    enable_dynamic_field=False,
                )

                # Add fields
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="chunk_number", datatype=DataType.INT64)
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)

                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    metric_type="COSINE",
                    consistency_level="Strong",
                )

                # Create index on vector field
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
                self.client.create_index(collection_name=self.collection_name, index_params=index_params)

                logger.info(f"Created Milvus collection: {self.collection_name}")
            else:
                logger.info(f"Milvus collection {self.collection_name} already exists")

            return True
        except Exception as e:
            logger.error(f"Error initializing MilvusVectorStore: {str(e)}")
            return False

    def _prepare_embeddings(self, embeddings: Union[np.ndarray, List[float]]) -> List[float]:
        """Convert embeddings to the expected format."""
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their embeddings."""
        try:
            # Prepare data for insertion
            data = []
            stored_ids = []

            for chunk in chunks:
                if not hasattr(chunk, "embedding") or chunk.embedding is None:
                    logger.error(f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}")
                    continue

                # Convert embedding to proper format
                vector = self._prepare_embeddings(chunk.embedding)

                # Prepare metadata as JSON string
                metadata_str = str(chunk.metadata) if chunk.metadata else "{}"

                data.append(
                    {
                        "document_id": chunk.document_id,
                        "chunk_number": chunk.chunk_number,
                        "content": chunk.content,
                        "metadata": metadata_str,
                        "vector": vector,
                    }
                )

                stored_ids.append(f"{chunk.document_id}-{chunk.chunk_number}")

            if not data:
                return True, []

            # Insert data into Milvus
            self.client.insert(collection_name=self.collection_name, data=data)

            logger.debug(f"{len(stored_ids)} embeddings stored in Milvus")
            return True, stored_ids

        except Exception as e:
            logger.error(f"Error storing embeddings in Milvus: {str(e)}")
            return False, []

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using vector similarity search."""
        try:
            # Convert query embedding to proper format
            query_vector = self._prepare_embeddings(query_embedding)

            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "limit": k,
                "output_fields": ["document_id", "chunk_number", "content", "metadata"],
            }

            # Add filter if doc_ids are specified
            if doc_ids:
                filter_expr = f"document_id in {doc_ids}"
                search_params["filter"] = filter_expr

            # Perform search
            results = self.client.search(**search_params)

            # Convert results to DocumentChunks
            chunks = []
            for result in results[0]:  # results is a list of lists
                try:
                    # Parse metadata
                    metadata = eval(result["entity"]["metadata"]) if result["entity"]["metadata"] else {}
                except Exception:
                    metadata = {}

                chunk = DocumentChunk(
                    document_id=result["entity"]["document_id"],
                    chunk_number=result["entity"]["chunk_number"],
                    content=result["entity"]["content"],
                    embedding=[],  # Don't return embeddings
                    metadata=metadata,
                    score=float(result["distance"]),
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error querying similar chunks in Milvus: {str(e)}")
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

            logger.debug(f"Found {len(chunks)} chunks in batch retrieval from Milvus")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by ID from Milvus: {str(e)}")
            return []

    async def delete_chunks_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks associated with a document."""
        try:
            # Delete all chunks for the specified document
            self.client.delete(collection_name=self.collection_name, filter=f"document_id == '{document_id}'")

            logger.info(f"Deleted all chunks for document {document_id} from Milvus")
            return True

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id} from Milvus: {str(e)}")
            return False

    def close(self):
        """Close the Milvus client connection."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing Milvus client: {e}")
