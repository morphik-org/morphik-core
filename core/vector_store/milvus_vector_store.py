import json
from collections import defaultdict
from logging import getLogger
from typing import List, Optional, Tuple

from pymilvus import DataType, MilvusClient

from core.config import get_settings
from core.models.chunk import DocumentChunk
from core.vector_store.base_vector_store import BaseVectorStore

settings = get_settings()

logger = getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    def __init__(self):
        self.client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_API_KEY)
        if not self.client.has_collection(collection_name="vector_store"):
            self._create_collection()
        self.client.load_collection(collection_name="vector_store")

    def _create_collection(self):
        """Create the vector_store collection with the proper schema."""
        # Create schema that matches our data structure
        schema = self.client.create_schema(
            auto_id=True,  # Auto-generate IDs
            enable_dynamic_fields=True,  # Allow dynamic fields
        )

        # Add fields that match our data structure
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIMENSIONS)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="chunk_number", datatype=DataType.INT32)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)

        # Create the collection
        self.client.create_collection(collection_name="vector_store", schema=schema)

        # Create index for vector field
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 1024})

        self.client.create_index(collection_name="vector_store", index_params=index_params)

        logger.info("Created vector_store collection with proper schema")

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
        data = []
        for chunk in chunks:
            # Truncate content if it's too long for Milvus VARCHAR field
            truncated_content = self._truncate_content(chunk.content)

            # Truncate metadata JSON if needed
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
            truncated_metadata = self._truncate_content(metadata_json, max_length=65000)

            # Create the data structure that Milvus expects
            # Note: we don't include 'id' since it's auto-generated
            chunk_data = {
                "document_id": chunk.document_id,
                "chunk_number": chunk.chunk_number,
                "content": truncated_content,
                "metadata": truncated_metadata,
                "vector": chunk.embedding,
            }
            data.append(chunk_data)

        try:
            result = self.client.insert(collection_name="vector_store", data=data)
            logger.info(f"Successfully inserted {len(data)} chunks into Milvus: {result}")
            return True, [f"{chunk.document_id}-{chunk.chunk_number}" for chunk in chunks]
        except Exception as e:
            logger.error(f"Error inserting chunks into Milvus: {e}")
            # Re-raise the exception so the caller knows the operation failed
            raise

    async def query_similar(self, query_embedding: List[float], k: int, doc_ids: Optional[List[str]] = None):
        res = self.client.search(
            collection_name="vector_store",
            data=[query_embedding],
            limit=k,
            output_fields=["document_id", "chunk_number", "content", "metadata"],
            filter=f"document_id in {doc_ids}" if doc_ids else None,
        )
        chunks = []
        for r in res[0]:
            # Parse metadata back from JSON string
            try:
                metadata = json.loads(r["metadata"]) if r.get("metadata") else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            chunk = DocumentChunk(
                document_id=r["document_id"],
                chunk_number=r["chunk_number"],
                content=r["content"],
                metadata=metadata,
                embedding=[],  # embedding is not returned
            )
            chunks.append(chunk)
        return chunks

    async def get_chunks_by_id(self, chunk_identifiers: List[Tuple[str, int]]):
        grouped = defaultdict(list)
        for doc_id, chunk_num in chunk_identifiers:
            grouped[doc_id].append(chunk_num)
        filter_statements = []
        for doc_id, chunk_nums in grouped.items():
            filter_statements.append(f"document_id == '{doc_id}' AND chunk_number IN {chunk_nums}")
        filter_exp = " OR ".join(filter_statements)
        res = self.client.query(
            collection_name="vector_store",
            filter=filter_exp,
            output_fields=["document_id", "chunk_number", "content", "metadata"],
        )
        chunks = []
        for r in res:
            # Parse metadata back from JSON string
            try:
                metadata = json.loads(r["metadata"]) if r.get("metadata") else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            chunk = DocumentChunk(
                document_id=r["document_id"],
                chunk_number=r["chunk_number"],
                content=r["content"],
                metadata=metadata,
                embedding=[],  # embedding is not returned
            )
            chunks.append(chunk)
        return chunks

    async def delete_chunks_by_document_id(self, document_id: str):
        res = self.client.delete(collection_name="vector_store", filter=f"document_id == '{document_id}'")
        logger.info(f"Deleted chunks on milvus: {res}")
        return True
