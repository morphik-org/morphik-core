import asyncio
import base64
import json
import logging
from typing import List, Optional, Tuple, Union

import fixed_dimensional_encoding as fde
import numpy as np
import torch
from turbopuffer import AsyncTurbopuffer

from core.config import get_settings
from core.models.chunk import DocumentChunk
from core.storage.base_storage import BaseStorage
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.storage.utils_file_extensions import detect_file_type

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Constants for external storage
MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"
DEFAULT_APP_ID = "default"  # Fallback for local usage when app_id is None


# external storage always enabled, no two ways about it
class FastMultiVectorStore(BaseVectorStore):
    def __init__(self, uri: str, tpuf_api_key: str, namespace: str = "public", region: str = "aws-us-west-2"):
        self.uri = uri
        self.tpuf_api_key = tpuf_api_key
        self.namespace = namespace
        self.tpuf = AsyncTurbopuffer(api_key=tpuf_api_key, region=region)
        self.ns = self.tpuf.namespace(namespace)
        self.storage = self._init_storage()
        self.fde_config = fde.FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=5,
            projection_dimension=16,
            projection_type="AMS_SKETCH",
        )

    def _init_storage(self) -> BaseStorage:
        """Initialize appropriate storage backend based on settings."""
        settings = get_settings()
        match settings.STORAGE_PROVIDER:
            case "aws-s3":
                logger.info("Initializing S3 storage for multi-vector chunks")
                return S3Storage(
                    aws_access_key=settings.AWS_ACCESS_KEY,
                    aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    default_bucket=MULTIVECTOR_CHUNKS_BUCKET,
                )
            case "local":
                logger.info("Initializing local storage for multi-vector chunks")
                storage_path = getattr(settings, "LOCAL_STORAGE_PATH", "./storage")
                return LocalStorage(storage_path=storage_path)
            case _:
                raise ValueError(f"Unsupported storage provider: {settings.STORAGE_PROVIDER}")

    def initialize(self):
        asyncio.run(self.ns.hint_cache_warm())

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        #  group fde calls for better cache hit rate
        embeddings = [
            fde.generate_document_encoding(np.array(chunk.embedding), self.fde_config).tolist() for chunk in chunks
        ]
        storage_keys = await asyncio.gather(*[self._save_chunk_to_storage(chunk) for chunk in chunks])
        stored_ids = [f"{chunk.document_id}-{chunk.chunk_number}" for chunk in chunks]
        result = await self.ns.write(
            upsert_columns={
                "id": stored_ids,
                "vector": embeddings,
                "document_id": [chunk.document_id for chunk in chunks],
                "chunk_number": [chunk.chunk_number for chunk in chunks],
                "content": storage_keys,
                "metadata": [json.dumps(chunk.metadata) for chunk in chunks],
            },
            distance_metric="cosine_distance",
        )
        logger.info(f"Stored {len(chunks)} chunks, tpuf ns: {result.model_dump_json()}")
        return True, stored_ids

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        elif isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        query_embedding = fde.generate_query_encoding(query_embedding, self.fde_config).tolist()
        result = await self.ns.query(
            filters=("document_id", "In", doc_ids),
            rank_by=("vector", "ANN", query_embedding),
            top_k=k,
            include_attributes=["id", "document_id", "chunk_number", "content", "metadata"],
        )
        storage_retrieval_tasks = [
            self._retrieve_content_from_storage(r["content"], json.loads(r["metadata"])) for r in result.rows
        ]
        contents = await asyncio.gather(*storage_retrieval_tasks)
        return [
            DocumentChunk(
                document_id=row["document_id"],
                embedding=[],
                chunk_number=row["chunk_number"],
                content=content,
                metadata=json.loads(row["metadata"]),
                score=1 - row["$dist"],
            )
            for row, content in zip(result.rows, contents)
        ]

    async def get_chunks_by_id(self, chunk_identifiers: List[Tuple[str, int]]) -> List[DocumentChunk]:
        result = await self.ns.query(
            filters=("id", "In", [f"{doc_id}-{chunk_num}" for doc_id, chunk_num in chunk_identifiers]),
            include_attributes=["id", "document_id", "chunk_number", "content", "metadata"],
        )
        storage_retrieval_tasks = [
            self._retrieve_content_from_storage(r["content"], json.loads(r["metadata"])) for r in result.rows
        ]
        contents = await asyncio.gather(*storage_retrieval_tasks)
        return [
            DocumentChunk(
                document_id=row["document_id"],
                embedding=[],
                chunk_number=row["chunk_number"],
                content=content,
                metadata=json.loads(row["metadata"]),
                score=0.0,
            )
            for row, content in zip(result.rows, contents)
        ]

    async def delete_chunks_by_document_id(self, document_id: str) -> bool:
        return await self.ns.write(delete_by_filter=("document_id", "=", document_id))

    async def _get_document_app_id(self, document_id: str) -> str:
        """Get app_id for a document, with caching."""
        if document_id in self._document_app_id_cache:
            return self._document_app_id_cache[document_id]

        try:
            query = "SELECT system_metadata->>'app_id' FROM documents WHERE external_id = %s"
            with self.get_connection() as conn:
                result = conn.execute(query, (document_id,)).fetchone()

            app_id = result[0] if result and result[0] else DEFAULT_APP_ID
            self._document_app_id_cache[document_id] = app_id
            return app_id
        except Exception as e:
            logger.warning(f"Failed to get app_id for document {document_id}: {e}")
            return DEFAULT_APP_ID

    def _determine_file_extension(self, content: str, chunk_metadata: Optional[str]) -> str:
        """Determine appropriate file extension based on content and metadata."""
        try:
            # Parse chunk metadata to check if it's an image
            if chunk_metadata:
                metadata = json.loads(chunk_metadata)
                is_image = metadata.get("is_image", False)

                if is_image:
                    # For images, auto-detect from base64 content
                    return detect_file_type(content)
                else:
                    # For text content, use .txt
                    return ".txt"
            else:
                # No metadata, try to auto-detect
                return detect_file_type(content)

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error parsing chunk metadata: {e}")
            # Fallback to auto-detection
            return detect_file_type(content)

    def _generate_storage_key(self, app_id: str, document_id: str, chunk_number: int, extension: str) -> str:
        """Generate storage key path."""
        return f"{app_id}/{document_id}/{chunk_number}{extension}"

    async def _store_content_externally(
        self, content: str, document_id: str, chunk_number: int, chunk_metadata: Optional[str]
    ) -> Optional[str]:
        """Store chunk content in external storage and return storage key."""
        if not self.storage:
            return None

        try:
            # Get app_id for this document
            app_id = await self._get_document_app_id(document_id)

            # Determine file extension
            extension = self._determine_file_extension(content, chunk_metadata)

            # Generate storage key
            storage_key = self._generate_storage_key(app_id, document_id, chunk_number, extension)

            # Store content in external storage
            if extension == ".txt":
                # For text content, store as-is without base64 encoding
                # Convert content to base64 for storage interface compatibility
                content_bytes = content.encode("utf-8")
                content_b64 = base64.b64encode(content_bytes).decode("utf-8")
                await self.storage.upload_from_base64(
                    content=content_b64, key=storage_key, content_type="text/plain", bucket=MULTIVECTOR_CHUNKS_BUCKET
                )
            else:
                # For images, content should already be base64
                await self.storage.upload_from_base64(
                    content=content, key=storage_key, bucket=MULTIVECTOR_CHUNKS_BUCKET
                )

            logger.debug(f"Stored chunk content externally with key: {storage_key}")
            return storage_key

        except Exception as e:
            logger.error(f"Failed to store content externally for {document_id}-{chunk_number}: {e}")
            return None

    async def _save_chunk_to_storage(self, chunk: DocumentChunk):
        return await self._store_content_externally(
            chunk.content, chunk.document_id, chunk.chunk_number, str(chunk.metadata)
        )

    def _is_storage_key(self, content: str) -> bool:
        """Check if content field contains a storage key rather than actual content."""
        # Storage keys are short paths with slashes, not base64/long content
        return (
            len(content) < 500 and "/" in content and not content.startswith("data:") and not content.startswith("http")
        )

    async def _retrieve_content_from_storage(self, storage_key: str, chunk_metadata: Optional[str]) -> str:
        """Retrieve content from external storage and convert to expected format."""
        logger.debug(f"Attempting to retrieve content from storage key: {storage_key}")

        if not self.storage:
            logger.warning(f"External storage not available for retrieving key: {storage_key}")
            return storage_key  # Return storage key as fallback

        try:
            # Download content from storage
            logger.debug(f"Downloading from bucket: {MULTIVECTOR_CHUNKS_BUCKET}, key: {storage_key}")
            if isinstance(self.storage, S3Storage):
                storage_key = f"{MULTIVECTOR_CHUNKS_BUCKET}/{storage_key}"
            try:
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=storage_key)
            except Exception:
                storage_key = f"{MULTIVECTOR_CHUNKS_BUCKET}/{storage_key}.txt"
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=storage_key)

            if not content_bytes:
                logger.error(f"No content downloaded for storage key: {storage_key}")
                return storage_key

            logger.debug(f"Downloaded {len(content_bytes)} bytes for key: {storage_key}")

            # Determine if this should be returned as base64 or text
            try:
                if chunk_metadata:
                    metadata = json.loads(chunk_metadata)
                    is_image = metadata.get("is_image", False)
                    logger.debug(f"Chunk metadata indicates is_image: {is_image}")

                    if is_image:
                        # For images, return as base64 string
                        result = base64.b64encode(content_bytes).decode("utf-8")
                        logger.debug(f"Returning image as base64, length: {len(result)}")
                        return result
                    result = content_bytes.decode("utf-8")
                    logger.debug(f"Returning text content, length: {len(result)}")
                    return result
                logger.debug("No metadata, auto-detecting content type")
                try:
                    result = content_bytes.decode("utf-8")
                    logger.debug(f"Auto-detected as text, length: {len(result)}")
                    return result
                except UnicodeDecodeError:
                    # If not valid UTF-8, treat as binary (image) and return base64
                    result = base64.b64encode(content_bytes).decode("utf-8")
                    logger.debug(f"Auto-detected as binary, returning base64, length: {len(result)}")
                    return result

            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error determining content type for {storage_key}: {e}")
                # Fallback: try text first, then base64
                try:
                    return content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    return base64.b64encode(content_bytes).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to retrieve content from storage key {storage_key}: {e}", exc_info=True)
            return storage_key  # Return storage key as fallback


# query_similar
# get_chunks_by_id
# store_embeddings
# delete_chunks_by_document_id
# initialize
