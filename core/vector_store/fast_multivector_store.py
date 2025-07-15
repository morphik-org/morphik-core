import asyncio
import base64
import json
import logging
import tempfile
import time
from contextlib import contextmanager
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg
import torch
from colpali_engine.models import ColQwen2_5_Processor
from psycopg_pool import ConnectionPool

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


if get_settings().MULTIVECTOR_STORE_PROVIDER == "morphik":
    import fixed_dimensional_encoding as fde
    from turbopuffer import AsyncTurbopuffer


# external storage always enabled, no two ways about it
class FastMultiVectorStore(BaseVectorStore):
    def __init__(self, uri: str, tpuf_api_key: str, namespace: str = "public", region: str = "aws-us-west-2"):
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://")
        self.uri = uri
        self.tpuf_api_key = tpuf_api_key
        self.namespace = namespace
        self.tpuf = AsyncTurbopuffer(api_key=tpuf_api_key, region=region)
        # TODO: Cache namespaces, and send a warming request
        self.ns = lambda app_id: self.tpuf.namespace(app_id)
        self.storage = self._init_storage()
        self.fde_config = fde.FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=5,
            projection_dimension=16,
            projection_type="AMS_SKETCH",
        )
        self._document_app_id_cache: Dict[str, str] = {}  # Cache for document app_ids
        self.pool: ConnectionPool = ConnectionPool(conninfo=self.uri, min_size=1, max_size=10, timeout=60)
        self.max_retries = 3
        self.retry_delay = 1.0
        self.processor: ColQwen2_5_Processor = ColQwen2_5_Processor.from_pretrained(
            "tsystems/colqwen2.5-3b-multilingual-v1.0"
        )
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

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
        return True

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        #  group fde calls for better cache hit rate
        embeddings = [
            fde.generate_document_encoding(np.array(chunk.embedding), self.fde_config).tolist() for chunk in chunks
        ]
        storage_keys = await asyncio.gather(*[self._save_chunk_to_storage(chunk, app_id) for chunk in chunks])
        stored_ids = [f"{chunk.document_id}-{chunk.chunk_number}" for chunk in chunks]
        doc_ids, chunk_numbers, metdatas, multivecs = [], [], [], []
        for chunk in chunks:
            doc_ids.append(chunk.document_id)
            chunk_numbers.append(chunk.chunk_number)
            metdatas.append(json.dumps(chunk.metadata))
            bucket, key = await self.save_multivector_to_storage(chunk)
            multivecs.append([bucket, key])
        result = await self.ns(app_id).write(
            upsert_columns={
                "id": stored_ids,
                "vector": embeddings,
                "document_id": doc_ids,
                "chunk_number": chunk_numbers,
                "content": storage_keys,
                "metadata": metdatas,
                "multivector": multivecs,
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
        app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        # --- Begin profiling ---
        t0 = time.perf_counter()

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        elif isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        # 1) Encode query embedding
        encoded_query_embedding = fde.generate_query_encoding(query_embedding, self.fde_config).tolist()
        t1 = time.perf_counter()
        logger.info(f"query_similar timing - encode_query: {(t1 - t0)*1000:.2f} ms")

        # 2) ANN search on Turbopuffer namespace
        result = await self.ns(app_id).query(
            filters=("document_id", "In", doc_ids),
            rank_by=("vector", "ANN", encoded_query_embedding),
            top_k=min(10 * k, 75),
            include_attributes=["id", "document_id", "chunk_number", "content", "metadata", "multivector"],
        )
        t2 = time.perf_counter()
        logger.info(f"query_similar timing - ns.query: {(t2 - t1)*1000:.2f} ms")

        # 3) Download multi-vectors
        multivector_retrieval_tasks = [
            self.load_multivector_from_storage(r["multivector"][0], r["multivector"][1]) for r in result.rows
        ]
        multivectors = await asyncio.gather(*multivector_retrieval_tasks)
        t3 = time.perf_counter()
        logger.info(f"query_similar timing - load_multivectors: {(t3 - t2)*1000:.2f} ms")

        # 4) Rerank using ColQwen2.5 processor
        scores = self.processor.score_multi_vector(
            [torch.from_numpy(query_embedding).float()], multivectors, device=self.device
        )[0]
        scores, idx = torch.topk(scores, min(k, len(scores)))
        scores, top_k_indices = scores.tolist(), idx.tolist()
        t4 = time.perf_counter()
        logger.info(f"query_similar timing - rerank_scoring: {(t4 - t3)*1000:.2f} ms")

        # 5) Retrieve chunk contents
        rows, storage_retrieval_tasks = [], []
        for i in top_k_indices:
            row = result.rows[i]
            rows.append(row)
            storage_retrieval_tasks.append(self._retrieve_content_from_storage(row["content"], row["metadata"]))
        contents = await asyncio.gather(*storage_retrieval_tasks)
        t5 = time.perf_counter()
        logger.info(f"query_similar timing - load_contents: {(t5 - t4)*1000:.2f} ms")

        # 6) Build return objects
        ret = [
            DocumentChunk(
                document_id=row["document_id"],
                embedding=[],
                chunk_number=row["chunk_number"],
                content=content,
                metadata=json.loads(row["metadata"]),
                score=score,
            )
            for score, row, content in zip(scores, rows, contents)
        ]
        t6 = time.perf_counter()
        logger.info(f"query_similar timing - build_chunks: {(t6 - t5)*1000:.2f} ms")
        logger.info(f"query_similar total time: {(t6 - t0)*1000:.2f} ms")

        return ret

    async def get_chunks_by_id(
        self, chunk_identifiers: List[Tuple[str, int]], app_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        result = await self.ns(app_id).query(
            filters=("id", "In", [f"{doc_id}-{chunk_num}" for doc_id, chunk_num in chunk_identifiers]),
            include_attributes=["id", "document_id", "chunk_number", "content", "metadata"],
            top_k=len(chunk_identifiers),
        )
        storage_retrieval_tasks = [
            self._retrieve_content_from_storage(r["content"], r["metadata"]) for r in result.rows
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

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        return await self.ns(app_id).write(delete_by_filter=("document_id", "Eq", document_id))

    async def save_multivector_to_storage(self, chunk: DocumentChunk) -> Tuple[str, str]:
        as_np = np.array(chunk.embedding)
        save_path = f"multivector/{chunk.document_id}/{chunk.chunk_number}.npy"
        with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
            np.save(temp_file, as_np)  # , allow_pickle=True)
            if isinstance(self.storage, S3Storage):
                self.storage.s3_client.upload_file(temp_file.name, MULTIVECTOR_CHUNKS_BUCKET, save_path)
                bucket, key = MULTIVECTOR_CHUNKS_BUCKET, save_path
            else:
                bucket, key = await self.storage.upload_file(temp_file.name, save_path)
            temp_file.close()
        return bucket, key

    async def load_multivector_from_storage(self, bucket: str, key: str) -> torch.Tensor:
        content = await self.storage.download_file(bucket, key)
        as_np = np.load(BytesIO(content))  # , allow_pickle=True)
        return torch.from_numpy(as_np).float()

    @contextmanager
    def get_connection(self):
        """Get a PostgreSQL connection with retry logic.

        Yields:
            A PostgreSQL connection object

        Raises:
            psycopg.OperationalError: If all connection attempts fail
        """
        attempt = 0
        last_error = None

        # Try to establish a new connection with retries
        while attempt < self.max_retries:
            try:
                # Borrow a pooled connection (blocking wait). Autocommit stays
                # disabled so we can batch-commit.
                conn = self.pool.getconn()

                try:
                    yield conn
                    return
                finally:
                    # Release connection back to the pool
                    try:
                        self.pool.putconn(conn)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
            except psycopg.OperationalError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(
                        f"Connection attempt {attempt} failed: {str(e)}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)

        # If we get here, all retries failed
        logger.error(f"All connection attempts failed after {self.max_retries} retries: {str(last_error)}")
        raise last_error

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
        self,
        content: str,
        document_id: str,
        chunk_number: int,
        chunk_metadata: Optional[str],
        app_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store chunk content in external storage and return storage key."""
        if not self.storage:
            return None

        try:
            # Use provided app_id or fall back to document lookup
            if app_id is None:
                logger.warning(f"No app_id provided for document {document_id}, falling back to database lookup")
                app_id = await self._get_document_app_id(document_id)
            else:
                logger.info(f"Using provided app_id: {app_id} for document {document_id}")

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

            logger.info(f"Stored chunk content externally with key: {storage_key}")
            return storage_key

        except Exception as e:
            logger.error(f"Failed to store content externally for {document_id}-{chunk_number}: {e}")
            return None

    async def _save_chunk_to_storage(self, chunk: DocumentChunk, app_id: Optional[str] = None):
        return await self._store_content_externally(
            chunk.content, chunk.document_id, chunk.chunk_number, str(chunk.metadata), app_id
        )

    def _is_storage_key(self, content: str) -> bool:
        """Check if content field contains a storage key rather than actual content."""
        # Storage keys are short paths with slashes, not base64/long content
        return (
            len(content) < 500 and "/" in content and not content.startswith("data:") and not content.startswith("http")
        )

    async def _retrieve_content_from_storage(self, storage_key: str, chunk_metadata: Optional[str]) -> str:
        """Retrieve content from external storage and convert to expected format."""
        logger.info(f"Attempting to retrieve content from storage key: {storage_key}")

        if not self.storage:
            logger.warning(f"External storage not available for retrieving key: {storage_key}")
            return storage_key  # Return storage key as fallback

        try:
            # Download content from storage
            logger.info(f"Downloading from bucket: {MULTIVECTOR_CHUNKS_BUCKET}, key: {storage_key}")
            key_possibilities = [
                storage_key,
                f"{storage_key}.txt",
                f"multivector-chunks/{storage_key}",
                f"multivector-chunks/{storage_key}.txt",
            ]
            download_tasks = [
                self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=key) for key in key_possibilities
            ]
            content_bytes_list = await asyncio.gather(*download_tasks, return_exceptions=True)
            content_bytes = None
            for potential_content_bytes in content_bytes_list:
                if isinstance(potential_content_bytes, Exception):
                    continue
                content_bytes = potential_content_bytes
                logger.info(f"Successfully downloaded content from storage key: {storage_key}")
                break
            if not content_bytes:
                logger.error(f"No content downloaded for storage key: {storage_key}")
                return storage_key

            logger.info(f"Downloaded {len(content_bytes)} bytes for key: {storage_key}")

            # Check if storage key ends with .txt (indicates content was stored as text)
            if storage_key.endswith(".txt"):
                # Content is stored as text (could be base64 string for images)
                result = content_bytes.decode("utf-8")
                logger.info(f"Retrieved text content from .txt file, length: {len(result)}")
                return result

            # For non-.txt files, determine content type
            try:
                if chunk_metadata:
                    metadata = json.loads(chunk_metadata)
                    is_image = metadata.get("is_image", False)
                    logger.info(f"Chunk metadata indicates is_image: {is_image}")

                    if is_image:
                        # For images, return as base64 string
                        result = base64.b64encode(content_bytes).decode("utf-8")
                        logger.info(f"Returning image as base64, length: {len(result)}")
                        return result
                    result = content_bytes.decode("utf-8")
                    logger.info(f"Returning text content, length: {len(result)}")
                    return result

                logger.info("No metadata, auto-detecting content type")
                try:
                    result = content_bytes.decode("utf-8")
                    logger.info(f"Auto-detected as text, length: {len(result)}")
                    return result
                except UnicodeDecodeError:
                    # If not valid UTF-8, treat as binary (image) and return base64
                    result = base64.b64encode(content_bytes).decode("utf-8")
                    logger.info(f"Auto-detected as binary, returning base64, length: {len(result)}")
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
