"""LocalMultiVectorStore: ChromaDB-based multi-vector store for local deployments.

This module provides a multi-vector store implementation using ChromaDB as the backend
instead of TurboPuffer. It maintains the same architecture as FastMultiVectorStore:
- FDE (Fixed Dimensional Encoding) for efficient ANN search
- External storage for chunk payloads and multi-vector tensors
- ColQwen2.5 re-ranking for final similarity scoring
"""

import asyncio
import json
import logging
import os
import time
from contextlib import contextmanager
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
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
from core.utils.fast_ops import bytes_to_data_uri, encode_base64

from .base_vector_store import BaseVectorStore
from .utils import (
    MULTIVECTOR_CHUNKS_BUCKET,
    build_store_metrics,
    derive_repaired_image_key,
    is_storage_key,
    normalize_storage_key,
    reset_pooled_connection,
    storage_provider_name,
)

logger = logging.getLogger(__name__)

# Attach a dedicated handler to capture multivector retrieval diagnostics once per process.
_multivector_log_path = os.path.join("logs", "local_multivector.log")
if not any(
    isinstance(handler, RotatingFileHandler)
    and getattr(handler, "baseFilename", "") == os.path.abspath(_multivector_log_path)
    for handler in logger.handlers
):
    os.makedirs("logs", exist_ok=True)
    _file_handler = RotatingFileHandler(
        _multivector_log_path,
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_file_handler)

# Constants for external storage
DEFAULT_APP_ID = "default"  # Fallback for local usage when app_id is None

# Import FDE conditionally - only when using local provider
if get_settings().MULTIVECTOR_STORE_PROVIDER == "local":
    import fixed_dimensional_encoding as fde


class FileCacheManager:
    """Manage local on-disk cache for blobs with eviction."""

    def __init__(self, enabled: bool, base_dir: Path, max_bytes: int):
        self.enabled = enabled
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        self._lock = asyncio.Lock()
        self._index_initialized = False
        self._cache_index: Dict[Path, Tuple[float, int]] = {}
        self._cache_total_size = 0
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_parts(self, raw: str) -> List[str]:
        if not raw:
            return []

        original = Path(raw)
        anchor = original.anchor
        candidate = raw
        if anchor and candidate.startswith(anchor):
            candidate = candidate[len(anchor) :]

        sanitized = Path(candidate)
        parts: List[str] = []
        for part in sanitized.parts:
            if part in ("", ".", ".."):
                continue
            parts.append(part)
        return parts

    def _path_for(self, namespace: str, bucket: str, key: str) -> Path:
        namespace_parts = self._normalize_parts(namespace or "_default")
        bucket_parts = self._normalize_parts(bucket or "_default")
        key_parts = self._normalize_parts(key)
        return self.base_dir.joinpath(*namespace_parts, *bucket_parts, *key_parts)

    async def get(self, namespace: str, bucket: str, key: str) -> Optional[bytes]:
        if not self.enabled:
            return None
        path = self._path_for(namespace, bucket, key)
        if not path.exists():
            return None
        try:
            data = await asyncio.to_thread(path.read_bytes)
            now = time.time()
            await asyncio.to_thread(self._touch_file, path, now)
            await self._record_access(path, now)
            return data
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read cache entry %s: %s", path, exc)
            return None

    async def put(self, namespace: str, bucket: str, key: str, data: bytes) -> None:
        if not self.enabled:
            return
        path = self._path_for(namespace, bucket, key)
        try:
            size = await asyncio.to_thread(self._write_file, path, data)
            now = time.time()
            await asyncio.to_thread(self._touch_file, path, now)
            await self._record_write(path, size, now)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to write cache entry %s: %s", path, exc)
            return

        try:
            await self._enforce_budget()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to enforce cache budget after writing %s: %s", path, exc)

    async def delete(self, namespace: str, bucket: str, key: str) -> None:
        if not self.enabled:
            return
        path = self._path_for(namespace, bucket, key)
        try:
            await asyncio.to_thread(self._remove_file, path)
            await self._record_delete(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to delete cache entry %s: %s", path, exc)

    async def delete_many(self, namespace: str, items: Iterable[Tuple[str, str]]) -> None:
        if not self.enabled:
            return
        tasks = [self.delete(namespace, bucket, key) for bucket, key in items]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _write_file(self, path: Path, data: bytes) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{path.name}.tmp-{uuid4().hex}"
        tmp_path = path.parent / tmp_name
        try:
            if isinstance(data, memoryview):
                payload = data.tobytes()
            elif isinstance(data, bytearray):
                payload = bytes(data)
            else:
                payload = data if isinstance(data, bytes) else bytes(data)
            with open(tmp_path, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            return len(payload)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def _touch_file(self, path: Path, atime: Optional[float] = None) -> None:
        now = atime or time.time()
        try:
            os.utime(path, (now, now))
        except FileNotFoundError:
            pass

    def _remove_file(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    async def _enforce_budget(self) -> None:
        if not self.enabled:
            return
        await self._ensure_index()
        async with self._lock:
            if self._cache_total_size <= self.max_bytes:
                return
            await asyncio.to_thread(self._enforce_budget_sync_indexed)

    def _scan_cache_dir(self) -> Tuple[Dict[Path, Tuple[float, int]], int]:
        total_size = 0
        index: Dict[Path, Tuple[float, int]] = {}
        for file_path in self.base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                stat = file_path.stat()
            except FileNotFoundError:
                continue
            index[file_path] = (stat.st_atime, stat.st_size)
            total_size += stat.st_size
        return index, total_size

    async def _ensure_index(self) -> None:
        if self._index_initialized:
            return
        async with self._lock:
            if self._index_initialized:
                return
            index, total_size = await asyncio.to_thread(self._scan_cache_dir)
            self._cache_index = index
            self._cache_total_size = total_size
            self._index_initialized = True

    async def _record_write(self, path: Path, size: int, atime: float) -> None:
        await self._ensure_index()
        async with self._lock:
            previous = self._cache_index.get(path)
            if previous:
                self._cache_total_size -= previous[1]
            self._cache_index[path] = (atime, size)
            self._cache_total_size += size

    async def _record_access(self, path: Path, atime: float) -> None:
        if not self._index_initialized:
            return
        async with self._lock:
            previous = self._cache_index.get(path)
            if not previous:
                return
            self._cache_index[path] = (atime, previous[1])

    async def _record_delete(self, path: Path) -> None:
        if not self._index_initialized:
            return
        async with self._lock:
            previous = self._cache_index.pop(path, None)
            if previous:
                self._cache_total_size -= previous[1]

    def _enforce_budget_sync_indexed(self) -> None:
        if self._cache_total_size <= self.max_bytes:
            return
        files: List[Tuple[float, int, Path]] = [
            (atime, size, file_path) for file_path, (atime, size) in self._cache_index.items()
        ]
        files.sort(key=lambda item: item[0])  # Oldest access time first
        for _, _, file_path in files:
            if self._cache_total_size <= self.max_bytes:
                break
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
            previous = self._cache_index.pop(file_path, None)
            if previous:
                self._cache_total_size -= previous[1]

        if self._cache_total_size > self.max_bytes and not files:
            index, total_size = self._scan_cache_dir()
            self._cache_index = index
            self._cache_total_size = total_size
            if self._cache_total_size > self.max_bytes:
                files = [(atime, size, file_path) for file_path, (atime, size) in self._cache_index.items()]
                files.sort(key=lambda item: item[0])
                for _, _, file_path in files:
                    if self._cache_total_size <= self.max_bytes:
                        break
                    try:
                        file_path.unlink()
                    except FileNotFoundError:
                        pass
                    previous = self._cache_index.pop(file_path, None)
                    if previous:
                        self._cache_total_size -= previous[1]


class LocalMultiVectorStore(BaseVectorStore):
    """Multi-vector store using ChromaDB for local deployments.

    This implementation uses ChromaDB as the vector database backend instead of TurboPuffer,
    making it suitable for local development and self-hosted deployments without requiring
    external API keys.

    Architecture:
    - ChromaDB for ANN search using FDE-encoded vectors
    - External storage (S3 or local) for chunk payloads
    - External storage for multi-vector tensors
    - ColQwen2.5 processor for re-ranking

    Args:
        uri: PostgreSQL connection URI for document metadata lookups
        chroma_host: ChromaDB server hostname (default: "localhost")
        chroma_port: ChromaDB server port (default: 8000)
        chroma_ssl: Enable SSL for ChromaDB connection (default: False)
        namespace: Default namespace/tenant for collections (default: "public")
    """

    def __init__(
        self,
        uri: str,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_ssl: bool = False,
        namespace: str = "public",
    ):
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://")
        self.uri = uri
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.chroma_ssl = chroma_ssl
        self.namespace = namespace

        # Initialize ChromaDB async client
        # Note: Client is created lazily on first use to support async initialization
        self._chroma_client: Optional[chromadb.AsyncHttpClient] = None
        self._chroma_client_lock = asyncio.Lock()

        # Collection cache to avoid repeated get_or_create calls
        self._collections: Dict[str, Any] = {}
        self._collection_lock = asyncio.Lock()

        # Initialize storage backends
        self.chunk_storage, self.chunk_bucket = self._init_chunk_storage()
        self.vector_storage, self.vector_bucket = self._init_vector_storage()
        # Maintain legacy attribute for backwards compatibility with other components
        self.storage = self.chunk_storage

        # Initialize cache
        cache_settings = get_settings()
        cache_enabled = cache_settings.CACHE_ENABLED
        cache_path = Path(cache_settings.CACHE_PATH or "./storage/cache")
        cache_limit = cache_settings.CACHE_MAX_BYTES
        self.cache = FileCacheManager(
            enabled=cache_enabled,
            base_dir=cache_path,
            max_bytes=cache_limit,
        )

        # Initialize FDE config for multi-vector encoding
        self.fde_config = fde.FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=5,
            projection_dimension=16,
            projection_type="AMS_SKETCH",
        )

        # Document app_id cache
        self._document_app_id_cache: Dict[str, str] = {}

        # PostgreSQL connection pool for metadata lookups
        self.pool: ConnectionPool = ConnectionPool(conninfo=self.uri, min_size=1, max_size=10, timeout=60)
        self.max_retries = 3
        self.retry_delay = 1.0

        # ColQwen2.5 processor for re-ranking
        self.processor: ColQwen2_5_Processor = ColQwen2_5_Processor.from_pretrained(
            "tsystems/colqwen2.5-3b-multilingual-v1.0"
        )
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        # Concurrency control for uploads
        try:
            self._multivector_upload_sem = asyncio.Semaphore(max(1, int(get_settings().S3_UPLOAD_CONCURRENCY)))
        except Exception:
            self._multivector_upload_sem = asyncio.Semaphore(16)

        self._last_store_metrics: Dict[str, Any] = {}

    async def _get_chroma_client(self) -> chromadb.AsyncHttpClient:
        """Get or create the ChromaDB async client."""
        if self._chroma_client is not None:
            return self._chroma_client

        async with self._chroma_client_lock:
            if self._chroma_client is not None:
                return self._chroma_client

            self._chroma_client = await chromadb.AsyncHttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                ssl=self.chroma_ssl,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
            logger.info(
                "Initialized ChromaDB async client: host=%s, port=%d, ssl=%s",
                self.chroma_host,
                self.chroma_port,
                self.chroma_ssl,
            )
            return self._chroma_client

    async def _get_collection(self, app_id: Optional[str] = None):
        """Get or create a ChromaDB collection for the given app_id.

        Collections are named based on app_id to provide multi-tenant isolation.
        """
        collection_name = self._collection_name(app_id)
        logger.debug("_get_collection called | app_id=%s | collection_name=%s", app_id, collection_name)

        if collection_name in self._collections:
            logger.debug("Returning cached collection: %s", collection_name)
            return self._collections[collection_name]

        async with self._collection_lock:
            if collection_name in self._collections:
                return self._collections[collection_name]

            logger.info(
                "Creating/getting ChromaDB collection | name=%s | host=%s | port=%d",
                collection_name,
                self.chroma_host,
                self.chroma_port,
            )
            client = await self._get_chroma_client()

            try:
                collection = await client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},  # Use cosine similarity
                )
                # Get initial count
                initial_count = await collection.count()
                logger.info(
                    "ChromaDB collection ready | name=%s | initial_count=%d | collection_id=%s",
                    collection_name,
                    initial_count,
                    getattr(collection, 'id', 'unknown'),
                )
                self._collections[collection_name] = collection
                return collection
            except Exception as e:
                logger.error(
                    "Failed to get/create ChromaDB collection | name=%s | error=%s",
                    collection_name,
                    str(e),
                    exc_info=True,
                )
                raise

    def _collection_name(self, app_id: Optional[str] = None) -> str:
        """Generate a valid ChromaDB collection name from app_id.

        ChromaDB collection names must:
        - Be 3-63 characters long
        - Start and end with alphanumeric
        - Contain only alphanumeric, underscore, or hyphen
        """
        base_name = app_id or DEFAULT_APP_ID
        # Sanitize the name
        sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in base_name)
        # Ensure it starts with alphanumeric
        if sanitized and not sanitized[0].isalnum():
            sanitized = "c" + sanitized
        # Ensure minimum length
        if len(sanitized) < 3:
            sanitized = sanitized + "_mv"
        # Truncate if too long
        if len(sanitized) > 63:
            sanitized = sanitized[:63]
        # Ensure it ends with alphanumeric
        while sanitized and not sanitized[-1].isalnum():
            sanitized = sanitized[:-1]
        return sanitized or "default_mv"

    async def close(self) -> None:
        """Release network and database resources held by the vector store."""
        logger.info("Closing LocalMultiVectorStore resources...")

        # Close ChromaDB client if initialized
        # Note: chromadb.AsyncHttpClient doesn't have an explicit close method
        # but we clear our reference
        self._chroma_client = None
        self._collections.clear()

        # Close PostgreSQL pool
        pool = getattr(self, "pool", None)
        if pool is not None and not pool.closed:
            try:
                await asyncio.to_thread(pool.close)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close PostgreSQL connection pool cleanly: %s", exc)

    def _init_chunk_storage(self) -> Tuple[BaseStorage, Optional[str]]:
        """Initialize storage backend for chunk payloads."""
        settings = get_settings()
        provider = settings.STORAGE_PROVIDER
        storage_path = settings.STORAGE_PATH or "./storage"
        bucket = (settings.S3_BUCKET or MULTIVECTOR_CHUNKS_BUCKET) if provider == "aws-s3" else ""

        logger.info("Initializing %s storage for chunk payloads", provider)
        storage = self._create_storage(provider, storage_path=storage_path, default_bucket=bucket)

        # Track meta for later reuse decisions
        self.chunk_storage_provider = provider
        self.chunk_storage_path = storage_path
        resolved_bucket = bucket if provider == "aws-s3" else ""
        return storage, resolved_bucket

    def _init_vector_storage(self) -> Tuple[BaseStorage, Optional[str]]:
        """Initialize storage backend for numpy multi-vector tensors."""
        settings = get_settings()
        provider = settings.STORAGE_PROVIDER
        storage_path = settings.STORAGE_PATH or "./storage"
        bucket = (settings.S3_BUCKET or MULTIVECTOR_CHUNKS_BUCKET) if provider == "aws-s3" else ""

        # Reuse chunk storage instance when configuration matches
        if provider == getattr(self, "chunk_storage_provider", None) and (
            (provider == "local" and storage_path == getattr(self, "chunk_storage_path", None))
            or (provider == "aws-s3" and bucket == getattr(self, "chunk_bucket", None))
        ):
            logger.info("Reusing chunk storage backend for vector tensors (matching configuration).")
            if provider == "local":
                return self.chunk_storage, ""
            return self.chunk_storage, getattr(self, "chunk_bucket", None)

        logger.info("Initializing %s storage for vector tensors", provider)
        storage = self._create_storage(provider, storage_path=storage_path, default_bucket=bucket)
        resolved_bucket = bucket if provider == "aws-s3" else ""
        return storage, resolved_bucket

    def _create_storage(
        self, provider: str, *, storage_path: Optional[str], default_bucket: Optional[str]
    ) -> BaseStorage:
        """Factory helper to instantiate storage implementations."""
        settings = get_settings()
        match provider:
            case "aws-s3":
                if not settings.AWS_ACCESS_KEY or not settings.AWS_SECRET_ACCESS_KEY:
                    raise ValueError("AWS credentials are required for S3 storage provider.")
                return S3Storage(
                    aws_access_key=settings.AWS_ACCESS_KEY,
                    aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    default_bucket=default_bucket or MULTIVECTOR_CHUNKS_BUCKET,
                    upload_concurrency=settings.S3_UPLOAD_CONCURRENCY,
                )
            case "local":
                path = storage_path or "./storage"
                return LocalStorage(storage_path=path)
            case _:
                raise ValueError(f"Unsupported storage provider: {provider}")

    def initialize(self):
        return True

    def latest_store_metrics(self) -> Dict[str, Any]:
        return dict(self._last_store_metrics) if self._last_store_metrics else {}

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Store document chunks with multi-vector embeddings in ChromaDB."""
        if not chunks:
            self._last_store_metrics = build_store_metrics(
                chunk_payload_backend=storage_provider_name(self.chunk_storage),
                multivector_backend=storage_provider_name(self.vector_storage),
                vector_store_backend="chromadb",
            )
            return True, []

        resolved_app_id = app_id
        if resolved_app_id is None:
            doc_id = chunks[0].document_id
            if all(chunk.document_id == doc_id for chunk in chunks):
                resolved_app_id = await self._get_document_app_id(doc_id)

        # Generate FDE encodings for ChromaDB ANN search
        embeddings = [
            fde.generate_document_encoding(np.array(chunk.embedding), self.fde_config).tolist() for chunk in chunks
        ]

        storage_metrics: Dict[str, Any] = build_store_metrics(
            chunk_payload_backend=storage_provider_name(self.chunk_storage),
            multivector_backend=storage_provider_name(self.vector_storage),
            vector_store_backend="chromadb",
            vector_store_rows=len(chunks),
        )

        # Store chunk payloads externally
        payload_start = time.perf_counter()
        storage_keys = await asyncio.gather(*[self._save_chunk_to_storage(chunk, resolved_app_id) for chunk in chunks])
        storage_metrics["chunk_payload_upload_s"] = time.perf_counter() - payload_start
        storage_metrics["chunk_payload_objects"] = sum(1 for key in storage_keys if key)

        # Prepare data for ChromaDB
        stored_ids = [f"{chunk.document_id}-{chunk.chunk_number}" for chunk in chunks]

        # Store multi-vectors externally
        async def _save_mv(c: DocumentChunk) -> Tuple[str, str, float]:
            async with self._multivector_upload_sem:
                return await self._save_multivector_to_storage_with_cache_time(c)

        multivector_start = time.perf_counter()
        multivecs_with_cache = await asyncio.gather(*[_save_mv(chunk) for chunk in chunks])
        storage_metrics["multivector_upload_s"] = time.perf_counter() - multivector_start
        storage_metrics["multivector_objects"] = len(multivecs_with_cache)
        storage_metrics["cache_write_s"] = sum(item[2] for item in multivecs_with_cache)
        storage_metrics["cache_write_objects"] = len(multivecs_with_cache)

        # Prepare metadata for ChromaDB
        # ChromaDB metadata must be flat (str, int, float, bool) - no nested objects
        metadatas = []
        for i, chunk in enumerate(chunks):
            mv_bucket, mv_key, _ = multivecs_with_cache[i]
            metadata = {
                "document_id": chunk.document_id,
                "chunk_number": chunk.chunk_number,
                "content_key": storage_keys[i] or "",
                "multivector_bucket": mv_bucket,
                "multivector_key": mv_key,
                "chunk_metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}",
            }
            metadatas.append(metadata)

        # Upsert to ChromaDB
        write_start = time.perf_counter()
        collection = await self._get_collection(resolved_app_id)
        collection_name = self._collection_name(resolved_app_id)

        logger.info(
            "ChromaDB upsert request | collection=%s | num_chunks=%d | ids=%s | embedding_dims=%s",
            collection_name,
            len(chunks),
            stored_ids[:5] if len(stored_ids) > 5 else stored_ids,  # Log first 5 IDs
            [len(e) for e in embeddings[:3]] if embeddings else [],  # Log embedding dimensions
        )
        logger.debug(
            "ChromaDB upsert details | metadatas_sample=%s",
            metadatas[:2] if metadatas else [],
        )

        try:
            upsert_result = await collection.upsert(
                ids=stored_ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            write_time = time.perf_counter() - write_start
            storage_metrics["vector_store_write_s"] = write_time

            # Verify the upsert by counting collection
            collection_count = await collection.count()
            logger.info(
                "ChromaDB upsert complete | collection=%s | write_time=%.3fs | result=%s | collection_count=%d",
                collection_name,
                write_time,
                upsert_result,
                collection_count,
            )
        except Exception as e:
            logger.error(
                "ChromaDB upsert FAILED | collection=%s | error=%s | error_type=%s",
                collection_name,
                str(e),
                type(e).__name__,
                exc_info=True,
            )
            raise

        self._last_store_metrics = storage_metrics
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB collection: {collection_name}")
        return True, stored_ids

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
        app_id: Optional[str] = None,
        skip_image_content: bool = False,
    ) -> List[DocumentChunk]:
        """Query for similar chunks using FDE + ChromaDB ANN + ColQwen2.5 re-ranking."""
        # --- Begin profiling ---
        t0 = time.perf_counter()

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        elif isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        # 1) Encode query embedding with FDE
        encoded_query_embedding = fde.generate_query_encoding(query_embedding, self.fde_config).tolist()
        t1 = time.perf_counter()
        logger.info(f"query_similar timing - encode_query: {(t1 - t0)*1000:.2f} ms")

        # 2) ANN search on ChromaDB collection
        collection = await self._get_collection(app_id)

        # Build where filter for document IDs if provided
        where_filter = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"document_id": {"$eq": doc_ids[0]}}
            else:
                where_filter = {"document_id": {"$in": doc_ids}}

        result = await collection.query(
            query_embeddings=[encoded_query_embedding],
            n_results=min(10 * k, 75),
            where=where_filter,
            include=["metadatas", "distances"],
        )
        t2 = time.perf_counter()
        logger.info(f"query_similar timing - collection.query: {(t2 - t1)*1000:.2f} ms")

        # 3) Process results
        if not result or not result["ids"] or not result["ids"][0]:
            logger.info(
                "query_similar: ChromaDB query returned no rows (doc_ids=%s, app_id=%s); returning empty result",
                doc_ids,
                app_id or self.namespace,
            )
            return []

        # Extract results (ChromaDB returns lists wrapped in another list for batch queries)
        result_ids = result["ids"][0]
        result_metadatas = result["metadatas"][0] if result["metadatas"] else [{}] * len(result_ids)
        result_distances = result["distances"][0] if result["distances"] else [0.0] * len(result_ids)

        # 4) Download multi-vectors for re-ranking
        multivector_retrieval_tasks = [
            self.load_multivector_from_storage(meta.get("multivector_bucket", ""), meta.get("multivector_key", ""))
            for meta in result_metadatas
        ]
        multivectors = await asyncio.gather(*multivector_retrieval_tasks)
        t3 = time.perf_counter()
        logger.info(f"query_similar timing - load_multivectors: {(t3 - t2)*1000:.2f} ms")

        # 5) Rerank using ColQwen2.5 processor
        scores = self.processor.score_multi_vector(
            [torch.from_numpy(query_embedding).float()], multivectors, device=self.device
        )[0]
        scores, idx = torch.topk(scores, min(k, len(scores)))
        scores, top_k_indices = scores.tolist(), idx.tolist()

        # Log reranking summary
        num_candidates = len(result_ids)
        try:
            selected_ids = [result_ids[i] for i in top_k_indices]
        except Exception:  # noqa: BLE001
            selected_ids = None
        logger.info(
            "ColPali rerank summary | requested_k=%d | candidates=%d | selected_positions=%s%s",
            k,
            num_candidates,
            top_k_indices,
            f" | selected_ids={selected_ids}" if selected_ids is not None else "",
        )
        t4 = time.perf_counter()
        logger.info(f"query_similar timing - rerank_scoring: {(t4 - t3)*1000:.2f} ms")

        # 6) Retrieve chunk contents
        rows, storage_retrieval_tasks, parsed_metadata = [], [], []
        for i in top_k_indices:
            meta = result_metadatas[i]
            rows.append(meta)
            chunk_metadata_str = meta.get("chunk_metadata", "{}")
            try:
                metadata = json.loads(chunk_metadata_str)
            except json.JSONDecodeError:
                metadata = {}
            parsed_metadata.append(metadata)

            content_key = meta.get("content_key", "")
            if skip_image_content and metadata.get("is_image") and self._is_storage_key(content_key):
                storage_retrieval_tasks.append(asyncio.sleep(0, result=content_key))
            else:
                storage_retrieval_tasks.append(self._retrieve_content_from_storage(content_key, chunk_metadata_str))

        contents = await asyncio.gather(*storage_retrieval_tasks)
        t5 = time.perf_counter()
        logger.info(f"query_similar timing - load_contents: {(t5 - t4)*1000:.2f} ms")

        # 7) Build return objects
        ret = [
            DocumentChunk(
                document_id=row.get("document_id", ""),
                embedding=[],
                chunk_number=row.get("chunk_number", 0),
                content=content,
                metadata=metadata,
                score=score,
            )
            for score, row, content, metadata in zip(scores, rows, contents, parsed_metadata)
        ]
        t6 = time.perf_counter()
        logger.info(f"query_similar timing - build_chunks: {(t6 - t5)*1000:.2f} ms")
        logger.info(f"query_similar total time: {(t6 - t0)*1000:.2f} ms")

        return ret

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
        app_id: Optional[str] = None,
        skip_image_content: bool = False,
    ) -> List[DocumentChunk]:
        """Retrieve specific chunks by document ID and chunk number."""
        if not chunk_identifiers:
            return []

        collection = await self._get_collection(app_id)
        ids_to_fetch = [f"{doc_id}-{chunk_num}" for doc_id, chunk_num in chunk_identifiers]

        result = await collection.get(
            ids=ids_to_fetch,
            include=["metadatas"],
        )

        if not result or not result["ids"]:
            return []

        storage_retrieval_tasks = []
        parsed_metadata = []
        for meta in result["metadatas"]:
            chunk_metadata_str = meta.get("chunk_metadata", "{}")
            try:
                metadata = json.loads(chunk_metadata_str)
            except json.JSONDecodeError:
                metadata = {}
            parsed_metadata.append(metadata)

            content_key = meta.get("content_key", "")
            if skip_image_content and metadata.get("is_image") and self._is_storage_key(content_key):
                storage_retrieval_tasks.append(asyncio.sleep(0, result=content_key))
            else:
                storage_retrieval_tasks.append(self._retrieve_content_from_storage(content_key, chunk_metadata_str))

        contents = await asyncio.gather(*storage_retrieval_tasks)

        return [
            DocumentChunk(
                document_id=meta.get("document_id", ""),
                embedding=[],
                chunk_number=meta.get("chunk_number", 0),
                content=content,
                metadata=metadata,
                score=0.0,
            )
            for meta, content, metadata in zip(result["metadatas"], contents, parsed_metadata)
        ]

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        """Delete all chunks associated with a document."""
        collection = await self._get_collection(app_id)
        storage_targets: Dict[str, Set[Tuple[str, str]]] = {"chunk": set(), "vector": set()}

        # Collect storage targets before deletion
        storage_available = self.chunk_storage or self.vector_storage
        if storage_available:
            storage_targets = await self._collect_storage_targets(collection, document_id, app_id)

        try:
            # Delete from ChromaDB using where filter
            await collection.delete(
                where={"document_id": {"$eq": document_id}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to delete ChromaDB rows for document %s in collection %s: %s",
                document_id,
                self._collection_name(app_id),
                exc,
            )
            return False

        # Delete external storage objects
        if storage_targets["chunk"] or storage_targets["vector"]:
            await self._delete_storage_targets(storage_targets, document_id)

        return True

    async def _save_multivector_to_storage_with_cache_time(self, chunk: DocumentChunk) -> Tuple[str, str, float]:
        """Save multi-vector tensor to storage and cache, returning (bucket, key, cache_time)."""
        # Use float32 - ColPali model outputs bfloat16 which shares float32's exponent range.
        as_np = np.asarray(chunk.embedding, dtype=np.float32)
        save_path = f"multivector/{chunk.document_id}/{chunk.chunk_number}.npy"
        # Save to BytesIO first so we can cache the bytes
        buffer = BytesIO()
        np.save(buffer, as_np)
        npy_bytes = buffer.getvalue()

        if isinstance(self.vector_storage, S3Storage):
            target_bucket = self.vector_bucket or self.vector_storage.default_bucket
            bucket, key = await self.vector_storage.upload_file(
                BytesIO(npy_bytes),
                save_path,
                content_type="application/octet-stream",
                bucket=target_bucket,
            )
        else:
            bucket_arg = "" if isinstance(self.vector_storage, LocalStorage) else (self.vector_bucket or "")
            bucket, key = await self.vector_storage.upload_file(npy_bytes, save_path, bucket=bucket_arg)
            if isinstance(self.vector_storage, LocalStorage):
                bucket = ""

        # Cache on ingest so retrieval hits cache immediately
        cache_start = time.perf_counter()
        await self.cache.put("vectors", bucket, key, npy_bytes)
        cache_write_time = time.perf_counter() - cache_start
        return bucket, key, cache_write_time

    async def save_multivector_to_storage(self, chunk: DocumentChunk) -> Tuple[str, str]:
        bucket, key, _ = await self._save_multivector_to_storage_with_cache_time(chunk)
        return bucket, key

    async def load_multivector_from_storage(self, bucket: str, key: str) -> torch.Tensor:
        """Load multi-vector tensor from storage or cache."""
        if not key:
            # Return empty tensor if no key provided
            return torch.zeros((1, 128), dtype=torch.float32)

        primary_bucket = bucket
        if isinstance(self.vector_storage, LocalStorage):
            storage_root = getattr(self.vector_storage, "storage_path", None)
            if storage_root is not None:
                storage_root_path = Path(storage_root)
                try:
                    bucket_path = Path(bucket) if bucket else None
                except Exception:
                    bucket_path = None

                if (
                    not bucket_path
                    or bucket_path == storage_root_path
                    or bucket_path.resolve() == storage_root_path.resolve()
                ):
                    primary_bucket = ""

        cache_bucket = primary_bucket or bucket
        cached_bytes = await self.cache.get("vectors", cache_bucket, key)
        if cached_bytes is not None:
            try:
                as_np = np.load(BytesIO(cached_bytes))
                return torch.from_numpy(as_np).float()
            except Exception as cache_exc:  # noqa: BLE001
                logger.warning(
                    "Vector cache entry for bucket %s key %s is invalid; purging and reloading: %s",
                    cache_bucket,
                    key,
                    cache_exc,
                )
                await self.cache.delete("vectors", cache_bucket, key)
                cached_bytes = None

        if cached_bytes is None:
            try:
                content = await self.vector_storage.download_file(primary_bucket, key)
            except Exception as primary_exc:  # noqa: BLE001
                if self.vector_storage is self.chunk_storage or not bucket:
                    raise
                logger.warning(
                    "Primary vector storage failed to load %s/%s, falling back to chunk storage: %s",
                    bucket,
                    key,
                    primary_exc,
                )
                content = await self.chunk_storage.download_file(bucket, key)
            await self.cache.put("vectors", cache_bucket, key, content)
            cached_bytes = content

        try:
            as_np = np.load(BytesIO(cached_bytes))
        except Exception as exc:  # noqa: BLE001
            await self.cache.delete("vectors", cache_bucket, key)
            logger.error(
                "Failed to deserialize vector content for bucket %s key %s after refresh: %s",
                cache_bucket,
                key,
                exc,
            )
            raise
        return torch.from_numpy(as_np).float()

    @contextmanager
    def get_connection(self):
        """Get a PostgreSQL connection with retry logic."""
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                conn = self.pool.getconn()
                try:
                    yield conn
                    return
                finally:
                    try:
                        if reset_pooled_connection(conn, logger):
                            self.pool.putconn(conn)
                        else:
                            conn.close()
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
            if chunk_metadata:
                metadata = json.loads(chunk_metadata)
                is_image = metadata.get("is_image", False)
                if is_image:
                    return detect_file_type(content)
                else:
                    return ".txt"
            else:
                return detect_file_type(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error parsing chunk metadata: {e}")
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
        if not self.chunk_storage:
            return None

        try:
            if app_id is None:
                if document_id not in self._document_app_id_cache:
                    logger.warning(f"No app_id provided for document {document_id}, falling back to database lookup")
                app_id = await self._get_document_app_id(document_id)
            else:
                logger.debug(f"Using provided app_id: {app_id} for document {document_id}")

            extension = self._determine_file_extension(content, chunk_metadata)
            storage_key = self._generate_storage_key(app_id, document_id, chunk_number, extension)

            if extension == ".txt":
                content_bytes = content.encode("utf-8")
                content_b64 = encode_base64(content_bytes)
                await self.chunk_storage.upload_from_base64(
                    content=content_b64, key=storage_key, content_type="text/plain", bucket=self.chunk_bucket or ""
                )
            else:
                await self.chunk_storage.upload_from_base64(
                    content=content, key=storage_key, bucket=self.chunk_bucket or ""
                )

            logger.debug(f"Stored chunk content externally with key: {storage_key}")
            return storage_key

        except Exception as e:
            logger.error(f"Failed to store content externally for {document_id}-{chunk_number}: {e}")
            return None

    async def _save_chunk_to_storage(self, chunk: DocumentChunk, app_id: Optional[str] = None):
        return await self._store_content_externally(
            chunk.content, chunk.document_id, chunk.chunk_number, json.dumps(chunk.metadata or {}), app_id
        )

    def _is_storage_key(self, content: str) -> bool:
        """Check if content field contains a storage key rather than actual content."""
        return is_storage_key(content)

    async def _download_chunk_bytes(self, bucket: str, storage_key: str) -> Optional[bytes]:
        """Attempt to fetch chunk payload bytes from storage."""
        candidate_order: List[str] = []
        base = storage_key
        candidate_order.append(base)
        _, ext = str(Path(base)), Path(base).suffix
        if not base.endswith(".txt"):
            candidate_order.append(f"{base}.txt")
        try:
            if ext:
                without_ext = base[: -len(ext)]
            else:
                without_ext = base
            if not without_ext.endswith(".txt"):
                candidate_order.append(f"{without_ext}.txt")
                candidate_order.append(f"{without_ext}.txt.txt")
        except Exception:
            pass
        if bucket:
            candidate_order.append(f"{bucket}/{base}")
            if not base.endswith(".txt"):
                candidate_order.append(f"{bucket}/{base}.txt")
            try:
                if ext:
                    without_ext = base[: -len(ext)]
                else:
                    without_ext = base
                if not without_ext.endswith(".txt"):
                    candidate_order.append(f"{bucket}/{without_ext}.txt")
                    candidate_order.append(f"{bucket}/{without_ext}.txt.txt")
            except Exception:
                pass

        for candidate_key in candidate_order:
            try:
                return await self.chunk_storage.download_file(bucket=bucket, key=candidate_key)
            except Exception:
                continue
        return None

    def _decode_chunk_bytes(self, content_bytes: bytes, storage_key: str, chunk_metadata: Optional[str]) -> str:
        """Convert raw chunk bytes into the format expected by callers."""
        if storage_key.endswith(".txt"):
            try:
                return content_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Failed to decode text chunk for key {storage_key}") from exc

        metadata: Dict[str, Any] = {}
        if chunk_metadata:
            try:
                metadata = json.loads(chunk_metadata)
            except json.JSONDecodeError as exc:
                logger.debug("Unable to parse chunk metadata for key %s: %s", storage_key, exc)

        if metadata.get("is_image"):
            try:
                as_text = content_bytes.decode("utf-8")
                if as_text.strip().startswith("data:") and "," in as_text:
                    return as_text
            except Exception:
                pass
            mime = metadata.get("mime_type")
            if not mime:
                b = content_bytes
                if b.startswith(b"\x89PNG\r\n\x1a\n"):
                    mime = "image/png"
                elif b.startswith(b"\xff\xd8"):
                    mime = "image/jpeg"
                elif b.startswith(b"GIF8"):
                    mime = "image/gif"
                elif b.startswith(b"BM"):
                    mime = "image/bmp"
                elif b.startswith(b"II*\x00") or b.startswith(b"MM\x00*"):
                    mime = "image/tiff"
                elif b.startswith(b"RIFF") and b"WEBP" in b[:16]:
                    mime = "image/webp"
                else:
                    mime = "image/png"
            return bytes_to_data_uri(content_bytes, mime)

        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return encode_base64(content_bytes)

    async def _collect_storage_targets(
        self, collection, document_id: str, app_id: Optional[str]
    ) -> Dict[str, Set[Tuple[str, str]]]:
        """Collect storage objects associated with a document before deletion."""
        targets: Dict[str, Set[Tuple[str, str]]] = {"chunk": set(), "vector": set()}
        collection_name = self._collection_name(app_id)

        try:
            # Query all chunks for this document
            result = await collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to collect storage targets for document %s in collection %s: %s",
                document_id,
                collection_name,
                exc,
            )
            return targets

        if not result or not result["metadatas"]:
            return targets

        for meta in result["metadatas"]:
            content_key = meta.get("content_key", "")
            if content_key and self._is_storage_key(content_key):
                bucket_name = self.chunk_bucket if self.chunk_bucket else ""
                targets["chunk"].add((bucket_name, normalize_storage_key(content_key)))

            mv_bucket = meta.get("multivector_bucket", "")
            mv_key = meta.get("multivector_key", "")
            if mv_key:
                normalized_bucket = mv_bucket if mv_bucket else ""
                targets["vector"].add((normalized_bucket, normalize_storage_key(mv_key)))

        return targets

    async def _delete_storage_targets(self, targets: Dict[str, Set[Tuple[str, str]]], document_id: str) -> None:
        """Delete external storage objects recorded for a document."""
        chunk_targets = [(bucket, key) for bucket, key in targets.get("chunk", set()) if key]
        vector_targets = [(bucket, key) for bucket, key in targets.get("vector", set()) if key]

        tasks = []
        task_meta: List[Tuple[str, str, str]] = []

        if self.chunk_storage and chunk_targets:
            for bucket, key in chunk_targets:
                tasks.append(self.chunk_storage.delete_file(bucket, key))
                task_meta.append(("chunk", bucket, key))

        if self.vector_storage and vector_targets:
            for bucket, key in vector_targets:
                tasks.append(self.vector_storage.delete_file(bucket, key))
                task_meta.append(("vector", bucket, key))

        if not tasks:
            await self.cache.delete_many("vectors", vector_targets)
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for meta, result in zip(task_meta, results):
            storage_type, bucket, key = meta
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to delete %s storage object %s/%s for document %s: %s",
                    storage_type,
                    bucket,
                    key,
                    document_id,
                    result,
                )

        await self.cache.delete_many("vectors", vector_targets)

    async def _retrieve_content_from_storage(self, storage_key: str, chunk_metadata: Optional[str]) -> str:
        """Retrieve content from external storage and convert to expected format."""
        if not storage_key:
            return ""

        logger.info(f"Attempting to retrieve content from storage key: {storage_key}")

        if not self.chunk_storage:
            logger.warning(f"External storage not available for retrieving key: {storage_key}")
            return storage_key

        try:
            metadata: Dict[str, Any] = {}
            if chunk_metadata:
                try:
                    metadata = json.loads(chunk_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            is_image = metadata.get("is_image", False)
            mime = metadata.get("mime_type") if is_image else None

            bucket_options: List[str] = []
            preferred_bucket = self.chunk_bucket if self.chunk_bucket else ""
            bucket_options.append(preferred_bucket)
            try:
                default_bucket = getattr(self.chunk_storage, "default_bucket", None)
                if default_bucket is not None and default_bucket not in bucket_options:
                    bucket_options.append(default_bucket)
            except Exception:
                pass
            try:
                for b in list(bucket_options):
                    if not b:
                        continue
                    alt_candidates = set()
                    if "-s3-" in b:
                        alt_candidates.add(b.replace("-s3-", "-"))
                    if b.endswith("-s3"):
                        alt_candidates.add(b[:-3])
                    if b.startswith("s3-"):
                        alt_candidates.add(b[3:])
                    for alt in alt_candidates:
                        if alt and alt not in bucket_options:
                            bucket_options.append(alt)
            except Exception:
                pass
            if MULTIVECTOR_CHUNKS_BUCKET not in bucket_options:
                bucket_options.append(MULTIVECTOR_CHUNKS_BUCKET)

            logger.info(f"Downloading from bucket candidates: {bucket_options}, key: {storage_key}")
            repaired_key = derive_repaired_image_key(storage_key, is_image=is_image, mime_type=mime)
            if repaired_key and repaired_key != storage_key:
                for bucket_candidate in bucket_options:
                    content_bytes = await self._download_chunk_bytes(bucket_candidate, repaired_key)
                    if content_bytes is None:
                        continue

                    logger.info(
                        "Successfully downloaded repaired content from bucket %s, key %s (len=%d)",
                        bucket_candidate,
                        repaired_key,
                        len(content_bytes),
                    )

                    try:
                        result = self._decode_chunk_bytes(content_bytes, repaired_key, chunk_metadata)
                    except Exception as decode_exc:  # noqa: BLE001
                        logger.error(
                            "Downloaded chunk content for key %s could not be decoded: %s",
                            repaired_key,
                            decode_exc,
                        )
                        raise

                    logger.info(
                        "Returning repaired chunk content for key %s (length=%d)",
                        repaired_key,
                        len(result),
                    )
                    return result

            for bucket_candidate in bucket_options:
                content_bytes = await self._download_chunk_bytes(bucket_candidate, storage_key)
                if content_bytes is None:
                    continue

                logger.info(
                    "Successfully downloaded content from bucket %s, key %s (len=%d)",
                    bucket_candidate,
                    storage_key,
                    len(content_bytes),
                )

                try:
                    result = self._decode_chunk_bytes(content_bytes, storage_key, chunk_metadata)
                except Exception as decode_exc:  # noqa: BLE001
                    logger.error(
                        "Downloaded chunk content for key %s could not be decoded: %s",
                        storage_key,
                        decode_exc,
                    )
                    raise

                logger.info(
                    "Returning downloaded chunk content for key %s (length=%d)",
                    storage_key,
                    len(result),
                )
                return result

            logger.error(f"No content downloaded for storage key: {storage_key}")
            return storage_key

        except Exception as e:
            logger.error(f"Failed to retrieve content from storage key {storage_key}: {e}", exc_info=True)
            return storage_key
