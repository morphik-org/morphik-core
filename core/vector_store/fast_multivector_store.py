import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from uuid import uuid4

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
from .multivector_block import BlockPointer, MultiVectorBlock, MultiVectorBlockEntry

logger = logging.getLogger(__name__)

# Attach a dedicated handler to capture multivector retrieval diagnostics once per process.
_multivector_log_path = os.path.join("logs", "multivector.log")
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
MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"
DEFAULT_APP_ID = "default"  # Fallback for local usage when app_id is None


if get_settings().MULTIVECTOR_STORE_PROVIDER == "morphik":
    import fixed_dimensional_encoding as fde
    from turbopuffer import AsyncTurbopuffer, NotFoundError
else:
    NotFoundError = Exception  # type: ignore[assignment]


class FileCacheManager:
    """Manage local on-disk cache for blobs with eviction."""

    def __init__(self, enabled: bool, base_dir: Path, max_bytes: int):
        self.enabled = enabled
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        self._lock = asyncio.Lock()
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
            await asyncio.to_thread(self._touch_file, path)
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
            await asyncio.to_thread(self._write_file, path, data)
            await asyncio.to_thread(self._touch_file, path)
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
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to delete cache entry %s: %s", path, exc)

    async def delete_many(self, namespace: str, items: Iterable[Tuple[str, str]]) -> None:
        if not self.enabled:
            return
        tasks = [self.delete(namespace, bucket, key) for bucket, key in items]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _write_file(self, path: Path, data: bytes) -> None:
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
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def _touch_file(self, path: Path) -> None:
        now = time.time()
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
        async with self._lock:
            await asyncio.to_thread(self._enforce_budget_sync)

    def _enforce_budget_sync(self) -> None:
        total_size = 0
        files: List[Tuple[float, int, Path]] = []
        for file_path in self.base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                stat = file_path.stat()
            except FileNotFoundError:
                continue
            total_size += stat.st_size
            files.append((stat.st_atime, stat.st_size, file_path))

        if total_size <= self.max_bytes:
            return

        files.sort(key=lambda item: item[0])  # Oldest access time first
        for _, file_size, file_path in files:
            if total_size <= self.max_bytes:
                break
            try:
                file_path.unlink()
                total_size -= file_size
            except FileNotFoundError:
                continue


# external storage always enabled, no two ways about it
class FastMultiVectorStore(BaseVectorStore):
    def __init__(self, uri: str, tpuf_api_key: str, namespace: str = "public", region: str = "aws-us-west-2"):
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://")
        self.uri = uri
        self.tpuf_api_key = tpuf_api_key
        self.namespace = namespace
        self.tpuf = AsyncTurbopuffer(api_key=tpuf_api_key, region=region, default_namespace="default2")
        # TODO: Cache namespaces, and send a warming request
        self.ns = lambda app_id: self.tpuf.namespace(app_id)
        self.chunk_storage, self.chunk_bucket = self._init_chunk_storage()
        self.vector_storage, self.vector_bucket = self._init_vector_storage()
        # Maintain legacy attribute for backwards compatibility with other components
        self.storage = self.chunk_storage
        settings = get_settings()
        cache_enabled = settings.CACHE_ENABLED
        cache_path = Path(settings.CACHE_PATH or "./storage/cache")
        cache_limit = settings.CACHE_MAX_BYTES
        self.cache = FileCacheManager(
            enabled=cache_enabled,
            base_dir=cache_path,
            max_bytes=cache_limit,
        )
        self.block_storage_enabled = bool(getattr(settings, "MULTIVECTOR_BLOCK_STORAGE_ENABLED", True))
        self.block_size = max(1, int(getattr(settings, "MULTIVECTOR_BLOCK_SIZE", 2048)))
        self.block_prefix = getattr(settings, "MULTIVECTOR_BLOCK_PREFIX", "multivector_blocks") or "multivector_blocks"
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

    async def close(self) -> None:
        """Release network and database resources held by the vector store."""
        logger.info("Closing FastMultiVectorStore resources...")

        tpuf_client = getattr(self, "tpuf", None)
        if tpuf_client is not None and hasattr(tpuf_client, "is_closed"):
            try:
                if not tpuf_client.is_closed():
                    await tpuf_client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close TurboPuffer client cleanly: %s", exc)

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
                )
            case "local":
                path = storage_path or "./storage"
                return LocalStorage(storage_path=path)
            case _:
                raise ValueError(f"Unsupported storage provider: {provider}")

    def _block_object_key(self, app_id: Optional[str]) -> str:
        namespace = (app_id or self.namespace).strip("/ ")
        prefix = (self.block_prefix or "multivector_blocks").strip("/ ")
        return f"{prefix}/{namespace}/{uuid4().hex}.bin"

    def _coerce_multivector_array(self, embedding: Any) -> Optional[np.ndarray]:
        """Convert raw embedding payload into a 2D numpy array or return None on failure."""
        try:
            arr = np.asarray(embedding, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to coerce embedding into array: %s", exc)
            return None

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.size == 0:
            return None
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    async def _upload_block(
        self, entries: List[MultiVectorBlockEntry], app_id: Optional[str]
    ) -> Tuple[str, str, int]:
        """Serialize and upload a multivector block, returning (bucket, key, count)."""
        block = MultiVectorBlock.build(entries)
        payload = block.to_bytes()
        object_key = self._block_object_key(app_id)
        bucket_arg = "" if isinstance(self.vector_storage, LocalStorage) else (self.vector_bucket or "")

        logger.debug(
            "Uploading multivector block %s with %d entries (%d bytes)",
            object_key,
            len(entries),
            len(payload),
        )

        bucket, saved_key = await self.vector_storage.upload_file(payload, object_key, bucket=bucket_arg)
        if isinstance(self.vector_storage, LocalStorage):
            bucket = ""

        await self.cache.delete("vectors", bucket, saved_key)
        return bucket, saved_key, len(entries)

    async def _flush_block_entries(
        self,
        staged: List[Tuple[int, DocumentChunk, np.ndarray]],
        app_id: Optional[str],
        results: List[Tuple[str, str]],
    ) -> None:
        """Upload staged entries as a block; fall back to legacy storage on failure."""
        try:
            entries = [
                MultiVectorBlockEntry(chunk.document_id, chunk.chunk_number, arr) for _, chunk, arr in staged
            ]
            bucket, base_key, _ = await self._upload_block(entries, app_id)
            for local_idx, (idx, _chunk, _arr) in enumerate(staged):
                results[idx] = (bucket, f"{base_key}#{local_idx}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to upload multivector block (%d entries); falling back to legacy objects: %s",
                len(staged),
                exc,
            )

        for result_idx, chunk, _ in staged:
            bucket, key = await self._save_legacy_multivector_to_storage(chunk)
            results[result_idx] = (bucket, key)

    def initialize(self):
        return True

    async def _save_legacy_multivector_to_storage(self, chunk: DocumentChunk) -> Tuple[str, str]:
        """Persist a single multivector as a standalone object (backwards-compatible path)."""
        as_np = np.array(chunk.embedding)
        save_path = f"multivector/{chunk.document_id}/{chunk.chunk_number}.npy"
        with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
            np.save(temp_file, as_np)
            if isinstance(self.vector_storage, S3Storage):
                target_bucket = self.vector_bucket or self.vector_storage.default_bucket
                self.vector_storage._ensure_bucket(target_bucket)  # type: ignore[attr-defined]
                self.vector_storage.s3_client.upload_file(temp_file.name, target_bucket, save_path)
                bucket, key = target_bucket, save_path
            else:
                bucket_arg = "" if isinstance(self.vector_storage, LocalStorage) else (self.vector_bucket or "")
                bucket, key = await self.vector_storage.upload_file(temp_file.name, save_path, bucket=bucket_arg)
                if isinstance(self.vector_storage, LocalStorage):
                    bucket = ""
            temp_file.close()
        await self.cache.delete("vectors", bucket, key)
        return bucket, key

    async def save_multivectors_to_blocks(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Store multivectors using block layout when enabled, falling back to legacy objects as needed.
        Returns (bucket, key) pairs aligned with input chunks.
        """
        if not chunks:
            return []

        if not self.block_storage_enabled:
            legacy_results = await asyncio.gather(*[self._save_legacy_multivector_to_storage(c) for c in chunks])
            return list(legacy_results)

        results: List[Tuple[str, str]] = [("", "") for _ in chunks]
        staged: List[Tuple[int, DocumentChunk, np.ndarray]] = []
        legacy_chunks: List[Tuple[int, DocumentChunk]] = []

        for idx, chunk in enumerate(chunks):
            coerced = self._coerce_multivector_array(chunk.embedding)
            if coerced is None:
                legacy_chunks.append((idx, chunk))
                continue

            staged.append((idx, chunk, coerced))
            if len(staged) >= self.block_size:
                await self._flush_block_entries(staged, app_id, results)
                staged = []

        if staged:
            await self._flush_block_entries(staged, app_id, results)

        if legacy_chunks:
            legacy_results = await asyncio.gather(
                *[self._save_legacy_multivector_to_storage(chunk) for _, chunk in legacy_chunks]
            )
            for (idx, _chunk), stored in zip(legacy_chunks, legacy_results):
                results[idx] = stored

        missing = [i for i, (_bucket, key) in enumerate(results) if not key]
        if missing:
            raise ValueError(f"Failed to store multivectors for positions: {missing}")

        return results

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        #  group fde calls for better cache hit rate
        embeddings = [
            fde.generate_document_encoding(np.array(chunk.embedding), self.fde_config).tolist() for chunk in chunks
        ]
        storage_keys = await asyncio.gather(*[self._save_chunk_to_storage(chunk, app_id) for chunk in chunks])
        stored_ids = [f"{chunk.document_id}-{chunk.chunk_number}" for chunk in chunks]
        doc_ids, chunk_numbers, metdatas = [], [], []
        multivector_refs = await self.save_multivectors_to_blocks(chunks, app_id)
        multivecs = [[bucket, key] for bucket, key in multivector_refs]
        block_keys = {key.split("#", 1)[0] for _, key in multivector_refs if "#" in key}
        legacy_multivectors = sum(1 for _bucket, key in multivector_refs if "#" not in key)
        for chunk in chunks:
            doc_ids.append(chunk.document_id)
            chunk_numbers.append(chunk.chunk_number)
            metdatas.append(json.dumps(chunk.metadata))
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
        logger.info(
            "Stored %d chunks (%d block objects, %d legacy objects), tpuf ns: %s",
            len(chunks),
            len(block_keys),
            legacy_multivectors,
            result.model_dump_json(),
        )
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
            consistency={"level": "eventual"},
        )
        t2 = time.perf_counter()
        logger.info(f"query_similar timing - ns.query: {(t2 - t1)*1000:.2f} ms")

        # 3) Download multi-vectors
        if not result.rows:
            logger.info(
                "query_similar: namespace query returned no rows (doc_ids=%s, app_id=%s); returning empty result",
                doc_ids,
                app_id or self.namespace,
            )
            return []

        multivectors = await self._load_multivectors_for_rows(result.rows)
        t3 = time.perf_counter()
        logger.info(f"query_similar timing - load_multivectors: {(t3 - t2)*1000:.2f} ms")

        # 4) Rerank using ColQwen2.5 processor
        scores = self.processor.score_multi_vector(
            [torch.from_numpy(query_embedding).float()], multivectors, device=self.device
        )[0]
        scores, idx = torch.topk(scores, min(k, len(scores)))
        scores, top_k_indices = scores.tolist(), idx.tolist()

        # Log which positions from the initial store results were selected after reranking
        # This shows if top-k chunks are consistently near the top or scattered throughout candidates
        num_candidates = len(result.rows)
        try:
            selected_ids = [result.rows[i].get("id") for i in top_k_indices]
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
        namespace = self.ns(app_id)
        storage_targets: Dict[str, Set[Tuple[str, str]]] = {"chunk": set(), "vector": set()}

        storage_available = self.chunk_storage or self.vector_storage
        if storage_available:
            storage_targets = await self._collect_storage_targets(namespace, document_id, app_id)

        try:
            await namespace.write(delete_by_filter=("document_id", "Eq", document_id))
        except NotFoundError:
            logger.info(
                "TurboPuffer namespace %s not found while deleting document %s",
                app_id or self.namespace,
                document_id,
            )
            storage_targets = {"chunk": set(), "vector": set()}
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to delete TurboPuffer rows for document %s in namespace %s: %s",
                document_id,
                app_id or self.namespace,
                exc,
            )
            return False

        if storage_targets["chunk"] or storage_targets["vector"]:
            await self._delete_storage_targets(storage_targets, document_id)

        return True

    async def save_multivector_to_storage(
        self, chunk: DocumentChunk, app_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """Compatibility wrapper that stores a single multivector using block or legacy layout."""
        stored = await self.save_multivectors_to_blocks([chunk], app_id)
        if not stored:
            raise RuntimeError("Failed to store multivector for chunk.")
        return stored[0]

    def _resolve_vector_bucket(self, bucket: str) -> Tuple[str, str]:
        """Normalize bucket for local storage and derive cache bucket."""
        bucket = bucket or ""
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
        return primary_bucket, cache_bucket

    async def _download_vector_content(self, bucket: str, key: str, use_cache: bool = True) -> bytes:
        """Download vector content with cache and storage fallback handling."""
        bucket = bucket or ""
        primary_bucket, cache_bucket = self._resolve_vector_bucket(bucket)

        if use_cache:
            cached_bytes = await self.cache.get("vectors", cache_bucket, key)
            if cached_bytes is not None:
                return cached_bytes

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
        return content

    async def _load_block(self, bucket: str, base_key: str) -> MultiVectorBlock:
        """Load and decode a block, refreshing cache on decode failure."""
        _, cache_bucket = self._resolve_vector_bucket(bucket)
        payload = await self._download_vector_content(bucket, base_key)
        try:
            return MultiVectorBlock.from_bytes(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Block decode failed for %s/%s, retrying with cache refresh: %s",
                bucket,
                base_key,
                exc,
            )
            await self.cache.delete("vectors", cache_bucket, base_key)
            payload = await self._download_vector_content(bucket, base_key, use_cache=False)
            return MultiVectorBlock.from_bytes(payload)

    async def _load_legacy_multivector(self, bucket: str, key: str) -> torch.Tensor:
        """Load a legacy per-chunk multivector object."""
        _, cache_bucket = self._resolve_vector_bucket(bucket)
        payload = await self._download_vector_content(bucket, key)
        try:
            as_np = np.load(BytesIO(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Legacy vector decode failed for %s/%s, refreshing cache: %s", bucket, key, exc)
            await self.cache.delete("vectors", cache_bucket, key)
            payload = await self._download_vector_content(bucket, key, use_cache=False)
            as_np = np.load(BytesIO(payload))
        return torch.from_numpy(as_np).float()

    async def _load_multivectors_for_rows(self, rows: List[Any]) -> List[torch.Tensor]:
        """Batch load multivectors, deduplicating block fetches and falling back to legacy objects."""
        pointers: List[BlockPointer] = []
        for row in rows:
            multivector_ref = self._row_get(row, "multivector")
            if not (isinstance(multivector_ref, (list, tuple)) and len(multivector_ref) == 2):
                raise ValueError(f"Row missing multivector reference: {row}")
            bucket, key = multivector_ref
            if not isinstance(bucket, str):
                bucket = bucket or ""
            if not isinstance(key, str):
                raise ValueError(f"Multivector key must be a string, got {type(key)}")
            pointers.append(BlockPointer.parse(bucket, key))

        block_groups: Dict[Tuple[str, str], List[Tuple[int, BlockPointer]]] = defaultdict(list)
        results: List[Optional[torch.Tensor]] = [None for _ in rows]
        legacy_indices: List[int] = []

        for idx, pointer in enumerate(pointers):
            if pointer.is_block and pointer.item_index is not None:
                block_groups[(pointer.bucket, pointer.base_key)].append((idx, pointer))
            else:
                legacy_indices.append(idx)

        if block_groups:
            block_tasks = {
                (bucket, base_key): asyncio.create_task(self._load_block(bucket, base_key))
                for bucket, base_key in block_groups
            }
            block_results = await asyncio.gather(*block_tasks.values(), return_exceptions=True)
            for (bucket, base_key), block_obj in zip(block_tasks.keys(), block_results):
                if isinstance(block_obj, Exception):
                    raise ValueError(f"Failed to load multivector block {bucket}/{base_key}: {block_obj}") from block_obj

                for idx, ptr in block_groups[(bucket, base_key)]:
                    try:
                        emb = block_obj.embedding_at(ptr.item_index or 0)
                        results[idx] = torch.from_numpy(emb).float()
                    except Exception as exc:  # noqa: BLE001
                        raise ValueError(
                            f"Failed to decode multivector index {ptr.item_index} in block {bucket}/{base_key}: {exc}"
                        ) from exc

        if legacy_indices:
            legacy_tasks = [
                self._load_legacy_multivector(pointers[i].bucket, pointers[i].raw_key) for i in legacy_indices
            ]
            legacy_results = await asyncio.gather(*legacy_tasks)
            for idx, tensor in zip(legacy_indices, legacy_results):
                results[idx] = tensor

        logger.debug(
            "Loaded %d multivectors via %d block(s) and %d legacy object(s)",
            len(results),
            len(block_groups),
            len(legacy_indices),
        )

        missing = [i for i, tensor in enumerate(results) if tensor is None]
        if missing:
            raise ValueError(f"Missing multivectors for result rows at positions: {missing}")

        return [tensor for tensor in results if tensor is not None]

    async def load_multivector_from_storage(self, bucket: str, key: str) -> torch.Tensor:
        pointer = BlockPointer.parse(bucket or "", key)
        if "#" in key and not pointer.is_block:
            raise ValueError(f"Malformed block pointer key: {key}")
        if pointer.is_block:
            block = await self._load_block(pointer.bucket, pointer.base_key)
            return torch.from_numpy(block.embedding_at(pointer.item_index or 0)).float()
        return await self._load_legacy_multivector(bucket, key)

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
        if not self.chunk_storage:
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
                await self.chunk_storage.upload_from_base64(
                    content=content_b64, key=storage_key, content_type="text/plain", bucket=self.chunk_bucket or ""
                )
            else:
                # For images, content should already be base64
                await self.chunk_storage.upload_from_base64(
                    content=content, key=storage_key, bucket=self.chunk_bucket or ""
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

    @staticmethod
    def _normalize_storage_key(key: str) -> str:
        base_key = key.split("#", 1)[0]
        if base_key.startswith(f"{MULTIVECTOR_CHUNKS_BUCKET}/"):
            return base_key[len(MULTIVECTOR_CHUNKS_BUCKET) + 1 :]
        return base_key

    async def _download_chunk_bytes(self, bucket: str, storage_key: str) -> Optional[bytes]:
        """Attempt to fetch chunk payload bytes from storage, considering legacy/variant keys.

        We try multiple combinations to handle historical key formats:
        - key (as stored currently)
        - key + ".txt" (legacy text objects)
        - f"{bucket}/{key}" (legacy keys that accidentally embedded the bucket name)
        - f"{bucket}/{key}.txt" (legacy with both bucket prefix and .txt suffix)
        """
        candidate_order: List[str] = []
        base = storage_key
        # Current expected key
        candidate_order.append(base)
        # Variants derived from base name
        _, ext = str(Path(base)), Path(base).suffix
        # Add base + .txt
        if not base.endswith(".txt"):
            candidate_order.append(f"{base}.txt")
        # Replace extension with .txt and .txt.txt
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
        # Legacy bucket-prefixed keys (only if bucket provided)
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
            # Preserve previous behavior by returning a data URI for images.
            # Use mime from metadata when available; otherwise sniff common headers.
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
            data_b64 = base64.b64encode(content_bytes).decode("utf-8")
            return f"data:{mime};base64,{data_b64}"

        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(content_bytes).decode("utf-8")

    @staticmethod
    def _row_get(row: Union[Dict[str, Any], object], field: str) -> Optional[Any]:
        """Helper to safely extract fields from TurboPuffer rows."""
        try:
            if isinstance(row, dict):
                return row.get(field)
            if hasattr(row, "__getitem__"):
                return row[field]  # type: ignore[index]
        except (KeyError, TypeError):
            pass
        return getattr(row, field, None)

    async def _collect_storage_targets(
        self, namespace, document_id: str, app_id: Optional[str]
    ) -> Dict[str, Set[Tuple[str, str]]]:
        """Collect storage objects associated with a document before deletion."""
        targets: Dict[str, Set[Tuple[str, str]]] = {"chunk": set(), "vector": set()}
        last_id: Optional[str] = None
        page_size = 500
        namespace_name = app_id or self.namespace

        while True:
            filters = (
                ("document_id", "Eq", document_id)
                if last_id is None
                else ("And", [("document_id", "Eq", document_id), ("id", "Gt", last_id)])
            )
            try:
                result = await namespace.query(
                    filters=filters,
                    include_attributes=["content", "multivector"],
                    rank_by=("id", "asc"),
                    top_k=page_size,
                )
            except NotFoundError:
                logger.info(
                    "TurboPuffer namespace %s not found while collecting storage targets for document %s",
                    namespace_name,
                    document_id,
                )
                return {"chunk": set(), "vector": set()}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to collect storage targets for document %s in namespace %s: %s",
                    document_id,
                    namespace_name,
                    exc,
                )
                break

            rows = result.rows or []
            last_seen_id: Optional[str] = None

            for row in rows:
                content = self._row_get(row, "content")
                if isinstance(content, str) and self._is_storage_key(content):
                    bucket_name = self.chunk_bucket if self.chunk_bucket else ""
                    targets["chunk"].add((bucket_name, self._normalize_storage_key(content)))

                multivector = self._row_get(row, "multivector")
                if isinstance(multivector, (list, tuple)) and len(multivector) == 2:
                    bucket, key = multivector
                    if isinstance(bucket, str) and isinstance(key, str):
                        pointer = BlockPointer.parse(bucket or "", key)
                        normalized_bucket = pointer.cache_bucket
                        targets["vector"].add((normalized_bucket, self._normalize_storage_key(pointer.base_key)))

                row_id = self._row_get(row, "id")
                if isinstance(row_id, str):
                    last_seen_id = row_id

            if len(rows) < page_size or last_seen_id is None:
                break

            last_id = last_seen_id

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
        logger.info(f"Attempting to retrieve content from storage key: {storage_key}")

        if not self.chunk_storage:
            logger.warning(f"External storage not available for retrieving key: {storage_key}")
            return storage_key  # Return storage key as fallback

        try:
            bucket_options: List[str] = []
            preferred_bucket = self.chunk_bucket if self.chunk_bucket else ""
            # 1) Preferred chunk bucket (from settings)
            bucket_options.append(preferred_bucket)
            # 2) Storage default bucket (if available and distinct)
            try:
                default_bucket = getattr(self.chunk_storage, "default_bucket", None)
                if default_bucket is not None and default_bucket not in bucket_options:
                    bucket_options.append(default_bucket)
            except Exception:
                pass
            # 2b) Heuristic alternates for historical naming (e.g., removing '-s3')
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
            # 3) Legacy constant bucket
            if MULTIVECTOR_CHUNKS_BUCKET not in bucket_options:
                bucket_options.append(MULTIVECTOR_CHUNKS_BUCKET)

            logger.info(f"Downloading from bucket candidates: {bucket_options}, key: {storage_key}")
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
            return storage_key  # Return storage key as fallback
