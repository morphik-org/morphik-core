"""Shared utilities for vector store implementations."""

from typing import Any, Dict, Optional

import psycopg

from core.storage.base_storage import BaseStorage
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage

MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"


def normalize_storage_key(key: str) -> str:
    """Strip bucket prefix if it is embedded in the key."""
    if key.startswith(f"{MULTIVECTOR_CHUNKS_BUCKET}/"):
        return key[len(MULTIVECTOR_CHUNKS_BUCKET) + 1 :]
    return key


def storage_provider_name(storage: Optional[BaseStorage]) -> str:
    if storage is None:
        return "none"
    if isinstance(storage, S3Storage):
        return "aws-s3"
    if isinstance(storage, LocalStorage):
        return "local"
    return storage.__class__.__name__.lower()


def build_store_metrics(
    *,
    chunk_payload_backend: str,
    multivector_backend: str,
    vector_store_backend: str,
    chunk_payload_upload_s: float = 0.0,
    chunk_payload_objects: int = 0,
    multivector_upload_s: float = 0.0,
    multivector_objects: int = 0,
    vector_store_write_s: float = 0.0,
    vector_store_rows: int = 0,
    cache_write_s: float = 0.0,
    cache_write_objects: int = 0,
) -> Dict[str, Any]:
    return {
        "chunk_payload_upload_s": chunk_payload_upload_s,
        "chunk_payload_objects": chunk_payload_objects,
        "chunk_payload_backend": chunk_payload_backend,
        "multivector_upload_s": multivector_upload_s,
        "multivector_objects": multivector_objects,
        "multivector_backend": multivector_backend,
        "vector_store_write_s": vector_store_write_s,
        "vector_store_backend": vector_store_backend,
        "vector_store_rows": vector_store_rows,
        "cache_write_s": cache_write_s,
        "cache_write_objects": cache_write_objects,
    }


def reset_pooled_connection(conn, logger=None) -> bool:
    """Ensure a pooled psycopg connection is idle before returning it."""
    try:
        status = conn.info.transaction_status
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Failed to read connection status: %s", exc)
        return False

    try:
        if status != psycopg.pq.TransactionStatus.IDLE:
            conn.rollback()
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Failed to rollback pooled connection: %s", exc)
        return False

    return True
