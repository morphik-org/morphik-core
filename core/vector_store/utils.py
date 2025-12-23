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


def is_storage_key(value: Any, *, require_extension: bool = False) -> bool:
    """Best-effort heuristic to detect storage keys in content fields."""
    if (
        not isinstance(value, str)
        or len(value) >= 500
        or "/" not in value
        or value.startswith("data:")
        or value.startswith("http")
    ):
        return False
    if any(ch.isspace() for ch in value):
        return False
    if any(ch in value for ch in ("(", ")", ";", "=", "{", "}")):
        return False
    if require_extension:
        _, tail = value.rsplit("/", 1)
        if "." not in tail:
            return False
    return True


def derive_repaired_image_key(storage_key: Any, *, is_image: bool, mime_type: Optional[str]) -> Optional[str]:
    """Derive a corrected image key when a legacy .txt key contains image data."""
    if not is_image or not isinstance(storage_key, str) or not storage_key.lower().endswith(".txt"):
        return None
    base = storage_key.rsplit(".", 1)[0]
    lower_base = base.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
    if any(lower_base.endswith(ext) for ext in image_exts):
        return base
    mime_to_ext = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return f"{base}{mime_to_ext.get(mime_type, '.png')}"


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
