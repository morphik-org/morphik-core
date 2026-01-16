from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DEFAULT_APP_ID = "default"


def normalize_app_id(app_id: Optional[str]) -> str:
    return app_id or DEFAULT_APP_ID


def extract_storage_bytes(metrics: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if not metrics:
        return 0, 0

    if metrics.get("mode") == "dual":
        fast = metrics.get("fast") or {}
        slow = metrics.get("slow") or {}
        fast_chunk = int(fast.get("chunk_payload_bytes") or 0)
        fast_mv = int(fast.get("multivector_bytes") or 0)
        slow_chunk = int(slow.get("chunk_payload_bytes") or 0)
        slow_mv = int(slow.get("multivector_bytes") or 0)
        chunk_bytes = max(fast_chunk, slow_chunk)
        multivector_bytes = max(fast_mv, slow_mv)
        return chunk_bytes, multivector_bytes

    chunk_bytes = int(metrics.get("chunk_payload_bytes") or 0)
    multivector_bytes = int(metrics.get("multivector_bytes") or 0)
    return chunk_bytes, multivector_bytes
