"""Shared utilities for vector store implementations."""

MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"


def normalize_storage_key(key: str) -> str:
    """Strip bucket prefix if it is embedded in the key."""
    if key.startswith(f"{MULTIVECTOR_CHUNKS_BUCKET}/"):
        return key[len(MULTIVECTOR_CHUNKS_BUCKET) + 1 :]
    return key
