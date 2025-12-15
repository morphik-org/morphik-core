"""Fast operations using Rust when available, with Python fallback.

This module provides optimized implementations for CPU-intensive operations.
When the morphik_rust extension is available, it uses SIMD-optimized Rust code.
Otherwise, it falls back to pure Python implementations.

Optimizations based on research from:
- Qdrant's binary quantization (40x speedup): https://qdrant.tech/articles/binary-quantization/
- SimSIMD library: https://github.com/ashvardanian/SimSIMD
- PyO3/rust-numpy best practices: https://github.com/PyO3/rust-numpy
"""

import base64 as _base64
import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Rust extension
try:
    import morphik_rust

    # Verify the extension has actual functions (not just a namespace package)
    if not hasattr(morphik_rust, "encode_base64"):
        raise ImportError("morphik_rust is a namespace package, not the compiled extension")

    HAS_RUST = True
    logger.info("morphik_rust extension loaded - using optimized Rust operations")
except ImportError:
    HAS_RUST = False
    logger.info("morphik_rust not available - using Python fallback")


# =============================================================================
# Base64 Operations (2-3x faster with Rust)
# =============================================================================


def encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string.

    Uses Rust SIMD-optimized implementation when available (2-3x faster).
    """
    if HAS_RUST:
        return morphik_rust.encode_base64(data)
    return _base64.b64encode(data).decode("utf-8")


def decode_base64(encoded: str) -> bytes:
    """Decode base64 string to bytes.

    Uses Rust SIMD-optimized implementation when available (2-3x faster).
    """
    if HAS_RUST:
        return morphik_rust.decode_base64(encoded)
    return _base64.b64decode(encoded)


def bytes_to_data_uri(data: bytes, mime_type: str = "image/png") -> str:
    """Convert bytes to a data URI string.

    Uses Rust implementation when available for faster base64 encoding.
    """
    if HAS_RUST:
        return morphik_rust.bytes_to_data_uri(data, mime_type)
    encoded = _base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def data_uri_to_bytes(data_uri: str) -> bytes:
    """Convert a data URI to bytes.

    Handles both "data:...;base64,..." format and raw base64 strings.
    Uses Rust implementation when available.
    """
    if HAS_RUST:
        return morphik_rust.data_uri_to_bytes(data_uri)

    # Python fallback
    if data_uri.startswith("data:"):
        # Extract base64 portion after the comma
        _, base64_str = data_uri.split(",", 1)
    else:
        base64_str = data_uri
    return _base64.b64decode(base64_str)


# =============================================================================
# Text Chunking (2-5x faster with Rust)
# =============================================================================


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into chunks with overlap.

    Uses Rust implementation when available (2-5x faster for large documents).
    Uses recursive character splitting with separators: ["\\n\\n", "\\n", ". ", " ", ""]
    """
    if HAS_RUST:
        return morphik_rust.split_text(text, chunk_size, chunk_overlap)

    # Python fallback - simplified recursive splitter
    separators = ["\n\n", "\n", ". ", " ", ""]
    return _split_recursive(text, chunk_size, chunk_overlap, separators)


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    """Python fallback for recursive text splitting."""
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    splits = text.split(sep) if sep else list(text)

    chunks = []
    current = ""
    for i, part in enumerate(splits):
        add_part = part + (sep if sep and i != len(splits) - 1 else "")
        if len(current + add_part) > chunk_size:
            if current:
                chunks.append(current)
            current = add_part
        else:
            current += add_part
    if current:
        chunks.append(current)

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size and len(separators) > 1:
            final_chunks.extend(_split_recursive(chunk, chunk_size, chunk_overlap, separators[1:]))
        else:
            final_chunks.append(chunk)

    if chunk_overlap > 0 and len(final_chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(final_chunks):
            if i > 0:
                prev = final_chunks[i - 1]
                overlap = prev[-chunk_overlap:]
                chunk = overlap + chunk
            overlapped.append(chunk)
        return overlapped
    return final_chunks


# =============================================================================
# Binary Quantization (5-10x faster with Rust, enables 40x faster search)
# =============================================================================


def binary_quantize(embeddings: Union[np.ndarray, List[np.ndarray]]) -> List[List[bool]]:
    """Quantize embeddings to binary (values > 0 become True).

    This enables 40x faster similarity search using Hamming distance.

    Args:
        embeddings: 2D numpy array of shape (batch_size, dim) or list of 1D arrays

    Returns:
        List of lists of booleans
    """
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)
    elif not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    if HAS_RUST:
        if embeddings.ndim == 1:
            return [morphik_rust.binary_quantize_1d(embeddings)]
        return morphik_rust.binary_quantize_batch(embeddings)

    # Python fallback
    return [(emb > 0).tolist() for emb in embeddings]


def binary_quantize_packed(embeddings: Union[np.ndarray, List[np.ndarray]]) -> List[bytes]:
    """Quantize embeddings to packed bytes (8 bits per byte).

    This is the most memory-efficient representation:
    - 128-dim embedding: 16 bytes instead of 512 bytes (32x compression)
    - Enables fast Hamming distance with XOR + popcount

    Args:
        embeddings: 2D numpy array of shape (batch_size, dim) or list of 1D arrays

    Returns:
        List of bytes objects with packed binary representation
    """
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)
    elif not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    if HAS_RUST:
        if embeddings.ndim == 1:
            return [morphik_rust.binary_quantize_packed(embeddings)]
        return morphik_rust.binary_quantize_batch_packed(embeddings)

    # Python fallback
    result = []
    for emb in embeddings:
        bits = emb > 0
        num_bytes = (len(bits) + 7) // 8
        packed = bytearray(num_bytes)
        for i, bit in enumerate(bits):
            if bit:
                packed[i // 8] |= 1 << (7 - (i % 8))
        result.append(bytes(packed))
    return result


def hamming_distance(a: bytes, b: bytes) -> int:
    """Compute Hamming distance between two packed binary vectors.

    Uses XOR + popcount which is very fast on modern CPUs.
    """
    if HAS_RUST:
        return morphik_rust.hamming_distance(a, b)

    # Python fallback
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def hamming_distance_batch(query: bytes, candidates: List[bytes]) -> List[int]:
    """Batch compute Hamming distances between a query and multiple candidates."""
    if HAS_RUST:
        return morphik_rust.hamming_distance_batch(query, candidates)

    # Python fallback
    return [hamming_distance(query, c) for c in candidates]


# =============================================================================
# Text Utilities
# =============================================================================


def count_tokens_whitespace(text: str) -> int:
    """Count tokens using whitespace splitting (fast approximation)."""
    if HAS_RUST:
        return morphik_rust.count_tokens_whitespace(text)
    return len(text.split())


def count_tokens_batch(texts: List[str]) -> List[int]:
    """Count tokens in multiple texts (parallel in Rust)."""
    if HAS_RUST:
        return morphik_rust.count_tokens_batch(texts)
    return [len(t.split()) for t in texts]


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    if HAS_RUST:
        return morphik_rust.normalize_whitespace(text)
    return " ".join(text.split())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using common delimiters."""
    if HAS_RUST:
        return morphik_rust.split_sentences(text)

    # Python fallback
    import re

    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def clean_control_chars(text: str) -> str:
    """Remove control characters (null bytes, etc.) that cause database issues."""
    if HAS_RUST:
        return morphik_rust.clean_control_chars(text)

    # Python fallback - check if character is a control char (ASCII 0-31, 127)
    # but keep newline, tab, carriage return
    import unicodedata

    return "".join(c for c in text if not (unicodedata.category(c).startswith("C") and c not in "\n\t\r"))


def clean_control_chars_batch(texts: List[str]) -> List[str]:
    """Batch clean control characters (parallel in Rust)."""
    if HAS_RUST:
        return morphik_rust.clean_control_chars_batch(texts)
    return [clean_control_chars(t) for t in texts]
