from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

_MAGIC = b"MVBL"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sBBI")  # magic, version, dtype_code, num_items
_INDEX_STRUCT = struct.Struct("<H I H I Q I")  # doc_id_len, chunk_number, num_modalities, dim, offset, length

# Keep the dtype space tiny for now; extend as needed.
_DTYPE_TO_CODE = {np.dtype(np.float16): 1}
_CODE_TO_DTYPE = {1: np.dtype(np.float16)}


@dataclass(frozen=True)
class MultiVectorBlockEntry:
    """Entry used while building a block."""

    doc_id: str
    chunk_number: int
    embedding: np.ndarray


@dataclass(frozen=True)
class MultiVectorBlockIndex:
    """Index metadata stored for every multivector inside a block."""

    doc_id: str
    chunk_number: int
    num_modalities: int
    dim: int
    offset: int  # Offset in number of dtype elements from the start of the payload
    length: int  # Number of dtype elements for this multivector


@dataclass(frozen=True)
class BlockPointer:
    """Pointer to a multivector inside a block object."""

    bucket: str
    base_key: str
    item_index: Optional[int] = None

    @property
    def cache_bucket(self) -> str:
        return self.bucket or ""

    @property
    def raw_key(self) -> str:
        if self.item_index is None:
            return self.base_key
        return f"{self.base_key}#{self.item_index}"

    @property
    def is_block(self) -> bool:
        return self.item_index is not None

    @classmethod
    def parse(cls, bucket: str, key: str) -> "BlockPointer":
        """Parse raw storage bucket/key into a structured pointer."""
        if "#" not in key:
            return cls(bucket=bucket, base_key=key, item_index=None)

        base, _, idx_str = key.rpartition("#")
        try:
            idx = int(idx_str)
        except ValueError:
            # Unexpected delimiter usage – treat as a plain object key.
            return cls(bucket=bucket, base_key=key, item_index=None)
        return cls(bucket=bucket, base_key=base, item_index=idx)


def _dtype_to_code(dtype: np.dtype) -> int:
    code = _DTYPE_TO_CODE.get(np.dtype(dtype))
    if code is None:
        raise ValueError(f"Unsupported dtype for block encoding: {dtype}")
    return code


def _code_to_dtype(code: int) -> np.dtype:
    dtype = _CODE_TO_DTYPE.get(code)
    if dtype is None:
        raise ValueError(f"Unknown dtype code in block payload: {code}")
    return dtype


class MultiVectorBlock:
    """Binary block containing many multivectors + lightweight index metadata."""

    def __init__(self, dtype: np.dtype, indices: List[MultiVectorBlockIndex], payload: bytes | memoryview):
        if not indices:
            raise ValueError("Cannot create a block without indices.")
        self.dtype = np.dtype(dtype)
        self.indices = indices
        self.payload = payload if isinstance(payload, memoryview) else memoryview(payload)

    @classmethod
    def build(cls, entries: Sequence[MultiVectorBlockEntry], dtype: np.dtype | None = None) -> "MultiVectorBlock":
        """Construct a block from in-memory entries."""
        if not entries:
            raise ValueError("Cannot build a block with zero entries.")

        target_dtype = np.dtype(dtype or np.float16)
        _dtype_to_code(target_dtype)  # Validate support early.

        payload_chunks: List[bytes] = []
        indices: List[MultiVectorBlockIndex] = []
        offset = 0

        for entry in entries:
            doc_id_bytes = entry.doc_id.encode("utf-8")
            if len(doc_id_bytes) > (2**16 - 1):
                raise ValueError(f"Document id too long for block entry: {entry.doc_id!r}")

            arr = np.asarray(entry.embedding, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.size == 0:
                raise ValueError("Embeddings must be 2D arrays with at least one element.")
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            arr_cast = arr.astype(target_dtype, copy=False)
            flat = arr_cast.reshape(-1)
            length = int(flat.size)
            indices.append(
                MultiVectorBlockIndex(
                    doc_id=entry.doc_id,
                    chunk_number=entry.chunk_number,
                    num_modalities=int(arr.shape[0]),
                    dim=int(arr.shape[1]),
                    offset=int(offset),
                    length=length,
                )
            )
            payload_chunks.append(flat.tobytes())
            offset += length

        payload = b"".join(payload_chunks)
        return cls(dtype=target_dtype, indices=indices, payload=payload)

    def to_bytes(self) -> bytes:
        """Serialize block to bytes."""
        dtype_code = _dtype_to_code(self.dtype)
        parts: List[bytes] = [
            _HEADER_STRUCT.pack(_MAGIC, _VERSION, dtype_code, len(self.indices)),
        ]
        for idx in self.indices:
            doc_bytes = idx.doc_id.encode("utf-8")
            parts.append(
                _INDEX_STRUCT.pack(
                    len(doc_bytes),
                    idx.chunk_number,
                    idx.num_modalities,
                    idx.dim,
                    idx.offset,
                    idx.length,
                )
            )
            parts.append(doc_bytes)

        parts.append(self.payload.tobytes())
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "MultiVectorBlock":
        """Parse a serialized block back into an in-memory structure."""
        buf = data if isinstance(data, memoryview) else memoryview(data)
        if len(buf) < _HEADER_STRUCT.size:
            raise ValueError("Block payload is too small to contain a header.")

        magic, version, dtype_code, num_items = _HEADER_STRUCT.unpack_from(buf, 0)
        if magic != _MAGIC:
            raise ValueError("Invalid block magic header.")
        if version != _VERSION:
            raise ValueError(f"Unsupported block version: {version}")

        dtype = _code_to_dtype(dtype_code)

        offset = _HEADER_STRUCT.size
        indices: List[MultiVectorBlockIndex] = []
        for _ in range(num_items):
            if offset + _INDEX_STRUCT.size > len(buf):
                raise ValueError("Block index table truncated.")
            doc_len, chunk_number, num_modalities, dim, vec_offset, length = _INDEX_STRUCT.unpack_from(buf, offset)
            offset += _INDEX_STRUCT.size

            doc_end = offset + doc_len
            if doc_end > len(buf):
                raise ValueError("Block index entry extends past payload.")
            doc_id = bytes(buf[offset:doc_end]).decode("utf-8", errors="replace")
            offset = doc_end

            indices.append(
                MultiVectorBlockIndex(
                    doc_id=doc_id,
                    chunk_number=chunk_number,
                    num_modalities=num_modalities,
                    dim=dim,
                    offset=vec_offset,
                    length=length,
                )
            )

        payload = buf[offset:]
        return cls(dtype=dtype, indices=indices, payload=payload)

    def embedding_at(self, index: int) -> np.ndarray:
        """Return multivector at index as float32 for downstream scoring."""
        if index < 0 or index >= len(self.indices):
            raise IndexError(f"Block index {index} out of range for {len(self.indices)} items.")
        meta = self.indices[index]
        start = meta.offset * self.dtype.itemsize
        end = start + meta.length * self.dtype.itemsize
        arr = np.frombuffer(self.payload[start:end], dtype=self.dtype, count=meta.length)
        return arr.reshape(meta.num_modalities, meta.dim).astype(np.float32)

    def iter_indices(self) -> Iterable[MultiVectorBlockIndex]:
        """Expose indices for diagnostics or metadata scans."""
        return iter(self.indices)
