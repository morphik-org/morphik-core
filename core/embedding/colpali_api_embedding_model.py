import base64
import io
import json
import logging
from collections import deque
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from httpx import AsyncClient, HTTPStatusError, Timeout  # replacing httpx.AsyncClient for clarity
from PIL.Image import Image

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)

# Define alias for a multivector: a list of embedding vectors
MultiVector = List[List[float]]


def partition_chunks(chunks: List[Chunk]) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    text_inputs: List[Tuple[int, str]] = []
    image_inputs: List[Tuple[int, str]] = []
    for idx, chunk in enumerate(chunks):
        if chunk.metadata.get("is_image"):
            content = chunk.content
            if content.startswith("data:"):
                content = content.split(",", 1)[1]
            image_inputs.append((idx, content))
        else:
            text_inputs.append((idx, chunk.content))
    return text_inputs, image_inputs


class ColpaliApiEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.settings = get_settings()
        # Use Morphik Embedding API key from settings
        self.api_key = self.settings.MORPHIK_EMBEDDING_API_KEY
        if not self.api_key:
            raise ValueError("MORPHIK_EMBEDDING_API_KEY must be set in settings")
        # Use the configured Morphik Embedding API domain
        domain = self.settings.MORPHIK_EMBEDDING_API_DOMAIN
        self.endpoint = f"{domain.rstrip('/')}/embeddings"
        # Batching is handled at a higher layer (streaming embed+store).
        # Here we issue at most one request per input type per batch.

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[MultiVector]:
        # Normalize to list
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        if not chunks:
            return []

        # Initialize result list with empty multivectors
        results: List[MultiVector] = [[] for _ in chunks]
        text_inputs, image_inputs = partition_chunks(chunks)

        # Image embeddings
        if image_inputs:
            image_results = await self._embed_inputs_with_backoff(list(image_inputs), "image")
            for idx, emb in image_results.items():
                results[idx] = emb

        # Text embeddings
        if text_inputs:
            text_results = await self._embed_inputs_with_backoff(list(text_inputs), "text")
            for idx, emb in text_results.items():
                results[idx] = emb

        return results

    async def embed_for_query(self, text: str) -> MultiVector:
        # Delegate to common API call helper for a single text input
        data = await self.call_api([text], "text")
        if not data:
            raise RuntimeError("No embeddings returned from Morphik Embedding API")
        return data[0]

    async def generate_embeddings(self, content: Union[str, Image]) -> np.ndarray:
        """Generate embeddings for either text or image content.

        Args:
            content: Either a text string or a PIL Image object.

        Returns:
            numpy array of embeddings.
        """
        if isinstance(content, Image):
            # Convert PIL Image to base64
            buffer = io.BytesIO()
            content.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            data = await self.call_api([image_b64], "image")
        else:
            data = await self.call_api([content], "text")

        if not data:
            raise RuntimeError("No embeddings returned from Morphik Embedding API")
        return np.array(data[0])

    async def call_api(self, inputs, input_type) -> List[MultiVector]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input_type": input_type, "inputs": inputs}
        timeout = Timeout(read=6000.0, connect=6000.0, write=6000.0, pool=6000.0)
        async with AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()

            # Load .npz from response content
            npz_data = np.load(io.BytesIO(resp.content))

            # Extract metadata
            count = int(npz_data["count"])
            returned_input_type = str(npz_data["input_type"])

            logger.debug(f"Received {count} embeddings for input_type: {returned_input_type}")

            # Extract embeddings in order
            embeddings = []
            for i in range(count):
                embedding_array = npz_data[f"emb_{i}"]
                # Convert numpy array to list of lists (MultiVector format)
                embeddings.append(embedding_array.tolist())

            return embeddings

    def latest_ingest_metrics(self) -> Dict[str, Any]:
        """API-backed implementation does not expose detailed metrics."""
        return {}

    async def _embed_inputs_with_backoff(
        self, indexed_inputs: List[Tuple[int, str]], input_type: str
    ) -> Dict[int, MultiVector]:
        """
        Embed inputs while dynamically shrinking the batch size to satisfy payload limits.

        Args:
            indexed_inputs: List of (original_index, payload) pairs.
            input_type: Either "text" or "image".

        Returns:
            Dictionary mapping original index to embedding result.
        """
        if not indexed_inputs:
            return {}

        results: Dict[int, MultiVector] = {}
        queue: deque[List[Tuple[int, str]]] = deque([indexed_inputs])

        while queue:
            batch = queue.popleft()
            if not batch:
                continue

            try:
                payload_inputs = [content for _, content in batch]
                data = await self.call_api(payload_inputs, input_type)
            except HTTPStatusError as exc:
                if exc.response.status_code == 413:
                    if len(batch) == 1:
                        size_bytes = self._estimate_payload_size(batch, input_type)
                        logger.error(
                            "ColPali API rejected single %s payload (size≈%s bytes) – cannot downsplit further.",
                            input_type,
                            size_bytes,
                        )
                        raise ValueError(
                            f"{input_type.title()} input exceeds ColPali API payload limit; "
                            "consider downsampling or splitting the source document."
                        ) from exc

                    mid = max(1, len(batch) // 2)
                    logger.warning(
                        "ColPali API returned 413 for %s batch of %s inputs (estimated %s bytes). "
                        "Retrying with %s and %s inputs.",
                        input_type,
                        len(batch),
                        self._estimate_payload_size(batch, input_type),
                        mid,
                        len(batch) - mid,
                    )
                    queue.appendleft(batch[mid:])
                    queue.appendleft(batch[:mid])
                    continue
                raise

            for (idx, _), embedding in zip(batch, data):
                results[idx] = embedding

        return results

    def _estimate_payload_size(self, batch: List[Tuple[int, str]], input_type: str) -> int:
        """
        Estimate the JSON payload size for a batch of inputs.

        Args:
            batch: List of (index, payload) tuples.
            input_type: String descriptor ("text" or "image").

        Returns:
            Integer byte estimate of the serialized payload.
        """
        try:
            payload = {"input_type": input_type, "inputs": [content for _, content in batch]}
            return len(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to estimate payload size: %s", exc)
            return sum(len(content) for _, content in batch)
