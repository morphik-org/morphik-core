import asyncio
import base64
import io
import json
import logging
import time
from collections import deque
from typing import Dict, List, Tuple, Union

import numpy as np
from httpx import AsyncClient, HTTPStatusError, Timeout
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
        self.api_key = self.settings.MORPHIK_EMBEDDING_API_KEY
        if not self.api_key:
            raise ValueError("MORPHIK_EMBEDDING_API_KEY must be set in settings")

        # MORPHIK_EMBEDDING_API_DOMAIN is always a list of endpoints
        if not self.settings.MORPHIK_EMBEDDING_API_DOMAIN:
            raise ValueError("MORPHIK_EMBEDDING_API_DOMAIN must contain at least one endpoint")
        self.endpoints = [f"{ep.rstrip('/')}/embeddings" for ep in self.settings.MORPHIK_EMBEDDING_API_DOMAIN]
        if len(self.endpoints) > 1:
            logger.info(f"ColPali API using {len(self.endpoints)} distributed endpoints")

        # Track endpoint health for failover
        self.healthy_endpoints: set[str] = set(self.endpoints)
        self._endpoint_latencies: dict[str, float] = {}
        self.endpoint = self.endpoints[0]
        self._latest_ingest_metrics: Dict[str, float] = {}

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[MultiVector]:
        ingest_start = time.monotonic()
        # Normalize to list
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        if not chunks:
            self._latest_ingest_metrics = {}
            return []

        # Initialize result list with empty multivectors
        results: List[MultiVector] = [[] for _ in chunks]
        text_inputs, image_inputs = partition_chunks(chunks)

        image_total = 0.0
        text_total = 0.0

        # Use distributed embedding when multiple endpoints available
        if len(self.endpoints) > 1:
            # Image embeddings (distributed)
            if image_inputs:
                image_start = time.monotonic()
                image_results = await self._embed_inputs_distributed(list(image_inputs), "image")
                image_total = time.monotonic() - image_start
                for idx, emb in image_results.items():
                    results[idx] = emb

            # Text embeddings (distributed)
            if text_inputs:
                text_start = time.monotonic()
                text_results = await self._embed_inputs_distributed(list(text_inputs), "text")
                text_total = time.monotonic() - text_start
                for idx, emb in text_results.items():
                    results[idx] = emb
        else:
            # Single endpoint - use existing backoff logic
            if image_inputs:
                image_start = time.monotonic()
                image_results = await self._embed_inputs_with_backoff(list(image_inputs), "image")
                image_total = time.monotonic() - image_start
                for idx, emb in image_results.items():
                    results[idx] = emb

            if text_inputs:
                text_start = time.monotonic()
                text_results = await self._embed_inputs_with_backoff(list(text_inputs), "text")
                text_total = time.monotonic() - text_start
                for idx, emb in text_results.items():
                    results[idx] = emb

        total_time = time.monotonic() - ingest_start
        self._latest_ingest_metrics = {
            "image_total": image_total,
            "text_total": text_total,
            "total": total_time,
            "image_count": float(len(image_inputs)),
            "text_count": float(len(text_inputs)),
            "endpoints": float(len(self.endpoints)),
        }
        return results

    async def _embed_inputs_distributed(
        self, indexed_inputs: List[Tuple[int, str]], input_type: str
    ) -> Dict[int, MultiVector]:
        """
        Distribute inputs across multiple endpoints and embed concurrently.

        Args:
            indexed_inputs: List of (original_index, payload) pairs.
            input_type: Either "text" or "image".

        Returns:
            Dictionary mapping original index to embedding result.
        """
        if not indexed_inputs:
            return {}

        # Use healthy endpoints, fall back to all if none healthy
        endpoints = list(self.healthy_endpoints) if self.healthy_endpoints else self.endpoints
        n_endpoints = len(endpoints)

        if n_endpoints == 1:
            # Single endpoint available - use backoff logic with actual healthy endpoint
            return await self._embed_batch_to_endpoint(endpoints[0], indexed_inputs, input_type)

        # Split inputs across endpoints (interleaved for balance)
        batches: list[list[tuple[int, str]]] = [[] for _ in range(n_endpoints)]
        for i, item in enumerate(indexed_inputs):
            batches[i % n_endpoints].append(item)

        # Filter to only non-empty endpoint-batch pairs
        endpoint_batches = [(ep, batch) for ep, batch in zip(endpoints, batches) if batch]

        logger.debug(f"Distributing {len(indexed_inputs)} {input_type} inputs across {len(endpoint_batches)} endpoints")

        # Call all endpoints concurrently
        tasks = [self._embed_batch_to_endpoint(endpoint, batch, input_type) for endpoint, batch in endpoint_batches]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results, collect failures
        merged: Dict[int, MultiVector] = {}
        failed_inputs: list[tuple[int, str]] = []

        for (endpoint, batch), result in zip(endpoint_batches, results_list):
            if isinstance(result, ValueError):
                # ValueError indicates a data issue (e.g., 413 payload too large), not endpoint failure
                # Re-raise immediately - don't mark endpoint unhealthy
                raise result
            elif isinstance(result, Exception):
                logger.warning(f"Endpoint {endpoint} failed: {result}")
                self.healthy_endpoints.discard(endpoint)
                failed_inputs.extend(batch)
            else:
                merged.update(result)

        # Retry failed inputs on remaining healthy endpoints
        if failed_inputs:
            if self.healthy_endpoints:
                logger.info(
                    f"Retrying {len(failed_inputs)} failed inputs on {len(self.healthy_endpoints)} healthy endpoints"
                )
                retry_results = await self._embed_inputs_distributed(failed_inputs, input_type)
                merged.update(retry_results)
            else:
                # All endpoints failed, reset health and raise
                logger.error("All ColPali endpoints failed, resetting health status")
                self.healthy_endpoints = set(self.endpoints)
                raise RuntimeError(
                    f"All {len(self.endpoints)} ColPali endpoints failed for {len(failed_inputs)} {input_type} inputs"
                )

        return merged

    async def _embed_batch_to_endpoint(
        self, endpoint: str, batch: List[Tuple[int, str]], input_type: str
    ) -> Dict[int, MultiVector]:
        """
        Embed a batch using a specific endpoint with backoff for 413 errors.

        Args:
            endpoint: The API endpoint URL.
            batch: List of (original_index, payload) pairs.
            input_type: Either "text" or "image".

        Returns:
            Dictionary mapping original index to embedding result.
        """
        results: Dict[int, MultiVector] = {}
        queue: deque[List[Tuple[int, str]]] = deque([batch])

        while queue:
            current_batch = queue.popleft()
            if not current_batch:
                continue

            try:
                start = time.monotonic()
                payload_inputs = [content for _, content in current_batch]
                data = await self._call_api_endpoint(endpoint, payload_inputs, input_type)
                elapsed = time.monotonic() - start
                self._endpoint_latencies[endpoint] = elapsed

                for (idx, _), embedding in zip(current_batch, data):
                    results[idx] = embedding

            except HTTPStatusError as exc:
                if exc.response.status_code == 413:
                    if len(current_batch) == 1:
                        size_bytes = self._estimate_payload_size(current_batch, input_type)
                        logger.error(
                            "ColPali API %s rejected single %s payload (size≈%s bytes)",
                            endpoint,
                            input_type,
                            size_bytes,
                        )
                        raise ValueError(
                            f"{input_type.title()} input exceeds ColPali API payload limit; "
                            "consider downsampling or splitting the source document."
                        ) from exc

                    # Split batch and retry on same endpoint
                    mid = max(1, len(current_batch) // 2)
                    logger.warning(
                        "ColPali API %s returned 413 for %s batch of %s inputs. Splitting.",
                        endpoint,
                        input_type,
                        len(current_batch),
                    )
                    queue.appendleft(current_batch[mid:])
                    queue.appendleft(current_batch[:mid])
                    continue
                raise

        return results

    async def _call_api_endpoint(self, endpoint: str, inputs: List[str], input_type: str) -> List[MultiVector]:
        """
        Call a specific ColPali API endpoint.

        Args:
            endpoint: The API endpoint URL.
            inputs: List of input payloads (base64 images or text).
            input_type: Either "text" or "image".

        Returns:
            List of MultiVector embeddings.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input_type": input_type, "inputs": inputs}
        timeout = Timeout(read=600.0, connect=30.0, write=600.0, pool=60.0)

        async with AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()

            # Load .npz from response content
            npz_data = np.load(io.BytesIO(resp.content))

            # Extract metadata
            count = int(npz_data["count"])
            returned_input_type = str(npz_data["input_type"])

            logger.debug(f"Endpoint {endpoint}: received {count} embeddings for input_type: {returned_input_type}")

            # Extract embeddings in order
            embeddings = []
            for i in range(count):
                embedding_array = npz_data[f"emb_{i}"]
                embeddings.append(embedding_array.tolist())

            return embeddings

    async def embed_for_query(self, text: str) -> MultiVector:
        # Use first healthy endpoint for queries (single text, fast)
        endpoint = next(iter(self.healthy_endpoints), self.endpoints[0])
        data = await self._call_api_endpoint(endpoint, [text], "text")
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
        endpoint = next(iter(self.healthy_endpoints), self.endpoints[0])

        if isinstance(content, Image):
            # Convert PIL Image to base64
            buffer = io.BytesIO()
            content.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            data = await self._call_api_endpoint(endpoint, [image_b64], "image")
        else:
            data = await self._call_api_endpoint(endpoint, [content], "text")

        if not data:
            raise RuntimeError("No embeddings returned from Morphik Embedding API")
        return np.array(data[0])

    async def call_api(self, inputs: List[str], input_type: str) -> List[MultiVector]:
        """Backward-compatible API call using first endpoint."""
        return await self._call_api_endpoint(self.endpoint, inputs, input_type)

    def latest_ingest_metrics(self) -> Dict[str, float]:
        """Return endpoint latency metrics from the most recent embed_for_ingestion call."""
        return dict(self._endpoint_latencies)

    def latest_ingest_timing(self) -> Dict[str, float]:
        """Return timing metrics from the most recent embed_for_ingestion call."""
        return dict(self._latest_ingest_metrics) if self._latest_ingest_metrics else {}

    async def _embed_inputs_with_backoff(
        self, indexed_inputs: List[Tuple[int, str]], input_type: str
    ) -> Dict[int, MultiVector]:
        """
        Embed inputs while dynamically shrinking the batch size to satisfy payload limits.
        Used for single-endpoint mode.

        Args:
            indexed_inputs: List of (original_index, payload) pairs.
            input_type: Either "text" or "image".

        Returns:
            Dictionary mapping original index to embedding result.
        """
        if not indexed_inputs:
            return {}

        return await self._embed_batch_to_endpoint(self.endpoint, indexed_inputs, input_type)

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
