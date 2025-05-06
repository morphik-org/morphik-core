import logging
from typing import List, Tuple, Union

from httpx import AsyncClient, Timeout  # replacing httpx.AsyncClient for clarity

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

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[MultiVector]:
        # Normalize to list
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        if not chunks:
            return []

        # Initialize result list with empty multivectors
        results: List[MultiVector] = [[] for _ in chunks]
        text_inputs, image_inputs = partition_chunks(chunks)

        # Batch image embeddings if needed
        if image_inputs:
            indices, inputs = zip(*image_inputs)
            data = await self.call_api(inputs, "image")
            for idx, emb in zip(indices, data):
                results[idx] = emb

        # Batch text embeddings if needed
        if text_inputs:
            indices, inputs = zip(*text_inputs)
            data = await self.call_api(inputs, "text")
            for idx, emb in zip(indices, data):
                results[idx] = emb

        return results

    async def embed_for_query(self, text: str) -> MultiVector:
        # Delegate to common API call helper for a single text input
        data = await self.call_api([text], "text")
        if not data:
            raise RuntimeError("No embeddings returned from Morphik Embedding API")
        return data[0]

    async def call_api(self, inputs, input_type) -> List[MultiVector]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input_type": input_type, "inputs": inputs}
        timeout = Timeout(read=6000.0, connect=6000.0, write=6000.0, pool=6000.0)
        async with AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("embeddings", [])
