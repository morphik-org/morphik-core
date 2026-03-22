"""MiniMax embedding model using the native MiniMax Embeddings API.

MiniMax's embo-01 model uses a different request/response format from OpenAI:
- Request: {"model": "embo-01", "texts": [...], "type": "db"|"query"}
- Response: {"vectors": [[...]], "total_tokens": N}

The ``type`` parameter distinguishes between embedding documents for storage ("db")
and embedding queries for search ("query"), which improves retrieval quality.
"""

import logging
import os
from typing import List, Union

import httpx

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)

MINIMAX_API_BASE = "https://api.minimax.io/v1"


class MiniMaxEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using the native MiniMax embedding API (embo-01)."""

    def __init__(self, model_key: str):
        settings = get_settings()
        self.model_key = model_key

        if not hasattr(settings, "REGISTERED_MODELS") or model_key not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.REGISTERED_MODELS[model_key]
        self.model_name = self.model_config.get("model_name", "embo-01")
        self.api_base = self.model_config.get("api_base", MINIMAX_API_BASE).rstrip("/")
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.dimensions = min(settings.VECTOR_DIMENSIONS, 2000)

        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY environment variable is not set")

        logger.info(
            "Initialized MiniMax embedding model: model_key=%s, model=%s, dimensions=%d",
            model_key,
            self.model_name,
            self.dimensions,
        )

    async def _embed(self, texts: List[str], embed_type: str) -> List[List[float]]:
        """Call the MiniMax embedding API.

        Args:
            texts: Texts to embed.
            embed_type: "db" for document storage, "query" for search queries.

        Returns:
            List of embedding vectors.
        """
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "texts": texts,
            "type": embed_type,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        if "vectors" not in data:
            base_resp = data.get("base_resp", {})
            raise ValueError(
                f"MiniMax embedding API error: "
                f"status_code={base_resp.get('status_code')}, "
                f"status_msg={base_resp.get('status_msg')}"
            )

        vectors = data["vectors"]

        # Validate dimensions
        if vectors and len(vectors[0]) != self.dimensions:
            logger.warning(
                "Embedding dimension mismatch: got %d, expected %d. "
                "Update VECTOR_DIMENSIONS in morphik.toml to match.",
                len(vectors[0]),
                self.dimensions,
            )

        return vectors

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """Embed chunks for storage using type='db'."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        texts = [chunk.content for chunk in chunks]
        if not texts:
            return []

        # Batch to respect rate limits
        batch_size = 50
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed(batch, embed_type="db")
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_for_query(self, text: str) -> List[float]:
        """Embed a single query for search using type='query'."""
        vectors = await self._embed([text], embed_type="query")
        if not vectors:
            return [0.0] * self.dimensions
        return vectors[0]
