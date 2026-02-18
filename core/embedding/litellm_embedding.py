import logging
from typing import List, Union

import litellm

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)
PGVECTOR_MAX_DIMENSIONS = 2000  # Maximum dimensions for pgvector


class LiteLLMEmbeddingModel(BaseEmbeddingModel):
    """
    LiteLLM embedding model implementation that provides unified access to various embedding providers.
    Uses registered models from the config file.
    """

    def __init__(self, model_key: str):
        """
        Initialize LiteLLM embedding model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_settings()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if not hasattr(settings, "REGISTERED_MODELS") or model_key not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.REGISTERED_MODELS[model_key]
        self.dimensions = min(settings.VECTOR_DIMENSIONS, 2000)
        model_name_lower = str(self.model_config.get("model_name", "")).lower()
        api_base_lower = str(self.model_config.get("api_base", "")).lower()
        self._is_local_provider = (
            any(
                indicator in api_base_lower
                for indicator in ("localhost", "127.0.0.1", "host.docker.internal", ":11434")
            )
            or "ollama" in model_name_lower
        )
        logger.info(f"Initialized LiteLLM embedding model with model_key={model_key}, config={self.model_config}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents using LiteLLM.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors (one per document)
        """
        if not texts:
            return []

        try:
            model_params = {"model": self.model_config["model_name"]}
            if self.model_config["model_name"] in [
                "text-embedding-3-large",
                "azure/text-embedding-3-large",
            ]:
                model_params["dimensions"] = PGVECTOR_MAX_DIMENSIONS

            # Add all model-specific parameters from the config
            for key, value in self.model_config.items():
                if key != "model_name":  # Skip as we've already handled it
                    model_params[key] = value

            # Ensure providers that don't require real API keys (e.g., Ollama, local OpenAI-compatible backends)
            # still pass a dummy key to LiteLLM/OpenAI client to avoid AuthenticationError.
            looks_like_local_provider = self._is_local_provider
            if looks_like_local_provider and "api_key" not in model_params:
                # Use a harmless placeholder; some LiteLLM providers demand a key even if backend ignores it
                model_params["api_key"] = get_settings().LITELLM_DUMMY_API_KEY

            # Call LiteLLM
            response = await litellm.aembedding(input=texts, **model_params)

            embeddings = [data["embedding"] for data in response.data]

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embeddings[0])}, expected {self.dimensions}. "
                    f"Please update your VECTOR_DIMENSIONS setting to match the actual dimension."
                )

            return embeddings
        except Exception as e:
            logger.error(
                f"Error generating embeddings with LiteLLM: {e}. "
                f"Context: model_key={self.model_key}, model_name={self.model_config.get('model_name')}, "
                f"api_base={self.model_config.get('api_base', 'default')}, num_texts={len(texts)}"
            )
            raise

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query using LiteLLM.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_documents([text])
        if not result:
            # In case of error, return zero vector
            return [0.0] * self.dimensions
        return result[0]

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """
        Generate embeddings for chunks to be ingested into the vector store.

        Args:
            chunks: Single chunk or list of chunks to embed

        Returns:
            List of embedding vectors (one per chunk)
        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        texts = [chunk.content for chunk in chunks]
        # Batch embedding to respect token limits
        batch_size = self._determine_batch_size()
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = await self.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    def _determine_batch_size(self) -> int:
        settings = get_settings()
        configured_batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", None)
        if isinstance(configured_batch_size, int) and configured_batch_size > 0:
            return configured_batch_size
        return 5 if self._is_local_provider else 100

    async def embed_for_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embed_query(text)
