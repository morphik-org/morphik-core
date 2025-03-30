from typing import List, Union
from openai import OpenAI

from core.models.chunk import Chunk
from core.embedding.base_embedding_model import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small", base_url: str = None):
        from core.config import get_settings

        settings = get_settings()
        # Use base_url param first, then global setting, then default
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif hasattr(settings, "OPENAI_BASE_URL") and settings.OPENAI_BASE_URL:
            self.client = OpenAI(api_key=api_key, base_url=settings.OPENAI_BASE_URL)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        chunks = [chunks] if isinstance(chunks, Chunk) else chunks
        text = [c.content for c in chunks]
        response = self.client.embeddings.create(model=self.model_name, input=text)

        return [item.embedding for item in response.data]

    async def embed_for_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)

        return response.data[0].embedding
