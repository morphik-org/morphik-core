from typing import List, Union
from ollama import AsyncClient
from core.embedding.base_embedding_model import BaseEmbeddingModel


class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = AsyncClient(host=base_url)

    async def embed_for_ingestion(
        self, text: Union[str, List[str]]
    ) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]

        embeddings: List[List[float]] = []
        for t in text:
            response = await self.client.embeddings(model=self.model_name, prompt=t)
            embedding = list(response["embedding"])
            embeddings.append(embedding)

        return embeddings

    async def embed_for_query(self, text: str) -> List[float]:
        response = await self.client.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]
