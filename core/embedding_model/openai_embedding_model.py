from typing import List, Union
from openai import OpenAI
from .base_embedding_model import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    async def embed_for_ingestion(
        self, text: Union[str, List[str]]
    ) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=text)

        return [item.embedding for item in response.data]

    async def embed_for_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)

        return response.data[0].embedding
