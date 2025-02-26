import base64
import io
from typing import List, Union

import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL.Image import Image, open as open_image

from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk


class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map=device,  # Automatically detect and use available device
            attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).eval()
        self.processor: ColQwen2Processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v1.0"
        )

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> np.ndarray:
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        contents = [open_image(io.BytesIO(base64.b64decode(chunk.content))) if chunk.metadata.get("is_image") else chunk.content for chunk in chunks]
        return [self.generate_embeddings(content) for content in contents]

    async def embed_for_query(self, text: str) -> torch.Tensor:
        return self.generate_embeddings(text)

    def generate_embeddings(self, content: str | Image) -> np.ndarray:
        if isinstance(content, Image):
            processed = self.processor.process_images([content]).to(self.model.device)
        else:
            processed = self.processor.process_queries([content]).to(self.model.device)

        with torch.no_grad():
            embeddings: torch.Tensor = self.model(**processed)

        return embeddings.to(torch.float32).numpy(force=True)[0]
