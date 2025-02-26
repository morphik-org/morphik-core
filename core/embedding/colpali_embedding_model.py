import base64
import io
from typing import List, Union

import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL.Image import Image, open as open_image

from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk
import logging

logger = logging.getLogger(__name__)

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

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        contents = []
        for chunk in chunks:
            if chunk.metadata.get("is_image"):
                try:
                    # Handle data URI format "data:image/png;base64,..."
                    content = chunk.content
                    if content.startswith('data:'):
                        # Extract the base64 part after the comma
                        content = content.split(',', 1)[1]
                    
                    # Now decode the base64 string
                    image_bytes = base64.b64decode(content)
                    image = open_image(io.BytesIO(image_bytes))
                    contents.append(image)
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    # Fall back to using the content as text
                    contents.append(chunk.content)
            else:
                contents.append(chunk.content)
        
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
