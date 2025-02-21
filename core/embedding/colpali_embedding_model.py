import base64
import io
import torch
from typing import List, Union
from PIL import Image
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from core.models.chunk import Chunk
from core.embedding.base_embedding_model import BaseEmbeddingModel


class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map=device,  # Automatically detect and use available device
            attn_implementation="eager" if device == "mps" else "flash_attention_2",
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[torch.Tensor]:
        if isinstance(chunks, Chunk):
            chunks = [chunks]
            
        embeddings_list = []
        with torch.no_grad():
            for chunk in chunks:
                image = Image.open(io.BytesIO(base64.b64decode(chunk.content)))
                processed_image = self.processor.process_images([image]).to(self.model.device)
                embedding = self.model(**processed_image)
                embeddings_list.append(embedding)
            
        return embeddings_list

    async def embed_for_query(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            processed_query = self.processor.process_queries([text]).to(self.model.device)
            embedding = self.model(**processed_query)
            return embedding
