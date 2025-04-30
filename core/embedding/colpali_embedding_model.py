import base64
import io
import logging
import time
from typing import List, Union

import numpy as np
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL.Image import Image
from PIL.Image import open as open_image

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ColpaliEmbeddingModel with device: {device}")
        start_time = time.time()
        self.model = ColQwen2_5.from_pretrained(
            "tsystems/colqwen2.5-3b-multilingual-v1.0",
            torch_dtype=torch.bfloat16,
            device_map=device,  # Automatically detect and use available device
            attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).eval()
        self.processor: ColQwen2_5_Processor = ColQwen2_5_Processor.from_pretrained(
            "tsystems/colqwen2.5-3b-multilingual-v1.0"
        )
        self.batch_size = 8  # Default batch size, adjust as needed
        self.settings = get_settings()
        self.mode = self.settings.MODE
        logger.info(f"Colpali running in mode: {self.mode}")
        total_init_time = time.time() - start_time
        logger.info(f"Colpali initialization time: {total_init_time:.2f} seconds")

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        job_start_time = time.time()
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        logger.info(f"Processing {len(chunks)} chunks for Colpali embedding in {self.mode} mode")

        if self.mode == "cloud":
            # GPU/Cloud path: Batching enabled
            images: List[Image] = []
            texts: List[str] = []
            sorting_start = time.time()
            for chunk in chunks:
                if chunk.metadata.get("is_image"):
                    try:
                        # Handle data URI format "data:image/png;base64,..."
                        content = chunk.content
                        if content.startswith("data:"):
                            # Extract the base64 part after the comma
                            content = content.split(",", 1)[1]

                        # Now decode the base64 string
                        image_bytes = base64.b64decode(content)
                        image = open_image(io.BytesIO(image_bytes))
                        images.append(image)
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                        # Fall back to using the content as text
                        texts.append(chunk.content)
                else:
                    texts.append(chunk.content)

            sorting_time = time.time() - sorting_start
            logger.info(
                f"Chunk sorting took {sorting_time:.2f}s - Found {len(images)} images and {len(texts)} text chunks"
            )

            # Process in batches
            embeddings = []

            # Process image batches
            if images:
                img_start = time.time()
                for i in range(0, len(images), self.batch_size):
                    batch = images[i : i + self.batch_size]
                    logger.debug(
                        f"Processing image batch {i//self.batch_size + 1}/"
                        f"{(len(images)-1)//self.batch_size + 1} with {len(batch)} images"
                    )
                    batch_start = time.time()
                    batch_embeddings = await self.generate_embeddings_batch_images(batch)
                    embeddings.extend(batch_embeddings)
                    batch_time = time.time() - batch_start
                    logger.debug(
                        f"Image batch {i//self.batch_size + 1} processing took {batch_time:.2f}s "
                        f"({batch_time/len(batch):.2f}s per image)"
                    )
                img_time = time.time() - img_start
                logger.info(f"All image embedding took {img_time:.2f}s ({img_time/len(images):.2f}s per image)")

            # Process text batches
            if texts:
                text_start = time.time()
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    logger.debug(
                        f"Processing text batch {i//self.batch_size + 1}/"
                        f"{(len(texts)-1)//self.batch_size + 1} with {len(batch)} texts"
                    )
                    batch_start = time.time()
                    batch_embeddings = await self.generate_embeddings_batch_texts(batch)
                    embeddings.extend(batch_embeddings)
                    batch_time = time.time() - batch_start
                    logger.debug(
                        f"Text batch {i//self.batch_size + 1} processing took {batch_time:.2f}s "
                        f"({batch_time/len(batch):.2f}s per text)"
                    )
                text_time = time.time() - text_start
                logger.info(f"All text embedding took {text_time:.2f}s ({text_time/len(texts):.2f}s per text)")

            # Note: The order of embeddings might not match the original chunk order
            # if images and texts were mixed. If order preservation is critical,
            # a different approach (e.g., processing all as single items or re-sorting)
            # would be needed.
            # For now, we assume the caller handles potential order changes.

        else:
            # Self-hosted/CPU path: Simple per-item processing
            contents = []
            for chunk in chunks:
                if chunk.metadata.get("is_image"):
                    try:
                        # Handle data URI format "data:image/png;base64,..."
                        content = chunk.content
                        if content.startswith("data:"):
                            # Extract the base64 part after the comma
                            content = content.split(",", 1)[1]

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
            embeddings = [await self.generate_embeddings(content) for content in contents]

        total_time = time.time() - job_start_time
        logger.info(
            f"Total Colpali embed_for_ingestion took {total_time:.2f}s for {len(chunks)} chunks "
            f"({total_time/len(chunks) if chunks else 0:.2f}s per chunk)"
        )
        return embeddings

    async def embed_for_query(self, text: str) -> torch.Tensor:
        start_time = time.time()
        result = await self.generate_embeddings(text)
        elapsed = time.time() - start_time
        logger.info(f"Colpali query embedding took {elapsed:.2f}s")
        return result

    async def generate_embeddings(self, content: Union[str, Image]) -> np.ndarray:
        start_time = time.time()
        content_type = "image" if isinstance(content, Image) else "text"
        process_start = time.time()
        if isinstance(content, Image):
            processed = self.processor.process_images([content]).to(self.model.device)
        else:
            processed = self.processor.process_queries([content]).to(self.model.device)

        process_time = time.time() - process_start

        model_start = time.time()

        with torch.no_grad():
            embeddings: torch.Tensor = self.model(**processed)

        model_time = time.time() - model_start

        convert_start = time.time()

        result = embeddings.to(torch.float32).numpy(force=True)[0]

        convert_time = time.time() - convert_start

        total_time = time.time() - start_time
        logger.debug(
            f"Generate embeddings ({content_type}): process={process_time:.2f}s, model={model_time:.2f}s, "
            f"convert={convert_time:.2f}s, total={total_time:.2f}s"
        )
        return result

    # ---- Batch processing methods (only used in 'cloud' mode) ----

    async def generate_embeddings_batch_images(self, images: List[Image]) -> List[np.ndarray]:
        batch_start_time = time.time()
        process_start = time.time()
        processed_images = self.processor.process_images(images).to(self.model.device)
        process_time = time.time() - process_start

        model_start = time.time()
        with torch.no_grad():
            image_embeddings = self.model(**processed_images)
        model_time = time.time() - model_start

        convert_start = time.time()
        image_embeddings_np = image_embeddings.to(torch.float32).numpy(force=True)
        result = [emb for emb in image_embeddings_np]
        convert_time = time.time() - convert_start

        total_batch_time = time.time() - batch_start_time
        logger.debug(
            f"Batch images ({len(images)}): process={process_time:.2f}s, model={model_time:.2f}s, "
            f"convert={convert_time:.2f}s, total={total_batch_time:.2f}s ({total_batch_time/len(images):.3f}s/image)"
        )
        return result

    async def generate_embeddings_batch_texts(self, texts: List[str]) -> List[np.ndarray]:
        batch_start_time = time.time()
        process_start = time.time()
        processed_texts = self.processor.process_queries(texts).to(self.model.device)
        process_time = time.time() - process_start

        model_start = time.time()
        with torch.no_grad():
            text_embeddings = self.model(**processed_texts)
        model_time = time.time() - model_start

        convert_start = time.time()
        text_embeddings_np = text_embeddings.to(torch.float32).numpy(force=True)
        result = [emb for emb in text_embeddings_np]
        convert_time = time.time() - convert_start

        total_batch_time = time.time() - batch_start_time
        logger.debug(
            f"Batch texts ({len(texts)}): process={process_time:.2f}s, model={model_time:.2f}s, "
            f"convert={convert_time:.2f}s, total={total_batch_time:.2f}s ({total_batch_time/len(texts):.3f}s/text)"
        )
        return result
