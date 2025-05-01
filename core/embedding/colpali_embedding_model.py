import base64
import io
import logging
import time
from typing import List, Tuple, Union

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
        self.settings = get_settings()
        self.mode = self.settings.MODE
        # Set batch size based on mode
        self.batch_size = 8 if self.mode == "cloud" else 1
        logger.info(f"Colpali running in mode: {self.mode} with batch size: {self.batch_size}")
        total_init_time = time.time() - start_time
        logger.info(f"Colpali initialization time: {total_init_time:.2f} seconds")

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        job_start_time = time.time()
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        if not chunks:
            return []

        logger.info(
            f"Processing {len(chunks)} chunks for Colpali embedding in {self.mode} mode (batch size: {self.batch_size})"
        )

        image_items: List[Tuple[int, Image]] = []
        text_items: List[Tuple[int, str]] = []
        sorting_start = time.time()

        for index, chunk in enumerate(chunks):
            if chunk.metadata.get("is_image"):
                try:
                    content = chunk.content
                    if content.startswith("data:"):
                        content = content.split(",", 1)[1]
                    image_bytes = base64.b64decode(content)
                    image = open_image(io.BytesIO(image_bytes))
                    image_items.append((index, image))
                except Exception as e:
                    logger.error(f"Error processing image chunk {index}: {str(e)}. Falling back to text.")
                    text_items.append((index, chunk.content))  # Fallback: treat content as text
            else:
                text_items.append((index, chunk.content))

        sorting_time = time.time() - sorting_start
        logger.info(
            f"Chunk sorting took {sorting_time:.2f}s - "
            f"Found {len(image_items)} images and {len(text_items)} text chunks"
        )

        # Initialize results array to preserve order
        results: List[np.ndarray | None] = [None] * len(chunks)

        # Process image batches
        if image_items:
            img_start = time.time()
            indices_to_process = [item[0] for item in image_items]
            images_to_process = [item[1] for item in image_items]
            for i in range(0, len(images_to_process), self.batch_size):
                batch_indices = indices_to_process[i : i + self.batch_size]
                batch_images = images_to_process[i : i + self.batch_size]
                logger.debug(
                    f"Processing image batch {i//self.batch_size + 1}/"
                    f"{(len(images_to_process)-1)//self.batch_size + 1} with {len(batch_images)} images"
                )
                batch_start = time.time()
                batch_embeddings = await self.generate_embeddings_batch_images(batch_images)
                # Place embeddings in the correct position in results
                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    results[original_index] = embedding
                batch_time = time.time() - batch_start
                logger.debug(
                    f"Image batch {i//self.batch_size + 1} processing took {batch_time:.2f}s "
                    f"({batch_time/len(batch_images):.2f}s per image)"
                )
            img_time = time.time() - img_start
            logger.info(f"All image embedding took {img_time:.2f}s ({img_time/len(images_to_process):.2f}s per image)")

        # Process text batches
        if text_items:
            text_start = time.time()
            indices_to_process = [item[0] for item in text_items]
            texts_to_process = [item[1] for item in text_items]
            for i in range(0, len(texts_to_process), self.batch_size):
                batch_indices = indices_to_process[i : i + self.batch_size]
                batch_texts = texts_to_process[i : i + self.batch_size]
                logger.debug(
                    f"Processing text batch {i//self.batch_size + 1}/"
                    f"{(len(texts_to_process)-1)//self.batch_size + 1} with {len(batch_texts)} texts"
                )
                batch_start = time.time()
                batch_embeddings = await self.generate_embeddings_batch_texts(batch_texts)
                # Place embeddings in the correct position in results
                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    results[original_index] = embedding
                batch_time = time.time() - batch_start
                logger.debug(
                    f"Text batch {i//self.batch_size + 1} processing took {batch_time:.2f}s "
                    f"({batch_time/len(batch_texts):.2f}s per text)"
                )
            text_time = time.time() - text_start
            logger.info(f"All text embedding took {text_time:.2f}s ({text_time/len(texts_to_process):.2f}s per text)")

        # Ensure all chunks were processed (handle potential None entries if errors occurred,
        # though unlikely with fallback)
        final_results = [res for res in results if res is not None]
        if len(final_results) != len(chunks):
            logger.warning(
                f"Number of embeddings ({len(final_results)}) does not match number of chunks "
                f"({len(chunks)}). Some chunks might have failed."
            )
            # Fill potential gaps if necessary, though the current logic should cover all chunks
            # For safety, let's reconstruct based on successfully processed indices, though it shouldn't be needed
            processed_indices = {idx for idx, _ in image_items} | {idx for idx, _ in text_items}
            if len(processed_indices) != len(chunks):
                logger.error("Mismatch in processed indices vs original chunks count. This indicates a logic error.")
            # Assuming results contains embeddings at correct original indices, filter out Nones
            final_results = [results[i] for i in range(len(chunks)) if results[i] is not None]

        total_time = time.time() - job_start_time
        logger.info(
            f"Total Colpali embed_for_ingestion took {total_time:.2f}s for {len(chunks)} chunks "
            f"({total_time/len(chunks) if chunks else 0:.2f}s per chunk)"
        )
        # Cast is safe because we filter out Nones, though Nones shouldn't occur with the fallback logic
        return final_results  # type: ignore

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
