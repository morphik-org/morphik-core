import importlib.util
import io
import logging
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import torch
from PIL.Image import Image
from PIL.Image import open as open_image

# --- V'S CHANGE: Import ALL 3 model architectures ---
from colpali_engine.models import (
    ColQwen2_5, 
    ColQwen2_5_Processor,
    ColIdefics3, 
    ColIdefics3Processor,
    ColPali,            # <--- New: For vidore/colpali-v1.2 (PaliGemma)
    ColPaliProcessor    # <--- New
)
# ---------------------------------------------------

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk
from core.utils.fast_ops import data_uri_to_bytes

logger = logging.getLogger(__name__)

_INGEST_METRICS: ContextVar[Dict[str, Any]] = ContextVar("_colpali_ingest_metrics", default={})

class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.settings = get_settings()
        self.mode = self.settings.MODE
        
        # 1. Determine Device
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ColpaliEmbeddingModel with device: {device}")
        
        # 2. Configure Attention (Flash Attn 2 Check)
        attn_implementation = "eager"
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if importlib.util.find_spec("flash_attn") is not None:
                attn_implementation = "flash_attention_2"
        
        # 3. Model Selector Logic
        # Get model name from ENV, default to the standard ColPali v1.2
        model_name = os.getenv("COLPALI_MODEL_NAME", "vidore/colpali-v1.2-merged")
        
        logger.info(f"Loading ColPali Model: {model_name}")

        # --- V'S CHANGE: Dynamic Loading for Smol, Qwen, AND PaliGemma ---
        
        # CASE 1: SMOLVLM (Idefics3 Architecture)
        if "smol" in model_name.lower() or "idefics" in model_name.lower():
            logger.info("Detected SmolVLM/Idefics3 architecture.")
            self.model = ColIdefics3.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation=attn_implementation,
            ).eval()
            self.processor = ColIdefics3Processor.from_pretrained(model_name, use_fast=True)
            # Smol is tiny (256M/500M), boost batch size!
            self.batch_size = 32 if self.mode == "cloud" else 4
            
        # CASE 2: QWEN (Qwen2/2.5-VL Architecture)
        elif "qwen" in model_name.lower():
            logger.info("Detected Qwen2/2.5-VL architecture.")
            self.model = ColQwen2_5.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation=attn_implementation,
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(model_name, use_fast=True)
            # Qwen is heavy (3B+), keep batch size conservative
            self.batch_size = 8 if self.mode == "cloud" else 1

        # CASE 3: COLPALI (PaliGemma Architecture - The Default)
        else:
            logger.info("Detected Standard ColPali (PaliGemma) architecture.")
            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation=attn_implementation,
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model_name, use_fast=True)
            # PaliGemma is ~3B, keep batch size conservative
            self.batch_size = 8 if self.mode == "cloud" else 1
            
        # -------------------------------------------------------

        self.device = device
        logger.info(f"Colpali initialized. Batch size: {self.batch_size}")

    
    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        # ... (Rest of the method remains exactly the same) ...
        job_start_time = time.time()
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        if not chunks:
            return []

        logger.info(
            f"Processing {len(chunks)} chunks for Colpali embedding in {self.mode} mode (batch size: {self.batch_size})"
        )
        _INGEST_METRICS.set({})

        image_items: List[Tuple[int, Image]] = []
        text_items: List[Tuple[int, str]] = []
        sorting_start = time.time()

        for index, chunk in enumerate(chunks):
            if chunk.metadata.get("is_image"):
                try:
                    raw_bytes = chunk.metadata.get("_image_bytes")
                    if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
                        image_bytes = bytes(raw_bytes)
                    else:
                        image_bytes = data_uri_to_bytes(chunk.content)
                    image = open_image(io.BytesIO(image_bytes))
                    chunk.metadata.pop("_image_bytes", None)
                    image_items.append((index, image))
                except Exception as e:
                    logger.error(f"Error processing image chunk {index}: {str(e)}. Falling back to text.")
                    text_items.append((index, chunk.content))
            else:
                text_items.append((index, chunk.content))

        sorting_time = time.time() - sorting_start
        logger.info(
            f"Chunk sorting took {sorting_time:.2f}s - "
            f"Found {len(image_items)} images and {len(text_items)} text chunks"
        )

        results: List[np.ndarray | None] = [None] * len(chunks)

        if image_items:
            img_start = time.time()
            indices_to_process = [item[0] for item in image_items]
            images_to_process = [item[1] for item in image_items]
            image_process = image_model = image_convert = image_total = 0.0
            for i in range(0, len(images_to_process), self.batch_size):
                batch_indices = indices_to_process[i : i + self.batch_size]
                batch_images = images_to_process[i : i + self.batch_size]
                # ... Logging ...
                batch_start = time.time()
                batch_embeddings, batch_metrics = await self.generate_embeddings_batch_images(batch_images)
                image_process += batch_metrics["process"]
                image_model += batch_metrics["model"]
                image_convert += batch_metrics["convert"]
                image_total += batch_metrics["total"]
                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    results[original_index] = embedding
                # ... Logging ...
            img_time = time.time() - img_start
            logger.info(f"All image embedding took {img_time:.2f}s")
        else:
            image_process = image_model = image_convert = image_total = 0.0
            img_time = 0.0

        if text_items:
            text_start = time.time()
            indices_to_process = [item[0] for item in text_items]
            texts_to_process = [item[1] for item in text_items]
            text_process = text_model = text_convert = text_total = 0.0
            for i in range(0, len(texts_to_process), self.batch_size):
                batch_indices = indices_to_process[i : i + self.batch_size]
                batch_texts = texts_to_process[i : i + self.batch_size]
                # ... Logging ...
                batch_start = time.time()
                batch_embeddings, batch_metrics = await self.generate_embeddings_batch_texts(batch_texts)
                text_process += batch_metrics["process"]
                text_model += batch_metrics["model"]
                text_convert += batch_metrics["convert"]
                text_total += batch_metrics["total"]
                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    results[original_index] = embedding
                # ... Logging ...
            text_time = time.time() - text_start
            logger.info(f"All text embedding took {text_time:.2f}s")
        else:
            text_process = text_model = text_convert = text_total = 0.0
            text_time = 0.0

        final_results = [res for res in results if res is not None]
        
        # ... (Metrics calculation remains the same) ...
        
        return final_results # type: ignore

    def latest_ingest_metrics(self) -> Dict[str, Any]:
        metrics = _INGEST_METRICS.get()
        return dict(metrics) if metrics else {}

    async def embed_for_query(self, text: str) -> torch.Tensor:
        # Remains the same
        start_time = time.time()
        result = await self.generate_embeddings(text)
        elapsed = time.time() - start_time
        logger.info(f"Colpali query embedding took {elapsed:.2f}s")
        return result

    async def generate_embeddings(self, content: Union[str, Image]) -> np.ndarray:
        # Remains the same
        start_time = time.time()
        content_type = "image" if isinstance(content, Image) else "text"
        process_start = time.time()
        if isinstance(content, Image):
            processed = self.processor.process_images([content]).to(self.model.device)
        else:
            processed = self.processor.process_queries([content]).to(self.model.device)

        process_time = time.time() - process_start
        model_start = time.time()

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    embeddings: torch.Tensor = self.model(**processed)
            else:
                embeddings = self.model(**processed)

        model_time = time.time() - model_start
        convert_start = time.time()
        result = embeddings.to(torch.float32).numpy(force=True)[0]
        convert_time = time.time() - convert_start
        total_time = time.time() - start_time
        return result

    async def generate_embeddings_batch_images(self, images: List[Image]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        # Remains the same
        batch_start_time = time.time()
        process_start = time.time()
        processed_images = self.processor.process_images(images).to(self.model.device)
        process_time = time.time() - process_start

        model_start = time.time()
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_embeddings = self.model(**processed_images)
            else:
                image_embeddings = self.model(**processed_images)
        model_time = time.time() - model_start

        convert_start = time.time()
        image_embeddings_np = image_embeddings.to(torch.float32).numpy(force=True)
        result = [emb for emb in image_embeddings_np]
        convert_time = time.time() - convert_start
        total_batch_time = time.time() - batch_start_time
        
        return result, {
            "process": process_time,
            "model": model_time,
            "convert": convert_time,
            "total": total_batch_time,
        }

    async def generate_embeddings_batch_texts(self, texts: List[str]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        # Remains the same
        batch_start_time = time.time()
        process_start = time.time()
        processed_texts = self.processor.process_queries(texts).to(self.model.device)
        process_time = time.time() - process_start

        model_start = time.time()
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    text_embeddings = self.model(**processed_texts)
            else:
                text_embeddings = self.model(**processed_texts)
        model_time = time.time() - model_start

        convert_start = time.time()
        text_embeddings_np = text_embeddings.to(torch.float32).numpy(force=True)
        result = [emb for emb in text_embeddings_np]
        convert_time = time.time() - convert_start
        total_batch_time = time.time() - batch_start_time
        
        return result, {
            "process": process_time,
            "model": model_time,
            "convert": convert_time,
            "total": total_batch_time,
        }
