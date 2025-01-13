# hugging face cache implementation.

from core.cache.base_cache import BaseCache
from typing import List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from core.models.completion import CompletionRequest, CompletionResponse


class HuggingFaceCache(BaseCache):
    """Hugging Face Cache implementation for cache-augmented generation"""

    def __init__(
        self,
        cache_path: Union[str, Path],
        model_name: str,
        device: str,
        default_max_new_tokens: int,
        use_fp16: bool = True,
    ):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.default_max_new_tokens = default_max_new_tokens
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.to(self.device)

        # Initialize cache
        self.kv_cache = None
        self.origin_len = None

    def get_kv_cache(self, prompt: str) -> DynamicCache:
        """Build KV cache from prompt"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        cache = DynamicCache()

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return cache

    def clean_up_cache(self, cache: DynamicCache, origin_len: int):
        """Clean up cache by removing appended tokens"""
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

    def generate(
        self, input_ids: torch.Tensor, past_key_values, max_new_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model and cache"""
        device = self.model.model.embed_tokens.weight.device
        origin_len = input_ids.shape[-1]
        input_ids = input_ids.to(device)
        output_ids = input_ids.clone()
        next_token = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens or self.default_max_new_tokens):
                out = self.model(
                    input_ids=next_token, past_key_values=past_key_values, use_cache=True
                )
                logits = out.logits[:, -1, :]
                token = torch.argmax(logits, dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, token], dim=-1)
                past_key_values = out.past_key_values
                next_token = token.to(device)

                if (
                    self.model.config.eos_token_id is not None
                    and token.item() == self.model.config.eos_token_id
                ):
                    break

        return output_ids[:, origin_len:]

    async def ingest(self, docs: List[str]) -> bool:
        """Ingest documents into cache"""
        try:
            # Create system prompt with documents
            system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers.
<|user|>
Context:
{' '.join(docs)}
Question:
""".strip()

            # Build the cache
            self.kv_cache = self.get_kv_cache(system_prompt)
            self.origin_len = self.kv_cache.key_cache[0].shape[-2]
            return True
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            return False

    async def update(self, new_doc: str) -> bool:
        """Update cache with new document"""
        try:
            if self.kv_cache is None:
                return await self.ingest([new_doc])

            # Clean up existing cache
            self.clean_up_cache(self.kv_cache, self.origin_len)

            # Add new document to cache
            input_ids = self.tokenizer(new_doc + "\n", return_tensors="pt").input_ids.to(
                self.device
            )
            _ = self.model(input_ids=input_ids, past_key_values=self.kv_cache, use_cache=True)
            return True
        except Exception as e:
            print(f"Error updating cache: {e}")
            return False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using cache-augmented generation"""
        try:
            if self.kv_cache is None:
                raise ValueError("Cache not initialized. Please ingest documents first.")

            # Clean up cache
            self.clean_up_cache(self.kv_cache, self.origin_len)

            # Generate completion
            input_ids = self.tokenizer(request.prompt + "\n", return_tensors="pt").input_ids.to(
                self.device
            )
            gen_ids = self.generate(input_ids, self.kv_cache)
            response = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            return CompletionResponse(text=response)
        except Exception as e:
            print(f"Error generating completion: {e}")
            return CompletionResponse(text=f"Error: {str(e)}")

    def save_cache(self) -> Path:
        """Save the KV cache to disk"""
        if self.kv_cache is None:
            raise ValueError("No cache to save")

        cache_dir = self.cache_path / "kv_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save key and value caches
        cache_data = {
            "key_cache": self.kv_cache.key_cache,
            "value_cache": self.kv_cache.value_cache,
            "origin_len": self.origin_len,
        }
        cache_path = cache_dir / "cache.pt"
        torch.save(cache_data, cache_path)
        return cache_path

    def load_cache(self, cache_path: Union[str, Path]) -> None:
        """Load KV cache from disk"""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found at {cache_path}")

        cache_data = torch.load(cache_path, map_location=self.device)

        self.kv_cache = DynamicCache()
        self.kv_cache.key_cache = cache_data["key_cache"]
        self.kv_cache.value_cache = cache_data["value_cache"]
        self.origin_len = cache_data["origin_len"]
