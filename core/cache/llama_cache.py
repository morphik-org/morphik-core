from pathlib import Path
from typing import List, Union
import pickle
from llama_cpp import CreateChatCompletionResponse, Llama, LlamaState

from core.models.completion import CompletionRequest, CompletionResponse
from core.cache.base_cache import BaseCache


class LlamaCache(BaseCache):
    """Cache-augmented generation implementation using llama.cpp."""

    def __init__(
        self,
        model_path: str,
        filename: str = "*q8_0.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ):
        """Initialize the Llama model and cache.

        Args:
            model_path: Path to the Llama model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
        """
        self.model = Llama.from_pretrained(
            repo_id=model_path,
            filename=filename,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.kv_cache: LlamaState = None
        self.cached_tokens = 0

    async def ingest(self, docs: List[str]) -> bool:
        """Ingest documents into the KV cache.

        Processes documents and stores their KV cache state.
        """
        try:
            # Concatenate docs with separator
            combined_text = "\n---\n".join(docs)

            # Generate and save initial KV cache state
            self.model.reset()
            tokens = self.model.tokenize(combined_text.encode())
            self.cached_tokens = len(tokens)

            # Create initial embedding and get KV cache
            self.model.eval(tokens)
            self.kv_cache = self.model.save_state()

            return True

        except Exception as e:
            print(f"Ingestion failed: {e}")
            return False

    async def update(self, new_doc: str) -> bool:
        """Update cache with a new document.

        Appends new document while maintaining KV cache state.
        """
        try:
            if self.kv_cache is None:
                return await self.ingest([new_doc])

            # Load existing cache state
            self.model.reset()
            self.model.load_state(self.kv_cache)

            # Process new document
            new_tokens = self.model.tokenize(f"\n---\n{new_doc}".encode())
            self.model.eval(new_tokens)

            # Update cache state
            self.kv_cache = self.model.save_state()
            self.cached_tokens += len(new_tokens)

            return True

        except Exception as e:
            print(f"Update failed: {e}")
            return False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using cached context.

        Uses the cached KV state for efficient generation.
        """
        try:
            self.model.reset()
            self.model.load_state(self.kv_cache)

            # self.model.create_chat_completion
            # Combine context chunks and query into prompt
            # prompt = "\n".join(request.context_chunks + [request.query])

            # Format prompt as chat message
            messages = [
                # {"role": "system", "content": "\n".join(request.context_chunks)},
                {"role": "user", "content": request.query}
            ]

            # Generate completion using chat completion API
            response: CreateChatCompletionResponse = self.model.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.9,
            )

            # Extract completion text from response
            completion_text = response["choices"][0]["message"]["content"]

            return CompletionResponse(
                completion=completion_text,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        except Exception as e:
            print(f"Completion failed: {e}")
            raise

    def save_cache(self) -> Path:
        """Save the KV cache state to disk."""
        if self.kv_cache is None:
            raise ValueError("No cache state to save")

        save_path = Path("llama_cache.pkl")
        cache_data = {"kv_cache": self.kv_cache, "cached_tokens": self.cached_tokens}

        with open(save_path, "wb") as f:
            pickle.dump(cache_data, f)

        return save_path

    def load_cache(self, cache_path: Union[str, Path]) -> None:
        """Load a previously saved KV cache state."""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        self.kv_cache = cache_data["kv_cache"]
        self.cached_tokens = cache_data["cached_tokens"]
