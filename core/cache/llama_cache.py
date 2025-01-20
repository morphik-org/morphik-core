import pickle
from core.cache.base_cache import BaseCache
from typing import Dict, Any, List
from core.models.completion import CompletionResponse
from core.models.documents import Document
from llama_cpp import Llama

INITIAL_SYSTEM_PROMPT = """[INST]<<SYS>>
You are a helpful assistant who can answer questions with the help of the provided documents.
Provided documents: {documents}
<</SYS>>[/INST]""".strip()

ADD_DOC_SYSTEM_PROMPT = """[INST]
I'm adding some additional documents for your reference:
{documents}

Please incorporate this new information along with what you already know from previous documents.
[/INST]""".strip()


class LlamaCache(BaseCache):
    def __init__(
        self,
        name: str,
        model: str,
        gguf_file: str,
        filters: Dict[str, Any],
        docs: List[Document],
        **kwargs
    ):
        # cache related
        self.name = name
        self.model = model
        self.filters = filters
        self.docs = docs

        # llama specific
        self.gguf_file = gguf_file
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)

        # late init (when we call _initialize)
        self.llama = None
        self.state = None
        self.cached_tokens = 0

        self._initialize(model, gguf_file, docs)

    def _initialize(self, model: str, gguf_file: str, docs: List[Document]) -> None:
        self.llama = Llama.from_pretrained(
            repo_id=model,
            filename=gguf_file,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

        documents = "\n".join(doc.system_metadata.get("content", "") for doc in docs)
        system_prompt = INITIAL_SYSTEM_PROMPT.format(documents=documents)

        # Get number of tokens in system prompt
        tokens = self.llama.tokenize(system_prompt.encode())
        self.cached_tokens = len(tokens)

        # Evaluate prompt to build KV cache
        self.llama.eval(tokens)
        self.state = self.llama.save_state()

    def add_docs(self, docs: List[Document]) -> bool:
        documents = "\n".join(doc.system_metadata.get("content", "") for doc in docs)
        system_prompt = ADD_DOC_SYSTEM_PROMPT.format(documents=documents)

        # Get number of new tokens
        new_tokens = self.llama.tokenize(system_prompt.encode())
        self.cached_tokens += len(new_tokens)

        # Evaluate prompt to update KV cache
        self.llama.eval(new_tokens)
        self.state = self.llama.save_state()

    def query(self, query: str) -> CompletionResponse:
        # Reset the context to our cached system prompt state
        self.llama.reset()
        self.llama.load_state(self.state)

        # Tokenize the query first
        tokens = self.llama.tokenize(query.encode())

        # Generate completion starting from cached state
        completion = self.llama.create_completion(
            tokens,
            max_tokens=None,  # Let model decide when to stop
            echo=False,  # Don't include prompt in output
        )

        return CompletionResponse(
            text=completion["choices"][0]["text"],
            usage={
                "prompt_tokens": self.cached_tokens + len(tokens),
                "completion_tokens": completion["usage"]["completion_tokens"],
                "total_tokens": completion["usage"]["total_tokens"],
            },
        )

    @property
    def saveable_state(self) -> bytes:
        return pickle.dumps(self.state)

    @classmethod
    def from_bytes(
        cls, name: str, cache_bytes: bytes, metadata: Dict[str, Any], **kwargs
    ) -> "LlamaCache":
        """Load a cache from its serialized state.

        Args:
            name: Name of the cache
            cache_bytes: Pickled state bytes
            metadata: Cache metadata including model info
            **kwargs: Additional arguments

        Returns:
            LlamaCache: Loaded cache instance
        """
        # Create new instance with metadata
        cache = cls(
            name=name,
            model=metadata["model"],
            gguf_file=metadata["model_file"],
            filters=metadata["filters"],
            docs=[Document(**doc) for doc in metadata["docs"]],
        )

        # Load the saved state
        cache.state = pickle.loads(cache_bytes)
        cache.llama.load_state(cache.state)

        return cache
