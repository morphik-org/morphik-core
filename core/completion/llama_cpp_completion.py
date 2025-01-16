from base_completion import BaseCompletion
from llama_cpp import Llama


class LlamaCppCompletion(BaseCompletion):
    def __init__(self, model_name: str, **kwargs):
        self.model = Llama.from_pretrained(repo_id=model_name, filename="*q8_0.gguf", verbose=False)
