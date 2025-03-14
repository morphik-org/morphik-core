from core.completion.base_completion import BaseCompletionModel
from core.models.completion import CompletionRequest, CompletionResponse
from ollama import AsyncClient

BASE_64_PREFIX = "data:image/png;base64,"

class OllamaCompletionModel(BaseCompletionModel):
    """Ollama completion model implementation"""

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.client = AsyncClient(host=base_url)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Ollama API"""
        # Construct prompt with context
        images, context = [], []
        for chunk in request.context_chunks:
            if chunk.startswith(BASE_64_PREFIX):
                image_b64 = chunk.split(',', 1)[1]
                images.append(image_b64)
            else:
                context.append(chunk)
        context = "\n\n".join(context)
        prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately.

<QUESTION>
{request.query}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
"""

        # Call Ollama API
        response = await self.client.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [images[0]] if images else [],
            }],
            options={
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        # Ollama doesn't provide token usage info, so we'll estimate based on characters
        completion_text = response["message"]["content"]
        char_to_token_ratio = 4  # Rough estimate
        estimated_prompt_tokens = len(prompt) // char_to_token_ratio
        estimated_completion_tokens = len(completion_text) // char_to_token_ratio

        return CompletionResponse(
            completion=completion_text,
            usage={
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": estimated_completion_tokens,
                "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
            },
        )
