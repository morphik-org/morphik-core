from .base_completion import BaseCompletionModel, CompletionRequest, CompletionResponse


class OpenAICompletionModel(BaseCompletionModel):
    """OpenAI completion model implementation"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Import here to avoid dependency if not using OpenAI
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI API"""
        # Construct prompt with context
        context = "\n\n".join(request.context_chunks)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
        ]

        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return CompletionResponse(
            completion=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
