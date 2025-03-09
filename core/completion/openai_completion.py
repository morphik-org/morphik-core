from .base_completion import BaseCompletionModel
from core.models.completion import CompletionRequest, CompletionResponse


class OpenAICompletionModel(BaseCompletionModel):
    """OpenAI completion model implementation"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Import here to avoid dependency if not using OpenAI
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI API"""
        # Process context chunks and handle images
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer questions accurately.",
            }
        ]

        # Build user message content
        user_message_content = []
        context_text = []

        for chunk in request.context_chunks:
            if chunk.startswith("data:image/"):
                # Handle image data URI
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": chunk,
                    }
                })
            else:
                context_text.append(chunk)
        
        max_num_images = min(3, len(user_message_content))
        user_message_content = user_message_content[:max_num_images] # limit the number of images to 3

        # Add text context if any
        if context_text:
            user_message_content.insert(0, {
                "type": "text",
                "text": f"Context:\n{'\\n\\n'.join(context_text)}\n\nQuestion: {request.query}"
            })
        else:
            user_message_content.insert(0, {
                "type": "text",
                "text": f"{request.query}"
            })

        messages.append({
            "role": "user",
            "content": user_message_content
        })

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
                "total_tokens": response.usage.total_tokens,
            },
            sources=request.sources,
        )
