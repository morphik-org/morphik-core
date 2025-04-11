import logging
import litellm
from .base_completion import BaseCompletionModel
from core.models.completion import CompletionRequest, CompletionResponse
from core.config import get_settings

logger = logging.getLogger(__name__)


class LiteLLMCompletionModel(BaseCompletionModel):
    """
    LiteLLM completion model implementation that provides unified access to various LLM providers.
    Uses registered models from the config file.
    """

    def __init__(self, model_key: str):
        """
        Initialize LiteLLM completion model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_settings()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if (
            not hasattr(settings, "REGISTERED_MODELS")
            or model_key not in settings.REGISTERED_MODELS
        ):
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.REGISTERED_MODELS[model_key]
        logger.info(
            f"Initialized LiteLLM completion model with model_key={model_key}, config={self.model_config}"
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using LiteLLM.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics
        """
        # Process context chunks and handle images
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer questions accurately.",
            }
        ]

        # Build user message content
        context_text = []
        image_urls = []

        for chunk in request.context_chunks:
            if chunk.startswith("data:image/"):
                # Handle image data URI
                image_urls.append(chunk)
                # Log image size for debugging
                logger.info(f"Found image data URI with size: {len(chunk)} bytes")
            else:
                context_text.append(chunk)

        context = "\n" + "\n\n".join(context_text) + "\n\n"

        # Create message content based on the template and available resources
        if request.prompt_template:
            # Use custom prompt template with placeholders for context and query
            formatted_text = request.prompt_template.format(
                context=context,
                question=request.query,
                query=request.query,  # Alternative name for the query
            )
            user_content = formatted_text
        elif context_text:
            user_content = f"Context: {context} Question: {request.query}"
        else:
            user_content = request.query

        # Add text content
        user_message = {"role": "user", "content": user_content}

        # Add images if the model supports vision capabilities
        is_vision_model = self.model_config.get("vision", False)
        if image_urls and is_vision_model:
            # For models that support images
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                ],
            }

            # Add up to 3 images
            for img_url in image_urls[:3]:
                user_message["content"].append({"type": "image_url", "image_url": {"url": img_url}})

        messages.append(user_message)

        # Prepare the completion request parameters
        model_params = {
            "model": self.model_config["model_name"],
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "num_retries": 3,
        }

        # Add all model-specific parameters from the config
        for key, value in self.model_config.items():
            if key not in ["model_name", "vision"]:  # Skip these as we've already handled them
                model_params[key] = value

        # Call LiteLLM
        logger.debug(f"Calling LiteLLM with params: {model_params}")
        
        # Log the actual messages being sent to the LLM for debugging
        logger.info(f"Sending {len(messages)} messages to LLM")
        for i, msg in enumerate(messages, 1):
            if isinstance(msg.get("content", ""), str):
                content_preview = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                logger.info(f"Message {i} - Role: {msg.get('role')}, Content: {content_preview}")
            elif isinstance(msg.get("content", ""), list):
                logger.info(f"Message {i} - Role: {msg.get('role')}, Content: [complex structured content with {len(msg.get('content', []))} items]")
        
        response = await litellm.acompletion(**model_params)

        # Format response to match CompletionResponse
        return CompletionResponse(
            completion=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )
