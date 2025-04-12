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

    def _create_system_message(self) -> dict:
        """Create the system message for the LLM."""
        return {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided context to answer questions accurately.",
        }

    def _process_context_chunks(self, context_chunks: list) -> tuple[list, list, str]:
        """
        Process context chunks and handle images.
        
        Returns:
            Tuple of (context_text, image_urls, formatted_context)
        """
        context_text = []
        image_urls = []

        for chunk in context_chunks:
            if chunk.startswith("data:image/"):
                # Handle image data URI
                image_urls.append(chunk)
                # Log image size for debugging
                logger.info(f"Found image data URI with size: {len(chunk)} bytes")
            else:
                context_text.append(chunk)

        context = "\n" + "\n\n".join(context_text) + "\n\n"
        return context_text, image_urls, context
        
    def _format_user_content(self, request: CompletionRequest, context_text: list, context: str) -> str:
        """Format user content based on the template and available resources."""
        if request.prompt_template:
            # Use custom prompt template with placeholders for context and query
            formatted_text = request.prompt_template.format(
                context=context,
                question=request.query,
                query=request.query,  # Alternative name for the query
            )
            return formatted_text
        elif context_text:
            return f"Context: {context} Question: {request.query}"
        else:
            return request.query

    def _prepare_user_message(self, user_content: str, image_urls: list) -> dict:
        """Prepare the user message with text and images if supported."""
        # Default text-only message
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

        return user_message

    def _prepare_model_params(self, messages: list, request: CompletionRequest) -> dict:
        """Prepare the parameters for the LiteLLM call."""
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
                
        return model_params

    def _log_messages(self, messages: list) -> None:
        """Log the details of messages being sent to the LLM."""
        logger.info(f"Sending {len(messages)} messages to LLM")
        for i, msg in enumerate(messages, 1):
            if isinstance(msg.get("content", ""), str):
                content_preview = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                logger.info(f"Message {i} - Role: {msg.get('role')}, Content: {content_preview}")
            elif isinstance(msg.get("content", ""), list):
                logger.info(f"Message {i} - Role: {msg.get('role')}, Content: [complex structured content with {len(msg.get('content', []))} items]")

    def _format_response(self, response) -> CompletionResponse:
        """Format the LiteLLM response to a CompletionResponse object."""
        return CompletionResponse(
            completion=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using LiteLLM.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics
        """
        # Initialize messages with system message
        messages = [self._create_system_message()]

        # Process context chunks and extract images
        context_text, image_urls, context = self._process_context_chunks(request.context_chunks)
        
        # Format user content
        user_content = self._format_user_content(request, context_text, context)
        
        # Prepare user message with text and possibly images
        user_message = self._prepare_user_message(user_content, image_urls)
        
        # Add user message to messages list
        messages.append(user_message)

        # Prepare model parameters for the API call
        model_params = self._prepare_model_params(messages, request)
        
        # Log the parameters and messages
        logger.debug(f"Calling LiteLLM with params: {model_params}")
        self._log_messages(messages)
        
        # Make the API call
        response = await litellm.acompletion(**model_params)

        # Format and return the response
        return self._format_response(response)
