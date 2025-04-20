import logging
import re  # Import re for parsing model name

import litellm

try:
    import ollama
except ImportError:
    ollama = None  # Make ollama import optional

from core.config import get_settings
from core.models.completion import CompletionRequest, CompletionResponse

from .base_completion import BaseCompletionModel

logger = logging.getLogger(__name__)


class LiteLLMCompletionModel(BaseCompletionModel):
    """
    LiteLLM completion model implementation that provides unified access to various LLM providers.
    Uses registered models from the config file. Can optionally use direct Ollama client.
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
        if not hasattr(settings, "REGISTERED_MODELS") or model_key not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.REGISTERED_MODELS[model_key]

        # Check if it's an Ollama model for potential direct usage
        self.is_ollama = "ollama" in self.model_config.get("model_name", "").lower()
        self.ollama_api_base = None
        self.ollama_base_model_name = None

        if self.is_ollama:
            if ollama is None:
                logger.warning("Ollama model selected, but 'ollama' library not installed. Falling back to LiteLLM.")
                self.is_ollama = False  # Fallback to LiteLLM if library missing
            else:
                self.ollama_api_base = self.model_config.get("api_base")
                if not self.ollama_api_base:
                    logger.warning(
                        f"Ollama model {self.model_key} selected for direct use, "
                        "but 'api_base' is missing in config. Falling back to LiteLLM."
                    )
                    self.is_ollama = False  # Fallback if api_base is missing
                else:
                    # Extract base model name (e.g., 'llama3.2' from 'ollama_chat/llama3.2')
                    match = re.search(r"[^/]+$", self.model_config["model_name"])
                    if match:
                        self.ollama_base_model_name = match.group(0)
                    else:
                        logger.warning(
                            f"Could not parse base model name from Ollama model "
                            f"{self.model_config['model_name']}. Falling back to LiteLLM."
                        )
                        self.is_ollama = False  # Fallback if name parsing fails

        logger.info(
            f"Initialized LiteLLM completion model with model_key={model_key}, "
            f"config={self.model_config}, is_ollama_direct={self.is_ollama}"
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using LiteLLM or direct Ollama client if configured.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics
        """
        # Process context chunks and handle images
        messages = [
            {
                "role": "system",
                "content": """You are Morphik's powerful query agent. Your role is to:

1. Analyze the provided context chunks from documents carefully
2. Use the context to answer questions accurately and comprehensively
3. Be clear and concise in your answers
4. When relevant, cite specific parts of the context to support your answers
5. For image-based queries, analyze the visual content in conjunction with any text context provided

Remember: Your primary goal is to provide accurate, context-aware responses that help users understand
and utilize the information in their documents effectively.""",
            }
        ]

        context_text = []
        image_urls = []  # For non-Ollama models (full data URI)
        ollama_image_data = []  # For Ollama models (raw base64)

        for chunk in request.context_chunks:
            if chunk.startswith("data:image/"):
                if self.is_ollama:
                    # For Ollama, strip the data URI prefix and just keep the base64 data
                    try:
                        base64_data = chunk.split(",", 1)[1]
                        ollama_image_data.append(base64_data)
                    except IndexError:
                        logger.warning(f"Could not parse base64 data from image chunk: {chunk[:50]}...")
                else:
                    image_urls.append(chunk)
            else:
                context_text.append(chunk)

        context = "\n" + "\n\n".join(context_text) + "\n\n"

        # Create message content based on the template and available resources
        if request.prompt_template:
            formatted_text = request.prompt_template.format(
                context=context,
                question=request.query,
                query=request.query,
            )
            user_content = formatted_text
        elif context_text:
            user_content = f"Context: {context} Question: {request.query}"
        else:
            user_content = request.query

        # --- Direct Ollama Call ---
        if self.is_ollama:
            logger.debug(f"Using direct Ollama client for model: {self.ollama_base_model_name}")
            client = ollama.AsyncClient(host=self.ollama_api_base)

            # Construct Ollama messages
            system_message = {"role": "system", "content": messages[0]["content"]}
            user_message_data = {"role": "user", "content": user_content}

            # Add images directly to the user message if available
            if ollama_image_data:
                if len(ollama_image_data) > 1:
                    logger.warning(
                        f"Ollama model {self.model_config['model_name']} only supports one image per message. "
                        "Using the first image and ignoring others."
                    )
                # Add 'images' key inside the user message dictionary
                user_message_data["images"] = [ollama_image_data[0]]

            ollama_messages = [system_message, user_message_data]

            # Construct Ollama options
            options = {
                "temperature": request.temperature,
                "num_predict": (
                    request.max_tokens if request.max_tokens is not None else -1
                ),  # Default to model's default if None
            }

            try:
                response = await client.chat(
                    model=self.ollama_base_model_name, messages=ollama_messages, options=options
                )

                # Map Ollama response to CompletionResponse
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)

                return CompletionResponse(
                    completion=response["message"]["content"],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    finish_reason=response.get("done_reason", "unknown"),  # Map done_reason if available
                )

            except Exception as e:
                logger.error(f"Error during direct Ollama call: {e}")
                raise

        # --- LiteLLM Call (Non-Ollama or Fallback) ---
        else:
            logger.debug(f"Using LiteLLM for model: {self.model_config['model_name']}")
            # Build messages for LiteLLM
            content_list = [{"type": "text", "text": user_content}]
            include_images = image_urls  # Use the collected full data URIs

            if include_images:
                NUM_IMAGES = min(3, len(image_urls))
                for img_url in image_urls[:NUM_IMAGES]:
                    content_list.append({"type": "image_url", "image_url": {"url": img_url}})

            # LiteLLM uses list content format
            user_message = {"role": "user", "content": content_list}
            # Use the system prompt defined earlier
            litellm_messages = [messages[0], user_message]

            # Prepare LiteLLM parameters
            model_params = {
                "model": self.model_config["model_name"],
                "messages": litellm_messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "num_retries": 3,
            }

            for key, value in self.model_config.items():
                if key != "model_name":
                    model_params[key] = value

            logger.debug(f"Calling LiteLLM with params: {model_params}")
            response = await litellm.acompletion(**model_params)

            return CompletionResponse(
                completion=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
            )
