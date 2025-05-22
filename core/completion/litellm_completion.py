import logging
import re  # Import re for parsing model name
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm

try:
    import ollama
except ImportError:
    ollama = None  # Make ollama import optional

from pydantic import BaseModel

from core.config import get_settings
from core.models.completion import CompletionRequest, CompletionResponse

from .base_completion import BaseCompletionModel

logger = logging.getLogger(__name__)


def get_system_message() -> Dict[str, str]:
    """Return the standard system message for Morphik's query agent."""
    return {
        "role": "system",
        "content": """You are Morphik's powerful query agent. Your role is to:

1. Analyze the provided context chunks from documents carefully
2. Use the context to answer questions accurately and comprehensively
3. Be clear and concise in your answers
4. When relevant, cite specific parts of the context to support your answers
5. For image-based queries, analyze the visual content in conjunction with any text context provided
6. Format your responses using Markdown.

Remember: Your primary goal is to provide accurate, context-aware responses that help users understand
and utilize the information in their documents effectively.""",
    }


def process_context_chunks(context_chunks: List[str], is_ollama: bool) -> Tuple[List[str], List[str], List[str]]:
    """
    Process context chunks and separate text from images.

    Args:
        context_chunks: List of context chunks which may include images
        is_ollama: Whether we're using Ollama (affects image processing)

    Returns:
        Tuple of (context_text, image_urls, ollama_image_data)
    """
    context_text = []
    image_urls = []  # For non-Ollama models (full data URI)
    ollama_image_data = []  # For Ollama models (raw base64)

    for chunk in context_chunks:
        if chunk.startswith("data:image/"):
            if is_ollama:
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

    return context_text, image_urls, ollama_image_data


def format_user_content(context_text: List[str], query: str, prompt_template: Optional[str] = None) -> str:
    """
    Format the user content based on context and query.

    Args:
        context_text: List of context text chunks
        query: The user query
        prompt_template: Optional template to format the content

    Returns:
        Formatted user content string
    """
    context = "\n" + "\n\n".join(context_text) + "\n\n" if context_text else ""

    if prompt_template:
        return prompt_template.format(
            context=context,
            question=query,
            query=query,
        )
    elif context_text:
        return f"Context: {context} Question: {query}"
    else:
        return query


def create_dynamic_model_from_schema(schema: Union[type, Dict]) -> Optional[type]:
    """
    Create a dynamic Pydantic model from a schema definition.

    Args:
        schema: Either a Pydantic BaseModel class or a JSON schema dict

    Returns:
        A Pydantic model class or None if schema format is not recognized
    """
    from pydantic import create_model

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    elif isinstance(schema, dict) and "properties" in schema:
        # Create a dynamic model from JSON schema
        field_definitions = {}
        schema_dict = schema

        for field_name, field_info in schema_dict.get("properties", {}).items():
            if isinstance(field_info, dict) and "type" in field_info:
                field_type = field_info.get("type")
                # Convert schema types to Python types
                if field_type == "string":
                    field_definitions[field_name] = (str, None)
                elif field_type == "number":
                    field_definitions[field_name] = (float, None)
                elif field_type == "integer":
                    field_definitions[field_name] = (int, None)
                elif field_type == "boolean":
                    field_definitions[field_name] = (bool, None)
                elif field_type == "array":
                    field_definitions[field_name] = (list, None)
                elif field_type == "object":
                    field_definitions[field_name] = (dict, None)
                else:
                    # Default to Any for unknown types
                    field_definitions[field_name] = (Any, None)

        # Create the dynamic model
        return create_model("DynamicQueryModel", **field_definitions)
    else:
        logger.warning(f"Unrecognized schema format: {schema}")
        return None


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

    async def _handle_structured_ollama(
        self,
        dynamic_model: type,
        system_message: Dict[str, str],
        user_content: str,
        ollama_image_data: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> CompletionResponse:
        """Handle structured output generation with Ollama."""
        try:
            client = ollama.AsyncClient(host=self.ollama_api_base)

            # Add images directly to content if available
            content_data = user_content
            if ollama_image_data and len(ollama_image_data) > 0:
                # Ollama image handling is limited; we can use only the first image
                content_data = {"content": user_content, "images": [ollama_image_data[0]]}

            # Create messages for Ollama
            messages = [system_message] + history_messages + [{"role": "user", "content": content_data}]

            # Get the JSON schema from the dynamic model
            format_schema = dynamic_model.model_json_schema()

            # Call Ollama directly with format parameter
            response = await client.chat(
                model=self.ollama_base_model_name,
                messages=messages,
                format=format_schema,
                options={
                    "temperature": request.temperature or 0.1,  # Lower temperature for structured output
                    "num_predict": request.max_tokens,
                },
            )

            # Parse the response into the dynamic model
            parsed_response = dynamic_model.model_validate_json(response["message"]["content"])

            # Extract token usage information
            usage = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            }

            return CompletionResponse(
                completion=parsed_response,
                usage=usage,
                finish_reason=response.get("done_reason", "stop"),
            )

        except Exception as e:
            logger.error(f"Error using Ollama for structured output: {e}")
            # Fall back to standard completion if structured output fails
            logger.warning("Falling back to standard Ollama completion without structured output")
            return None

    async def _handle_structured_litellm(
        self,
        dynamic_model: type,
        system_message: Dict[str, str],
        user_content: str,
        image_urls: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> CompletionResponse:
        """Handle structured output generation with LiteLLM."""
        import instructor
        from instructor import Mode

        try:
            # Use instructor with litellm
            client = instructor.from_litellm(litellm.acompletion, mode=Mode.JSON)

            # Create content list with text and images
            content_list = [{"type": "text", "text": user_content}]

            # Add images if available
            if image_urls:
                NUM_IMAGES = min(3, len(image_urls))
                for img_url in image_urls[:NUM_IMAGES]:
                    content_list.append({"type": "image_url", "image_url": {"url": img_url}})

            # Create messages for instructor
            messages = [system_message] + history_messages + [{"role": "user", "content": content_list}]

            # Extract model configuration
            model = self.model_config.get("model_name")
            model_kwargs = {k: v for k, v in self.model_config.items() if k != "model_name"}

            # Override with completion request parameters
            if request.temperature is not None:
                model_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                model_kwargs["max_tokens"] = request.max_tokens

            # Add format forcing for structured output
            model_kwargs["response_format"] = {"type": "json_object"}

            # Call instructor with litellm
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=dynamic_model,
                **model_kwargs,
            )

            # Get token usage from response
            completion_tokens = model_kwargs.get("response_tokens", 0)
            prompt_tokens = model_kwargs.get("prompt_tokens", 0)

            return CompletionResponse(
                completion=response,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                finish_reason="stop",
            )

        except Exception as e:
            logger.error(f"Error using instructor with LiteLLM: {e}")
            # Fall back to standard completion if instructor fails
            logger.warning("Falling back to standard LiteLLM completion without structured output")
            return None

    async def _handle_standard_ollama(
        self,
        user_content: str,
        ollama_image_data: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> CompletionResponse:
        """Handle standard (non-structured) output generation with Ollama."""
        logger.debug(f"Using direct Ollama client for model: {self.ollama_base_model_name}")
        client = ollama.AsyncClient(host=self.ollama_api_base)

        # Construct Ollama messages
        system_message = {"role": "system", "content": get_system_message()["content"]}
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

        ollama_messages = [system_message] + history_messages + [user_message_data]

        # Construct Ollama options
        options = {
            "temperature": request.temperature,
            "num_predict": (
                request.max_tokens if request.max_tokens is not None else -1
            ),  # Default to model's default if None
        }

        try:
            response = await client.chat(model=self.ollama_base_model_name, messages=ollama_messages, options=options)

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

    async def _handle_standard_litellm(
        self,
        user_content: str,
        image_urls: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> CompletionResponse:
        """Handle standard (non-structured) output generation with LiteLLM."""
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
        litellm_messages = [get_system_message()] + history_messages + [user_message]

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

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using LiteLLM or direct Ollama client if configured.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics
        """
        # Process context chunks and handle images
        context_text, image_urls, ollama_image_data = process_context_chunks(request.context_chunks, self.is_ollama)

        # Format user content
        user_content = format_user_content(context_text, request.query, request.prompt_template)

        history_messages = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]

        # Check if structured output is requested
        structured_output = request.schema is not None

        # If structured output is requested, use instructor to handle it
        if structured_output:
            # Get dynamic model from schema
            dynamic_model = create_dynamic_model_from_schema(request.schema)

            # If schema format is not recognized, log warning and fall back to text completion
            if not dynamic_model:
                logger.warning(f"Unrecognized schema format: {request.schema}. Falling back to text completion.")
                structured_output = False
            else:
                logger.info(f"Using structured output with model: {dynamic_model.__name__}")

                # Create system and user messages with enhanced instructions for structured output
                system_message = {
                    "role": "system",
                    "content": get_system_message()["content"]
                    + "\n\nYou MUST format your response according to the required schema.",
                }

                # Create enhanced user message that includes schema information
                enhanced_user_content = (
                    user_content + "\n\nPlease format your response according to the required schema."
                )

                # Try structured output based on model type
                if self.is_ollama:
                    response = await self._handle_structured_ollama(
                        dynamic_model,
                        system_message,
                        enhanced_user_content,
                        ollama_image_data,
                        request,
                        history_messages,
                    )
                    if response:
                        return response
                    structured_output = False  # Fall back if structured output failed
                else:
                    response = await self._handle_structured_litellm(
                        dynamic_model,
                        system_message,
                        enhanced_user_content,
                        image_urls,
                        request,
                        history_messages,
                    )
                    if response:
                        return response
                    structured_output = False  # Fall back if structured output failed

        # If we're here, either structured output wasn't requested or instructor failed
        # Proceed with standard completion based on model type
        if self.is_ollama:
            return await self._handle_standard_ollama(user_content, ollama_image_data, request, history_messages)
        else:
            return await self._handle_standard_litellm(user_content, image_urls, request, history_messages)
