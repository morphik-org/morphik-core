import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

import litellm
from pydantic import BaseModel

from core.config import get_settings
from ee.llm_key_router import get_current_overrides

logger = logging.getLogger(__name__)
global_settings_ref = get_settings()


# Helper to determine provider from model name string (can be moved to a util module later)
def _get_provider_from_model_name(model_name_str: Optional[str]) -> Optional[str]:
    if not model_name_str:
        return None
    if "gpt-" in model_name_str.lower() or "openai" in model_name_str.lower():
        return "openai"
    if "claude-" in model_name_str.lower() or "anthropic" in model_name_str.lower():
        return "anthropic"
    if "azure_" in model_name_str.lower() or "azure" == model_name_str.lower():
        return "azure"
    if "ollama" in model_name_str.lower():
        return "ollama"
    return None  # Default if no specific provider pattern is matched


class BaseRule(BaseModel, ABC):
    """Base model for all rules"""

    type: str
    stage: Literal["post_parsing", "post_chunking"]

    @abstractmethod
    async def apply(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], str]:
        """
        Apply the rule to the content.

        Args:
            content: The content to apply the rule to
            metadata: Optional existing metadata that may be used or modified by the rule

        Returns:
            tuple[Dict[str, Any], str]: (metadata, modified_content)
        """
        pass


class MetadataOutput(BaseModel):
    """Model for metadata extraction results"""

    # This model will be dynamically extended based on the schema


class MetadataExtractionRule(BaseRule):
    """Rule for extracting metadata using a schema"""

    type: Literal["metadata_extraction"]
    schema: Dict[str, Any]
    use_images: bool = False

    async def apply(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], str]:
        """Extract metadata according to schema, respecting user's model for 'rules' stage."""
        import instructor
        from pydantic import create_model

        all_user_settings = get_current_overrides()

        # Determine the model key for the 'rules' stage
        rules_stage_settings = all_user_settings.get("rules", {})
        user_defined_rules_model_key = rules_stage_settings.get("model_name")

        final_rules_model_key = user_defined_rules_model_key or global_settings_ref.RULES_MODEL

        # Create a dynamic Pydantic model based on the schema
        # This allows instructor to validate the output against our schema
        field_definitions = {}
        for field_name, field_info in self.schema.items():
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
            else:
                # Default to Any if no type specified
                field_definitions[field_name] = (Any, None)

        # Create the dynamic model
        DynamicMetadataModel = create_model("DynamicMetadataModel", **field_definitions)

        # Create a more explicit instruction that clearly shows expected output format
        schema_descriptions = []
        for field_name, field_config in self.schema.items():
            field_type = field_config.get("type", "string") if isinstance(field_config, dict) else "string"
            description = (
                field_config.get("description", "No description") if isinstance(field_config, dict) else field_config
            )
            schema_descriptions.append(f"- {field_name}: {description} (type: {field_type})")

        schema_text = "\n".join(schema_descriptions)

        # Adjust prompt based on whether it's a chunk or full document and whether it's an image
        if self.use_images:
            # For image rules, we do NOT embed the base64 image data in the textual prompt.
            # The image will be sent separately via the `image_url` entry in the vision message.
            # Keeping the textual prompt concise avoids blowing up the context window.
            prompt = f"""
            Extract metadata from the following image according to this schema:

            {schema_text}

            The image is attached below.

            Follow these guidelines:
            1. Extract all requested information as simple strings, numbers, or booleans
            (not as objects or nested structures)
            2. If information is not present, indicate this with null instead of making something up
            3. Answer directly with the requested information - don't include explanations or reasoning
            4. Be concise but accurate in your extractions
            """
            prompt_context = "image" if self.stage == "post_chunking" else "document with images"
        else:
            prompt_context = "chunk of text" if self.stage == "post_chunking" else "text"
            prompt = f"""
            Extract metadata from the following {prompt_context} according to this schema:

            {schema_text}

            Text to extract from:
            {content}

            Follow these guidelines:
            1. Extract all requested information as simple strings, numbers, or booleans
            (not as objects or nested structures)
            2. If information is not present, indicate this with null instead of making something up
            3. Answer directly with the requested information - don't include explanations or reasoning
            4. Be concise but accurate in your extractions
            """

        # Get the model configuration from registered_models FOR THE FINAL MODEL KEY
        model_config = global_settings_ref.REGISTERED_MODELS.get(final_rules_model_key, {})
        if not model_config:
            raise ValueError(f"Model '{final_rules_model_key}' not found in registered_models configuration")

        # Prepare base64 data for vision model if this is an image rule
        vision_messages = []
        if self.use_images:
            try:
                # For image content, check if it's a base64 string
                # Handle data URI format "data:image/png;base64,..."
                if content.startswith("data:"):
                    content_type, content = content.split(";base64,", 1)

                # User message with image content
                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}},
                        ],
                    }
                ]
            except Exception as e:
                logger.error(f"Error preparing image content for vision model: {str(e)}")
                # Fall back to text-only if image processing fails
                vision_messages = []

        system_message = {
            "role": "system",
            "content": (
                "You are a metadata extraction assistant. Extract structured metadata "
                f"from {'images' if self.use_images else 'text'} "
                "precisely following the provided schema. Always return the metadata as direct values "
                "(strings, numbers, booleans), not as objects with additional properties."
            ),
        }

        # If we have vision messages, use those, otherwise use standard text message
        messages = []
        if vision_messages and self.use_images:
            messages = [system_message] + vision_messages
        else:
            user_message = {"role": "user", "content": prompt}
            messages = [system_message, user_message]

        # Use instructor with litellm to get structured responses
        client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

        try:
            model_for_call = model_config.get("model_name")
            model_kwargs = {k: v for k, v in model_config.items() if k != "model_name"}

            # Inject stage-specific API key override for "rules" stage
            try:
                # rules_stage_settings already fetched
                user_selected_provider = rules_stage_settings.get("provider")
                if user_selected_provider:
                    user_api_key = rules_stage_settings.get(f"{user_selected_provider}_api_key")
                    current_model_provider = _get_provider_from_model_name(model_config.get("model_name"))
                    if user_api_key and current_model_provider == user_selected_provider:
                        model_kwargs["api_key"] = user_api_key
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to apply API key override for metadata rule: {e}")

            # Use instructor's client to create a structured response
            response = await client.chat.completions.create(
                model=model_for_call,
                messages=messages,
                response_model=DynamicMetadataModel,
                **model_kwargs,
            )

            # Convert pydantic model to dict
            extracted_metadata = response.model_dump()

        except Exception as e:
            logger.error(f"Error in instructor metadata extraction: {str(e)}")
            extracted_metadata = {}

        # Metadata extraction doesn't modify content
        return extracted_metadata, content


class TransformationOutput(BaseModel):
    """Model for text transformation results"""

    transformed_text: str


class NaturalLanguageRule(BaseRule):
    """Rule for transforming content using natural language"""

    type: Literal["natural_language"]
    prompt: str

    async def apply(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], str]:
        """Transform content according to prompt, respecting user's model for 'rules' stage."""
        import instructor

        all_user_settings = get_current_overrides()

        # Determine the model key for the 'rules' stage
        rules_stage_settings = all_user_settings.get("rules", {})
        user_defined_rules_model_key = rules_stage_settings.get("model_name")

        final_rules_model_key = user_defined_rules_model_key or global_settings_ref.RULES_MODEL

        # Adjust prompt based on whether it's a chunk or full document
        prompt_context = "chunk of text" if self.stage == "post_chunking" else "text"

        prompt = f"""
        Your task is to transform the following {prompt_context} according to this instruction:
        {self.prompt}

        Text to transform:
        {content}

        Perform the transformation and return only the transformed text.
        """

        # Get the model configuration from registered_models FOR THE FINAL MODEL KEY
        model_config = global_settings_ref.REGISTERED_MODELS.get(final_rules_model_key, {})
        if not model_config:
            raise ValueError(f"Model '{final_rules_model_key}' not found in registered_models configuration")

        system_message = {
            "role": "system",
            "content": (
                "You are a text transformation assistant. Transform text precisely following "
                "the provided instructions."
            ),
        }

        user_message = {"role": "user", "content": prompt}

        # Use instructor with litellm to get structured responses
        client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

        try:
            model_for_call = model_config.get("model_name")
            model_kwargs = {k: v for k, v in model_config.items() if k != "model_name"}

            # Inject stage-specific API key override for "rules" stage
            try:
                # rules_stage_settings already fetched
                user_selected_provider = rules_stage_settings.get("provider")
                if user_selected_provider:
                    user_api_key = rules_stage_settings.get(f"{user_selected_provider}_api_key")
                    current_model_provider = _get_provider_from_model_name(model_config.get("model_name"))
                    if user_api_key and current_model_provider == user_selected_provider:
                        model_kwargs["api_key"] = user_api_key
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to apply API key override for NL rule: {e}")

            # Use instructor's client to create a structured response
            response = await client.chat.completions.create(
                model=model_for_call,
                messages=[system_message, user_message],
                response_model=TransformationOutput,
                **model_kwargs,
            )

            # Extract the transformed text from the response model
            transformed_text = response.transformed_text

        except Exception as e:
            logger.error(f"Error in instructor text transformation: {str(e)}")
            transformed_text = content  # Return original content on error

        # Natural language rules modify content, don't add metadata directly
        return {}, transformed_text
