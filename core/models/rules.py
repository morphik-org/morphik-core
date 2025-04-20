import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

import litellm
from pydantic import BaseModel

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseRule(BaseModel, ABC):
    """Base model for all rules"""

    type: str

    @abstractmethod
    async def apply(self, content: str) -> tuple[Dict[str, Any], str]:
        """
        Apply the rule to the content.

        Args:
            content: The content to apply the rule to

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

    async def apply(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract metadata according to schema"""
        import instructor
        from pydantic import create_model

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

        prompt = f"""
        Extract metadata from the following text according to this schema:

        {schema_text}

        Text to extract from:
        {content}

        Follow these guidelines:
        1. Extract all requested information as simple strings, numbers, or booleans (not as objects or nested structures)
        2. If information is not present, indicate this with null instead of making something up
        3. Answer directly with the requested information - don't include explanations or reasoning
        4. Be concise but accurate in your extractions
        """

        # Get the model configuration from registered_models
        model_config = settings.REGISTERED_MODELS.get(settings.RULES_MODEL, {})
        if not model_config:
            raise ValueError(f"Model '{settings.RULES_MODEL}' not found in registered_models configuration")

        system_message = {
            "role": "system",
            "content": "You are a metadata extraction assistant. Extract structured metadata from text precisely following the provided schema. Always return the metadata as direct values (strings, numbers, booleans), not as objects with additional properties.",
        }

        user_message = {"role": "user", "content": prompt}

        # Use instructor with litellm to get structured responses
        client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

        try:
            # Extract the model name for instructor call
            model = model_config.get("model_name")

            # Prepare additional parameters from model config
            model_kwargs = {k: v for k, v in model_config.items() if k != "model_name"}

            # Use instructor's client to create a structured response
            response = await client.chat.completions.create(
                model=model,
                messages=[system_message, user_message],
                response_model=DynamicMetadataModel,
                **model_kwargs,
            )

            # Convert pydantic model to dict
            metadata = response.model_dump()

        except Exception as e:
            logger.error(f"Error in instructor metadata extraction: {str(e)}")
            metadata = {}

        return metadata, content


class TransformationOutput(BaseModel):
    """Model for text transformation results"""

    transformed_text: str


class NaturalLanguageRule(BaseRule):
    """Rule for transforming content using natural language"""

    type: Literal["natural_language"]
    prompt: str

    async def apply(self, content: str) -> tuple[Dict[str, Any], str]:
        """Transform content according to prompt"""
        import instructor

        prompt = f"""
        Your task is to transform the following text according to this instruction:
        {self.prompt}

        Text to transform:
        {content}

        Perform the transformation and return only the transformed text.
        """

        # Get the model configuration from registered_models
        model_config = settings.REGISTERED_MODELS.get(settings.RULES_MODEL, {})
        if not model_config:
            raise ValueError(f"Model '{settings.RULES_MODEL}' not found in registered_models configuration")

        system_message = {
            "role": "system",
            "content": "You are a text transformation assistant. Transform text precisely following the provided instructions.",
        }

        user_message = {"role": "user", "content": prompt}

        # Use instructor with litellm to get structured responses
        client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

        try:
            # Extract the model name for instructor call
            model = model_config.get("model_name")

            # Prepare additional parameters from model config
            model_kwargs = {k: v for k, v in model_config.items() if k != "model_name"}

            # Use instructor's client to create a structured response
            response = await client.chat.completions.create(
                model=model,
                messages=[system_message, user_message],
                response_model=TransformationOutput,
                **model_kwargs,
            )

            # Extract the transformed text from the response model
            transformed_text = response.transformed_text

        except Exception as e:
            logger.error(f"Error in instructor text transformation: {str(e)}")
            transformed_text = content  # Return original content on error

        return {}, transformed_text
