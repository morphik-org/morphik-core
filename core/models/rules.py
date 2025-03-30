from typing import Dict, Any, Literal
from pydantic import BaseModel
from abc import ABC, abstractmethod
from core.config import get_settings
from openai import AsyncOpenAI
from ollama import AsyncClient
import json
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize the appropriate client based on settings
if settings.RULES_PROVIDER == "openai":
    # Use global OpenAI base URL if provided
    if hasattr(settings, "OPENAI_BASE_URL") and settings.OPENAI_BASE_URL:
        rules_client = AsyncOpenAI(base_url=settings.OPENAI_BASE_URL)
    else:
        rules_client = AsyncOpenAI()
else:  # ollama
    rules_client = AsyncClient(host=settings.COMPLETION_OLLAMA_BASE_URL)


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


class MetadataExtractionRule(BaseRule):
    """Rule for extracting metadata using a schema"""

    type: Literal["metadata_extraction"]
    schema: Dict[str, Any]

    async def apply(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract metadata according to schema"""
        prompt = f"""
        Extract metadata from the following text according to this schema:
        {self.schema}

        Text to extract from:
        {content}

        Return ONLY a JSON object with the extracted metadata.
        """

        if settings.RULES_PROVIDER == "openai":
            response = await rules_client.chat.completions.create(
                model=settings.RULES_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a metadata extraction assistant. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            metadata = json.loads(response.choices[0].message.content)
        else:  # ollama
            response = await rules_client.chat(
                model=settings.RULES_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a metadata extraction assistant. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                format="json",
            )
            content_str = response["message"]["content"]
            logger.debug(f"Ollama raw response: {content_str}")
            metadata = json.loads(content_str)

        return metadata, content


class NaturalLanguageRule(BaseRule):
    """Rule for transforming content using natural language"""

    type: Literal["natural_language"]
    prompt: str

    async def apply(self, content: str) -> tuple[Dict[str, Any], str]:
        """Transform content according to prompt"""
        prompt = f"""
        Your task is to transform the following text according to this instruction:
        {self.prompt}
        
        Text to transform:
        {content}
        
        Return ONLY the transformed text.
        """

        if settings.RULES_PROVIDER == "openai":
            response = await rules_client.chat.completions.create(
                model=settings.RULES_MODEL,
                messages=[
                    {"role": "system", "content": "You are a text transformation assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            transformed_text = response.choices[0].message.content
        else:  # ollama
            response = await rules_client.chat(
                model=settings.RULES_MODEL,
                messages=[
                    {"role": "system", "content": "You are a text transformation assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            transformed_text = response["message"]["content"]

        return {}, transformed_text
