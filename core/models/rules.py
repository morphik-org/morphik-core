from typing import Dict, Any, Literal
from pydantic import BaseModel
from abc import ABC, abstractmethod
from core.completion.base_completion import BaseCompletionModel, CompletionRequest


class BaseRule(BaseModel, ABC):
    """Base model for all rules"""

    type: str

    @abstractmethod
    async def apply(
        self, content: str, completion_model: BaseCompletionModel
    ) -> tuple[Dict[str, Any], str]:
        """
        Apply the rule to the content.

        Args:
            content: The content to apply the rule to
            completion_model: The completion model to use for NL operations

        Returns:
            tuple[Dict[str, Any], str]: (metadata, modified_content)
        """
        pass


class MetadataExtractionRule(BaseRule):
    """Rule for extracting metadata using a schema"""

    type: Literal["metadata_extraction"]
    schema: Dict[str, Any]

    async def apply(
        self, content: str, completion_model: BaseCompletionModel
    ) -> tuple[Dict[str, Any], str]:
        """Extract metadata according to schema"""
        prompt = f"""
        Extract metadata from the following text according to this schema:
        {self.schema}

        Text to extract from:
        {content}

        Return ONLY a JSON object with the extracted metadata.
        """

        response = await completion_model.complete(
            CompletionRequest(prompt=prompt, json_response=True)
        )

        try:
            metadata = response.content if isinstance(response.content, dict) else {}
            return metadata, content
        except Exception as e:
            return {}, content


class NaturalLanguageRule(BaseRule):
    """Rule for transforming content using natural language"""

    type: Literal["natural_language"]
    prompt: str

    async def apply(
        self, content: str, completion_model: BaseCompletionModel
    ) -> tuple[Dict[str, Any], str]:
        """Transform content according to prompt"""
        prompt = f"""
        Your task is to transform the following text according to this instruction:
        {self.prompt}
        
        Text to transform:
        {content}
        
        Return ONLY the transformed text.
        """

        response = await completion_model.complete(CompletionRequest(prompt=prompt))

        return {}, response.content or content
