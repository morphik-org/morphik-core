from typing import Dict, Any, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel


class Rule(ABC):
    """Base class for all rules that can be applied during document ingestion"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary format for API requests"""
        pass


class MetadataExtractionRule(Rule):
    """Server-side rule for extracting metadata using a schema"""

    def __init__(self, schema: Union[Type[BaseModel], Dict[str, Any]]):
        self.schema = schema

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            # Convert Pydantic model to dict schema
            schema_dict = self.schema.model_json_schema()
        else:
            # Assume it's already a dict schema
            schema_dict = self.schema

        return {"type": "metadata_extraction", "schema": schema_dict}


class NaturalLanguageRule(Rule):
    """Server-side rule for transforming content using natural language"""

    def __init__(self, prompt: str):
        """
        Args:
            prompt: Instruction for how to transform the content
                   e.g. "Remove any personal information" or "Convert to bullet points"
        """
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "natural_language", "prompt": self.prompt}


__all__ = ["Rule", "MetadataExtractionRule", "NaturalLanguageRule"]
