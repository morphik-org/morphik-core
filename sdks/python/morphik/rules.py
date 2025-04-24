from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Type, Union

from pydantic import BaseModel


class Rule(ABC):
    """Base class for all rules that can be applied during document ingestion"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary format for API requests"""
        pass


class MetadataExtractionRule(Rule):
    """Server-side rule for extracting metadata using a schema"""

    def __init__(self, schema: Union[Type[BaseModel], Dict[str, Any]], stage: Literal["post_parsing", "post_chunking"]):
        """
        Args:
            schema: Pydantic model or dict schema defining metadata fields to extract
            stage: When to apply the rule - either "post_parsing" (full document text) or
                  "post_chunking" (individual chunks)
        """
        self.schema = schema
        self.stage = stage

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            # Convert Pydantic model to dict schema
            schema_dict = self.schema.model_json_schema()
        else:
            # Assume it's already a dict schema
            schema_dict = self.schema

        return {"type": "metadata_extraction", "schema": schema_dict, "stage": self.stage}


class NaturalLanguageRule(Rule):
    """Server-side rule for transforming content using natural language"""

    def __init__(self, prompt: str, stage: Literal["post_parsing", "post_chunking"]):
        """
        Args:
            prompt: Instruction for how to transform the content
                   e.g. "Remove any personal information" or "Convert to bullet points"
            stage: When to apply the rule - either "post_parsing" (full document text) or
                  "post_chunking" (individual chunks)
        """
        self.prompt = prompt
        self.stage = stage

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "natural_language", "prompt": self.prompt, "stage": self.stage}


__all__ = ["Rule", "MetadataExtractionRule", "NaturalLanguageRule"]
