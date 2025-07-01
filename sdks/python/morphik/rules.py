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

    def __init__(
        self,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        stage: Literal["post_parsing", "post_chunking"] = "post_parsing",
        use_images: bool = False,
    ):
        """
        Args:
            schema: Pydantic model or dict schema defining metadata fields to extract
            stage: When to apply the rule - either "post_parsing" (full document text) or
                  "post_chunking" (individual chunks). Defaults to "post_parsing" for backward compatibility.
            use_images: Whether to process image chunks instead of text chunks. Defaults to False.
        """
        self.schema = schema
        self.stage = stage
        self.use_images = use_images

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            # Convert Pydantic model to dict schema
            json_schema = self.schema.model_json_schema()
            # Extract the properties from the JSON schema format
            # to match the expected server format
            schema_dict = {}
            if "properties" in json_schema:
                for field_name, field_info in json_schema["properties"].items():
                    # Get field description from the Pydantic field
                    description = field_info.get("description", f"No description for {field_name}")
                    field_type = field_info.get("type", "string")
                    schema_dict[field_name] = {
                        "type": field_type,
                        "description": description
                    }
            else:
                # Fallback if properties not found
                schema_dict = json_schema
        else:
            # Assume it's already a dict schema
            schema_dict = self.schema

        return {
            "type": "metadata_extraction",
            "schema": schema_dict,
            "stage": self.stage,
            "use_images": self.use_images,
        }


class NaturalLanguageRule(Rule):
    """Server-side rule for transforming content using natural language"""

    def __init__(self, prompt: str, stage: Literal["post_parsing", "post_chunking"] = "post_parsing"):
        """
        Args:
            prompt: Instruction for how to transform the content
                   e.g. "Remove any personal information" or "Convert to bullet points"
            stage: When to apply the rule - either "post_parsing" (full document text) or
                  "post_chunking" (individual chunks). Defaults to "post_parsing" for backward compatibility.
        """
        self.prompt = prompt
        self.stage = stage

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "natural_language", "prompt": self.prompt, "stage": self.stage}


__all__ = ["Rule", "MetadataExtractionRule", "NaturalLanguageRule"]
