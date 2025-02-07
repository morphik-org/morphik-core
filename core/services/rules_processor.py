from typing import Dict, Any, List, Tuple
import logging
from core.models.rules import BaseRule, MetadataExtractionRule, NaturalLanguageRule
from core.completion.base_completion import BaseCompletionModel
from pydantic import BaseModel
from core.completion.ollama_completion import OllamaCompletionModel
from core.completion.openai_completion import OpenAICompletionModel
from core.config import get_settings
from core.completion.base_completion import CompletionRequest
import json

logger = logging.getLogger(__name__)


class RuleResponse(BaseModel):
    """Schema for rule processing responses."""

    metadata: Dict[str, Any] = {}  # Optional metadata extracted from text
    modified_text: str  # The actual modified text - REQUIRED

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    # Example for PII extraction
                    "metadata": {},
                    "modified_text": "Original text with PII removed",
                },
                {
                    # Example for text shortening
                    "metadata": {},
                    "modified_text": "key1, key2, key3, important_concept",
                },
            ]
        }
    }


class RulesProcessor:
    """Processes rules during document ingestion"""

    def __init__(self, completion_model: BaseCompletionModel):
        self.completion_model = completion_model

    async def process_rules(
        self, content: str, rules: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Process a list of rules on content.

        Args:
            content: The document content
            rules: List of rule dictionaries

        Returns:
            Tuple[Dict[str, Any], str]: (extracted_metadata, modified_content)
        """
        metadata = {}
        modified_content = content

        try:
            # Parse all rules first to fail fast if any are invalid
            parsed_rules = [self._parse_rule(rule) for rule in rules]

            # Apply rules in order
            for rule in parsed_rules:
                try:
                    rule_metadata, modified_content = await rule.apply(
                        modified_content, self.completion_model
                    )
                    metadata.update(rule_metadata)
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule}: {str(e)}")
                    # Continue with next rule on failure
                    continue

        except Exception as e:
            logger.error(f"Failed to process rules: {str(e)}")
            return metadata, content

        return metadata, modified_content

    def _parse_rule(self, rule_dict: Dict[str, Any]) -> BaseRule:
        """Parse a rule dictionary into a rule object"""
        rule_type = rule_dict.get("type")

        if rule_type == "metadata_extraction":
            return MetadataExtractionRule(**rule_dict)
        elif rule_type == "natural_language":
            return NaturalLanguageRule(**rule_dict)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    async def _apply_natural_language_rule(self, rule: NaturalLanguageRule, content: str) -> str:
        """Apply a natural language rule to transform content"""
        prompt = f"""
        Your task is to transform the following text according to this instruction:
        {rule.prompt}
        
        Text to transform:
        {content}
        
        Transformed text:
        """

        request = CompletionRequest(
            query=prompt,
            context_chunks=[],  # No context needed for transformation
            max_tokens=None,  # Let model decide
            temperature=0.0,  # Deterministic
        )

        response = await self.completion_model.complete(request)
        return response.completion.strip()

    async def _apply_metadata_extraction_rule(
        self, rule: MetadataExtractionRule, content: str
    ) -> Dict[str, Any]:
        """Apply a metadata extraction rule to extract metadata"""
        schema_str = json.dumps(rule.schema, indent=2)

        prompt = f"""
        Your task is to extract metadata from the following text according to this schema:
        {schema_str}
        
        Text to extract from:
        {content}
        
        Return ONLY a valid JSON object matching the schema. Nothing else.
        """

        request = CompletionRequest(
            query=prompt,
            context_chunks=[],  # No context needed for extraction
            max_tokens=None,  # Let model decide
            temperature=0.0,  # Deterministic
        )

        response = await self.completion_model.complete(request)
        try:
            return json.loads(response.completion.strip())
        except json.JSONDecodeError:
            raise ValueError("Failed to extract valid metadata")
