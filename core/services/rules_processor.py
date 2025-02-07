from typing import Dict, Any, List, Tuple
import logging
from core.models.rules import BaseRule, MetadataExtractionRule, NaturalLanguageRule
from core.completion.base_completion import BaseCompletionModel
from pydantic import BaseModel

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
