import logging
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from core.config import get_settings
from core.models.rules import BaseRule, MetadataExtractionRule, NaturalLanguageRule

logger = logging.getLogger(__name__)
settings = get_settings()


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

    def __init__(self):
        logger.debug(
            f"Initializing RulesProcessor with {settings.RULES_PROVIDER} provider using model {settings.RULES_MODEL}"
        )

    async def process_rules(self, content: str, rules: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        """
        Process a list of rules on content.

        Args:
            content: The document content
            rules: List of rule dictionaries

        Returns:
            Tuple[Dict[str, Any], str]: (extracted_metadata, modified_content)
        """
        logger.debug(f"Processing {len(rules)} rules on content of length {len(content)}")
        metadata = {}
        modified_content = content

        try:
            # Parse all rules first to fail fast if any are invalid
            parsed_rules = [self._parse_rule(rule) for rule in rules]
            logger.debug(f"Successfully parsed {len(parsed_rules)} rules: {[r.type for r in parsed_rules]}")

            # Apply rules in order
            for i, rule in enumerate(parsed_rules, 1):
                try:
                    logger.debug(f"Applying rule {i}/{len(parsed_rules)}: {rule.type}")
                    rule_metadata, modified_content = await rule.apply(modified_content)
                    logger.debug(f"Rule {i} extracted metadata: {rule_metadata}")
                    metadata.update(rule_metadata)
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule.type}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Failed to process rules: {str(e)}")
            return metadata, content

        logger.debug(f"Completed processing {len(rules)} rules. Final metadata: {metadata}")
        return metadata, modified_content

    def _parse_rule(self, rule_dict: Dict[str, Any]) -> BaseRule:
        """Parse a rule dictionary into a rule object"""
        rule_type = rule_dict.get("type")
        logger.debug(f"Parsing rule of type: {rule_type}")

        if rule_type == "metadata_extraction":
            return MetadataExtractionRule(**rule_dict)
        elif rule_type == "natural_language":
            return NaturalLanguageRule(**rule_dict)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")
