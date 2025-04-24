import logging
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from core.config import get_settings
from core.models.chunk import Chunk
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

    def _parse_rule(self, rule_dict: Dict[str, Any]) -> BaseRule:
        """Parse a rule dictionary into a rule object"""
        rule_type = rule_dict.get("type")
        stage = rule_dict.get("stage")

        if not stage:
            # Handle missing stage - default to post_parsing for backward compatibility
            logger.warning(f"Rule is missing 'stage' field, defaulting to 'post_parsing': {rule_dict}")
            rule_dict["stage"] = "post_parsing"
            stage = "post_parsing"

        if stage not in ["post_parsing", "post_chunking"]:
            raise ValueError(f"Invalid stage '{stage}' in rule: {rule_dict}")

        logger.debug(f"Parsing rule of type: {rule_type}, stage: {stage}")

        if rule_type == "metadata_extraction":
            return MetadataExtractionRule(**rule_dict)
        elif rule_type == "natural_language":
            return NaturalLanguageRule(**rule_dict)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    async def process_document_rules(self, content: str, rules: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        """
        Process rules intended for the post-parsing stage (full document text).

        Args:
            content: The full document content (text)
            rules: The original list of rule dictionaries

        Returns:
            Tuple[Dict[str, Any], str]: (extracted_metadata, modified_content)
        """
        logger.debug(f"Processing document-level rules (post_parsing) on content length {len(content)}")
        document_metadata = {}
        modified_content = content
        parsed_rules = []

        # Process rules for post_parsing stage
        for rule_dict in rules:
            try:
                # Let _parse_rule handle stage defaulting
                rule = self._parse_rule(rule_dict)

                # Only include rules for post_parsing stage
                if rule.stage == "post_parsing":
                    parsed_rules.append(rule)
            except ValueError as e:
                logger.warning(f"Skipping invalid document rule: {e}")
                continue  # Skip this rule

        logger.debug(f"Applying {len(parsed_rules)} post_parsing rules.")
        for i, rule in enumerate(parsed_rules, 1):
            try:
                logger.debug(f"Applying document rule {i}/{len(parsed_rules)}: {rule.type}")
                # Pass document metadata accumulated so far
                rule_metadata, modified_content = await rule.apply(modified_content, document_metadata)
                logger.debug(f"Rule {i} extracted metadata: {rule_metadata}")
                document_metadata.update(rule_metadata)  # Accumulate metadata
            except Exception as e:
                logger.error(f"Failed to apply document rule {rule.type}: {str(e)}")
                # Decide: stop or continue? Let's continue for robustness.
                continue

        logger.debug(f"Finished post_parsing rules. Final metadata: {document_metadata}")
        return document_metadata, modified_content

    async def process_chunk_rules(self, chunk: Chunk, rules: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Chunk]:
        """
        Process rules intended for the post-chunking stage on a single chunk.
        Modifies the chunk's content if necessary and returns any extracted metadata.

        Args:
            chunk: The Chunk object to process
            rules: The original list of rule dictionaries

        Returns:
            Tuple[Dict[str, Any], Chunk]: (extracted_metadata_for_doc, potentially_modified_chunk)
        """
        logger.debug(f"Processing chunk-level rules (post_chunking) on chunk content length {len(chunk.content)}")
        parsed_rules = []

        # Process rules for post_chunking stage
        for rule_dict in rules:
            try:
                # Let _parse_rule handle stage defaulting
                rule = self._parse_rule(rule_dict)

                # Only include rules for post_chunking stage
                if rule.stage == "post_chunking":
                    parsed_rules.append(rule)
            except ValueError as e:
                logger.warning(f"Skipping invalid chunk rule: {e}")
                continue

        if not parsed_rules:
            return {}, chunk  # No applicable rules, return empty metadata and original chunk

        logger.debug(f"Applying {len(parsed_rules)} post_chunking rules to chunk.")
        modified_content = chunk.content
        # Metadata extracted by rules in *this chunk* to be aggregated at the document level
        aggregated_chunk_rule_metadata = {}

        for i, rule in enumerate(parsed_rules, 1):
            try:
                logger.debug(f"Applying chunk rule {i}/{len(parsed_rules)}: {rule.type}")
                # Pass original chunk metadata (if any) for context, but don't modify it here
                rule_metadata, rule_modified_content = await rule.apply(modified_content, chunk.metadata)

                # If it's a metadata rule, aggregate its findings
                if isinstance(rule, MetadataExtractionRule):
                    logger.debug(f"Rule {i} (Metadata) extracted metadata: {rule_metadata}")
                    # Aggregate metadata - simple update, last one wins for a key
                    aggregated_chunk_rule_metadata.update(rule_metadata)

                # If it's an NL rule, update the content for the next rule in this chunk
                elif isinstance(rule, NaturalLanguageRule):
                    logger.debug(f"Rule {i} (NL) modified content.")
                    modified_content = rule_modified_content

            except Exception as e:
                logger.error(f"Failed to apply chunk rule {rule.type}: {str(e)}")
                continue

        # Update the chunk's content if it was modified by NL rules
        # Note: We are NOT updating chunk.metadata here with aggregated_chunk_rule_metadata
        chunk.content = modified_content

        logger.debug(f"Finished post_chunking rules for chunk. Metadata to aggregate: {aggregated_chunk_rule_metadata}")
        # Return the aggregated metadata from this chunk and the potentially modified chunk
        return aggregated_chunk_rule_metadata, chunk
