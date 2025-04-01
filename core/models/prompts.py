"""
Models for customizing prompts used in entity extraction, entity resolution, and query generation.

These models provide a structured way to override default prompts used by the system,
enabling customization for domain-specific applications without modifying core code.

## Available Customization Points

### Entity Extraction
Customizes how entities are extracted from text:
- **Prompt Template**: Replace the default prompt with your domain-specific instructions
- **Examples**: Provide examples of entities specific to your domain

### Entity Resolution
Customizes how variants of the same entity are grouped together:
- **Prompt Template**: Adjust how the LLM resolves variants of the same entity
- **Examples**: Provide domain-specific examples of canonical forms and their variants

### Query Generation
Customizes how responses are generated:
- **Prompt Template**: Adjust the prompt used for generating responses to queries

## Placeholder Variables

Each prompt template supports specific placeholder variables:

- **Entity Extraction**:
  - `{content}`: The text from which entities will be extracted
  - `{examples}`: Formatted examples of entities to extract

- **Entity Resolution**:
  - `{entities_str}`: String representation of extracted entities
  - `{examples_json}`: JSON representation of entity resolution examples

- **Query**:
  - System-specific placeholders for query context, chunks, etc.

## Usage Examples

1. Customizing entity extraction for a scientific domain:
```python
from core.models.prompts import EntityExtractionPromptOverride, EntityExtractionExample, GraphPromptOverrides

# Define the override for entity extraction
extraction_override = EntityExtractionPromptOverride(
    prompt_template="Extract scientific entities from the following text: {content}\\n{examples}",
    examples=[
        EntityExtractionExample(label="CRISPR-Cas9", type="TECHNOLOGY"),
        EntityExtractionExample(label="Alzheimer's", type="DISEASE")
    ]
)

# Option 1: Using the typed container model (recommended)
prompt_overrides = GraphPromptOverrides(entity_extraction=extraction_override)

# Option 2: Using a dictionary (also supported)
prompt_overrides_dict = {"entity_extraction": extraction_override}
```

2. Customizing entity resolution for a legal domain:
```python
from core.models.prompts import EntityResolutionPromptOverride, EntityResolutionExample, GraphPromptOverrides

# Define the override for entity resolution
resolution_override = EntityResolutionPromptOverride(
    examples=[
        EntityResolutionExample(
            canonical="Supreme Court of the United States", 
            variants=["SCOTUS", "Supreme Court", "US Supreme Court"]
        )
    ]
)

# Create a complete override object
prompt_overrides = GraphPromptOverrides(
    entity_resolution=resolution_override
)
```

3. Combining multiple overrides for a query operation:
```python
from core.models.prompts import (
    EntityExtractionPromptOverride, 
    EntityResolutionPromptOverride,
    QueryPromptOverride,
    QueryPromptOverrides
)

# Create a comprehensive override for a query operation
query_overrides = QueryPromptOverrides(
    entity_extraction=EntityExtractionPromptOverride(...),
    entity_resolution=EntityResolutionPromptOverride(...),
    query=QueryPromptOverride(
        prompt_template="Using the provided information, answer the question in the style of a professor: {question}"
    )
)
```

## Usage Context

- For graph operations (create_graph, update_graph), use GraphPromptOverrides, which only accepts 
  entity_extraction and entity_resolution customizations.

- For query operations, use QueryPromptOverrides, which additionally accepts query prompt
  customizations.

Both model types provide validation to ensure only appropriate overrides are used in each context.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator


class EntityExtractionExample(BaseModel):
    """
    Example entity for guiding entity extraction.

    Used to provide domain-specific examples to the LLM of what entities to extract.
    These examples help steer the extraction process toward entities relevant to your domain.
    """

    label: str = Field(..., description="The entity label (e.g., 'John Doe', 'Apple Inc.')")
    type: str = Field(
        ..., description="The entity type (e.g., 'PERSON', 'ORGANIZATION', 'PRODUCT')"
    )
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional properties of the entity (e.g., {'role': 'CEO', 'age': 42})",
    )


class EntityResolutionExample(BaseModel):
    """
    Example for entity resolution, showing how variants should be grouped.

    Entity resolution is the process of identifying when different references
    (variants) in text refer to the same real-world entity. These examples
    help the LLM understand domain-specific patterns for resolving entities.
    """

    canonical: str = Field(..., description="The canonical (standard/preferred) form of the entity")
    variants: List[str] = Field(
        ..., description="List of variant forms that should resolve to the canonical form"
    )


class EntityExtractionPromptOverride(BaseModel):
    """
    Configuration for customizing entity extraction prompts.

    This allows you to override both the prompt template used for entity extraction
    and provide domain-specific examples of entities to be extracted.

    If only examples are provided (without a prompt_template), they will be
    incorporated into the default prompt. If only prompt_template is provided,
    it will be used with default examples (if any).
    """

    prompt_template: Optional[str] = Field(
        None,
        description="Custom prompt template, supports {content} and {examples} placeholders. "
        "The {content} placeholder will be replaced with the text to analyze, and "
        "{examples} will be replaced with formatted examples.",
    )
    examples: Optional[List[EntityExtractionExample]] = Field(
        None,
        description="Examples of entities to extract, used to guide the LLM toward "
        "domain-specific entity types and patterns.",
    )


class EntityResolutionPromptOverride(BaseModel):
    """
    Configuration for customizing entity resolution prompts.

    Entity resolution identifies and groups variant forms of the same entity.
    This override allows you to customize how this process works by providing
    a custom prompt template and/or domain-specific examples.

    If only examples are provided (without a prompt_template), they will be
    incorporated into the default prompt. If only prompt_template is provided,
    it will be used with default examples (if any).
    """

    prompt_template: Optional[str] = Field(
        None,
        description="Custom prompt template that supports {entities_str} and {examples_json} placeholders. "
        "The {entities_str} placeholder will be replaced with the extracted entities, and "
        "{examples_json} will be replaced with JSON-formatted examples of entity resolution groups.",
    )
    examples: Optional[List[EntityResolutionExample]] = Field(
        None,
        description="Examples of entity resolution groups showing how variants of the same entity "
        "should be resolved to their canonical forms. This is particularly useful for "
        "domain-specific terminology, abbreviations, and naming conventions.",
    )


class QueryPromptOverride(BaseModel):
    """
    Configuration for customizing query prompts.

    This allows you to customize how responses are generated during query operations.
    Query prompts guide the LLM on how to format and style responses, what tone to use,
    and how to incorporate retrieved information into the response.
    """

    prompt_template: Optional[str] = Field(
        None,
        description="Custom prompt template for generating responses to queries. "
        "The exact placeholders available depend on the query context, but "
        "typically include {question}, {context}, and other system-specific variables. "
        "Use this to control response style, format, and tone.",
    )


class PromptOverrides(BaseModel):
    """
    Generic container for all prompt overrides.

    This is a base class that contains all possible override types.
    For specific operations, use the more specialized GraphPromptOverrides
    or QueryPromptOverrides classes, which enforce context-specific validation.
    """

    entity_extraction: Optional[EntityExtractionPromptOverride] = Field(
        None,
        description="Overrides for entity extraction prompts - controls how entities are identified in text",
    )
    entity_resolution: Optional[EntityResolutionPromptOverride] = Field(
        None,
        description="Overrides for entity resolution prompts - controls how variant forms are grouped",
    )
    query: Optional[QueryPromptOverride] = Field(
        None,
        description="Overrides for query prompts - controls response generation style and format",
    )


class GraphPromptOverrides(BaseModel):
    """
    Container for graph-related prompt overrides.

    Use this class when customizing prompts for graph operations like
    create_graph() and update_graph(), which only support entity extraction
    and entity resolution customizations.

    This class enforces that only graph-relevant override types are used.
    """

    entity_extraction: Optional[EntityExtractionPromptOverride] = Field(
        None,
        description="Overrides for entity extraction prompts - controls how entities are identified in text during graph operations",
    )
    entity_resolution: Optional[EntityResolutionPromptOverride] = Field(
        None,
        description="Overrides for entity resolution prompts - controls how variant forms are grouped during graph operations",
    )

    @model_validator(mode="after")
    def validate_graph_fields(self) -> "GraphPromptOverrides":
        """Ensure only graph-related fields are present."""
        allowed_fields = {"entity_extraction", "entity_resolution"}
        for field in self.model_fields:
            if field not in allowed_fields and getattr(self, field, None) is not None:
                raise ValueError(f"Field '{field}' is not allowed in graph prompt overrides")
        return self


class QueryPromptOverrides(BaseModel):
    """
    Container for query-related prompt overrides.

    Use this class when customizing prompts for query operations, which may
    include customizations for entity extraction, entity resolution, and
    the query/response generation itself.

    This is the most feature-complete override class, supporting all customization types.
    """

    entity_extraction: Optional[EntityExtractionPromptOverride] = Field(
        None,
        description="Overrides for entity extraction prompts - controls how entities are identified in text during queries",
    )
    entity_resolution: Optional[EntityResolutionPromptOverride] = Field(
        None,
        description="Overrides for entity resolution prompts - controls how variant forms are grouped during queries",
    )
    query: Optional[QueryPromptOverride] = Field(
        None,
        description="Overrides for query prompts - controls response generation style, format, and tone",
    )
