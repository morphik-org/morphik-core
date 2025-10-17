from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field


class EntityExtractionExample(BaseModel):
    """
    Example entity for guiding entity extraction.

    Used to provide domain-specific examples to the LLM of what entities to extract.
    These examples help steer the extraction process toward entities relevant to your domain.
    """

    label: str = Field(..., description="The entity label (e.g., 'John Doe', 'Apple Inc.')")
    type: str = Field(..., description="The entity type (e.g., 'PERSON', 'ORGANIZATION', 'PRODUCT')")
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
    variants: List[str] = Field(..., description="List of variant forms that should resolve to the canonical form")


class EntityExtractionPromptOverride(BaseModel):
    """
    Configuration for customizing entity extraction prompts.

    This allows you to override both the prompt template used for entity extraction
    and provide domain-specific examples of entities to be extracted.

    If only examples are provided (without a prompt_template), they will be
    incorporated into the default prompt. If only prompt_template is provided,
    it will be used with default examples (if any).

    Required placeholders:
    - {content}: Will be replaced with the text to analyze for entity extraction
    - {examples}: Will be replaced with formatted examples of entities to extract

    Example prompt template:
    ```
    Extract entities from the following text. Look for entities similar to these examples:

    {examples}

    Text to analyze:
    {content}

    Extracted entities (in JSON format):
    ```
    """

    prompt_template: Optional[str] = Field(
        None,
        description="Custom prompt template, MUST include both {content} and {examples} placeholders. "
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

    Required placeholders:
    - {entities_str}: Will be replaced with the extracted entities
    - {examples_json}: Will be replaced with JSON-formatted examples of entity resolution groups

    Example prompt template:
    ```
    I have extracted the following entities:

    {entities_str}

    Below are examples of how different entity references can be grouped together:

    {examples_json}

    Group the above entities by resolving which mentions refer to the same entity.
    Return the results in JSON format.
    ```
    """

    prompt_template: Optional[str] = Field(
        None,
        description="Custom prompt template that MUST include both {entities_str} and {examples_json} placeholders. "
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

    Required placeholders:
    - {question}: Will be replaced with the user's query
    - {context}: Will be replaced with the retrieved content/context

    Example prompt template:
    ```
    Answer the following question based on the provided information.

    Question: {question}

    Context:
    {context}

    Answer:
    ```
    """

    prompt_template: Optional[str] = Field(
        None,
        description="Custom prompt template for generating responses to queries. "
        "REQUIRED PLACEHOLDERS: {question} and {context} must be included in the template. "
        "The {question} placeholder will be replaced with the user query, and "
        "{context} will be replaced with the retrieved content. "
        "Use this to control response style, format, and tone.",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt that replaces Morphik's default query agent instructions. "
        "Use this to fully control the assistant's behavior when generating responses.",
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

    model_config = {"extra": "forbid"}  # This will cause validation error for extra fields

    entity_extraction: Optional[EntityExtractionPromptOverride] = Field(
        None,
        description="Overrides for entity extraction prompts - controls how entities are identified in text during graph operations",
    )
    entity_resolution: Optional[EntityResolutionPromptOverride] = Field(
        None,
        description="Overrides for entity resolution prompts - controls how variant forms are grouped during graph operations",
    )


def validate_prompt_template_placeholders(prompt_type: str, template: str) -> None:
    """
    Validate that a prompt template contains all required placeholders.

    Args:
        prompt_type: The type of prompt ("query", "entity_extraction", or "entity_resolution")
        template: The prompt template to validate

    Raises:
        ValueError: If any required placeholders are missing
    """
    if not template:
        return

    if prompt_type == "query":
        required = ["{question}", "{context}"]
    elif prompt_type == "entity_extraction":
        required = ["{content}", "{examples}"]
    elif prompt_type == "entity_resolution":
        required = ["{entities_str}", "{examples_json}"]
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    missing = [p for p in required if p not in template]
    if missing:
        raise ValueError(f"Required placeholders {missing} are missing from {prompt_type} prompt template")


def validate_prompt_overrides(prompt_overrides):
    """
    Validate that all prompt templates in the prompt_overrides have the required placeholders.

    This function is meant to be called from API endpoints to provide better error messages
    for incorrectly formatted prompt templates.

    Args:
        prompt_overrides: The prompt overrides object (can be of type QueryPromptOverrides or GraphPromptOverrides)
                         or a dictionary representation

    Raises:
        ValueError: If any required placeholders are missing from any templates or if invalid fields are present
    """
    if not prompt_overrides:
        return

    # First, validate field names
    # This handles dictionary inputs that haven't been validated by Pydantic models yet
    if isinstance(prompt_overrides, dict):
        # Determine allowed fields based on whether 'query' is one of expected fields
        # If GraphPromptOverrides: only entity_extraction and entity_resolution are allowed
        # If QueryPromptOverrides: entity_extraction, entity_resolution, and query are allowed
        is_graph_context = "query" not in prompt_overrides and any(
            key in {"entity_extraction", "entity_resolution"} for key in prompt_overrides
        )

        allowed_fields = {"entity_extraction", "entity_resolution"}
        if not is_graph_context:
            allowed_fields.add("query")

        # Check for invalid fields
        for field in prompt_overrides:
            if field not in allowed_fields:
                context_type = "graph" if is_graph_context else "query"
                raise ValueError(f"Field '{field}' is not allowed in {context_type} prompt overrides")

        # Validate query prompt template if present
        if "query" in prompt_overrides and prompt_overrides["query"] and "prompt_template" in prompt_overrides["query"]:
            validate_prompt_template_placeholders("query", prompt_overrides["query"]["prompt_template"])

        # Validate entity_extraction prompt template if present
        if (
            "entity_extraction" in prompt_overrides
            and prompt_overrides["entity_extraction"]
            and "prompt_template" in prompt_overrides["entity_extraction"]
        ):
            validate_prompt_template_placeholders(
                "entity_extraction", prompt_overrides["entity_extraction"]["prompt_template"]
            )

        # Validate entity_resolution prompt template if present
        if (
            "entity_resolution" in prompt_overrides
            and prompt_overrides["entity_resolution"]
            and "prompt_template" in prompt_overrides["entity_resolution"]
        ):
            validate_prompt_template_placeholders(
                "entity_resolution", prompt_overrides["entity_resolution"]["prompt_template"]
            )

    else:
        # Object is a model instance, validate with attribute access
        # Validate query prompt template if present
        if hasattr(prompt_overrides, "query") and prompt_overrides.query and prompt_overrides.query.prompt_template:
            validate_prompt_template_placeholders("query", prompt_overrides.query.prompt_template)

        # Validate entity_extraction prompt template if present
        if (
            hasattr(prompt_overrides, "entity_extraction")
            and prompt_overrides.entity_extraction
            and prompt_overrides.entity_extraction.prompt_template
        ):
            validate_prompt_template_placeholders(
                "entity_extraction", prompt_overrides.entity_extraction.prompt_template
            )

        # Validate entity_resolution prompt template if present
        if (
            hasattr(prompt_overrides, "entity_resolution")
            and prompt_overrides.entity_resolution
            and prompt_overrides.entity_resolution.prompt_template
        ):
            validate_prompt_template_placeholders(
                "entity_resolution", prompt_overrides.entity_resolution.prompt_template
            )


class QueryPromptOverrides(BaseModel):
    """
    Container for query-related prompt overrides.

    Use this class when customizing prompts for query operations, which may
    include customizations for entity extraction, entity resolution, and
    the query/response generation itself.

    This is the most feature-complete override class, supporting all customization types.

    Available customizations:
    - entity_extraction: Customize how entities are identified in text
    - entity_resolution: Customize how entity variants are grouped
    - query: Customize response generation style, format, and tone

    Each type has its own required placeholders. See the specific class documentation
    for details and examples.
    """

    model_config = {"extra": "forbid"}  # This will cause validation error for extra fields

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


def validate_prompt_overrides_with_http_exception(
    prompt_overrides=None, operation_type: str = None, error: ValueError = None
):
    """
    Validate prompt overrides and raise appropriate HTTP exceptions if validation fails.

    This function centralizes validation and error handling for prompt overrides in API endpoints.
    It can be used in two ways:
    1. For proactive validation: Pass prompt_overrides and it will validate and raise exceptions if needed
    2. For error handling: Pass an existing error (e) to properly format and raise the HTTP exception

    Args:
        prompt_overrides: The prompt overrides object to validate
        operation_type: Type of operation (e.g., "query", "graph") to customize error messages
        error: An existing ValueError to handle (used for exception handling blocks)

    Raises:
        HTTPException: With appropriate status code and detail message if validation fails
    """
    # If we're handling an existing error
    if error:
        e = error
        error_msg = str(e).lower()
    # If we're doing validation
    elif prompt_overrides:
        try:
            validate_prompt_overrides(prompt_overrides)
            return  # Validation passed, return without raising exception
        except ValueError as validation_error:
            e = validation_error
            error_msg = str(e).lower()
    else:
        return  # Nothing to validate or handle

    # Handle field validation errors
    if (
        ("not allowed in" in error_msg and "prompt overrides" in error_msg)
        or "extra fields not permitted" in error_msg
        or "field is not allowed" in error_msg
    ):

        # Customize message based on operation type
        if operation_type == "query":
            detail = (
                f"Invalid field in query prompt overrides: {str(e)}. "
                f"For query operations, valid fields are 'entity_extraction', 'entity_resolution', and 'query'."
            )
        elif operation_type == "graph":
            detail = (
                f"Invalid field in graph prompt overrides: {str(e)}. "
                f"For graph operations, only 'entity_extraction' and 'entity_resolution' overrides are supported."
            )
        else:
            detail = f"Invalid field in prompt overrides: {str(e)}."

        raise HTTPException(status_code=422, detail=detail)

    # Handle placeholder validation errors
    elif "required placeholders" in error_msg and "are missing" in error_msg:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prompt override: {str(e)}. Make sure all required placeholders are included in your prompt templates.",
        )

    # Default error handling
    raise HTTPException(status_code=400, detail=str(e))
