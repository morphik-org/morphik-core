from typing import Dict, Any, List, Literal, Optional, Union, BinaryIO
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class Document(BaseModel):
    """Document metadata model"""

    external_id: str = Field(..., description="Unique document identifier")
    content_type: str = Field(..., description="Content type of the document")
    filename: Optional[str] = Field(None, description="Original filename if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
    storage_info: Dict[str, str] = Field(
        default_factory=dict, description="Storage-related information"
    )
    system_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="System-managed metadata"
    )
    access_control: Dict[str, Any] = Field(
        default_factory=dict, description="Access control information"
    )
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of document chunks")

    # Client reference for update methods
    _client = None

    def update_with_text(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> "Document":
        """
        Update this document with new text content using the specified strategy.

        Args:
            content: The new content to add
            filename: Optional new filename for the document
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata
        """
        if self._client is None:
            raise ValueError(
                "Document instance not connected to a client. Use a document returned from a Morphik client method."
            )

        return self._client.update_document_with_text(
            document_id=self.external_id,
            content=content,
            filename=filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

    def update_with_file(
        self,
        file: "Union[str, bytes, BinaryIO, Path]",
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rules: Optional[List] = None,
        update_strategy: str = "add",
        use_colpali: Optional[bool] = None,
    ) -> "Document":
        """
        Update this document with content from a file using the specified strategy.

        Args:
            file: File to add (path string, bytes, file object, or Path)
            filename: Name of the file
            metadata: Additional metadata to update (optional)
            rules: Optional list of rules to apply to the content
            update_strategy: Strategy for updating the document (currently only 'add' is supported)
            use_colpali: Whether to use multi-vector embedding

        Returns:
            Document: Updated document metadata
        """
        if self._client is None:
            raise ValueError(
                "Document instance not connected to a client. Use a document returned from a Morphik client method."
            )

        return self._client.update_document_with_file(
            document_id=self.external_id,
            file=file,
            filename=filename,
            metadata=metadata,
            rules=rules,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

    def update_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> "Document":
        """
        Update this document's metadata only.

        Args:
            metadata: Metadata to update

        Returns:
            Document: Updated document metadata
        """
        if self._client is None:
            raise ValueError(
                "Document instance not connected to a client. Use a document returned from a Morphik client method."
            )

        return self._client.update_document_metadata(
            document_id=self.external_id, metadata=metadata
        )


class ChunkResult(BaseModel):
    """Query result at chunk level"""

    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Parent document ID")
    chunk_number: int = Field(..., description="Chunk sequence number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    content_type: str = Field(..., description="Content type")
    filename: Optional[str] = Field(None, description="Original filename")
    download_url: Optional[str] = Field(None, description="URL to download full document")


class DocumentContent(BaseModel):
    """Represents either a URL or content string"""

    type: Literal["url", "string"] = Field(..., description="Content type (url or string)")
    value: str = Field(..., description="The actual content or URL")
    filename: Optional[str] = Field(None, description="Filename when type is url")

    @field_validator("filename")
    def filename_only_for_url(cls, v, values):
        if values.data.get("type") == "string" and v is not None:
            raise ValueError("filename can only be set when type is url")
        if values.data.get("type") == "url" and v is None:
            raise ValueError("filename is required when type is url")
        return v


class DocumentResult(BaseModel):
    """Query result at document level"""

    score: float = Field(..., description="Relevance score")
    document_id: str = Field(..., description="Document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    content: DocumentContent = Field(..., description="Document content or URL")


class ChunkSource(BaseModel):
    """Source information for a chunk used in completion"""

    document_id: str = Field(..., description="ID of the source document")
    chunk_number: int = Field(..., description="Chunk number within the document")
    score: Optional[float] = Field(None, description="Relevance score")


class CompletionResponse(BaseModel):
    """Completion response model"""

    completion: str
    usage: Dict[str, int]
    sources: List[ChunkSource] = Field(
        default_factory=list, description="Sources of chunks used in the completion"
    )
    metadata: Optional[Dict[str, Any]] = None


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    content: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    use_colpali: bool = Field(default=False)


class Entity(BaseModel):
    """Represents an entity in a knowledge graph"""

    id: str = Field(..., description="Unique entity identifier")
    label: str = Field(..., description="Display label for the entity")
    type: str = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    document_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    chunk_sources: Dict[str, List[int]] = Field(
        default_factory=dict, description="Source chunk numbers by document ID"
    )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


class Relationship(BaseModel):
    """Represents a relationship between entities in a knowledge graph"""

    id: str = Field(..., description="Unique relationship identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    document_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    chunk_sources: Dict[str, List[int]] = Field(
        default_factory=dict, description="Source chunk numbers by document ID"
    )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return self.id == other.id


class Graph(BaseModel):
    """Represents a knowledge graph"""

    id: str = Field(..., description="Unique graph identifier")
    name: str = Field(..., description="Graph name")
    entities: List[Entity] = Field(default_factory=list, description="Entities in the graph")
    relationships: List[Relationship] = Field(
        default_factory=list, description="Relationships in the graph"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")
    document_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Document filters used to create the graph"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    owner: Dict[str, str] = Field(default_factory=dict, description="Graph owner information")
    access_control: Dict[str, List[str]] = Field(
        default_factory=dict, description="Access control information"
    )


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
