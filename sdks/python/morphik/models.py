from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


class StorageFileInfo(BaseModel):
    """Information about a file stored in storage."""

    bucket: str
    key: str
    version: int = Field(default=1, description="Incremented on each file replacement")
    filename: Optional[str] = None
    content_type: Optional[str] = None
    timestamp: Optional[datetime] = None


class Document(BaseModel):
    """Document metadata model"""

    external_id: str = Field(..., description="Unique document identifier")
    content_type: str = Field(..., description="Content type of the document")
    filename: Optional[str] = Field(None, description="Original filename if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
    metadata_types: Dict[str, str] = Field(default_factory=dict, description="Per-field metadata type hints")
    storage_info: Dict[str, str] = Field(default_factory=dict, description="Storage-related information")
    system_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="System-managed metadata (status, progress, timestamps)",
    )
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Ingestion-generated metadata")
    storage_files: List[StorageFileInfo] = Field(default_factory=list, description="Files associated with the document")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of document chunks")
    folder_name: Optional[str] = Field(None, description="Folder scope for the document")
    end_user_id: Optional[str] = Field(None, description="End-user scope for the document")
    app_id: Optional[str] = Field(None, description="App identifier for the document")

    # Client reference for update methods
    _client = None

    @property
    def status(self) -> Dict[str, Any]:
        """Get the latest processing status of the document from the API.

        Returns:
            Dict[str, Any]: Status information including current status, potential errors, and other metadata
        """
        if self._client is None:
            raise ValueError(
                "Document instance not connected to a client. Use a document returned from a Morphik client method."
            )
        return self._client.get_document_status(self.external_id)

    @property
    def is_processing(self) -> bool:
        """Check if the document is still being processed."""
        return self.status.get("status") == "processing"

    @property
    def is_ingested(self) -> bool:
        """Check if the document has completed processing."""
        return self.status.get("status") == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if document processing has failed."""
        return self.status.get("status") == "failed"

    @property
    def error(self) -> Optional[str]:
        """Get the error message if processing failed."""
        status_info = self.status
        return status_info.get("error") if status_info.get("status") == "failed" else None

    def wait_for_completion(self, timeout_seconds=300, check_interval_seconds=2, progress_callback=None):
        """Wait for document processing to complete.

        Args:
            timeout_seconds: Maximum time to wait for completion (default: 300 seconds)
            check_interval_seconds: Time between status checks (default: 2 seconds)
            progress_callback: Optional callback function that receives progress updates.
                               Called with (current_step, total_steps, step_name, percentage)

        Returns:
            Document: Updated document with the latest status

        Raises:
            TimeoutError: If processing doesn't complete within the timeout period
            ValueError: If processing fails with an error
        """
        if self._client is None:
            raise ValueError(
                "Document instance not connected to a client. Use a document returned from a Morphik client method."
            )
        return self._client.wait_for_document_completion(
            self.external_id, timeout_seconds, check_interval_seconds, progress_callback
        )

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

        return self._client.update_document_metadata(document_id=self.external_id, metadata=metadata)


class DocumentQueryResponse(BaseModel):
    """Response model for Morphik On-the-Fly document queries with optional ingestion follow-up."""

    structured_output: Optional[Any] = Field(
        default=None,
        description="Structured output returned by Morphik On-the-Fly when a schema is provided.",
    )
    extracted_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured output coerced to metadata when possible.",
    )
    text_output: Optional[str] = Field(
        default=None,
        description="Unstructured text output when no schema is enforced.",
    )
    ingestion_enqueued: bool = Field(
        default=False,
        description="True when the document was queued for ingestion after extraction.",
    )
    ingestion_document: Optional[Document] = Field(
        default=None,
        description="Queued document stub when ingestion_enqueued is true.",
    )
    input_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original metadata supplied alongside the request.",
    )
    combined_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata that would be used if ingestion is performed (original metadata merged with extracted fields when available).",
    )
    ingestion_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized ingestion options applied to this request.",
    )

    model_config = ConfigDict(extra="ignore")


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

    completion: Optional[Union[str, Dict[str, Any], None]] = Field(
        None, description="Generated text completion or structured output"
    )
    usage: Dict[str, int]
    sources: List[ChunkSource] = Field(default_factory=list, description="Sources of chunks used in the completion")
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = Field(None, description="Reason the generation finished (e.g., 'stop', 'length')")


class FolderCount(BaseModel):
    """Document count for a folder"""

    folder: Optional[str] = Field(None, description="Folder name (None for root)")
    count: int = Field(..., description="Number of documents in folder")


class ListDocsResponse(BaseModel):
    """Response model for list_documents with pagination and aggregates"""

    documents: List[Document] = Field(default_factory=list, description="List of documents")
    skip: int = Field(..., description="Pagination offset used")
    limit: int = Field(..., description="Limit used")
    returned_count: int = Field(..., description="Number of documents in this response")
    total_count: Optional[int] = Field(None, description="Total matching documents (if include_total_count=True)")
    has_more: bool = Field(False, description="Whether more documents exist beyond this page")
    next_skip: Optional[int] = Field(None, description="Skip value for next page")
    status_counts: Optional[Dict[str, int]] = Field(None, description="Document counts by status")
    folder_counts: Optional[List[FolderCount]] = Field(None, description="Document counts by folder")


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    model_config = ConfigDict(extra="allow")

    content: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    metadata_types: Optional[Dict[str, str]] = Field(default=None)
    rules: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        exclude=True,
        json_schema_extra={"deprecated": True},
    )
    use_colpali: bool = Field(default=False)


class Entity(BaseModel):
    """Represents an entity in a knowledge graph"""

    id: str = Field(..., description="Unique entity identifier")
    label: str = Field(..., description="Display label for the entity")
    type: str = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    document_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    chunk_sources: Dict[str, List[int]] = Field(default_factory=dict, description="Source chunk numbers by document ID")

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
    chunk_sources: Dict[str, List[int]] = Field(default_factory=dict, description="Source chunk numbers by document ID")

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
    relationships: List[Relationship] = Field(default_factory=list, description="Relationships in the graph")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")
    system_metadata: Dict[str, Any] = Field(default_factory=dict, description="System-managed metadata")
    document_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    filters: Optional[Dict[str, Any]] = Field(None, description="Document filters used to create the graph")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    _client: Any | None = PrivateAttr(default=None)

    # ---------------- Convenience helpers ----------------
    @property
    def status(self) -> str | None:
        """Return processing status if available."""
        return self.system_metadata.get("status") if self.system_metadata else None

    @property
    def is_processing(self) -> bool:
        return self.system_metadata.get("status") == "processing"

    @property
    def is_completed(self) -> bool:
        return self.system_metadata.get("status") == "completed"

    @property
    def is_failed(self) -> bool:
        return self.system_metadata.get("status") == "failed"

    @property
    def error(self) -> str | None:
        return self.system_metadata.get("error") if self.system_metadata else None

    def wait_for_completion(self, timeout_seconds: int = 300, check_interval_seconds: int = 2) -> "Graph":
        """Poll the server until the graph processing is finished."""
        import time

        if not self._client:
            raise RuntimeError("Graph object has no client reference for polling")

        start = time.time()
        while time.time() - start < timeout_seconds:
            refreshed = self._client.get_graph(self.name)
            if refreshed.is_completed:
                return refreshed
            if refreshed.is_failed:
                raise RuntimeError(refreshed.error or "Graph creation failed")
            time.sleep(check_interval_seconds)
        raise TimeoutError("Timed out waiting for graph completion")


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
        "REQUIRED PLACEHOLDERS: {question} and {context} must be included in the template. "
        "Use this to control response style, format, and tone.",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt that replaces Morphik's default query agent instructions.",
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
        description="Overrides for entity extraction prompts - controls how entities are identified in text "
        "during graph operations",
    )
    entity_resolution: Optional[EntityResolutionPromptOverride] = Field(
        None,
        description="Overrides for entity resolution prompts - controls how variant forms are grouped "
        "during graph operations",
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
        description="Overrides for entity extraction prompts - controls how entities are identified in text "
        "during queries",
    )
    entity_resolution: Optional[EntityResolutionPromptOverride] = Field(
        None,
        description="Overrides for entity resolution prompts - controls how variant forms are grouped during queries",
    )
    query: Optional[QueryPromptOverride] = Field(
        None,
        description="Overrides for query prompts - controls response generation style, format, and tone",
    )


class FolderInfo(BaseModel):
    """Folder metadata model"""

    id: str = Field(..., description="Unique folder identifier")
    name: str = Field(..., description="Folder name")
    description: Optional[str] = Field(None, description="Folder description")
    document_ids: Optional[List[str]] = Field(default_factory=list, description="IDs of documents in the folder")
    system_metadata: Dict[str, Any] = Field(default_factory=dict, description="System-managed metadata")
    rules: List[Dict[str, Any]] = Field(default_factory=list, description="Rules associated with the folder")
    app_id: Optional[str] = Field(None, description="Application ID associated with the folder")
    end_user_id: Optional[str] = Field(None, description="End user ID associated with the folder")


class DocumentPagesResponse(BaseModel):
    """Response for document pages extraction endpoint"""

    document_id: str = Field(..., description="ID of the document")
    pages: List[str] = Field(..., description="List of page contents as base64 encoded strings")
    start_page: int = Field(..., description="Start page number (1-indexed)")
    end_page: int = Field(..., description="End page number (1-indexed)")
    total_pages: int = Field(..., description="Total number of pages in the document")


class ChunkGroup(BaseModel):
    """Represents a group of chunks: one main match + its padding chunks"""

    main_chunk: ChunkResult = Field(..., description="The primary matched chunk")
    padding_chunks: List[ChunkResult] = Field(default_factory=list, description="Surrounding context chunks")
    total_chunks: int = Field(..., description="Total number of chunks in this group")


class GroupedChunkResponse(BaseModel):
    """Response that includes both flat results and grouped results for UI"""

    chunks: List[ChunkResult] = Field(..., description="Flat list of all chunks (for backward compatibility)")
    groups: List[ChunkGroup] = Field(..., description="Grouped chunks for UI display")
    total_results: int = Field(..., description="Total number of unique chunks")
    has_padding: bool = Field(..., description="Whether padding was applied to any results")


class FolderSummary(BaseModel):
    """Summary information for a folder"""

    id: str = Field(..., description="Unique folder identifier")
    name: str = Field(..., description="Folder name")
    description: Optional[str] = Field(None, description="Folder description")
    doc_count: int = Field(default=0, description="Number of documents in folder")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class FolderDocumentInfo(BaseModel):
    """Document count and status information for a folder"""

    total_count: Optional[int] = Field(None, description="Total document count")
    status_counts: Optional[Dict[str, int]] = Field(None, description="Document counts by status")
    documents: Optional[List[Document]] = Field(None, description="Paginated list of documents")


class FolderDetails(BaseModel):
    """Folder details with optional document summary"""

    folder: FolderInfo = Field(..., description="Folder information")
    document_info: Optional[FolderDocumentInfo] = Field(None, description="Document statistics and list")


class FolderDetailsResponse(BaseModel):
    """Response wrapping folder detail entries"""

    folders: List[FolderDetails] = Field(..., description="List of folder details")
