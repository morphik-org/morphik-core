from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models.completion import ChunkSource
from core.models.documents import Document
from core.models.prompts import GraphPromptOverrides, QueryPromptOverrides


class ListDocumentsRequest(BaseModel):
    """Request model for listing documents"""

    document_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filters with operator support: $and, $or, $nor, $not, $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $type, $regex, $contains. Implicit equality uses JSONB containment; explicit operators support typed comparisons.",
    )
    skip: int = Field(default=0, ge=0, description="Number of documents to skip before returning results.")
    limit: int = Field(default=1000, gt=0, description="Maximum number of documents to return.")


class ListDocsRequest(BaseModel):
    """Flexible request model for listing documents with projection and aggregates."""

    document_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filters with operator support: $and, $or, $nor, $not, $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $type, $regex, $contains. Implicit equality uses JSONB containment; explicit operators support typed comparisons.",
    )
    skip: int = Field(default=0, ge=0, description="Number of documents to skip")
    limit: int = Field(default=100, ge=0, description="Maximum number of documents to return")
    return_documents: bool = Field(default=True, description="When false, only aggregates are returned")
    include_total_count: bool = Field(default=False, description="Include total number of matching documents when true")
    include_status_counts: bool = Field(
        default=False, description="Include document counts grouped by processing status when true"
    )
    include_folder_counts: bool = Field(
        default=False, description="Include document counts grouped by folder when true"
    )
    completed_only: bool = Field(
        default=False,
        description="When true, only documents with completed processing status are returned and counted",
    )
    sort_by: Optional[Literal["created_at", "updated_at", "filename", "external_id"]] = Field(
        default="updated_at", description="Field to sort the results by"
    )
    sort_direction: Literal["asc", "desc"] = Field(default="desc", description="Sort direction for the results")
    fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of fields to project for each document (dot notation supported). "
            "Derived fields such as 'page_count' are also supported."
        ),
    )


class FolderDetailsRequest(BaseModel):
    """Request model for retrieving folder details with document statistics."""

    identifiers: Optional[List[str]] = Field(
        default=None,
        description="List of folder IDs or names. If omitted, returns details for all accessible folders.",
    )
    include_document_count: bool = Field(default=True, description="Include total document count when true")
    include_status_counts: bool = Field(
        default=False, description="Include document counts grouped by status when true"
    )
    include_documents: bool = Field(
        default=False, description="Include a paginated list of documents for each folder when true"
    )
    document_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters applied when computing folder document statistics",
    )
    document_skip: int = Field(
        default=0,
        ge=0,
        description="Number of documents to skip within each folder when include_documents is true",
    )
    document_limit: int = Field(
        default=25,
        ge=0,
        description="Maximum number of documents to return per folder when include_documents is true",
    )
    document_fields: Optional[List[str]] = Field(
        default=None,
        description="Optional list of fields to project for folder documents (dot notation supported)",
    )
    sort_by: Optional[Literal["created_at", "updated_at", "filename", "external_id"]] = Field(
        default="updated_at", description="Field to sort folder documents by when include_documents is true"
    )
    sort_direction: Literal["asc", "desc"] = Field(
        default="desc", description="Sort direction for folder documents when include_documents is true"
    )


class SearchDocumentsRequest(BaseModel):
    """Request model for searching documents by name"""

    query: str = Field(..., min_length=1, description="Search query for document names/filenames")
    limit: int = Field(default=10, ge=1, le=100, description="Number of documents to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters for documents")
    folder_name: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional folder scope for the search. Accepts a single folder name or a list of folder names.",
    )
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the search")


class RetrieveRequest(BaseModel):
    """Base retrieve request model"""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language query used to retrieve relevant chunks or documents.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Metadata filters supporting logical operators ($and/$or/$not/$nor) "
            "and field predicates ($eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists/$type/$regex/$contains)."
        ),
    )
    k: int = Field(
        default=4,
        gt=0,
        description="Maximum number of chunks or documents to return.",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum similarity score a result must meet before it is returned.",
    )
    use_reranking: Optional[bool] = Field(
        default=None,
        description="When provided, overrides the workspace reranking configuration for this request.",
    )
    use_colpali: Optional[bool] = Field(
        default=None,
        description="When provided, uses Morphik's finetuned ColPali style embeddings (recommended to be True for high quality retrieval).",
    )
    output_format: Optional[Literal["base64", "url"]] = Field(
        default="base64",
        description="How to return image chunks: base64 data URI (default) or a presigned URL",
    )
    padding: int = Field(
        default=0,
        ge=0,
        description="Number of additional chunks/pages to retrieve before and after matched chunks (ColPali only)",
    )
    graph_name: Optional[str] = Field(
        None, description="Name of the graph to use for knowledge graph-enhanced retrieval"
    )
    hop_depth: Optional[int] = Field(1, description="Number of relationship hops to traverse in the graph", ge=1, le=3)
    include_paths: Optional[bool] = Field(False, description="Whether to include relationship paths in the response")
    folder_name: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional folder scope for the operation. Accepts a single folder name or a list of folder names.",
    )
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")


class CompletionQueryRequest(RetrieveRequest):
    """Request model for completion generation"""

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allowed in the generated completion.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature passed to the completion model (None uses provider default).",
    )
    prompt_overrides: Optional[QueryPromptOverrides] = Field(
        None,
        description="Optional customizations for entity extraction, resolution, and query prompts",
    )
    schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = Field(
        None,
        description="Schema for structured output, can be a Pydantic model or JSON schema dict",
    )
    chat_id: Optional[str] = Field(
        None,
        description="Optional chat session ID for persisting conversation history",
    )
    stream_response: Optional[bool] = Field(
        False,
        description="Whether to stream the response back in chunks",
    )
    llm_config: Optional[Dict[str, Any]] = Field(
        None,
        description="LiteLLM-compatible model configuration (e.g., model name, API key, base URL)",
    )
    inline_citations: Optional[bool] = Field(
        False,
        description="Whether to include inline citations with filename and page number in the response",
    )


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        ...,
        description="Raw text content to store as a document.",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename hint used when inferring MIME type or displaying the document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined metadata stored with the document (JSON-serializable).",
    )
    metadata_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional per-field type hints: 'string', 'number', 'decimal', 'datetime', 'date', 'boolean', 'array', 'object'. Enables typed comparisons with $eq, $gt, etc. Types are inferred if omitted.",
    )
    use_colpali: Optional[bool] = Field(
        default=None,
        description="When provided, uses Morphik's finetuned ColPali style embeddings (recommended to be True for high quality retrieval).",
    )
    folder_name: Optional[str] = Field(None, description="Optional folder scope for the operation")
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")


class MetadataUpdateRequest(BaseModel):
    """Request payload for metadata-only document updates."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata fields to merge into the document.")
    metadata_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional per-field type hints: 'string', 'number', 'decimal', 'datetime', 'date', 'boolean', 'array', 'object'. Enables typed comparisons.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_plain_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict) and "metadata" not in value:
            return {"metadata": value}
        return value


class CreateGraphRequest(BaseModel):
    """Request model for creating a graph"""

    name: str = Field(..., description="Name of the graph to create")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters to determine which documents to include"
    )
    documents: Optional[List[str]] = Field(None, description="Optional list of specific document IDs to include")
    prompt_overrides: Optional[GraphPromptOverrides] = Field(
        None,
        description="Optional customizations for entity extraction and resolution prompts",
        json_schema_extra={
            "example": {
                "entity_extraction": {
                    "prompt_template": "Extract entities from the following text: {content}\n{examples}",
                    "examples": [{"label": "Example", "type": "ENTITY"}],
                }
            }
        },
    )
    folder_name: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional folder scope for the operation. Accepts a single folder name or a list of folder names.",
    )
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")


class UpdateGraphRequest(BaseModel):
    """Request model for updating a graph"""

    additional_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional metadata filters to determine which new documents to include",
    )
    additional_documents: Optional[List[str]] = Field(
        None, description="Optional list of additional document IDs to include"
    )
    prompt_overrides: Optional[GraphPromptOverrides] = Field(
        None, description="Optional customizations for entity extraction and resolution prompts"
    )
    folder_name: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional folder scope for the operation. Accepts a single folder name or a list of folder names.",
    )
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion"""

    documents: List[Document]
    errors: List[Dict[str, str]]


class DocumentQueryResponse(BaseModel):
    """Response model for document query with optional ingestion follow-up."""

    structured_output: Optional[Any] = Field(
        default=None, description="Raw structured output returned from Morphik On-the-Fly (may be list/dict)"
    )
    extracted_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured output coerced to metadata when possible"
    )
    text_output: Optional[str] = Field(
        default=None, description="Raw text returned from Morphik On-the-Fly when no schema is provided"
    )
    ingestion_enqueued: bool = Field(
        default=False, description="True when the document was queued for ingestion after extraction"
    )
    ingestion_document: Optional[Document] = Field(
        default=None, description="Queued document stub when ingestion_enqueued is true"
    )
    input_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Original metadata supplied alongside the request"
    )
    combined_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata that would be used if ingestion is performed"
    )
    ingestion_options: Dict[str, Any] = Field(
        default_factory=dict, description="Normalized ingestion options applied to this request"
    )


class BatchIngestJobResponse(BaseModel):
    """Response model for batch ingestion jobs"""

    status: str = Field(..., description="Status of the batch operation")
    documents: List[Document] = Field(..., description="List of created documents with processing status")
    timestamp: str = Field(..., description="ISO-formatted timestamp")


class GenerateUriRequest(BaseModel):
    """Request model for generating a cloud URI"""

    app_id: str = Field(..., description="ID of the application")
    name: str = Field(..., description="Name of the application")
    user_id: str = Field(..., description="ID of the user who owns the app")
    expiry_days: int = Field(default=30, description="Number of days until the token expires")
    org_id: Optional[str] = Field(None, description="Optional organization identifier for multi-tenant control planes")
    created_by_user_id: Optional[str] = Field(
        None,
        description="ID of the admin or service user that initiated the request",
    )


class DocumentPagesRequest(BaseModel):
    """Request model for extracting pages from a document"""

    document_id: str = Field(..., description="ID of the document to extract pages from")
    start_page: int = Field(..., ge=1, description="Starting page number (1-indexed)")
    end_page: int = Field(..., ge=1, description="Ending page number (1-indexed)")


class RequeueIngestionJob(BaseModel):
    """Job descriptor for requeuing an ingestion task."""

    external_id: str = Field(..., description="External identifier of the document to requeue")
    use_colpali: Optional[bool] = Field(
        default=None,
        description="When provided, uses Morphik's finetuned ColPali style embeddings (recommended to be True for high quality retrieval).",
    )


class RequeueIngestionRequest(BaseModel):
    """Request payload for requeueing ingestion jobs."""

    jobs: List[RequeueIngestionJob] = Field(
        default_factory=list, description="Collection of jobs to requeue, each with optional ColPali override."
    )
    include_all: bool = Field(
        default=False,
        description="When true, requeue every accessible document whose status matches `statuses` (defaults to processing/failed).",
    )
    statuses: Optional[List[str]] = Field(
        default=None,
        description="Processing statuses to include when `include_all` is true. Defaults to ['processing', 'failed'].",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of documents to auto-select from the provided statuses when include_all is true.",
    )


class BatchDocumentsRequest(BaseModel):
    """Request model for batch document retrieval."""

    document_ids: List[str] = Field(default_factory=list, description="List of document IDs to retrieve")
    folder_name: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional folder scope for the operation. Accepts a single folder name or a list of folder names.",
    )
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")


class BatchChunksRequest(BaseModel):
    """Request model for batch chunk retrieval."""

    sources: List[ChunkSource] = Field(default_factory=list, description="List of chunk sources to retrieve")
    folder_name: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional folder scope for the operation. Accepts a single folder name or a list of folder names.",
    )
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")
    use_colpali: Optional[bool] = Field(None, description="Whether to use ColPali embeddings for retrieval")
    output_format: Optional[Literal["base64", "url"]] = Field(
        None, description="How to return image chunks: base64 data URI (default) or a presigned URL"
    )
