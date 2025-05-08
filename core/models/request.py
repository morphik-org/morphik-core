from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from core.models.documents import Document
from core.models.prompts import GraphPromptOverrides, QueryPromptOverrides


class RetrieveRequest(BaseModel):
    """Base retrieve request model"""

    query: str = Field(..., min_length=1)
    filters: Optional[Dict[str, Any]] = None
    k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.0)
    use_reranking: Optional[bool] = None  # If None, use default from config
    use_colpali: Optional[bool] = None
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

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    prompt_overrides: Optional[QueryPromptOverrides] = Field(
        None,
        description="Optional customizations for entity extraction, resolution, and query prompts",
    )
    schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = Field(
        None,
        description="Schema for structured output, can be a Pydantic model or JSON schema dict",
    )


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    content: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    use_colpali: Optional[bool] = None
    folder_name: Optional[str] = Field(None, description="Optional folder scope for the operation")
    end_user_id: Optional[str] = Field(None, description="Optional end-user scope for the operation")


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


# Add these classes before the extract_folder_data endpoint
class MetadataExtractionRuleRequest(BaseModel):
    """Request model for metadata extraction rule"""

    type: str = "metadata_extraction"  # Only metadata_extraction supported for now
    schema: Dict[str, Any]


class SetFolderRuleRequest(BaseModel):
    """Request model for setting folder rules"""

    rules: List[MetadataExtractionRuleRequest]


class AgentQueryRequest(BaseModel):
    """Request model for agent queries"""

    query: str = Field(..., description="Natural language query for the Morphik agent")
    rich: bool = Field(default=False, description="Whether to request a rich JSON response with citations")
    ground: bool = Field(
        default=False, description="Whether to perform a grounding pass on the rich response (requires rich=True)"
    )
