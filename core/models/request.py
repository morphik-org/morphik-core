from typing import Dict, List, Optional, Any, Literal
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
    hop_depth: Optional[int] = Field(
        1, description="Number of relationship hops to traverse in the graph", ge=1, le=3
    )
    include_paths: Optional[bool] = Field(
        False, description="Whether to include relationship paths in the response"
    )


class ChatMessage(BaseModel):
    """A message in a chat conversation"""
    
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="The role of the message sender"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion generation"""
    
    messages: List[ChatMessage] = Field(
        ..., description="The messages in the chat conversation", min_items=1
    )
    end_user_id: str = Field(
        ..., description="Identifier for the end user for whom the chat is intended", min_length=1
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional identifier for the conversation"
    )
    remember: Optional[bool] = Field(
        None, description="Whether to remember this conversation for future reference"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters for documents"
    )
    k: int = Field(
        default=4, gt=0, description="Number of chunks to retrieve"
    )
    temperature: Optional[float] = Field(
        None, description="Temperature for completion generation", ge=0, le=2
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens for completion generation"
    )
    use_colpali: Optional[bool] = Field(
        None, description="Whether to use multi-vector embeddings (ColPali style embeddings)"
    )


class CompletionQueryRequest(RetrieveRequest):
    """Request model for completion generation"""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    prompt_overrides: Optional[QueryPromptOverrides] = Field(
        None,
        description="Optional customizations for entity extraction, resolution, and query prompts"
    )


class IngestTextRequest(BaseModel):
    """Request model for ingesting text content"""

    content: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    use_colpali: Optional[bool] = None


class CreateGraphRequest(BaseModel):
    """Request model for creating a graph"""

    name: str = Field(..., description="Name of the graph to create")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters to determine which documents to include"
    )
    documents: Optional[List[str]] = Field(
        None, description="Optional list of specific document IDs to include"
    )
    prompt_overrides: Optional[GraphPromptOverrides] = Field(
        None,
        description="Optional customizations for entity extraction and resolution prompts",
        json_schema_extra={"example": {
            "entity_extraction": {
                "prompt_template": "Extract entities from the following text: {content}\n{examples}", 
                "examples": [{"label": "Example", "type": "ENTITY"}]
            }
        }}
    )


class UpdateGraphRequest(BaseModel):
    """Request model for updating a graph"""
    
    additional_filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional additional metadata filters to determine which new documents to include"
    )
    additional_documents: Optional[List[str]] = Field(
        None, description="Optional list of additional document IDs to include"
    )
    prompt_overrides: Optional[GraphPromptOverrides] = Field(
        None,
        description="Optional customizations for entity extraction and resolution prompts"
    )


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion"""
    documents: List[Document]
    errors: List[Dict[str, str]]


class GenerateUriRequest(BaseModel):
    """Request model for generating a cloud URI"""
    app_id: str = Field(..., description="ID of the application")
    name: str = Field(..., description="Name of the application")
    user_id: str = Field(..., description="ID of the user who owns the app")
    expiry_days: int = Field(default=30, description="Number of days until the token expires")
