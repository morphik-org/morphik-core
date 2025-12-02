from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

from .chat import ChatMessage

# Type variable for any Pydantic model
PydanticT = TypeVar("PydanticT", bound=BaseModel)


class StructuredCompletion(BaseModel):
    """Structured completion object for schema-based responses"""

    class Config:
        extra = "allow"  # Allow additional properties


class ChunkSource(BaseModel):
    """Source information for a chunk used in completion"""

    document_id: str
    chunk_number: int
    score: Optional[float] = None


class CompletionResponse(BaseModel):
    """Response from completion generation"""

    completion: Union[str, StructuredCompletion]
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    sources: List[ChunkSource] = []
    metadata: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    """Request for completion generation"""

    model_config = {"populate_by_name": True}

    query: str
    context_chunks: List[str]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.3
    prompt_template: Optional[str] = None
    folder_name: Optional[str] = None
    end_user_id: Optional[str] = None
    response_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = Field(default=None, alias="schema")
    chat_history: Optional[List[ChatMessage]] = None
    stream_response: Optional[bool] = False
    llm_config: Optional[Dict[str, Any]] = None
    inline_citations: Optional[bool] = False
    chunk_metadata: Optional[List[Dict[str, Any]]] = None  # Metadata for each chunk including filename and page
    system_prompt: Optional[str] = None
