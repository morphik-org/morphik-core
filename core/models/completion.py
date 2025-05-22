from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from .chat import ChatMessage

# Type variable for any Pydantic model
PydanticT = TypeVar("PydanticT", bound=BaseModel)


class ChunkSource(BaseModel):
    """Source information for a chunk used in completion"""

    document_id: str
    chunk_number: int
    score: Optional[float] = None


class CompletionResponse(BaseModel):
    """Response from completion generation"""

    completion: Union[str, PydanticT]
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    sources: List[ChunkSource] = []
    metadata: Optional[Dict] = None


class CompletionRequest(BaseModel):
    """Request for completion generation"""

    query: str
    context_chunks: List[str]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    prompt_template: Optional[str] = None
    folder_name: Optional[str] = None
    end_user_id: Optional[str] = None
    schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    chat_history: Optional[List[ChatMessage]] = None
