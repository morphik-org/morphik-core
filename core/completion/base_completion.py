from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from pydantic import BaseModel


class CompletionResponse(BaseModel):
    """Response from completion generation"""
    completion: str
    usage: Dict[str, int]


class CompletionRequest(BaseModel):
    """Request for completion generation"""
    query: str
    context_chunks: List[str]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7


class BaseCompletionModel(ABC):
    """Base class for completion models"""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion from query and context"""
        pass
