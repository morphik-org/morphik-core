from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    status: str
    message: str


class ModelsResponse(BaseModel):
    """Response for available models endpoint"""

    chat_models: List[Dict[str, Any]]
    embedding_models: List[Dict[str, Any]]
    default_models: Dict[str, Optional[str]]
    providers: List[str]


class OAuthCallbackResponse(BaseModel):
    """Response for OAuth callback endpoint"""

    status: str
    message: Optional[str] = None


class FolderDeleteResponse(BaseModel):
    """Response for folder deletion endpoint"""

    status: str
    message: str


class FolderRuleResponse(BaseModel):
    """Response for folder rule setting endpoint"""

    status: str
    message: str


class DocumentAddToFolderResponse(BaseModel):
    """Response for adding document to folder endpoint"""

    status: str
    message: str


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion endpoint"""

    status: str
    message: str


class DocumentDownloadUrlResponse(BaseModel):
    """Response for document download URL endpoint"""

    download_url: str
    expires_in: int


class DocumentFileResponse(BaseModel):
    """Response for document file endpoint"""

    file_data: bytes
    content_type: str
    filename: str


class ChatResponse(BaseModel):
    """Response for chat endpoint"""

    chat_id: str
    messages: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion endpoint"""

    completion: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    sources: List[Dict[str, Any]] = []


class ChatTitleResponse(BaseModel):
    """Response for chat title update endpoint"""

    status: str
    message: str
    title: str
