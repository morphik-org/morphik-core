from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.models.folders import Folder


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    status: str
    message: str


class ServiceStatus(BaseModel):
    """Status of an individual service"""

    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: Optional[str] = None
    response_time_ms: Optional[float] = None


class DetailedHealthCheckResponse(BaseModel):
    """Response for detailed health check endpoint"""

    status: str  # "healthy", "unhealthy", "degraded"
    services: List[ServiceStatus]
    timestamp: str


class ModelsResponse(BaseModel):
    """Response for available models endpoint"""

    chat_models: List[Dict[str, Any]]
    embedding_models: List[Dict[str, Any]]
    default_models: Dict[str, Optional[str]]
    providers: List[str]


class FolderDeleteResponse(BaseModel):
    """Response for folder deletion endpoint"""

    status: str
    message: str


class DocumentPagesResponse(BaseModel):
    """Response for document pages extraction endpoint"""

    document_id: str
    pages: List[str]  # Base64-encoded images
    start_page: int
    end_page: int
    total_pages: int


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


class ChatTitleResponse(BaseModel):
    """Response for chat title update endpoint"""

    status: str
    message: str
    title: str


class FolderCount(BaseModel):
    """Count of documents grouped by folder name."""

    folder: Optional[str]
    count: int


class ListDocsResponse(BaseModel):
    """Flexible response for listing documents with aggregates."""

    documents: List[Any] = Field(default_factory=list)
    skip: int
    limit: int
    returned_count: int
    total_count: Optional[int] = None
    has_more: bool = False
    next_skip: Optional[int] = None
    status_counts: Optional[Dict[str, int]] = None
    folder_counts: Optional[List[FolderCount]] = None


class FolderDocumentInfo(BaseModel):
    """Document summary for a folder."""

    documents: List[Any] = Field(default_factory=list)
    document_count: Optional[int] = None
    status_counts: Optional[Dict[str, int]] = None
    skip: int = 0
    limit: int = 0
    returned_count: int = 0
    has_more: bool = False
    next_skip: Optional[int] = None


class FolderDetails(BaseModel):
    """Folder details with optional document summary."""

    folder: Folder
    document_info: Optional[FolderDocumentInfo] = None


class FolderDetailsResponse(BaseModel):
    """Response wrapping folder detail entries."""

    folders: List[FolderDetails]


class RequeueIngestionResult(BaseModel):
    """Result information for an individual requeued ingestion job."""

    external_id: str = Field(..., description="Document external identifier")
    status: str = Field(..., description="Outcome status for this job")
    message: Optional[str] = Field(default=None, description="Optional human-readable message describing the outcome")


class RequeueIngestionResponse(BaseModel):
    """Response payload for requeueing ingestion jobs."""

    results: List[RequeueIngestionResult] = Field(..., description="Per-document outcomes")
