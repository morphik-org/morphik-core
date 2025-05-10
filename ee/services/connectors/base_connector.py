from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ConnectorFile(BaseModel):
    id: str
    name: str
    is_folder: bool = False
    mime_type: Optional[str] = None
    size: Optional[int] = None  # in bytes
    modified_date: Optional[str] = None  # ISO 8601 string
    # Add other common fields if identifiable, like a path or icon_link


class ConnectorAuthStatus(BaseModel):
    is_authenticated: bool
    message: Optional[str] = None
    auth_url: Optional[str] = None  # URL to redirect user if not authenticated


class BaseConnector(ABC):
    connector_type: str

    @abstractmethod
    def __init__(self, user_morphik_id: str):
        self.user_morphik_id = user_morphik_id

    @abstractmethod
    async def get_auth_status(self) -> ConnectorAuthStatus:
        """Checks if the user is currently authenticated with this connector."""
        pass

    @abstractmethod
    async def initiate_auth(self) -> Dict[str, Any]:
        """Starts the authentication flow (e.g., returns OAuth URL and state)."""
        pass

    @abstractmethod
    async def finalize_auth(self, auth_response_data: Dict[str, Any]) -> bool:
        """Completes authentication and securely stores credentials."""
        pass

    @abstractmethod
    async def list_files(
        self, path: Optional[str] = None, page_token: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Lists files/folders. Should return {"files": List[ConnectorFile], "next_page_token": "..."}."""
        pass

    @abstractmethod
    async def download_file_by_id(self, file_id: str) -> Optional[BytesIO]:
        """Downloads a file by its connector-specific ID and returns its content as BytesIO."""
        pass

    @abstractmethod
    async def get_file_metadata_by_id(self, file_id: str) -> Optional[ConnectorFile]:
        """Gets metadata for a specific file by its ID."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Removes stored credentials for the user for this connector."""
        pass
