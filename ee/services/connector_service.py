import logging
from typing import Optional, Dict, Type, List

# from core.models.auth import AuthContext # Morphik's AuthContext - Assuming this path is correct
# For now, let's use a placeholder if the actual AuthContext is not available for type hinting
try:
    from core.models.auth import AuthContext
except ImportError:

    class AuthContext:  # type: ignore
        user_id: Optional[str]
        entity_id: Optional[str]


from .connectors.base_connector import BaseConnector
from .connectors.google_drive_connector import GoogleDriveConnector
from .connectors.github_connector import GitHubConnector
from .connectors.zotero_connector import ZoteroConnector

from redis.asyncio import Redis

from core.services.document_service import DocumentService
from ee.services.connectors.base_connector import ConnectorAuthStatus

logger = logging.getLogger(__name__)


class ConnectorService:
    def __init__(self, auth_context: AuthContext):
        self.auth_context = auth_context
        # Ensure user_id and entity_id are Optional in AuthContext definition for this logic
        self.user_identifier = auth_context.user_id if auth_context.user_id else auth_context.entity_id
        if not self.user_identifier:
            raise ValueError("User identifier is missing from AuthContext.")

    async def get_connector(self, connector_type: str) -> BaseConnector:
        logger.info(f"Getting connector of type '{connector_type}' for user '{self.user_identifier}'")
        if connector_type == "google_drive":
            return GoogleDriveConnector(user_morphik_id=self.user_identifier)
        elif connector_type == "github":
            return GitHubConnector(user_morphik_id=self.user_identifier)
        elif connector_type == "zotero":
            return ZoteroConnector(user_morphik_id=self.user_identifier)
        # Add elif for other connectors here in the future
        else:
            logger.error(f"Unsupported connector type: {connector_type}")
            raise ValueError(f"Unsupported connector type: {connector_type}")

    async def ingest_file_from_connector(
        self,
        connector_type: str,
        file_id: str,
        document_service: DocumentService,
        auth: AuthContext,
        redis: Redis,
        folder_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        rules: Optional[List] = None,
    ):
        connector = await self.get_connector(connector_type)
        if not connector:
            raise ValueError(f"Connector '{connector_type}' not found or initialized.")

        file_content_bytes = await connector.download_file_by_id(file_id)
        if not file_content_bytes:
            raise ValueError(f"Failed to download file with id '{file_id}'.")

        file_metadata = await connector.get_file_metadata_by_id(file_id)
        if not file_metadata:
            raise ValueError(f"Failed to get metadata for file with id '{file_id}'.")

        # Prepare metadata
        final_metadata = {"source": connector_type, "connector_file_id": file_id}
        if metadata:
            final_metadata.update(metadata)
        
        # Use the injected document_service
        doc = await document_service.ingest_file_content(
            file_content_bytes=file_content_bytes.getvalue(),
            filename=file_metadata.name,
            content_type=file_metadata.mime_type,
            metadata=final_metadata,
            auth=auth,
            redis=redis,
            folder_name=folder_name,
            rules=rules,
            use_colpali=False,  # Or determine based on file type
        )

        return {"document_id": doc.external_id, "status_path": f"/documents/{doc.external_id}/status"}
