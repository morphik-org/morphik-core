import logging
from typing import Optional

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
        # Add elif for other connectors here in the future
        else:
            logger.error(f"Unsupported connector type: {connector_type}")
            raise ValueError(f"Unsupported connector type: {connector_type}")
