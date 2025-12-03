import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyzotero import zotero

from ee.config import EESettings, get_ee_settings

from .base_connector import BaseConnector, ConnectorAuthStatus, ConnectorFile

logger = logging.getLogger(__name__)


class ZoteroConnector(BaseConnector):
    connector_type = "zotero"

    def __init__(self, user_morphik_id: str):
        super().__init__(user_morphik_id)
        self.ee_settings: EESettings = get_ee_settings()
        self.zot_client: Optional[zotero.Zotero] = None
        self.credentials: Optional[Dict[str, str]] = None
        self._load_credentials()

    def _get_user_credentials_path(self) -> Path:
        """Get the path to store user credentials."""
        # Use the same directory structure as Google Drive but for Zotero
        return self._build_storage_path(self.ee_settings.GOOGLE_TOKEN_STORAGE_PATH, "zotero_creds_", ".json")

    def _save_credentials(self) -> None:
        """Save credentials to file."""
        if not self.credentials:
            logger.error(f"Attempted to save null credentials for user {self.user_morphik_id}.")
            return

        creds_path = self._get_user_credentials_path()
        try:
            with open(creds_path, "w") as creds_file:
                json.dump(self.credentials, creds_file)
            logger.info(f"Successfully saved Zotero credentials for user {self.user_morphik_id}")
        except Exception as e:
            logger.error(f"Failed to save Zotero credentials for user {self.user_morphik_id}: {e}")

    def _load_credentials(self) -> None:
        """Load credentials from file and initialize Zotero client."""
        creds_path = self._get_user_credentials_path()

        if not creds_path.exists():
            logger.info(f"No Zotero credentials file found for user {self.user_morphik_id}")
            self.credentials = None
            self.zot_client = None
            return

        try:
            with open(creds_path, "r") as creds_file:
                self.credentials = json.load(creds_file)

            # Validate required fields
            required_fields = ["library_id", "library_type", "api_key"]
            if not all(field in self.credentials for field in required_fields):
                logger.error(f"Invalid Zotero credentials format for user {self.user_morphik_id}")
                self.credentials = None
                self.zot_client = None
                return

            # Initialize Zotero client
            self.zot_client = zotero.Zotero(
                library_id=self.credentials["library_id"],
                library_type=self.credentials["library_type"],
                api_key=self.credentials["api_key"],
            )

            # Test the connection
            try:
                self.zot_client.key_info()
                logger.info(f"Successfully loaded Zotero credentials for user {self.user_morphik_id}")
            except Exception as e:
                logger.error(f"Zotero credentials invalid for user {self.user_morphik_id}: {e}")
                self.credentials = None
                self.zot_client = None
                # Remove invalid credentials file
                creds_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to load Zotero credentials for user {self.user_morphik_id}: {e}")
            self.credentials = None
            self.zot_client = None

    async def get_auth_status(self) -> ConnectorAuthStatus:
        """Check if the user is currently authenticated with Zotero."""
        if self.zot_client and self.credentials:
            try:
                # Test the connection by getting key info
                _ = self.zot_client.key_info()
                return ConnectorAuthStatus(
                    is_authenticated=True,
                    message=f"Successfully connected to Zotero library ({self.credentials['library_type']})",
                )
            except Exception as e:
                logger.error(f"Zotero connection test failed for user {self.user_morphik_id}: {e}")

        return ConnectorAuthStatus(
            is_authenticated=False, message="Not authenticated with Zotero. Please provide your Zotero API credentials."
        )

    async def initiate_auth(self) -> Dict[str, Any]:
        """Start the Zotero authentication flow by indicating what credentials are needed."""
        return {
            "auth_type": "manual_credentials",
            "required_fields": [
                {
                    "name": "library_id",
                    "label": "Library ID",
                    "description": "Your personal library ID or group library ID",
                    "type": "text",
                    "required": True,
                },
                {
                    "name": "library_type",
                    "label": "Library Type",
                    "description": "Type of library",
                    "type": "select",
                    "options": [
                        {"value": "user", "label": "Personal Library"},
                        {"value": "group", "label": "Group Library"},
                    ],
                    "required": True,
                },
                {
                    "name": "api_key",
                    "label": "API Key",
                    "description": "Your Zotero API key",
                    "type": "password",
                    "required": True,
                },
            ],
            "instructions": "You can find your library ID and create an API key at https://www.zotero.org/settings/keys",
        }

    async def finalize_auth(self, auth_response_data: Dict[str, Any]) -> bool:
        """Complete Zotero authentication by validating and storing credentials."""
        try:
            library_id = auth_response_data.get("library_id")
            library_type = auth_response_data.get("library_type")
            api_key = auth_response_data.get("api_key")

            if not all([library_id, library_type, api_key]):
                logger.error(f"Missing required Zotero credentials for user {self.user_morphik_id}")
                return False

            if library_type not in ["user", "group"]:
                logger.error(f"Invalid library_type '{library_type}' for user {self.user_morphik_id}")
                return False

            # Test the credentials by creating a client and checking key info
            test_client = zotero.Zotero(library_id=library_id, library_type=library_type, api_key=api_key)

            # Validate credentials work
            test_client.key_info()

            logger.info(f"Zotero key validation successful for user {self.user_morphik_id}")

            # Save credentials
            self.credentials = {"library_id": library_id, "library_type": library_type, "api_key": api_key}
            self.zot_client = test_client
            self._save_credentials()

            return True

        except Exception as e:
            logger.error(f"Zotero authentication failed for user {self.user_morphik_id}: {e}")
            return False

    async def list_files(
        self, path: Optional[str] = None, page_token: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """List Zotero items/collections."""
        if not self.zot_client:
            logger.error(f"User {self.user_morphik_id} is not authenticated with Zotero")
            raise ConnectionError("Not authenticated with Zotero")

        try:
            connector_files: List[ConnectorFile] = []

            # Handle pagination
            start = 0
            limit = kwargs.get("limit", 100)
            if page_token:
                try:
                    start = int(page_token)
                except ValueError:
                    start = 0

            if path == "collections" or path is None:
                # List collections as folders
                collections = self.zot_client.collections(start=start, limit=limit)
                for collection in collections:
                    connector_file = ConnectorFile(
                        id=f"collection_{collection['key']}",
                        name=collection["data"]["name"],
                        is_folder=True,
                        mime_type="application/x-zotero-collection",
                        size=None,
                        modified_date=collection["data"].get("dateModified"),
                    )
                    connector_files.append(connector_file)

            # List items (always include items)
            if path and path.startswith("collection_"):
                # Items in a specific collection
                collection_key = path.replace("collection_", "")
                items = self.zot_client.collection_items(collection_key, start=start, limit=limit)
            else:
                # Top-level items
                items = self.zot_client.top(start=start, limit=limit)

            for item in items:
                item_data = item["data"]

                # Determine if this item has attachments
                mime_type = "application/x-zotero-item"
                if item_data.get("itemType") == "attachment":
                    mime_type = item_data.get("contentType", "application/octet-stream")

                connector_file = ConnectorFile(
                    id=item["key"],
                    name=item_data.get("title", item_data.get("filename", "Untitled")),
                    is_folder=False,
                    mime_type=mime_type,
                    size=None,  # Zotero doesn't provide file sizes in item listings
                    modified_date=item_data.get("dateModified"),
                )
                connector_files.append(connector_file)

            # Calculate next page token
            next_page_token = None
            if len(connector_files) == limit:
                next_page_token = str(start + limit)

            return {"files": connector_files, "next_page_token": next_page_token}

        except Exception as e:
            logger.error(f"Error listing Zotero items for user {self.user_morphik_id}: {e}")
            raise ConnectionError(f"Failed to list Zotero items: {e}")

    async def download_file_by_id(self, file_id: str) -> Optional[BytesIO]:
        """Download a Zotero item by its ID."""
        if not self.zot_client:
            logger.error(f"User {self.user_morphik_id} is not authenticated with Zotero")
            return None

        try:
            # For Zotero, downloading typically means getting file attachments
            # First, check if this is a direct attachment
            item = self.zot_client.item(file_id)
            item_data = item["data"]

            if item_data.get("itemType") == "attachment":
                # This is an attachment, try to download it
                try:
                    file_content = self.zot_client.file(file_id)
                    if isinstance(file_content, bytes):
                        return BytesIO(file_content)
                    logger.warning(f"Attachment {file_id} returned non-bytes content")
                    return None
                except Exception as e:
                    logger.error(f"Failed to download attachment {file_id}: {e}")
                    return None
            else:
                # This is a regular item, look for attachments
                children = self.zot_client.children(file_id)
                for child in children:
                    if child["data"].get("itemType") == "attachment":
                        try:
                            file_content = self.zot_client.file(child["key"])
                            if isinstance(file_content, bytes):
                                return BytesIO(file_content)
                        except Exception as e:
                            logger.warning(f"Failed to download attachment {child['key']}: {e}")
                            continue

                logger.warning(f"No downloadable attachments found for item {file_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading Zotero file {file_id} for user {self.user_morphik_id}: {e}")
            return None

    async def get_file_metadata_by_id(self, file_id: str) -> Optional[ConnectorFile]:
        """Get metadata for a specific Zotero item."""
        if not self.zot_client:
            logger.error(f"User {self.user_morphik_id} is not authenticated with Zotero")
            return None

        try:
            item = self.zot_client.item(file_id)
            item_data = item["data"]

            # Determine mime type
            mime_type = "application/x-zotero-item"
            if item_data.get("itemType") == "attachment":
                mime_type = item_data.get("contentType", "application/octet-stream")

            connector_file = ConnectorFile(
                id=item["key"],
                name=item_data.get("title", item_data.get("filename", "Untitled")),
                is_folder=False,
                mime_type=mime_type,
                size=None,  # Zotero doesn't provide file sizes in API
                modified_date=item_data.get("dateModified"),
            )

            return connector_file

        except Exception as e:
            logger.error(f"Error getting Zotero item metadata {file_id} for user {self.user_morphik_id}: {e}")
            return None

    async def disconnect(self) -> bool:
        """Remove stored Zotero credentials for the user."""
        creds_path = self._get_user_credentials_path()

        if creds_path.exists():
            try:
                creds_path.unlink()
                logger.info(f"Successfully deleted Zotero credentials for user {self.user_morphik_id}")
            except OSError as e:
                logger.error(f"Error deleting Zotero credentials for user {self.user_morphik_id}: {e}")
                return False

        self.credentials = None
        self.zot_client = None
        return True
