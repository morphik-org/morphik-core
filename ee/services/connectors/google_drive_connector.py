# This is the beginning of the GoogleDriveConnector implementation.

import logging
import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.exceptions import GoogleAuthError  # Added for error handling
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload  # Added for file download

from ee.config import EESettings, get_ee_settings

from .base_connector import BaseConnector, ConnectorAuthStatus, ConnectorFile

logger = logging.getLogger(__name__)


class GoogleDriveConnector(BaseConnector):
    connector_type = "google_drive"
    _flow_state: Optional[str] = None  # Used to store OAuth state temporarily on the instance

    def __init__(self, user_morphik_id: str):
        super().__init__(user_morphik_id)
        self.ee_settings: EESettings = get_ee_settings()
        self.creds: Optional[Credentials] = None
        self.service: Optional[Resource] = None
        self._load_credentials()

    def _get_user_token_path(self) -> Path:
        token_dir = Path(self.ee_settings.GOOGLE_TOKEN_STORAGE_PATH)
        # Create the token directory if it doesn't exist
        os.makedirs(token_dir, exist_ok=True)
        return token_dir / f"gdrive_token_{self.user_morphik_id}.pickle"

    def _save_credentials(self) -> None:
        if not self.creds:
            logger.error(f"Attempted to save null credentials for user {self.user_morphik_id}.")
            return
        token_path = self._get_user_token_path()
        try:
            with open(token_path, "wb") as token_file:
                pickle.dump(self.creds, token_file)
            logger.info(f"Successfully saved token for user {self.user_morphik_id} to {token_path}")
        except Exception as e:
            logger.error(f"Failed to save token for user {self.user_morphik_id} to {token_path}: {e}")

    def _load_credentials(self) -> None:
        token_path = self._get_user_token_path()
        loaded_creds: Optional[Credentials] = None

        if token_path.exists():
            try:
                with open(token_path, "rb") as token_file:
                    loaded_creds = pickle.load(token_file)
                logger.info(f"Successfully loaded token for user {self.user_morphik_id} from {token_path}")
            except Exception as e:
                logger.error(f"Failed to load token for user {self.user_morphik_id} from {token_path}: {e}")
                # Optionally, attempt to delete corrupted token file: token_path.unlink(missing_ok=True)
                self.creds = None
                self.service = None
                return

        if loaded_creds:
            self.creds = loaded_creds  # Assign initially, will be updated if refreshed
            if self.creds.expired and self.creds.refresh_token:
                try:
                    logger.info(f"Refreshing token for user {self.user_morphik_id}")
                    self.creds.refresh(Request())
                    # self.creds is now updated with new access token and potentially new refresh token
                    self._save_credentials()  # Save the refreshed (and now current) credentials
                except Exception as e:
                    logger.error(f"Failed to refresh token for user {self.user_morphik_id}: {e}. Clearing credentials.")
                    self.creds = None
                    # Optionally, delete the token file that failed to refresh: token_path.unlink(missing_ok=True)
            elif not self.creds.valid:
                logger.warning(f"Loaded token for user {self.user_morphik_id} is invalid and has no refresh token.")
                self.creds = None  # Clear invalid credentials
        else:  # No token file found
            logger.info(f"No token file found for user {self.user_morphik_id} at {token_path}.")
            self.creds = None

        # If we have valid credentials (either loaded & valid, or refreshed & valid)
        if self.creds and self.creds.valid:
            try:
                # cache_discovery=False is useful for development to avoid issues with stale discovery documents.
                # For production, you might remove it or make it configurable.
                self.service = build("drive", "v3", credentials=self.creds, cache_discovery=False)
                logger.info(f"Google Drive service initialized for user {self.user_morphik_id}")
            except Exception as e:
                logger.error(f"Failed to build Google Drive service for user {self.user_morphik_id}: {e}")
                self.service = None
                # Consider if self.creds should be cleared if service build fails despite creds being 'valid'
                # self.creds = None
        else:
            self.service = None
            if (
                self.creds
            ):  # If creds exist but are not valid (e.g. refresh failed, or invalid from start with no refresh)
                logger.warning(f"Credentials for user {self.user_morphik_id} are not valid. Service not built.")
                self.creds = None  # Ensure self.creds is None if service could not be built due to invalid creds

    async def get_auth_status(self) -> ConnectorAuthStatus:
        if self.creds and self.creds.valid and self.service:
            return ConnectorAuthStatus(is_authenticated=True, message="Successfully connected to Google Drive.")

        # According to the plan, if not authenticated, this should return an auth_url.
        # We can obtain this by preparing parts of the initiate_auth logic here or calling a helper.
        # For now, let's call a part of initiate_auth that just generates the URL without side effects.
        # The `initiate_auth` method itself will store state and return the full dict.
        auth_details = await self._prepare_auth_details_for_status()
        return ConnectorAuthStatus(
            is_authenticated=False,
            message="Not authenticated with Google Drive. Please initiate authorization.",
            auth_url=auth_details.get("authorization_url"),
        )

    async def _prepare_auth_details_for_status(self) -> Dict[str, Any]:
        """Helper to generate auth URL for status without full initiate_auth side effects."""
        if not self.ee_settings.GOOGLE_CLIENT_ID or not self.ee_settings.GOOGLE_CLIENT_SECRET:
            # This case should ideally be caught earlier or handled by system health checks.
            logger.error("Google Client ID or Secret is not configured for status check.")
            return {}

        client_config = {
            "web": {
                "client_id": self.ee_settings.GOOGLE_CLIENT_ID,
                "client_secret": self.ee_settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.ee_settings.GOOGLE_REDIRECT_URI],
            }
        }
        try:
            flow = Flow.from_client_config(
                client_config=client_config,
                scopes=self.ee_settings.GOOGLE_SCOPES,
                redirect_uri=self.ee_settings.GOOGLE_REDIRECT_URI,
            )
            authorization_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
            return {"authorization_url": authorization_url}
        except Exception as e:
            logger.error(f"Error preparing auth URL for status: {e}")
            return {}

    async def initiate_auth(self) -> Dict[str, Any]:
        if not self.ee_settings.GOOGLE_CLIENT_ID or not self.ee_settings.GOOGLE_CLIENT_SECRET:
            logger.error("Google Client ID or Secret is not configured.")
            # It might be better to raise an HTTPException from the router if this is a web flow.
            raise ValueError("Google API credentials not configured. Cannot initiate authentication.")

        client_config = {
            "web": {
                "client_id": self.ee_settings.GOOGLE_CLIENT_ID,
                "client_secret": self.ee_settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.ee_settings.GOOGLE_REDIRECT_URI],
            }
        }

        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=self.ee_settings.GOOGLE_SCOPES,
            redirect_uri=self.ee_settings.GOOGLE_REDIRECT_URI,
        )

        authorization_url, state = flow.authorization_url(
            access_type="offline",  # Request refresh token
            prompt="consent",  # Ensure user sees consent screen, good for dev/re-auth
        )
        # Storing state on instance: In a real scenario, this state would be stored in the user's session
        # by the calling router (e.g., request.session['oauth_state'] = state)
        # and then passed back to this method in auth_response_data.
        # For this connector, we assume the calling router will handle state persistence
        # and pass it via auth_response_data[\"state\"].
        # self._flow_state is a conceptual placeholder for where this connector might expect it.
        # The actual state comparison should use the state passed in auth_response_data
        # against the state from the user's session (managed by the router).
        self._flow_state = state
        logger.info(f"Generated authorization_url for user {self.user_morphik_id}, state: {state[:10]}...")
        return {"authorization_url": authorization_url, "state": state}

    async def finalize_auth(self, auth_response_data: Dict[str, Any]) -> bool:
        received_state = auth_response_data.get("state")
        authorization_response_url = auth_response_data.get("authorization_response_url")

        # The router should have stored the original state (e.g., self._flow_state from initiate_auth)
        # in the user's session and passed it back in auth_response_data as, for example,
        # auth_response_data[\"original_state_from_session\"].
        # For this example, we'll directly use self._flow_state, assuming the router populated it or
        # passed the correct one.
        # A more robust implementation in the router would be:
        # original_state = request.session.pop('oauth_state', None)
        # if not received_state or not original_state or received_state != original_state:
        #     logger.error("OAuth state mismatch or missing.")
        #     # raise HTTPException(status_code=400, detail="Invalid OAuth state")

        # Here, we use self._flow_state as if it's the state retrieved from the session by the caller.
        # This implies that the caller (router) correctly managed and provided the state.

        if not authorization_response_url:
            logger.error(f"Authorization response URL not provided for user {self.user_morphik_id}.")
            return False

        if not self.ee_settings.GOOGLE_CLIENT_ID or not self.ee_settings.GOOGLE_CLIENT_SECRET:
            logger.error("Google Client ID or Secret is not configured for finalize_auth.")
            return False

        client_config = {
            "web": {
                "client_id": self.ee_settings.GOOGLE_CLIENT_ID,
                "client_secret": self.ee_settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",  # Google's token endpoint
                "redirect_uris": [self.ee_settings.GOOGLE_REDIRECT_URI],
            }
        }

        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=self.ee_settings.GOOGLE_SCOPES,
            state=received_state,  # Pass the received state to the flow for validation
            redirect_uri=self.ee_settings.GOOGLE_REDIRECT_URI,
        )

        try:
            logger.info(
                f"Fetching token for user {self.user_morphik_id} using response: {authorization_response_url[:100]}..."
            )
            # The fetch_token method will use the authorization code from the URL.
            flow.fetch_token(authorization_response=authorization_response_url)

            self.creds = flow.credentials
            if not self.creds:
                logger.error(f"Credential object not created after fetch_token for user {self.user_morphik_id}")
                return False

            self._save_credentials()
            self._load_credentials()  # This will also build the service if creds are valid

            if self.creds and self.creds.valid and self.service:
                logger.info(f"Successfully finalized authentication for user {self.user_morphik_id}.")
                return True
            else:
                logger.error(f"Auth finalized but credentials invalid or service not built for {self.user_morphik_id}.")
                return False

        except GoogleAuthError as e:  # More specific Google auth errors
            logger.error(f"Google authentication error during token fetch for user {self.user_morphik_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error fetching token or saving credentials for user {self.user_morphik_id}: {e}")
            # Potentially clear self.creds if token fetch failed partway
            self.creds = None
            return False

    async def list_files(
        self, path: Optional[str] = None, page_token: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        if not (self.creds and self.creds.valid and self.service):
            logger.error(
                f"User {self.user_morphik_id} is not authenticated or Google Drive service "
                f"is not available for list_files."
            )
            # In a real API, you might raise an HTTPException or a custom exception.
            # For now, returning an empty state or error structure if that's preferred.
            # However, the spec implies an error if not authenticated.
            raise ConnectionError("Not authenticated or Google Drive service not available.")

        folder_id_to_query = path if path else "root"
        q_filter = kwargs.get("q_filter")
        page_size = kwargs.get("page_size", 100)

        query_parts = [f"'{folder_id_to_query}' in parents", "trashed = false"]
        if q_filter:
            query_parts.append(q_filter)

        query = " and ".join(query_parts)

        connector_files: List[ConnectorFile] = []
        next_page_token_to_return: Optional[str] = None

        try:
            logger.info(
                f"Listing files for user {self.user_morphik_id} in folder '{folder_id_to_query}', "
                f"pageToken: {page_token}, pageSize: {page_size}, query: '{query}'"
            )
            results = (
                self.service.files()
                .list(
                    q=query,
                    pageSize=page_size,
                    pageToken=page_token,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, kind)",
                    # orderBy="folder, name" # Optional: to sort folders first, then by name
                )
                .execute()
            )

            items = results.get("files", [])
            for item in items:
                is_folder = item.get("mimeType") == "application/vnd.google-apps.folder"
                # Google Drive API doesn't return 'size' for folders or Google Docs native types.
                # Pydantic model has size as Optional[int], so None is acceptable.
                file_size = item.get("size")
                if file_size is not None:
                    try:
                        file_size = int(file_size)
                    except ValueError:
                        logger.warning(
                            f"Could not convert size '{item.get('size')}' to int for file "
                            f"'{item.get('name')}'. Setting to None."
                        )
                        file_size = None

                connector_file = ConnectorFile(
                    id=item["id"],
                    name=item["name"],
                    is_folder=is_folder,
                    mime_type=item.get("mimeType"),
                    size=file_size,
                    modified_date=item.get("modifiedTime"),  # ISO 8601 string
                )
                connector_files.append(connector_file)

            next_page_token_to_return = results.get("nextPageToken")

        except Exception as e:
            logger.error(f"Error listing files from Google Drive for user {self.user_morphik_id}: {e}")
            # Depending on the error, you might want to re-raise or return a specific error structure.
            # For now, we'll let it propagate if it's a critical API error or return empty if it's caught.
            # For robustness, specific Google API errors should be caught.
            # Example: from googleapiclient.errors import HttpError
            # except HttpError as he: logger.error(...)
            # For simplicity, catching generic Exception here.
            raise ConnectionError(f"Failed to list files from Google Drive: {e}")

        return {"files": connector_files, "next_page_token": next_page_token_to_return}

    async def download_file_by_id(self, file_id: str) -> Optional[BytesIO]:
        logger.info(f"Attempting to download file with ID '{file_id}' for user '{self.user_morphik_id}'.")
        if not self.service or not self.creds or not self.creds.valid:
            logger.warning(
                f"Google Drive service not available or credentials invalid for user '{self.user_morphik_id}'. "
                f"Cannot download file '{file_id}'. Triggering credential load."
            )
            self._load_credentials()  # Attempt to reload/refresh credentials
            if not self.service or not self.creds or not self.creds.valid:
                logger.error(
                    f"Credential reload failed or service still unavailable for user '{self.user_morphik_id}'. "
                    f"Download of file '{file_id}' aborted."
                )
                return None

        try:
            file_metadata_request = self.service.files().get(fileId=file_id, fields="id, name, mimeType")
            file_metadata = file_metadata_request.execute()
            mime_type = file_metadata.get("mimeType")

            request = None
            if mime_type and "google-apps" in mime_type:
                # Handle Google Workspace documents (Docs, Sheets, Slides) by exporting them
                # For simplicity, exporting as PDF. This could be made more configurable.
                # Common export MIME types:
                # Google Docs: 'application/pdf', 'application/vnd.oasis.opendocument.text',
                #   'text/plain', 'application/rtf', 'application/zip' (html)
                # Google Sheets: 'application/pdf',
                #   'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' (xlsx)
                # Google Slides: 'application/pdf',
                #   'application/vnd.openxmlformats-officedocument.presentationml.presentation' (pptx)

                export_mime_type = "application/pdf"  # Default export type
                if mime_type == "application/vnd.google-apps.document":
                    export_mime_type = "application/pdf"  # Or 'text/plain', 'application/rtf', etc.
                elif mime_type == "application/vnd.google-apps.spreadsheet":
                    # XLSX
                    xlsx_mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    export_mime_type = xlsx_mime_type
                elif mime_type == "application/vnd.google-apps.presentation":
                    # PDF or PPTX
                    export_mime_type = "application/pdf"
                    # Alt PPTX MIME:
                    # "application/vnd.openxmlformats-officedocument.presentationml.presentation"

                logger.info(
                    f"Exporting Google Workspace file '{file_id}' (type: {mime_type}) "
                    f"as {export_mime_type} for user '{self.user_morphik_id}'."
                )
                request = self.service.files().export_media(fileId=file_id, mimeType=export_mime_type)
            else:
                # For other file types, use get_media
                logger.info(
                    f"Downloading binary file '{file_id}' (type: {mime_type}) for user '{self.user_morphik_id}'."
                )
                request = self.service.files().get_media(fileId=file_id)

            if request is None:  # Should not happen if logic is correct
                logger.error(f"Download request could not be created for file ID {file_id}")
                return None

            file_content_bytes_io = BytesIO()
            downloader = MediaIoBaseDownload(file_content_bytes_io, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.debug(f"Download {int(status.progress() * 100)}% for file_id {file_id}.")

            file_content_bytes_io.seek(0)  # Reset stream position to the beginning
            logger.info(f"Successfully downloaded file '{file_id}' for user '{self.user_morphik_id}'.")
            return file_content_bytes_io

        except GoogleAuthError as e:  # Catch auth-specific errors explicitly
            logger.error(f"Authentication error downloading file '{file_id}' for user '{self.user_morphik_id}': {e}")
            # Potentially try to re-authenticate or guide user to re-auth
            self.creds = None  # Invalidate current creds
            self.service = None
            self._load_credentials()  # Attempt to refresh
            return None
        except Exception as e:
            # Consider specific GDrive API error types (e.g., HttpError from googleapiclient.errors)
            logger.error(f"Error downloading file '{file_id}' for user '{self.user_morphik_id}': {e}")
            # For example, if e.resp.status == 404, file not found.
            return None

    async def get_file_metadata_by_id(self, file_id: str) -> Optional[ConnectorFile]:
        logger.info(f"Fetching metadata for file ID '{file_id}' for user '{self.user_morphik_id}'.")
        if not self.service or not self.creds or not self.creds.valid:
            logger.warning(
                f"Google Drive service not available or credentials invalid for user '{self.user_morphik_id}'. "
                f"Cannot get metadata for file '{file_id}'. Triggering credential load."
            )
            self._load_credentials()  # Attempt to reload/refresh credentials
            if not self.service or not self.creds or not self.creds.valid:
                logger.error(
                    f"Credential reload failed or service still unavailable for user '{self.user_morphik_id}'. "
                    f"Metadata fetch for file '{file_id}' aborted."
                )
                return None

        try:
            # Added webViewLink, iconLink, parents for more comprehensive metadata fetching
            fields_to_request = "id, name, mimeType, size, modifiedTime, trashed, webViewLink, iconLink, parents"
            file_resource = self.service.files().get(fileId=file_id, fields=fields_to_request).execute()

            if file_resource.get("trashed"):
                logger.info(f"File '{file_id}' (Name: {file_resource.get('name')}) is trashed. Not returning metadata.")
                return None

            mime_type = file_resource.get("mimeType")
            size_str = file_resource.get("size")
            size_int: Optional[int] = None
            if size_str:
                try:
                    size_int = int(size_str)
                except ValueError:
                    logger.warning(f"Could not convert size '{size_str}' to int for file '{file_id}'.")

            modified_date = file_resource.get("modifiedTime")  # ISO 8601 format (RFC3339)

            # Consider adding webViewLink and iconLink to ConnectorFile Pydantic model if generally useful
            # path=file_resource.get("webViewLink"), # Example if path could be webViewLink
            # icon_link=file_resource.get("iconLink") # Example if ConnectorFile has icon_link

            connector_file = ConnectorFile(
                id=file_resource["id"],
                name=file_resource.get("name", "Unnamed File"),
                is_folder=(mime_type == "application/vnd.google-apps.folder"),
                mime_type=mime_type,
                size=size_int,
                modified_date=modified_date,
            )
            logger.info(f"Successfully fetched metadata for file '{file_id}'. Name: '{connector_file.name}'")
            return connector_file

        except GoogleAuthError as e:
            logger.error(
                f"Authentication error fetching metadata for file '{file_id}' for user '{self.user_morphik_id}': {e}"
            )
            self.creds = None  # Invalidate credentials
            self.service = None
            self._load_credentials()  # Attempt to refresh
            return None
        except Exception as e:  # Handles HttpError (e.g., 404 Not Found) and other API errors
            logger.error(
                f"API error fetching metadata for file '{file_id}' for user '{self.user_morphik_id}': {e}. "
                f"Type: {type(e).__name__}"
            )
            return None

    async def disconnect(self) -> bool:
        token_path = self._get_user_token_path()
        if token_path.exists():
            try:
                token_path.unlink()
                logger.info(f"Successfully deleted token file for user {self.user_morphik_id} at {token_path}")
                self.creds = None
                self.service = None
                return True
            except OSError as e:  # More specific exception for file operations
                logger.error(f"Error deleting token file for user {self.user_morphik_id} at {token_path}: {e}")
                return False
        logger.info(
            f"No token file found to delete for user {self.user_morphik_id} at {token_path}. Considered success."
        )
        self.creds = None  # Ensure state is cleared even if file didn't exist
        self.service = None
        return True
