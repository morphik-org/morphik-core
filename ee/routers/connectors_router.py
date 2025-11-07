import json
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from core.auth_utils import verify_token

# Attempt to import global document_service and redis_pool dependency from core.api
# This is a simplification; a more robust solution might use app.state or a dedicated dependency module
from core.dependencies import get_document_service, get_redis_pool
from core.models.auth import AuthContext

# Import DocumentService type for dependency injection hint
from core.services.document_service import DocumentService
from ee.services.connector_service import ConnectorService
from ee.services.connectors.base_connector import ConnectorAuthStatus, ConnectorFile  # Importing specific models

# from starlette.datastructures import URL  # Will be needed for oauth2callback

# Connector models defined locally below

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ee/connectors",
    tags=["EE - Connectors"],
)


# Dependency to get ConnectorService
async def get_connector_service(auth: AuthContext = Depends(verify_token)) -> ConnectorService:
    # Should be caught by verify_token but as a safeguard
    if not auth.user_id and not auth.entity_id:
        logger.error("AuthContext is missing user_id and entity_id in get_connector_service.")
        raise HTTPException(status_code=401, detail="Invalid authentication context.")
    try:
        return ConnectorService(auth_context=auth)
    except ValueError as e:
        logger.error(f"Failed to initialize ConnectorService: {e}")
        # User-friendly error message
        raise HTTPException(status_code=500, detail="Connector service initialization error.")


# Placeholder for IngestFromConnectorRequest Pydantic model
class IngestFromConnectorRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_id: str
    morphik_folder_name: Optional[str] = None
    morphik_end_user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # New field for custom metadata


class ConnectorIngestRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_id: str
    folder_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GitHubRepositoryIngestRequest(BaseModel):
    connector_type: str = Field(
        default="github",
        json_schema_extra={"x-stainless-param": "body_connector_type"},
    )
    repo_path: str  # Format: "owner/repo"
    folder_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    include_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    compress: bool = True
    force: bool = False  # Force re-ingestion even if repository already exists


class ConnectorAuthRequest(BaseModel):
    connector_type: str


class ConnectorAuthResponse(BaseModel):
    connector_type: str
    auth_response_data: Dict[str, Any]


class ConnectorListFilesRequest(BaseModel):
    connector_type: str
    path: Optional[str] = None
    page_token: Optional[str] = None


# Add request model for manual credentials
class ManualCredentialsRequest(BaseModel):
    """Request model for manual credential submission."""

    credentials: Dict[str, Any]


# Models for auth initiation responses
class CredentialFieldOption(BaseModel):
    value: str
    label: str


class CredentialField(BaseModel):
    name: str
    label: str
    description: str
    type: str  # "text", "password", "select"
    required: bool
    options: Optional[List[CredentialFieldOption]] = None


class ManualCredentialsAuthResponse(BaseModel):
    auth_type: str  # "manual_credentials"
    required_fields: List[CredentialField]
    instructions: Optional[str] = None


class OAuthAuthResponse(BaseModel):
    authorization_url: str


# Union type for auth responses
AuthInitiateResponse = Union[ManualCredentialsAuthResponse, OAuthAuthResponse]


# Endpoints will be added below


@router.get("/{connector_type}/auth_status", response_model=ConnectorAuthStatus)
async def get_auth_status_for_connector(
    connector_type: str, connector_service: ConnectorService = Depends(get_connector_service)
):
    """Checks the current authentication status for the given connector type."""
    try:
        connector = await connector_service.get_connector(connector_type)
        status = await connector.get_auth_status()
        return status
    except ValueError as e:
        # Handle cases where the connector type is unsupported or other init errors
        logger.error(
            f"Value error getting auth status for {connector_type} for user {connector_service.user_identifier}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e))
    except ConnectionError as e:
        # Handle cases where the connector itself has issues connecting (e.g. to external service if checked early)
        logger.error(f"Connection error for {connector_type} for user {connector_service.user_identifier}: {e}")
        raise HTTPException(status_code=503, detail=f"Connector service unavailable: {str(e)}")
    except Exception as e:
        logger.exception(
            f"Unexpected error getting auth status for {connector_type} "
            f"for user {connector_service.user_identifier}: {e}"
        )
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@router.get("/{connector_type}/auth/initiate_url", response_model=AuthInitiateResponse)
async def get_initiate_auth_url(
    request: Request,
    connector_type: str,
    app_redirect_uri: Optional[str] = None,
    service: ConnectorService = Depends(get_connector_service),
):
    """Return the provider's *authorization_url* for the given connector.

    The method mirrors the logic of the `/auth/initiate` endpoint but sends a
    JSON payload instead of a redirect so that browsers can stay on the same
    origin until they intentionally navigate away.

    For OAuth-based connectors, this returns authorization_url.
    For manual credential connectors, this returns the credential form specification.
    """

    try:
        connector = await service.get_connector(connector_type)
        auth_details = await connector.initiate_auth()

        # Check if this is a manual credentials flow
        if auth_details.get("auth_type") == "manual_credentials":
            # For manual credentials, return the form specification directly
            return ManualCredentialsAuthResponse(**auth_details)

        # For OAuth flows, continue with existing logic
        authorization_url = auth_details.get("authorization_url")
        state = auth_details.get("state")

        if not authorization_url or not state:
            logger.error(
                "Connector '%s' did not return authorization URL or state for user '%s'.",
                connector_type,
                service.user_identifier,
            )
            raise HTTPException(status_code=500, detail="Failed to initiate authentication with the provider.")

        # Store state and connector type in session for later validation.
        request.session["oauth_state"] = state
        request.session["connector_type_for_callback"] = connector_type
        if app_redirect_uri:
            request.session["app_redirect_uri"] = app_redirect_uri

        # Store AuthContext for the callback as a JSON string
        auth_context_json_str = service.auth_context.model_dump_json()  # Use .model_dump_json()
        request.session["oauth_auth_context_json"] = auth_context_json_str  # Store as JSON string

        logger.info("Prepared auth URL for '%s' for user '%s'.", connector_type, service.user_identifier)

        return OAuthAuthResponse(authorization_url=authorization_url)

    except ValueError as ve:
        logger.warning("Auth URL preparation for '%s' failed: %s", connector_type, ve)
        if "Unsupported connector type" in str(ve):
            raise HTTPException(status_code=404, detail=str(ve))
        raise HTTPException(status_code=500, detail=str(ve))
    except NotImplementedError:
        raise HTTPException(status_code=501, detail=f"Connector '{connector_type}' not fully implemented.")
    except Exception as exc:
        logger.exception("Error preparing auth URL for '%s': %s", connector_type, exc)
        raise HTTPException(status_code=500, detail="Internal server error preparing authentication URL.")


@router.get("/{connector_type}/oauth2callback", response_model=None)
async def connector_oauth_callback(
    request: Request,  # For accessing session and query parameters
    connector_type: str,  # From path, to verify against session
    code: Optional[str] = None,  # OAuth code from query parameters
    state: Optional[str] = None,  # State from query parameters
    error: Optional[str] = None,  # Optional error from OAuth provider
    error_description: Optional[str] = None,  # Optional error description
):
    """
    Handles the OAuth 2.0 callback from the authentication provider.
    Validates state, finalizes authentication, and stores credentials.
    """
    logger.info(
        f"Received OAuth callback for '{connector_type}'. Code: {'SET' if code else 'NOT_SET'}, "
        f"State: {'SET' if state else 'NOT_SET'}, Error: {error}"
    )

    if error:
        logger.error(f"OAuth provider returned error for '{connector_type}': {error} - {error_description}")
        raise HTTPException(status_code=400, detail=f"OAuth provider error: {error_description or error}")

    # --- Session Data Retrieval and Validation ---
    stored_state = request.session.pop("oauth_state", None)
    stored_connector_type = request.session.pop("connector_type_for_callback", None)
    auth_context_json_str = request.session.pop("oauth_auth_context_json", None)

    if not stored_state or not state or stored_state != state:
        logger.error(
            f"OAuth state mismatch for '{connector_type}'. Expected: '{stored_state}', "
            f"Received: '{state}'. IP: {request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(status_code=400, detail="Invalid OAuth state. Authentication failed.")

    if not stored_connector_type or stored_connector_type != connector_type:
        logger.error(
            f"Connector type mismatch in OAuth callback. Expected: '{stored_connector_type}', Path: '{connector_type}'."
        )
        raise HTTPException(status_code=400, detail="Connector type mismatch during OAuth callback.")

    if not auth_context_json_str:
        logger.error(f"AuthContext not found in session during OAuth callback for '{connector_type}'.")
        raise HTTPException(status_code=400, detail="Authentication context missing. Please restart the auth flow.")

    # --- Service and Connector Initialization ---
    try:
        auth_context = AuthContext(**json.loads(auth_context_json_str))
        service = ConnectorService(auth_context=auth_context)
        # connector variable will be defined by calling service.get_connector
        # but we need to ensure it's done within a try block that can handle connector-specific errors.
    except Exception as e:  # Covers AuthContext reconstruction and ConnectorService instantiation
        logger.error(f"Failed to reconstruct AuthContext or instantiate ConnectorService: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during authentication setup.")

    # --- Code Validation (can now use service.user_identifier if needed in logs) ---
    if not code:
        user_id_for_log = service.user_identifier if "service" in locals() else "unknown user"
        logger.error(
            f"Authorization code not found in OAuth callback for '{connector_type}' for user '{user_id_for_log}'."
        )
        raise HTTPException(status_code=400, detail="Authorization code missing from provider callback.")

    # Reconstruct the full authorization response URL that the provider redirected to.
    authorization_response_url = str(request.url)
    logger.debug(f"Full authorization_response_url for '{connector_type}': {authorization_response_url}")

    try:
        # Now get the connector, as service is initialized
        connector = await service.get_connector(connector_type)

        auth_data = {
            "authorization_response_url": authorization_response_url,
            "state": state,
        }

        success = await connector.finalize_auth(auth_data)  # Correctly call on connector

        if success:
            logger.info(
                f"Successfully finalized authentication for '{connector_type}' for user '{service.user_identifier}'."
            )
            app_redirect_uri = request.session.pop("app_redirect_uri", None)
            if app_redirect_uri:
                logger.info(f"Redirecting to frontend app_redirect_uri: {app_redirect_uri}")
                return RedirectResponse(url=app_redirect_uri)
            else:
                logger.info("No app_redirect_uri found, showing generic success page.")
                html_content = """
                <html><head><title>Authentication Successful</title></head>
                <body><h1>Authentication Successful</h1>
                <p>You have successfully authenticated. You can now close this window and return to the application.</p>
                </body></html>
                """
                return HTMLResponse(content=html_content)
        else:
            logger.error(
                f"Failed to finalize auth for '{connector_type}' with user '{service.user_identifier}' "
                f"(connector returned False)."
            )
            raise HTTPException(status_code=500, detail="Failed to finalize authentication with the provider.")

    except ValueError as ve:
        logger.error(f"Error during OAuth callback for '{connector_type}' for user '{service.user_identifier}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except NotImplementedError:
        logger.error(f"Connector '{connector_type}' is not fully implemented for auth finalization.")
        raise HTTPException(status_code=501, detail=f"Connector '{connector_type}' not fully implemented.")
    except Exception as e:
        user_id_for_log = service.user_identifier if "service" in locals() else "unknown user"
        logger.exception(
            f"Unexpected error during OAuth callback for '{connector_type}' for user '{user_id_for_log}': {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error during authentication callback.")


@router.post("/{connector_type}/auth/finalize", response_model=Dict[str, Any])
async def finalize_manual_auth(
    connector_type: str,
    credentials_request: ManualCredentialsRequest,
    service: ConnectorService = Depends(get_connector_service),
):
    """Finalize authentication using manual credentials.

    This endpoint is used for connectors that require manual credential input
    (like Zotero) instead of OAuth flows.
    """
    try:
        connector = await service.get_connector(connector_type)

        # Attempt to finalize authentication with the provided credentials
        success = await connector.finalize_auth(credentials_request.credentials)

        if success:
            logger.info(
                f"Successfully finalized manual authentication for '{connector_type}' for user '{service.user_identifier}'."
            )
            return {"status": "success", "message": f"Successfully authenticated with {connector_type}."}
        else:
            logger.error(
                f"Failed to finalize manual auth for '{connector_type}' with user '{service.user_identifier}' "
                f"(connector returned False)."
            )
            raise HTTPException(status_code=400, detail="Invalid credentials provided.")

    except ValueError as ve:
        logger.error(f"Error during manual auth for '{connector_type}' for user '{service.user_identifier}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except NotImplementedError:
        logger.error(f"Connector '{connector_type}' is not fully implemented for manual auth finalization.")
        raise HTTPException(status_code=501, detail=f"Connector '{connector_type}' not fully implemented.")
    except Exception as e:
        user_id_for_log = service.user_identifier if "service" in locals() else "unknown user"
        logger.exception(
            f"Unexpected error during manual auth for '{connector_type}' for user '{user_id_for_log}': {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error during manual authentication.")


# Response model for list_files
class FileListResponse(BaseModel):
    files: List[ConnectorFile]
    next_page_token: Optional[str] = None


@router.get("/{connector_type}/files", response_model=FileListResponse)
async def list_files_for_connector(
    connector_type: str,
    path: Optional[str] = None,  # Connector-specific path (e.g., folder_id)
    page_token: Optional[str] = None,
    q_filter: Optional[str] = None,  # Connector-specific search/filter query string
    page_size: int = 100,  # Default page size, can be overridden by query param
    service: ConnectorService = Depends(get_connector_service),
):
    """Lists files and folders from the specified connector."""
    try:
        connector = await service.get_connector(connector_type)
        # Pass all relevant parameters to the connector's list_files method
        # The connector itself will decide how to use them (e.g. in **kwargs or named params)
        file_listing = await connector.list_files(
            path=path,
            page_token=page_token,
            q_filter=q_filter,  # Pass the filter query
            page_size=page_size,  # Pass the page size
        )
        # Ensure the response from the connector matches the FileListResponse model.
        # The connector.list_files method should return a dict like:
        # {"files": [ConnectorFile, ...], "next_page_token": "..."}
        return file_listing
    except ValueError as ve:  # Raised by get_connector or if connector has issues
        logger.error(f"Error listing files for '{connector_type}' for user '{service.user_identifier}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except ConnectionError as ce:  # If connector.list_files raises connection issues
        logger.error(
            f"Connection error listing files for '{connector_type}' for user '{service.user_identifier}': {ce}"
        )
        raise HTTPException(status_code=503, detail=f"Connector service unavailable: {str(ce)}")
    except NotImplementedError:
        logger.error(f"Connector '{connector_type}' does not support listing files or is not fully implemented.")
        raise HTTPException(status_code=501, detail=f"File listing not implemented for connector '{connector_type}'.")
    except Exception as e:
        logger.exception(
            f"Unexpected error listing files for '{connector_type}' for user '{service.user_identifier}': {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error listing files.")


@router.post("/{connector_type}/ingest", status_code=202)
async def ingest_file(
    connector_type: str,
    ingest_request: ConnectorIngestRequest,
    auth: AuthContext = Depends(verify_token),
    redis: Redis = Depends(get_redis_pool),
    document_service: DocumentService = Depends(get_document_service),
):
    """Ingest a single file from a connector."""
    try:
        connector_service_instance = ConnectorService(auth_context=auth)
        result = await connector_service_instance.ingest_file_from_connector(
            connector_type=connector_type,
            file_id=ingest_request.file_id,
            folder_name=ingest_request.folder_name,
            document_service=document_service,
            auth=auth,
            redis=redis,
            metadata=ingest_request.metadata,
            rules=getattr(ingest_request, "rules", None),
        )
        return result
    except Exception as e:
        logger.error(f"Error ingesting file {ingest_request.file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{connector_type}/ingest-repository", status_code=202)
async def ingest_repository(
    connector_type: str,
    ingest_request: GitHubRepositoryIngestRequest,
    auth: AuthContext = Depends(verify_token),
    redis: Redis = Depends(get_redis_pool),
    document_service: DocumentService = Depends(get_document_service),
):
    """Ingest an entire GitHub repository."""
    logger.info(
        f"Repository ingestion endpoint called with connector_type={connector_type}, repo_path={ingest_request.repo_path}"
    )
    connector_service_instance = ConnectorService(auth_context=auth)
    connector = await connector_service_instance.get_connector(ingest_request.connector_type)
    if connector.connector_type != "github":
        raise HTTPException(status_code=400, detail="Repository ingestion is only supported for GitHub")

    auth_status = await connector.get_auth_status()
    if not auth_status.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated with connector")

    try:
        documents = await connector.ingest_repository(
            repo_path=ingest_request.repo_path,
            document_service=document_service,
            auth_context=auth,
            redis=redis,
            folder_name=ingest_request.folder_name,
            metadata=ingest_request.metadata,
            include_patterns=ingest_request.include_patterns,
            ignore_patterns=ingest_request.ignore_patterns,
            compress=ingest_request.compress,
            force=ingest_request.force,
        )
        return {"status": "Repository ingestion started", "documents": documents}
    except Exception as e:
        logger.error(f"Error ingesting repository {ingest_request.repo_path}: {e}")
        error_detail = str(e)
        if hasattr(e, "detail"):
            error_detail = e.detail
        raise HTTPException(status_code=500, detail=f"Failed to ingest repository: {error_detail}")


@router.post("/status")
async def get_status(
    auth_request: ConnectorAuthRequest,
    auth: AuthContext = Depends(verify_token),
    connector_service: ConnectorService = Depends(get_connector_service),
):
    """Get the authentication status for a connector."""
    connector = await connector_service.get_connector(auth_request.connector_type, auth.user_id)
    return await connector.get_auth_status()


@router.post("/initiate-auth")
async def initiate_auth(
    auth_request: ConnectorAuthRequest,
    auth: AuthContext = Depends(verify_token),
    connector_service: ConnectorService = Depends(get_connector_service),
):
    """Initiate the OAuth flow for a connector."""
    connector = await connector_service.get_connector(auth_request.connector_type, auth.user_id)
    return await connector.initiate_auth()


@router.post("/finalize-auth")
async def finalize_auth(
    auth_response: ConnectorAuthResponse,
    auth: AuthContext = Depends(verify_token),
    connector_service: ConnectorService = Depends(get_connector_service),
):
    """Finalize the OAuth flow and exchange the code for a token."""
    connector = await connector_service.get_connector(auth_response.connector_type, auth.user_id)
    await connector.finalize_auth(auth_response.auth_response_data)
    return {"status": "success"}


@router.post("/disconnect")
async def disconnect(
    auth_request: ConnectorAuthRequest,
    auth: AuthContext = Depends(verify_token),
    connector_service: ConnectorService = Depends(get_connector_service),
):
    """Disconnect from a connector and remove credentials."""
    connector = await connector_service.get_connector(auth_request.connector_type, auth.user_id)
    await connector.disconnect()
    return {"status": "success"}


@router.post("/list-files")
async def list_files(
    list_request: ConnectorListFilesRequest,
    auth: AuthContext = Depends(verify_token),
    connector_service: ConnectorService = Depends(get_connector_service),
):
    """List files from a connector."""
    connector = await connector_service.get_connector(list_request.connector_type, auth.user_id)
    return await connector.list_files(path=list_request.path, page_token=list_request.page_token)
