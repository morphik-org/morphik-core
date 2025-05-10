import logging
from typing import Any, Dict, List, Optional

import arq  # Added for Redis
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

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
    file_id: str
    morphik_folder_name: Optional[str] = None
    morphik_end_user_id: Optional[str] = None


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


@router.get("/{connector_type}/auth/initiate")  # No specific response model, as it redirects
async def initiate_connector_authentication(
    request: Request,  # FastAPI Request object to access session
    connector_type: str,
    service: ConnectorService = Depends(get_connector_service),
):
    """
    Initiates the OAuth 2.0 authentication flow for the specified connector.
    Stores state in session and redirects user to the provider's authorization URL.
    """
    try:
        connector = await service.get_connector(connector_type)
        auth_details = await connector.initiate_auth()

        authorization_url = auth_details.get("authorization_url")
        state = auth_details.get("state")

        if not authorization_url or not state:
            logger.error(
                f"Connector '{connector_type}' did not return authorization URL or state "
                f"for user '{service.user_identifier}'."
            )
            raise HTTPException(status_code=500, detail="Failed to initiate authentication with the provider.")

        # Store state and connector type in session for later validation in the callback
        request.session["oauth_state"] = state
        request.session["connector_type_for_callback"] = connector_type
        logger.info(
            f"Initiating auth for '{connector_type}' for user '{service.user_identifier}'. "
            f"Redirecting to: {authorization_url[:70]}..."
        )
        return RedirectResponse(url=authorization_url)

    except ValueError as ve:  # Raised by get_connector for unsupported type or by connector for config issues
        logger.warning(f"Auth initiation for '{connector_type}' failed: {ve} for user '{service.user_identifier}'")
        # Determine if it's a 404 (unsupported) or 500 (config error within connector)
        if "Unsupported connector type" in str(ve):
            raise HTTPException(status_code=404, detail=str(ve))
        else:
            # E.g. Google Client ID/Secret not configured in GoogleDriveConnector
            raise HTTPException(
                status_code=500, detail=f"Configuration error for connector '{connector_type}': {str(ve)}"
            )
    except NotImplementedError:
        logger.error(f"Connector '{connector_type}' is not fully implemented for auth initiation.")
        raise HTTPException(status_code=501, detail=f"Connector '{connector_type}' not fully implemented.")
    except Exception as e:
        logger.exception(
            f"Error initiating auth for connector '{connector_type}' for user '{service.user_identifier}': {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error initiating authentication.")


@router.get("/{connector_type}/oauth2callback")
async def connector_oauth_callback(
    request: Request,  # For accessing session and query parameters
    connector_type: str,  # From path, to verify against session
    code: Optional[str] = None,  # OAuth code from query parameters
    state: Optional[str] = None,  # State from query parameters
    error: Optional[str] = None,  # Optional error from OAuth provider
    error_description: Optional[str] = None,  # Optional error description
    service: ConnectorService = Depends(get_connector_service),
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
        # You might want to redirect to a frontend error page here
        raise HTTPException(status_code=400, detail=f"OAuth provider error: {error_description or error}")

    stored_state = request.session.pop("oauth_state", None)
    stored_connector_type = request.session.pop("connector_type_for_callback", None)

    if not stored_state or not state or stored_state != state:
        logger.error(
            f"OAuth state mismatch for '{connector_type}'. Expected: '{stored_state}', "
            f"Received: '{state}'. IP: {request.client.host if request.client else 'unknown'}"
        )
        # Redirect to an error page or show a generic error
        raise HTTPException(status_code=400, detail="Invalid OAuth state. Authentication failed.")

    if not stored_connector_type or stored_connector_type != connector_type:
        logger.error(
            f"Connector type mismatch in OAuth callback. Expected: '{stored_connector_type}', Path: '{connector_type}'."
        )
        raise HTTPException(status_code=400, detail="Connector type mismatch during OAuth callback.")

    if not code:
        logger.error(f"Authorization code not found in OAuth callback for '{connector_type}'.")
        raise HTTPException(status_code=400, detail="Authorization code missing from provider callback.")

    # Reconstruct the full authorization response URL that the provider redirected to.
    # The `google-auth-oauthlib` flow.fetch_token expects the full URL.
    authorization_response_url = str(request.url)
    logger.debug(f"Full authorization_response_url for '{connector_type}': {authorization_response_url}")

    try:
        connector = await service.get_connector(connector_type)

        # The `finalize_auth` method expects a dictionary with the full response URL and the validated state.
        auth_data = {
            "authorization_response_url": authorization_response_url,
            "state": state,  # Pass the received (and validated) state to the connector
        }

        success = await connector.finalize_auth(auth_data)

        if success:
            logger.info(
                f"Successfully finalized authentication for '{connector_type}' for user '{service.user_identifier}'."
            )
            # In a real app, redirect to a frontend page indicating success
            # For example: return RedirectResponse(url="/profile?auth_success="+connector_type)
            # Redirect to the frontend connections page with a success indicator
            frontend_redirect_url = f"/ee/connections?auth_success={connector_type}"
            logger.info(f"Redirecting to frontend: {frontend_redirect_url}")
            return RedirectResponse(url=frontend_redirect_url)
        else:
            logger.error(
                f"Failed to finalize auth for '{connector_type}' with user '{service.user_identifier}' "
                f"(connector returned False)."
            )
            # Redirect to a frontend error page
            raise HTTPException(status_code=500, detail="Failed to finalize authentication with the provider.")

    except ValueError as ve:  # Raised by get_connector or if connector has issues not caught by its finalize_auth
        logger.error(f"Error during OAuth callback for '{connector_type}' for user '{service.user_identifier}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except NotImplementedError:
        logger.error(f"Connector '{connector_type}' is not fully implemented for auth finalization.")
        raise HTTPException(status_code=501, detail=f"Connector '{connector_type}' not fully implemented.")
    except Exception as e:
        logger.exception(
            f"Unexpected error during OAuth callback for '{connector_type}' for user '{service.user_identifier}': {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error during authentication callback.")


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


@router.post("/{connector_type}/ingest", response_model=Dict[str, Any])
async def ingest_file_from_connector(
    connector_type: str,
    ingest_request: IngestFromConnectorRequest,
    auth_context: AuthContext = Depends(verify_token),  # For DocumentService & connector
    connector_service: ConnectorService = Depends(get_connector_service),
    doc_service: DocumentService = Depends(get_document_service),
    redis_pool_instance: arq.ArqRedis = Depends(get_redis_pool),  # Retained as per original router structure
):
    """Downloads a file from the connector and ingests it into Morphik via DocumentService."""
    logger.info(
        f"Ingesting file_id: {ingest_request.file_id} from connector: {connector_type} "
        f"for user: {auth_context.user_id or auth_context.entity_id}"
    )
    try:
        connector = await connector_service.get_connector(connector_type)

        # 1. Get file metadata from connector
        file_metadata = await connector.get_file_metadata_by_id(ingest_request.file_id)
        if not file_metadata:
            logger.error(f"File not found via connector: {ingest_request.file_id}")
            raise HTTPException(status_code=404, detail="File not found via connector.")

        # 2. Download file content from connector
        file_content_stream = await connector.download_file_by_id(ingest_request.file_id)
        if not file_content_stream:
            logger.error(f"Failed to download file from connector: {ingest_request.file_id}")
            raise HTTPException(status_code=500, detail="Failed to download file from connector.")

        file_content_bytes = file_content_stream.getvalue()  # Assuming BytesIO

        # 3. Ingest into Morphik using DocumentService
        # The doc_service.ingest_file_content method is expected to handle storage and enqueueing the processing job.
        # It will need redis for enqueuing, which is why redis_pool_instance is passed.
        morphik_doc = await doc_service.ingest_file_content(
            file_content_bytes=file_content_bytes,
            filename=file_metadata.name,
            content_type=file_metadata.mime_type,
            metadata=file_metadata.model_dump(),  # Pass connector file metadata
            auth=auth_context,  # Pass the auth context
            redis=redis_pool_instance,  # Pass redis for the service to use
            folder_name=ingest_request.morphik_folder_name,
            end_user_id=ingest_request.morphik_end_user_id,
            # rules and use_colpali can be added if needed
        )

        return {
            "message": f"File '{file_metadata.name}' successfully queued for ingestion.",
            "morphik_document_id": morphik_doc.external_id,
            "status_path": f"/documents/{morphik_doc.external_id}/status",
        }

    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except ValueError as ve:  # E.g., unsupported connector type
        logger.error(f"Value error during ingestion from {connector_type}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error ingesting from connector '{connector_type}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during file ingestion.")


@router.post("/{connector_type}/disconnect", response_model=Dict[str, Any])
async def disconnect_from_connector(
    connector_type: str,
    service: ConnectorService = Depends(get_connector_service),
):
    """Disconnects the user from the specified connector by removing stored credentials."""
    user_log_id = service.user_identifier
    logger.info(f"Attempting to disconnect from '{connector_type}' for user '{user_log_id}'.")
    try:
        connector = await service.get_connector(connector_type)
        success = await connector.disconnect()

        if success:
            logger.info(f"Successfully disconnected from '{connector_type}' for user '{user_log_id}'.")
            return {"status": "success", "message": f"Successfully disconnected from {connector_type}."}
        else:
            logger.warning(
                f"Disconnection from '{connector_type}' for user '{user_log_id}' "
                f"indicated failure by connector (returned False)."
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to disconnect from {connector_type}. An issue occurred on the server."
            )

    except ValueError as ve:  # From get_connector
        logger.error(f"ValueError during disconnect from '{connector_type}' for user '{user_log_id}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except ConnectionError as ce:  # If connector.disconnect had an issue
        logger.error(f"ConnectionError during disconnect from '{connector_type}' for user '{user_log_id}': {ce}")
        raise HTTPException(status_code=503, detail=f"Connector service error during disconnect: {str(ce)}")
    except Exception as e:
        logger.exception(f"Unexpected error during disconnect from '{connector_type}' for user '{user_log_id}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during disconnect operation.")
