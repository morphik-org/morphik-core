import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query

from core.auth_utils import verify_token
from core.config import get_settings
from core.limits_utils import check_and_increment_limits
from core.models.auth import AuthContext
from core.models.graph import Graph
from core.models.prompts import validate_prompt_overrides_with_http_exception
from core.models.request import CreateGraphRequest, UpdateGraphRequest
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# ---------------------------------------------------------------------------
# Router initialization & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/graph", tags=["Graph"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()


# Helper function to normalize folder_name parameter
def normalize_folder_name(folder_name: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """Convert string 'null' to None for folder_name parameter."""
    if folder_name is None:
        return None
    if isinstance(folder_name, str):
        return None if folder_name.lower() == "null" else folder_name
    if isinstance(folder_name, list):
        return [None if f.lower() == "null" else f for f in folder_name]
    return folder_name


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------


@router.post("/create", response_model=Graph)
@telemetry.track(operation_type="create_graph", metadata_resolver=telemetry.create_graph_metadata)
async def create_graph(
    request: CreateGraphRequest,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """Create a new graph based on document contents.

    The graph is created asynchronously. A stub graph record is returned with
    ``status = "processing"`` while a background task extracts entities and
    relationships.

    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="graph")

        # Enforce usage limits (cloud mode)
        if settings.MODE == "cloud" and auth.user_id:
            await check_and_increment_limits(auth, "graph", 1)

        # --------------------
        # Build system filters
        # --------------------
        system_filters: Dict[str, Any] = {}
        if request.folder_name is not None:
            normalized_folder_name = normalize_folder_name(request.folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if request.end_user_id:
            system_filters["end_user_id"] = request.end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        # Note: Don't add auth.app_id here - it's already handled in document retrieval

        # --------------------
        # Create stub graph
        # --------------------
        graph_stub = Graph(
            id=str(uuid.uuid4()),
            name=request.name,
            filters=request.filters,
            folder_name=system_filters.get("folder_name"),
            end_user_id=system_filters.get("end_user_id"),
            app_id=auth.app_id,
        )

        # Mark graph as processing
        graph_stub.system_metadata["status"] = "processing"
        graph_stub.system_metadata["created_at"] = datetime.now(UTC)
        graph_stub.system_metadata["updated_at"] = datetime.now(UTC)

        # Store the stub graph so clients can poll for status
        success = await document_service.db.store_graph(graph_stub, auth)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create graph stub")

        # --------------------
        # Background processing
        # --------------------
        async def _build_graph_async():
            try:
                await document_service.update_graph(
                    name=request.name,
                    auth=auth,
                    additional_filters=None,  # original filters already on stub
                    additional_documents=request.documents,
                    prompt_overrides=request.prompt_overrides,
                    system_filters=system_filters,
                    is_initial_build=True,  # Indicate this is the initial build
                )
            except Exception as e:
                logger.error(f"Graph creation failed for {request.name}: {e}")
                # Update graph status to failed
                existing = await document_service.db.get_graph(request.name, auth, system_filters=system_filters)
                if existing:
                    existing.system_metadata["status"] = "failed"
                    existing.system_metadata["error"] = str(e)
                    existing.system_metadata["updated_at"] = datetime.now(UTC)
                    await document_service.db.update_graph(existing)

        import asyncio

        asyncio.create_task(_build_graph_async())

        # Return the stub graph immediately
        return graph_stub
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="graph", error=e)


@router.get("/{name}", response_model=Graph)
@telemetry.track(operation_type="get_graph", metadata_resolver=telemetry.get_graph_metadata)
async def get_graph(
    name: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
) -> Graph:
    """
    Get a graph by name.

    This endpoint retrieves a graph by its name if the user has access to it.
    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        # Note: Don't add auth.app_id here - it's already handled in document retrieval

        graph = await document_service.db.get_graph(name, auth, system_filters)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
        return graph
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[Graph])
@telemetry.track(operation_type="list_graphs", metadata_resolver=telemetry.list_graphs_metadata)
async def list_graphs(
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
) -> List[Graph]:
    """
    List all graphs the user has access to.

    This endpoint retrieves all graphs the user has access to.

    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        # Note: Don't add auth.app_id here - it's already handled in document retrieval

        return await document_service.db.list_graphs(auth, system_filters)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}/visualization", response_model=Dict[str, Any])
@telemetry.track(operation_type="get_graph_visualization", metadata_resolver=telemetry.get_graph_metadata)
async def get_graph_visualization(
    name: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get graph visualization data.

    This endpoint retrieves the nodes and links data needed for graph visualization.
    It works with both local and API-based graph services.
    """
    try:
        return await document_service.get_graph_visualization_data(
            name=name,
            auth=auth,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting graph visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{name}/update", response_model=Graph)
@telemetry.track(operation_type="update_graph", metadata_resolver=telemetry.update_graph_metadata)
async def update_graph(
    name: str,
    request: UpdateGraphRequest,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """
    Update an existing graph with new documents.

    This endpoint processes additional documents based on the original graph filters
    and/or new filters/document IDs, extracts entities and relationships, and
    updates the graph with new information.

    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="graph")

        # Create system filters for folder and user scoping
        system_filters = {}
        if request.folder_name:
            system_filters["folder_name"] = request.folder_name
        if request.end_user_id:
            system_filters["end_user_id"] = request.end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        # Note: Don't add auth.app_id here - it's already handled in document retrieval

        return await document_service.update_graph(
            name=name,
            auth=auth,
            additional_filters=request.additional_filters,
            additional_documents=request.additional_documents,
            prompt_overrides=request.prompt_overrides,
            system_filters=system_filters,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="graph", error=e)
    except Exception as e:
        logger.error(f"Error updating graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
@telemetry.track(operation_type="delete_graph")
async def delete_graph(
    name: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Delete a graph by name."""
    try:
        # Check if graph exists first
        graph = await document_service.db.get_graph(name, auth)
        if not graph:
            # Return success if graph doesn't exist (idempotent delete)
            return {"status": "success", "message": f"Graph '{name}' deleted successfully"}

        # Get the graph service
        graph_service = document_service.graph_service

        # Check if it's the MorphikGraphService
        from core.services.morphik_graph_service import MorphikGraphService

        if isinstance(graph_service, MorphikGraphService):
            # Use the graph service to delete, which will handle both API and database deletion
            try:
                success = await graph_service.delete_graph(name, auth)
            except ValueError as e:
                # Handle "graph not found" from MorphikGraphService
                if "not found" in str(e).lower():
                    return {"status": "success", "message": f"Graph '{name}' deleted successfully"}
                raise
        else:
            # Fallback to database-only deletion
            success = await document_service.db.delete_graph(name, auth)

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete graph '{name}'")

        return {"status": "success", "message": f"Graph '{name}' deleted successfully"}

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}/status", response_model=Dict[str, Any])
@telemetry.track(operation_type="get_graph_status")
async def get_graph_status(
    name: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight endpoint to check graph status with automatic status synchronization.

    This endpoint is designed for polling during graph creation/update operations.
    For MorphikGraphService backends, it synchronizes the local graph status with
    the remote service status automatically.
    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Get the local graph record
        graph = await document_service.db.get_graph(name, auth, system_filters)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")

        # Check if we're using MorphikGraphService and if the graph is still processing
        graph_service = document_service.graph_service
        from core.services.morphik_graph_service import MorphikGraphService

        if isinstance(graph_service, MorphikGraphService) and graph.system_metadata.get("status") == "processing":
            # Sync with remote status
            try:
                remote_graph = await graph_service.get_graph(name, auth)
                if remote_graph:
                    # Update local graph with remote status
                    graph.system_metadata["status"] = remote_graph.system_metadata.get("status", "processing")
                    if "error" in remote_graph.system_metadata:
                        graph.system_metadata["error"] = remote_graph.system_metadata["error"]
                    if "node_count" in remote_graph.system_metadata:
                        graph.system_metadata["node_count"] = remote_graph.system_metadata["node_count"]
                    if "edge_count" in remote_graph.system_metadata:
                        graph.system_metadata["edge_count"] = remote_graph.system_metadata["edge_count"]
                    graph.system_metadata["updated_at"] = datetime.now(UTC)

                    # Save the updated status
                    await document_service.db.update_graph(graph)
            except Exception as e:
                logger.warning(f"Failed to sync graph status with remote service: {e}")
                # Continue with local status if sync fails

        # Build response
        response = {
            "status": graph.system_metadata.get("status", "unknown"),
            "created_at": graph.system_metadata.get("created_at"),
            "updated_at": graph.system_metadata.get("updated_at"),
        }

        # Add optional fields if present
        if "error" in graph.system_metadata:
            response["error"] = graph.system_metadata["error"]
        if "node_count" in graph.system_metadata:
            response["node_count"] = graph.system_metadata["node_count"]
        if "edge_count" in graph.system_metadata:
            response["edge_count"] = graph.system_metadata["edge_count"]

        # Add a message based on status
        if response["status"] == "processing":
            response["message"] = "Graph is being processed"
        elif response["status"] == "completed":
            response["message"] = "Graph processing completed successfully"
        elif response["status"] == "failed":
            response["message"] = "Graph processing failed"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{workflow_id}/status", response_model=Dict[str, Any])
@telemetry.track(operation_type="check_workflow_status", metadata_resolver=telemetry.workflow_status_metadata)
async def check_workflow_status(
    workflow_id: str,
    run_id: Optional[str] = None,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Check the status of a graph build/update workflow.

    This endpoint polls the external graph API to check the status of an async operation.
    """
    try:
        # Get the graph service (either local or API-based)
        graph_service = document_service.graph_service

        # Check if it's the MorphikGraphService
        from core.services.morphik_graph_service import MorphikGraphService

        if isinstance(graph_service, MorphikGraphService):
            # Use the new check_workflow_status method
            result = await graph_service.check_workflow_status(workflow_id=workflow_id, run_id=run_id, auth=auth)

            # If the workflow is completed, update the corresponding graph status
            if result.get("status") == "completed":
                # Extract graph_id from workflow_id (format: "build-update-{graph_name}-...")
                # This is a simple heuristic, adjust based on actual workflow_id format
                parts = workflow_id.split("-")
                if len(parts) >= 3:
                    graph_name = parts[2]
                    try:
                        # Find and update the graph
                        graphs = await document_service.db.list_graphs(auth)
                        for graph in graphs:
                            if graph.name == graph_name or workflow_id in graph.system_metadata.get("workflow_id", ""):
                                graph.system_metadata["status"] = "completed"
                                # Clear workflow tracking data
                                graph.system_metadata.pop("workflow_id", None)
                                graph.system_metadata.pop("run_id", None)
                                await document_service.db.update_graph(graph)
                                logger.info(f"Auto-updated graph '{graph.name}' status to completed")
                                break
                    except Exception as e:
                        logger.warning(f"Failed to update graph status after workflow completion: {e}")
            elif result.get("status") == "failed":
                # Also handle failed status updates
                parts = workflow_id.split("-")
                if len(parts) >= 3:
                    graph_name = parts[2]
                    try:
                        graphs = await document_service.db.list_graphs(auth)
                        for graph in graphs:
                            if graph.name == graph_name or workflow_id in graph.system_metadata.get("workflow_id", ""):
                                graph.system_metadata["status"] = "failed"
                                graph.system_metadata["error"] = result.get("error", "Workflow failed")
                                await document_service.db.update_graph(graph)
                                logger.info(f"Auto-updated graph '{graph.name}' status to failed")
                                break
                    except Exception as e:
                        logger.warning(f"Failed to update graph status after workflow failure: {e}")

            return result
        else:
            # For local graph service, workflows complete synchronously
            return {"status": "completed", "result": {"message": "Local graph operations complete synchronously"}}

    except Exception as e:
        logger.error(f"Error checking workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
