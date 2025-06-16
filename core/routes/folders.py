import logging
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException

from core.auth_utils import verify_token
from core.config import get_settings
from core.models.auth import AuthContext
from core.models.folders import Folder, FolderCreate, FolderSummary
from core.models.request import SetFolderRuleRequest
from core.services.telemetry import TelemetryService
from core.services_init import document_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/folders", tags=["Folders"])
settings = get_settings()
telemetry = TelemetryService()


# Helper function to normalize folder_name parameter
def normalize_folder_name(
    folder_name: Optional[Union[str, List[str]]]
) -> Optional[Union[str, List[str]]]:
    """Convert string 'null' to None for folder_name parameter."""
    if folder_name is None:
        return None
    if isinstance(folder_name, str):
        return None if folder_name.lower() == "null" else folder_name
    if isinstance(folder_name, list):
        return [None if f.lower() == "null" else f for f in folder_name]
    return folder_name


@router.post("/", response_model=Folder)
@telemetry.track(
    operation_type="create_folder",
    metadata_resolver=telemetry.create_folder_metadata
)
async def create_folder(
    folder_create: FolderCreate,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """Create a new folder.

    Args:
        folder_create: FolderCreate object with folder details
        auth: Authentication context

    Returns:
        Folder: The created folder

    Raises:
        HTTPException: 400 if folder already exists, 403 if access denied
    """
    try:
        # Check if folder already exists
        existing = await document_service.db.get_folder(
            folder_create.name, auth
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Folder '{folder_create.name}' already exists"
            )

        # Create folder
        folder = await document_service.db.create_folder(
            folder_create.name,
            folder_create.description,
            auth,
        )
        return folder
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create folder: {str(e)}"
        )


@router.get("/", response_model=List[Folder])
@telemetry.track(
    operation_type="list_folders",
    metadata_resolver=telemetry.list_folders_metadata
)
async def list_folders(
    auth: AuthContext = Depends(verify_token),
) -> List[Folder]:
    """List all folders accessible to the user.

    Args:
        auth: Authentication context

    Returns:
        List[Folder]: List of folders
    """
    try:
        folders = await document_service.db.list_folders(auth)
        return folders
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/summary", response_model=List[FolderSummary])
@telemetry.track(operation_type="list_folders_summary")
async def list_folder_summaries(
    auth: AuthContext = Depends(verify_token)
) -> List[FolderSummary]:
    """List folder summaries with document counts.

    Args:
        auth: Authentication context

    Returns:
        List[FolderSummary]: List of folder summaries
    """
    try:
        summaries = await document_service.db.list_folder_summaries(auth)
        return summaries
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{folder_id}", response_model=Folder)
@telemetry.track(
    operation_type="get_folder",
    metadata_resolver=telemetry.get_folder_metadata
)
async def get_folder(
    folder_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """Get a specific folder by ID or name.

    Args:
        folder_id: Folder ID or name
        auth: Authentication context

    Returns:
        Folder: The requested folder

    Raises:
        HTTPException: 404 if not found, 403 if access denied
    """
    try:
        # Try to get by ID first, then by name
        folder = await document_service.db.get_folder_by_id(folder_id, auth)
        if not folder:
            folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(
                status_code=404,
                detail=f"Folder '{folder_id}' not found"
            )
        return folder
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{folder_name}")
@telemetry.track(
    operation_type="delete_folder",
    metadata_resolver=telemetry.delete_folder_metadata
)
async def delete_folder(
    folder_name: str,
    auth: AuthContext = Depends(verify_token),
):
    """Delete a folder and optionally its documents.

    Args:
        folder_name: Name of the folder to delete
        auth: Authentication context

    Returns:
        Dict with deletion status
    """
    try:
        # Get folder
        folder = await document_service.db.get_folder(folder_name, auth)
        if not folder:
            raise HTTPException(
                status_code=404,
                detail=f"Folder '{folder_name}' not found"
            )

        # Check if folder has documents
        documents = await document_service.db.get_documents(
            auth, filters={"folder_name": folder_name}
        )

        if documents:
            # Move documents to no folder
            for doc in documents:
                doc.system_metadata["folder_name"] = None
                await document_service.db.update_document(
                    doc.external_id,
                    {"system_metadata": doc.system_metadata},
                    auth,
                )

        # Delete folder
        await document_service.db.delete_folder(folder.id, auth)

        return {
            "status": "deleted",
            "folder": folder_name,
            "documents_updated": len(documents),
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{folder_id}/documents/{document_id}")
@telemetry.track(
    operation_type="add_document_to_folder",
    metadata_resolver=telemetry.add_document_to_folder_metadata
)
async def add_document_to_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """Add a document to a folder.

    Args:
        folder_id: Folder ID or name
        document_id: Document ID
        auth: Authentication context

    Returns:
        Dict with operation status
    """
    try:
        # Get folder
        folder = await document_service.db.get_folder_by_id(folder_id, auth)
        if not folder:
            folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(
                status_code=404,
                detail=f"Folder '{folder_id}' not found"
            )

        # Update document
        await document_service.db.add_document_to_folder(
            document_id, folder.name, auth
        )

        return {
            "status": "added",
            "folder": folder.name,
            "document": document_id
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/{folder_id}/documents/{document_id}")
@telemetry.track(
    operation_type="remove_document_from_folder",
    metadata_resolver=telemetry.remove_document_from_folder_metadata
)
async def remove_document_from_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """Remove a document from a folder.

    Args:
        folder_id: Folder ID or name
        document_id: Document ID
        auth: Authentication context

    Returns:
        Dict with operation status
    """
    try:
        # Get folder
        folder = await document_service.db.get_folder_by_id(folder_id, auth)
        if not folder:
            folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(
                status_code=404,
                detail=f"Folder '{folder_id}' not found"
            )

        # Update document to remove folder
        doc = await document_service.db.get_document(document_id, auth)
        if doc:
            doc.system_metadata["folder_name"] = None
            await document_service.db.update_document(
                document_id,
                {"system_metadata": doc.system_metadata},
                auth,
            )

        return {
            "status": "removed",
            "folder": folder.name,
            "document": document_id
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{folder_id}/set_rule")
@telemetry.track(
    operation_type="set_folder_rule",
    metadata_resolver=telemetry.set_folder_rule_metadata
)
async def set_folder_rule(
    folder_id: str,
    request: SetFolderRuleRequest,
    auth: AuthContext = Depends(verify_token),
    apply_to_existing: bool = True,
):
    """Set a rule for a folder.

    Args:
        folder_id: Folder ID or name
        request: SetFolderRuleRequest with rule details
        auth: Authentication context
        apply_to_existing: Whether to apply rule to existing documents

    Returns:
        Dict with operation status
    """
    try:
        # Get folder
        folder = await document_service.db.get_folder_by_id(folder_id, auth)
        if not folder:
            folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(
                status_code=404,
                detail=f"Folder '{folder_id}' not found"
            )

        # Store rule in folder metadata
        if not folder.metadata:
            folder.metadata = {}
        if "rules" not in folder.metadata:
            folder.metadata["rules"] = []

        folder.metadata["rules"].append({
            "rule": request.rule,
            "type": request.type,
        })

        # Update folder
        await document_service.db.update_folder(
            folder.id,
            {"metadata": folder.metadata},
            auth,
        )

        # Apply to existing documents if requested
        documents_updated = 0
        if apply_to_existing and request.rule:
            documents = await document_service.db.get_documents(
                auth,
                filters={"folder_name": folder.name}
            )
            for doc in documents:
                if "rules" not in doc.metadata:
                    doc.metadata["rules"] = []
                doc.metadata["rules"].append(request.rule)
                await document_service.db.update_document(
                    doc.external_id,
                    {"metadata": doc.metadata},
                    auth,
                )
                documents_updated += 1

        return {
            "status": "rule_set",
            "folder": folder.name,
            "rule": request.rule,
            "documents_updated": documents_updated,
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))