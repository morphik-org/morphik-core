import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from core.auth_utils import verify_token
from core.database.postgres_database import InvalidMetadataFilterError
from core.models.auth import AuthContext
from core.models.folders import Folder, FolderCreate, FolderSummary
from core.models.request import FolderDetailsRequest
from core.models.responses import (
    DocumentAddToFolderResponse,
    DocumentDeleteResponse,
    FolderDeleteResponse,
    FolderDetails,
    FolderDetailsResponse,
    FolderDocumentInfo,
)
from core.routes.utils import project_document_fields
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# ---------------------------------------------------------------------------
# Router initialization & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/folders", tags=["Folders"])
logger = logging.getLogger(__name__)
telemetry = TelemetryService()


async def _resolve_folder(identifier: str, auth: AuthContext) -> Folder:
    """
    Resolve a folder identifier that might be either an ID or a name.
    """
    folder = await document_service.db.get_folder(identifier, auth)
    if folder:
        return folder

    folder = await document_service.db.get_folder_by_name(identifier, auth)
    if folder:
        return folder

    raise HTTPException(status_code=404, detail=f"Folder {identifier} not found")


# ---------------------------------------------------------------------------
# Folder management endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=Folder)
@telemetry.track(operation_type="create_folder", metadata_resolver=telemetry.create_folder_metadata)
async def create_folder(
    folder_create: FolderCreate,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Create a new folder.
    """
    try:
        # Validate folder name - no slashes allowed (nested folders not supported)
        if "/" in folder_create.name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid folder name '{folder_create.name}'. Folder names cannot contain '/'. "
                    f"Nested folders are not supported. Use '_' instead to denote subfolders "
                    f"(e.g., 'folder_subfolder_subsubfolder')."
                ),
            )

        # Create a folder object with explicit ID
        folder_id = str(uuid.uuid4())
        logger.info(f"Creating folder with ID: {folder_id}, auth.user_id: {auth.user_id}")

        # Set up access control with user_id
        folder = Folder(
            id=folder_id, name=folder_create.name, description=folder_create.description, app_id=auth.app_id
        )

        # Store in database
        success = await document_service.db.create_folder(folder, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create folder")

        return folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[Folder])
@telemetry.track(operation_type="list_folders", metadata_resolver=telemetry.list_folders_metadata)
async def list_folders(
    auth: AuthContext = Depends(verify_token),
) -> List[Folder]:
    """
    List all folders the user has access to.
    """
    try:
        folders = await document_service.db.list_folders(auth)
        return folders
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/details", response_model=FolderDetailsResponse)
async def folder_details(
    request: FolderDetailsRequest,
    auth: AuthContext = Depends(verify_token),
) -> FolderDetailsResponse:
    """
    Retrieve folder metadata with optional document statistics and projections.
    """

    try:
        if request.identifiers:
            resolved: Dict[str, Folder] = {}
            for identifier in request.identifiers:
                folder = await _resolve_folder(identifier, auth)
                key = folder.id or folder.name or identifier
                resolved[key] = folder
            target_folders = list(resolved.values())
        else:
            target_folders = await document_service.db.list_folders(auth)

        folder_entries: List[FolderDetails] = []

        for folder in target_folders:
            document_info: Optional[FolderDocumentInfo] = None

            if request.include_documents or request.include_document_count or request.include_status_counts:
                doc_result: Dict[str, Any]
                if not folder.name:
                    doc_result = {
                        "documents": [],
                        "returned_count": 0,
                        "total_count": 0 if request.include_document_count else None,
                        "status_counts": {},
                        "has_more": False,
                        "next_skip": None,
                    }
                else:
                    doc_result = await document_service.db.list_documents_flexible(
                        auth=auth,
                        skip=request.document_skip if request.include_documents else 0,
                        limit=request.document_limit if request.include_documents else 0,
                        filters=request.document_filters,
                        system_filters={"folder_name": folder.name},
                        include_total_count=request.include_document_count,
                        include_status_counts=request.include_status_counts,
                        include_folder_counts=False,
                        return_documents=request.include_documents,
                        sort_by=request.sort_by,
                        sort_direction=request.sort_direction,
                    )

                documents_payload: List[Any] = []
                if request.include_documents:
                    for document in doc_result.get("documents", []):
                        if hasattr(document, "model_dump"):
                            doc_dict = document.model_dump(mode="json")
                        elif hasattr(document, "dict"):
                            doc_dict = document.dict()
                        else:
                            doc_dict = dict(document)
                        documents_payload.append(project_document_fields(doc_dict, request.document_fields))

                returned_count = doc_result.get("returned_count", len(documents_payload))
                has_more = doc_result.get("has_more", False)
                next_skip = doc_result.get("next_skip")
                if request.include_documents and next_skip is None and has_more:
                    next_skip = request.document_skip + returned_count

                document_count = None
                if request.include_document_count:
                    document_count = doc_result.get("total_count")
                if document_count is None and request.include_documents:
                    document_count = returned_count

                status_counts = doc_result.get("status_counts") if request.include_status_counts else None

                document_info = FolderDocumentInfo(
                    documents=documents_payload,
                    document_count=document_count,
                    status_counts=status_counts,
                    skip=request.document_skip if request.include_documents else 0,
                    limit=request.document_limit if request.include_documents else 0,
                    returned_count=returned_count,
                    has_more=has_more,
                    next_skip=next_skip,
                )

            folder_entries.append(FolderDetails(folder=folder, document_info=document_info))

        return FolderDetailsResponse(folders=folder_entries)

    except HTTPException:
        raise
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error retrieving folder details: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/summary", response_model=List[FolderSummary])
@telemetry.track(operation_type="list_folders_summary")
async def list_folder_summaries(auth: AuthContext = Depends(verify_token)) -> List[FolderSummary]:
    """Return compact folder list (id, name, doc_count, updated_at)."""

    try:
        summaries = await document_service.db.list_folders_summary(auth)
        return summaries  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing folder summaries: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{folder_id_or_name}", response_model=Folder)
@telemetry.track(operation_type="get_folder", metadata_resolver=telemetry.get_folder_metadata)
async def get_folder(
    folder_id_or_name: str,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Get a folder by ID or name.
    """
    try:
        return await _resolve_folder(folder_id_or_name, auth)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{folder_id_or_name}", response_model=FolderDeleteResponse)
@telemetry.track(operation_type="delete_folder", metadata_resolver=telemetry.delete_folder_metadata)
async def delete_folder(
    folder_id_or_name: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Delete a folder and all associated documents.
    """
    try:
        folder = await _resolve_folder(folder_id_or_name, auth)
        folder_id = folder.id
        if not folder_id:
            raise HTTPException(status_code=500, detail="Folder is missing an ID")

        document_ids = folder.document_ids or []
        removal_results = await asyncio.gather(
            *[
                document_service.db.remove_document_from_folder(folder_id, document_id, auth)
                for document_id in document_ids
            ]
        )
        if not all(removal_results):
            failed = [doc for doc, stat in zip(document_ids, removal_results) if not stat]
            msg = "Failed to remove the following documents from folder: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # folder is empty now
        delete_tasks = [document_service.delete_document(document_id, auth) for document_id in document_ids]
        stati = await asyncio.gather(*delete_tasks, return_exceptions=True)

        failed_docs = []
        for doc_id, result in zip(document_ids, stati):
            if isinstance(result, Exception):
                logger.error("Error deleting document %s while deleting folder %s: %s", doc_id, folder_id, result)
                failed_docs.append(doc_id)
            elif not result:
                failed_docs.append(doc_id)

        if failed_docs:
            msg = "Failed to delete the following documents: " + ", ".join(failed_docs)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # just remove the folder too now.
        status = await document_service.db.delete_folder(folder_id, auth)
        if not status:
            logger.error(f"Failed to delete folder {folder_id}")
            raise HTTPException(status_code=500, detail=f"Failed to delete folder {folder_id}")
        return {"status": "success", "message": f"Folder {folder.name} ({folder_id}) deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Folder-Document association endpoints
# ---------------------------------------------------------------------------


@router.post("/{folder_id_or_name}/documents/{document_id}", response_model=DocumentAddToFolderResponse)
@telemetry.track(operation_type="add_document_to_folder", metadata_resolver=telemetry.add_document_to_folder_metadata)
async def add_document_to_folder(
    folder_id_or_name: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Add a document to a folder.
    """
    try:
        folder = await _resolve_folder(folder_id_or_name, auth)
        success = await document_service.db.add_document_to_folder(folder.id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to folder")

        return {
            "status": "success",
            "message": f"Document {document_id} added to folder {folder.name} ({folder.id})",
        }
    except Exception as e:
        logger.error(f"Error adding document to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{folder_id_or_name}/documents/{document_id}", response_model=DocumentDeleteResponse)
@telemetry.track(
    operation_type="remove_document_from_folder", metadata_resolver=telemetry.remove_document_from_folder_metadata
)
async def remove_document_from_folder(
    folder_id_or_name: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Remove a document from a folder.
    """
    try:
        folder = await _resolve_folder(folder_id_or_name, auth)
        success = await document_service.db.remove_document_from_folder(folder.id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove document from folder")

        return {
            "status": "success",
            "message": f"Document {document_id} removed from folder {folder.name} ({folder.id})",
        }
    except Exception as e:
        logger.error(f"Error removing document from folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))
