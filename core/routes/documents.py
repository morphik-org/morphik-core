import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile

from core.auth_utils import verify_token
from core.config import get_settings
from core.database.postgres_database import InvalidMetadataFilterError
from core.models.auth import AuthContext
from core.models.documents import Document
from core.models.request import DocumentPagesRequest, IngestTextRequest, ListDocsRequest, ListDocumentsRequest
from core.models.responses import (
    DocumentDeleteResponse,
    DocumentDownloadUrlResponse,
    DocumentPagesResponse,
    FolderCount,
    ListDocsResponse,
)
from core.routes.utils import project_document_fields
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# ---------------------------------------------------------------------------
# Router initialization & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/documents", tags=["Documents"])
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
# Document CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=List[Document])
async def list_documents(
    request: ListDocumentsRequest,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = Query(None),
):
    """
    List accessible documents.

    `request.document_filters` may include the following operators in addition to equality checks (which also match
    scalars inside JSON arrays): `$and`, `$or`, `$nor`, `$not`, `$in`, `$nin`, `$exists`, `$regex`, and `$contains`.
    Filters can be nested arbitrarily. Regex filters accept the optional `i` flag; `$contains` performs substring
    matches with optional `case_sensitive` overrides.
    Example:

    ```json
    {
      "$and": [
        {"department": "sales"},
        {"$or": [{"status": "approved"}, {"priority": {"$in": ["high", "urgent"]}}]}
      ]
    }
    ```

    Args:
        request: Request body containing filters and pagination
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        List[Document]: List of accessible documents
    """
    # Create system filters for folder and user scoping
    system_filters = {}

    # Normalize folder_name parameter (convert string "null" to None)
    if folder_name is not None:
        normalized_folder_name = normalize_folder_name(folder_name)
        system_filters["folder_name"] = normalized_folder_name
    if end_user_id:
        system_filters["end_user_id"] = end_user_id
    # Note: auth.app_id is already handled in _build_access_filter_optimized

    try:
        return await document_service.db.get_documents(
            auth, request.skip, request.limit, filters=request.document_filters, system_filters=system_filters
        )
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/list_docs", response_model=ListDocsResponse)
async def list_docs(
    request: ListDocsRequest,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = Query(None),
) -> ListDocsResponse:
    """
    Flexible document listing endpoint with support for aggregates, projections, and advanced pagination.

    `request.document_filters` supports equality plus `$and`, `$or`, `$nor`, `$not`, `$in`, `$nin`, `$exists`,
    `$regex`, and `$contains`, with arbitrary nesting. Scalar comparisons match array elements automatically.
    Example:

    ```json
    {
      "$and": [
        {"category": "patent"},
        {"$nor": [{"status": "archived"}, {"priority": {"$in": ["low", "medium"]}}]}
      ]
    }
    ```

    Use the `folder_name` and `end_user_id` query parameters to scope system metadata instead of embedding those
    keys in the filter payload.
    """
    try:
        system_filters: Dict[str, Any] = {}
        if folder_name is not None:
            system_filters["folder_name"] = normalize_folder_name(folder_name)
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        db_result = await document_service.db.list_documents_flexible(
            auth=auth,
            skip=request.skip,
            limit=request.limit,
            filters=request.document_filters,
            system_filters=system_filters,
            status_filter=["completed"] if request.completed_only else None,
            include_total_count=request.include_total_count,
            include_status_counts=request.include_status_counts,
            include_folder_counts=request.include_folder_counts,
            return_documents=request.return_documents,
            sort_by=request.sort_by,
            sort_direction=request.sort_direction,
        )

        documents_payload: List[Any] = []
        if request.return_documents:
            raw_documents = db_result.get("documents", [])
            for document in raw_documents:
                if hasattr(document, "model_dump"):
                    doc_dict = document.model_dump(mode="json")
                elif hasattr(document, "dict"):
                    doc_dict = document.dict()
                else:
                    doc_dict = dict(document)
                documents_payload.append(project_document_fields(doc_dict, request.fields))

        total_count = db_result.get("total_count")
        returned_count = db_result.get("returned_count", len(documents_payload))
        has_more = db_result.get("has_more", False)
        next_skip = db_result.get("next_skip")

        if next_skip is None and has_more:
            next_skip = request.skip + returned_count

        folder_counts_raw = db_result.get("folder_counts")
        folder_counts: Optional[List[FolderCount]] = None
        if folder_counts_raw:
            folder_counts = [
                FolderCount(folder=item.get("folder"), count=item.get("count", 0)) for item in folder_counts_raw
            ]

        return ListDocsResponse(
            documents=documents_payload,
            skip=request.skip,
            limit=request.limit,
            returned_count=returned_count,
            total_count=total_count,
            has_more=has_more,
            next_skip=next_skip,
            status_counts=db_result.get("status_counts") if request.include_status_counts else None,
            folder_counts=folder_counts,
        )
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """Retrieve a single document by its external identifier.

    Args:
        document_id: External ID of the document to fetch.
        auth: Authentication context used to verify access rights.

    Returns:
        The :class:`Document` metadata if found.
    """
    try:
        doc = await document_service.db.get_document(document_id, auth)
        logger.debug(f"Found document: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document: {e}")
        raise e


@router.get("/{document_id}/status", response_model=Dict[str, Any])
async def get_document_status(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Get the processing status of a document.

    Args:
        document_id: ID of the document to check
        auth: Authentication context

    Returns:
        Dict containing status information for the document
    """
    try:
        doc = await document_service.db.get_document(document_id, auth)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Extract status information
        status = doc.system_metadata.get("status", "unknown")

        response = {
            "document_id": doc.external_id,
            "status": status,
            "filename": doc.filename,
            "created_at": doc.system_metadata.get("created_at"),
            "updated_at": doc.system_metadata.get("updated_at"),
        }

        # Add progress information if processing
        if status == "processing" and "progress" in doc.system_metadata:
            response["progress"] = doc.system_metadata["progress"]

        # Add error information if failed
        if status == "failed":
            response["error"] = doc.system_metadata.get("error", "Unknown error")

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document status: {str(e)}")


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
@telemetry.track(operation_type="delete_document", metadata_resolver=telemetry.document_delete_metadata)
async def delete_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Delete a document and all associated data.

    This endpoint deletes a document and all its associated data, including:
    - Document metadata
    - Document content in storage
    - Document chunks and embeddings in vector store

    Args:
        document_id: ID of the document to delete
        auth: Authentication context (must have write access to the document)

    Returns:
        Deletion status
    """
    try:
        success = await document_service.delete_document(document_id, auth)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or delete failed")
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/filename/{filename}", response_model=Document)
async def get_document_by_filename(
    filename: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
):
    """
    Get document by filename.

    Args:
        filename: Filename of the document to retrieve
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        Document: Document metadata if found and accessible
    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        doc = await document_service.db.get_document_by_filename(filename, auth, system_filters)
        logger.debug(f"Found document by filename: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document by filename: {e}")
        raise e


@router.get("/{document_id}/download_url", response_model=DocumentDownloadUrlResponse)
async def get_document_download_url(
    document_id: str,
    auth: AuthContext = Depends(verify_token),
    expires_in: int = Query(3600, description="URL expiration time in seconds"),
):
    """
    Get a download URL for a specific document.

    Args:
        document_id: External ID of the document
        auth: Authentication context
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Dictionary containing the download URL and metadata
    """
    try:
        # Get the document
        doc = await document_service.db.get_document(document_id, auth)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if document has storage info
        if not doc.storage_info or not doc.storage_info.get("bucket") or not doc.storage_info.get("key"):
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        # Generate download URL
        download_url = await document_service.storage.get_download_url(
            doc.storage_info["bucket"], doc.storage_info["key"], expires_in=expires_in
        )

        return {
            "document_id": doc.external_id,
            "filename": doc.filename,
            "content_type": doc.content_type,
            "download_url": download_url,
            "expires_in": expires_in,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting download URL for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting download URL: {str(e)}")


@router.get("/{document_id}/file", response_model=None)
async def download_document_file(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Download the actual file content for a document.
    This endpoint is used for local storage when file:// URLs cannot be accessed by browsers.

    Args:
        document_id: External ID of the document
        auth: Authentication context

    Returns:
        StreamingResponse with the file content
    """
    try:
        logger.info(f"Attempting to download file for document ID: {document_id}")
        logger.info(f"Auth context: entity_id={auth.entity_id}, app_id={auth.app_id}")

        # Get the document
        doc = await document_service.db.get_document(document_id, auth)
        logger.info(f"Document lookup result: {doc is not None}")

        if not doc:
            logger.warning(f"Document not found in database: {document_id}")
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        logger.info(f"Found document: {doc.filename}, content_type: {doc.content_type}")
        logger.info(f"Storage info: {doc.storage_info}")

        # Check if document has storage info
        if not doc.storage_info or not doc.storage_info.get("bucket") or not doc.storage_info.get("key"):
            logger.warning(f"Document has no storage info: {document_id}")
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        # Download file content from storage
        logger.info(f"Downloading from bucket: {doc.storage_info['bucket']}, key: {doc.storage_info['key']}")
        file_content = await document_service.storage.download_file(doc.storage_info["bucket"], doc.storage_info["key"])

        logger.info(f"Successfully downloaded {len(file_content)} bytes")

        # Create streaming response

        from fastapi.responses import StreamingResponse

        def generate():
            yield file_content

        return StreamingResponse(
            generate(),
            media_type=doc.content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f"inline; filename=\"{doc.filename or 'document'}\"",
                "Content-Length": str(len(file_content)),
            },
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found in storage for document {document_id}: {e}")
        raise HTTPException(status_code=404, detail=f"File not found in storage: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading document file {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


# ---------------------------------------------------------------------------
# Document update endpoints
# ---------------------------------------------------------------------------


@router.post("/{document_id}/update_text", response_model=Document)
@telemetry.track(operation_type="update_document_text", metadata_resolver=telemetry.document_update_text_metadata)
async def update_document_text(
    document_id: str,
    request: IngestTextRequest,
    update_strategy: str = "add",
    auth: AuthContext = Depends(verify_token),
):
    """
    Update a document with new text content using the specified strategy.

    Args:
        document_id: ID of the document to update
        request: Text content and metadata for the update
        update_strategy: Strategy for updating the document (default: 'add')
        auth: Authentication context

    Returns:
        Document: Updated document metadata
    """
    try:
        doc = await document_service.update_document(
            document_id=document_id,
            auth=auth,
            content=request.content,
            file=None,
            filename=request.filename,
            metadata=request.metadata,
            rules=request.rules,
            update_strategy=update_strategy,
            use_colpali=request.use_colpali,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{document_id}/update_file", response_model=Document)
@telemetry.track(operation_type="update_document_file", metadata_resolver=telemetry.document_update_file_metadata)
async def update_document_file(
    document_id: str,
    file: UploadFile,
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    update_strategy: str = Form("add"),
    use_colpali: Optional[bool] = Form(None),
    auth: AuthContext = Depends(verify_token),
):
    """
    Update a document with content from a file using the specified strategy.

    Args:
        document_id: ID of the document to update
        file: File to add to the document
        metadata: JSON string of metadata to merge with existing metadata
        rules: JSON string of rules to apply to the content
        update_strategy: Strategy for updating the document (default: 'add')
        use_colpali: Whether to use multi-vector embedding
        auth: Authentication context

    Returns:
        Document: Updated document metadata
    """
    try:
        metadata_dict = json.loads(metadata)
        rules_list = json.loads(rules)

        doc = await document_service.update_document(
            document_id=document_id,
            auth=auth,
            content=None,
            file=file,
            filename=file.filename,
            metadata=metadata_dict,
            rules=rules_list,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/{document_id}/update_metadata", response_model=Document)
@telemetry.track(
    operation_type="update_document_metadata",
    metadata_resolver=telemetry.document_update_metadata_resolver,
)
async def update_document_metadata(
    document_id: str, metadata_updates: Dict[str, Any], auth: AuthContext = Depends(verify_token)
):
    """
    Update only a document's metadata.

    Args:
        document_id: ID of the document to update
        metadata_updates: New metadata to merge with existing metadata
        auth: Authentication context

    Returns:
        Document: Updated document metadata
    """
    try:
        doc = await document_service.update_document(
            document_id=document_id,
            auth=auth,
            content=None,
            file=None,
            filename=None,
            metadata=metadata_updates,
            rules=[],
            update_strategy="add",
            use_colpali=None,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


# TODO: add @telemetry.track(operation_type="extract_document_pages", metadata_resolver=telemetry.document_pages_metadata)
@router.post("/pages", response_model=DocumentPagesResponse)
async def extract_document_pages(
    request: DocumentPagesRequest,
    auth: AuthContext = Depends(verify_token),
):
    """
    Extract specific pages from a document (PDF or PowerPoint) as base64-encoded images.

    Args:
        request: Request containing document_id, start_page, and end_page
        auth: Authentication context

    Returns:
        DocumentPagesResponse: Base64-encoded images of the requested pages
    """
    try:
        # Get the document
        doc = await document_service.db.get_document(request.document_id, auth)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if document has storage info
        if not doc.storage_info or not doc.storage_info.get("bucket") or not doc.storage_info.get("key"):
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        # Validate page range
        if request.start_page > request.end_page:
            raise HTTPException(status_code=400, detail="start_page must be less than or equal to end_page")

        # Determine document type by content_type or filename
        content_type = (doc.content_type or "").lower()
        filename = (doc.filename or "").lower()
        _, ext = os.path.splitext(filename)

        is_pdf = content_type == "application/pdf" or ext == ".pdf"
        is_ppt = content_type in {
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        } or ext in {".ppt", ".pptx", ".pps", ".ppsx"}

        # Extract pages using appropriate handler
        if is_pdf:
            pages_data = await document_service.extract_pdf_pages(
                doc.storage_info["bucket"], doc.storage_info["key"], request.start_page, request.end_page
            )
        elif is_ppt:
            # Assume PPT/PPTX were ingested via ColPali: fetch image chunks for the page range
            if not getattr(document_service, "colpali_vector_store", None):
                raise HTTPException(status_code=400, detail="ColPali is required for PowerPoint page extraction")

            start_idx = max(0, request.start_page - 1)
            end_idx = max(0, request.end_page - 1)
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            identifiers = [(doc.external_id, i) for i in range(start_idx, end_idx + 1)]
            try:
                chunks = await document_service.colpali_vector_store.get_chunks_by_id(identifiers, auth.app_id)
            except Exception as e:
                logger.error(f"Failed to retrieve ColPali chunks for {doc.external_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve slide images")

            # Order by chunk_number and extract base64 image content
            chunks_sorted = sorted(chunks, key=lambda c: c.chunk_number)
            pages_b64 = [c.content for c in chunks_sorted if isinstance(c.content, str) and c.content]

            # Provide a best-effort total_pages placeholder (not authoritative)
            pages_data = {"pages": pages_b64, "total_pages": request.end_page}
        else:
            raise HTTPException(status_code=400, detail="Unsupported document type for page extraction")

        return DocumentPagesResponse(
            document_id=request.document_id,
            pages=pages_data["pages"],
            start_page=request.start_page,
            end_page=request.end_page,
            total_pages=pages_data["total_pages"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting pages from document {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting pages: {str(e)}")
