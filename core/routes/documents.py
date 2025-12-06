import logging
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query

from core.auth_utils import verify_token
from core.config import get_settings
from core.database.postgres_database import InvalidMetadataFilterError
from core.models.auth import AuthContext
from core.models.documents import Document
from core.models.request import DocumentPagesRequest, ListDocsRequest
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
from core.utils.folder_utils import normalize_folder_name
from core.utils.typed_metadata import TypedMetadataError

# ---------------------------------------------------------------------------
# Router initialization & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()


# ---------------------------------------------------------------------------
# Document CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/list_docs", response_model=ListDocsResponse)
async def list_docs(
    request: ListDocsRequest,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = Query(None),
) -> ListDocsResponse:
    """
    Flexible document listing with aggregates, projections, and advanced pagination.

    **Supported operators**: `$and`, `$or`, `$nor`, `$not`, `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`,
    `$in`, `$nin`, `$exists`, `$type`, `$regex`, `$contains`.

    **Implicit equality** (backwards compatible, JSONB containment):
    ```json
    {"status": "active"}
    ```

    **Explicit operators** (typed comparisons for number, decimal, datetime, date):
    ```json
    {"priority": {"$gte": 40}, "end_date": {"$lt": "2025-01-01"}}
    ```

    Use `folder_name` and `end_user_id` query parameters to scope system metadata.
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

    Returns the :class:`Document` metadata if found or raises 404.
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


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
@telemetry.track(operation_type="delete_document", metadata_resolver=telemetry.document_delete_metadata)
async def delete_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Delete a document and all associated data.

    This endpoint deletes a document and all its associated data, including:
    - Document metadata
    - Document content in storage
    - Document chunks and embeddings in vector store
    """
    try:
        success = await document_service.delete_document(document_id, auth)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or delete failed")
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{document_id}/download_url", response_model=DocumentDownloadUrlResponse)
async def get_document_download_url(
    document_id: str,
    auth: AuthContext = Depends(verify_token),
    expires_in: int = Query(3600, description="URL expiration time in seconds"),
):
    """
    Get a download URL for a specific document.
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


# TODO: add @telemetry.track(operation_type="extract_document_pages", metadata_resolver=telemetry.document_pages_metadata)
@router.post("/pages", response_model=DocumentPagesResponse)
async def extract_document_pages(
    request: DocumentPagesRequest,
    auth: AuthContext = Depends(verify_token),
):
    """
    Extract specific pages from a document (PDF, PowerPoint, or Word) as base64-encoded images.
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
        is_word = content_type in {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        } or ext in {".doc", ".docx"}

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
        elif is_word:
            # Fetch image chunks for DOC/DOCX from the multi-vector store, same as PPT
            if not getattr(document_service, "colpali_vector_store", None):
                raise HTTPException(status_code=400, detail="ColPali is required for Word page extraction")

            start_idx = max(0, request.start_page - 1)
            end_idx = max(0, request.end_page - 1)
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            identifiers = [(doc.external_id, i) for i in range(start_idx, end_idx + 1)]
            try:
                chunks = await document_service.colpali_vector_store.get_chunks_by_id(identifiers, auth.app_id)
            except Exception as e:
                logger.error(f"Failed to retrieve ColPali chunks for {doc.external_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve document page images")

            chunks_sorted = sorted(chunks, key=lambda c: c.chunk_number)
            pages_b64 = [c.content for c in chunks_sorted if isinstance(c.content, str) and c.content]
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
