import logging
from typing import Optional

import arq
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.auth_utils import verify_token
from core.database.metadata_filters import InvalidMetadataFilterError
from core.dependencies import get_redis_pool
from core.models.auth import AuthContext
from core.models.responses import DocumentDeleteResponse
from core.models.v2 import V2ChunkResult, V2IngestResponse, V2RetrieveRequest, V2RetrieveResponse
from core.routes.utils import parse_json_dict
from core.services_init import document_service, v2_document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["V2"])


def _require_app_id(auth: AuthContext) -> None:
    if not auth.app_id:
        raise HTTPException(status_code=403, detail="app_id is required for v2 endpoints")


@router.post("/documents", response_model=V2IngestResponse)
async def ingest_document_v2(
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    metadata: str = Form("{}"),
    metadata_types: str = Form("{}"),
    folder_path: Optional[str] = Form(None),
    end_user_id: Optional[str] = Form(None),
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> V2IngestResponse:
    try:
        _require_app_id(auth)
        metadata_dict = parse_json_dict(metadata, "metadata", default={})
        metadata_types_dict = parse_json_dict(metadata_types, "metadata_types", default={})

        file_bytes = None
        resolved_filename = filename
        content_type = None
        if file is not None:
            file_bytes = await file.read()
            resolved_filename = resolved_filename or file.filename or "uploaded_file"
            content_type = file.content_type

        result = await v2_document_service.ingest_document(
            file_bytes=file_bytes,
            filename=resolved_filename,
            content=content,
            content_type=content_type,
            metadata=metadata_dict,
            metadata_types=metadata_types_dict,
            folder_path=folder_path,
            end_user_id=end_user_id,
            auth=auth,
            redis=redis,
        )

        return V2IngestResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            status=result.get("status"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during v2 ingestion: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error during v2 ingestion: {exc}")


@router.post("/retrieve/chunks", response_model=V2RetrieveResponse)
async def retrieve_chunks_v2(
    request: V2RetrieveRequest,
    auth: AuthContext = Depends(verify_token),
) -> V2RetrieveResponse:
    _require_app_id(auth)
    filters = request.filters
    document_ids = filters.document_ids if filters else None
    folder_paths = filters.folder_paths if filters else None
    metadata_filters = filters.metadata if filters else None

    try:
        results = await v2_document_service.retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            auth=auth,
            document_ids=document_ids,
            folder_paths=folder_paths,
            metadata_filters=metadata_filters,
            end_user_id=request.end_user_id,
        )
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = [
        V2ChunkResult(
            chunk_id=item["id"],
            document_id=item["document_id"],
            page_number=item.get("page_number"),
            chunk_number=item.get("chunk_number"),
            score=item["score"],
            content=item["content"],
        )
        for item in results
    ]

    return V2RetrieveResponse(query=request.query, chunks=chunks)


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document_v2(
    document_id: str,
    auth: AuthContext = Depends(verify_token),
) -> DocumentDeleteResponse:
    try:
        _require_app_id(auth)
        success = await document_service.delete_document(document_id, auth)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or delete failed")
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Error deleting v2 document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {exc}")
