import logging
from typing import Literal, Optional

import arq
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from core.auth_utils import verify_token
from core.dependencies import get_redis_pool
from core.models.auth import AuthContext
from core.models.responses import MigrationIngestResponse
from core.routes.utils import parse_bool, parse_json_dict
from core.services_init import ingestion_service
from core.utils.typed_metadata import TypedMetadataError

router = APIRouter(prefix="/migrate", tags=["Migration"])
logger = logging.getLogger(__name__)


@router.post("/document", response_model=MigrationIngestResponse)
async def migrate_document(
    file: UploadFile,
    source_document_id: str = Form(...),
    metadata: str = Form("{}"),
    metadata_types: str = Form("{}"),
    use_colpali: Optional[bool] = Form(None),
    folder_name: Optional[str] = Form(None),
    end_user_id: Optional[str] = Form(None),
    on_conflict: Literal["skip", "fail"] = Form("skip"),
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> MigrationIngestResponse:
    """Ingest a migrated source document while preserving its document ID."""
    source_document_id = (source_document_id or "").strip()
    if not source_document_id:
        raise HTTPException(status_code=400, detail="source_document_id is required")

    try:
        existing_doc = await ingestion_service.db.get_document(source_document_id, auth)
        if existing_doc is not None:
            if on_conflict == "skip":
                return MigrationIngestResponse(status="skipped", document=existing_doc)
            raise HTTPException(
                status_code=409,
                detail=f"Document {source_document_id} already exists in the target app",
            )

        metadata_dict = parse_json_dict(metadata, "metadata", default={})
        metadata_types_dict = parse_json_dict(metadata_types, "metadata_types", default={})
        file_content = await file.read()
        filename = file.filename or "uploaded_file"

        document = await ingestion_service.ingest_file_content(
            file_content_bytes=file_content,
            filename=filename,
            content_type=file.content_type,
            metadata=metadata_dict,
            auth=auth,
            redis=redis,
            metadata_types=metadata_types_dict,
            folder_name=folder_name,
            end_user_id=end_user_id,
            use_colpali=parse_bool(use_colpali),
            external_id=source_document_id,
        )
        return MigrationIngestResponse(status="created", document=document)
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error migrating document %s: %s", source_document_id, exc)
        raise HTTPException(status_code=500, detail=f"Error migrating document: {str(exc)}")
