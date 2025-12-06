import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional, Union

import arq
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from core.auth_utils import verify_token
from core.config import get_settings
from core.dependencies import get_redis_pool
from core.limits_utils import check_and_increment_limits
from core.models.auth import AuthContext
from core.models.documents import Document, StorageFileInfo
from core.routes.utils import warn_if_legacy_rules
from core.services.telemetry import TelemetryService
from core.services_init import ingestion_service, storage
from core.storage.utils_file_extensions import detect_file_type
from core.utils.typed_metadata import TypedMetadataError, normalize_metadata

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()


def _str2bool(value: Union[bool, str]) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes", "y", "on"}


@router.post("/file", response_model=Document)
@telemetry.track(operation_type="queue_ingest_file", metadata_resolver=telemetry.ingest_file_metadata)
async def ingest_file(
    request: Request,
    file: UploadFile,
    metadata: str = Form("{}"),
    metadata_types: str = Form("{}"),
    auth: AuthContext = Depends(verify_token),
    use_colpali: Optional[bool] = Form(None),
    folder_name: Optional[str] = Form(None),
    end_user_id: Optional[str] = Form(None),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> Document:
    """Queue a single file for ingestion."""
    try:
        await warn_if_legacy_rules(request, "/ingest/file", logger)
        metadata_dict = json.loads(metadata or "{}")
        metadata_types_dict = json.loads(metadata_types or "{}") if metadata_types else {}
        if metadata_types_dict is None:
            metadata_types_dict = {}
        if not isinstance(metadata_types_dict, dict):
            raise HTTPException(status_code=400, detail="metadata_types must be a JSON object")

        use_colpali_bool = _str2bool(use_colpali) if use_colpali is not None else True

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        ingestion_service._enforce_no_user_mutable_fields(metadata_dict, folder_name, context="ingest")

        # Create document stub (processing)
        doc = Document(
            content_type=file.content_type,
            filename=file.filename,
            metadata=metadata_dict,
            system_metadata={"status": "processing"},
            folder_name=folder_name,
            end_user_id=end_user_id,
            app_id=auth.app_id,
        )

        metadata_payload = dict(metadata_dict)
        metadata_payload.setdefault("external_id", doc.external_id)
        if folder_name is not None:
            metadata_payload["folder_name"] = folder_name
        normalized_metadata, normalized_types = normalize_metadata(metadata_payload, metadata_types_dict)
        doc.metadata = normalized_metadata
        doc.metadata_types = normalized_types

        app_db = ingestion_service.db
        success = await app_db.store_document(doc, auth)
        if not success:
            raise Exception("Failed to store document metadata")

        if folder_name:
            try:
                await ingestion_service._ensure_folder_exists(folder_name, doc.external_id, auth)
            except Exception as folder_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to add document %s to folder %s immediately after ingest: %s",
                    doc.external_id,
                    folder_name,
                    folder_exc,
                )

        file_content = await file.read()

        if settings.MODE == "cloud" and auth.user_id:
            await check_and_increment_limits(auth, "storage_file", 1, verify_only=True)
            await check_and_increment_limits(auth, "storage_size", len(file_content), verify_only=True)

        safe_filename = Path(file.filename or "").name or "uploaded_file"
        file_key = f"ingest_uploads/{uuid.uuid4()}/{safe_filename}"
        if not Path(file_key).suffix:
            detected_ext = detect_file_type(file_content)
            if detected_ext:
                file_key = f"{file_key}{detected_ext}"

        bucket, stored_key = await storage.upload_file(
            file_content,
            file_key,
            file.content_type,
            bucket="",
        )

        doc.storage_info = {"bucket": bucket, "key": stored_key}
        doc.storage_files = [
            StorageFileInfo(
                bucket=bucket,
                key=stored_key,
                version=1,
                filename=safe_filename,
                content_type=file.content_type,
                timestamp=datetime.now(UTC),
            )
        ]

        await app_db.update_document(
            document_id=doc.external_id,
            updates={"storage_info": doc.storage_info, "storage_files": doc.storage_files},
            auth=auth,
        )

        if settings.MODE == "cloud" and auth.user_id:
            try:
                await check_and_increment_limits(auth, "storage_file", 1)
                await check_and_increment_limits(auth, "storage_size", len(file_content))
            except Exception as rec_exc:  # noqa: BLE001
                logger.error("Failed to record storage usage: %s", rec_exc)

        auth_dict = {
            "entity_type": auth.entity_type.value,
            "entity_id": auth.entity_id,
            "app_id": auth.app_id,
            "permissions": list(auth.permissions),
            "user_id": auth.user_id,
        }

        job = await redis.enqueue_job(
            "process_ingestion_job",
            _job_id=f"ingest:{doc.external_id}",
            document_id=doc.external_id,
            file_key=stored_key,
            bucket=bucket,
            original_filename=file.filename,
            content_type=file.content_type,
            metadata_json=json.dumps(metadata_dict),
            metadata_types_json=json.dumps(metadata_types_dict),
            auth_dict=auth_dict,
            use_colpali=use_colpali_bool,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        if job is None:
            logger.info("File ingestion job already queued (doc=%s)", doc.external_id)
        else:
            logger.info("File ingestion job queued (job_id=%s, doc=%s)", job.job_id, doc.external_id)

        return doc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(exc)}")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during file ingestion: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error during file ingestion: {str(exc)}")
