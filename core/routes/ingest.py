import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import arq
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from core.auth_utils import verify_token
from core.config import get_settings
from core.dependencies import get_redis_pool
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.auth import AuthContext
from core.models.documents import Document
from core.models.request import (
    BatchIngestResponse,
    DocumentQueryResponse,
    IngestionOptions,
    IngestTextRequest,
    RequeueIngestionRequest,
)
from core.models.responses import RequeueIngestionResponse, RequeueIngestionResult
from core.routes.utils import warn_if_legacy_rules
from core.services.ingestion_service import IngestionService
from core.services.morphik_on_the_fly_structured_output import (
    MorphikOnTheFlyContentError,
    generate_morphik_on_the_fly_content,
)
from core.services.telemetry import TelemetryService
from core.services_init import ingestion_service, storage
from core.storage.utils_file_extensions import detect_file_type
from core.utils.folder_utils import normalize_folder_path
from core.utils.typed_metadata import TypedMetadataError, normalize_metadata

# ---------------------------------------------------------------------------
# Router initialisation & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()

MORPHIK_ON_THE_FLY_MAX_DOCUMENT_BYTES = 20 * 1024 * 1024  # 20 MB limit for inline uploads to Morphik On-the-Fly


def _parse_bool(value: Optional[Union[str, bool]]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _normalize_folder_inputs(folder_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return canonical folder path with leading slash and the leaf name."""
    if folder_name is None:
        return None, None
    try:
        folder_path = normalize_folder_path(folder_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if folder_path == "/":
        raise HTTPException(status_code=400, detail="Cannot ingest into root folder '/'")
    parts = [p for p in folder_path.strip("/").split("/") if p]
    leaf = parts[-1] if parts else None
    return folder_path, leaf


# ---------------------------------------------------------------------------
# /ingest/text
# ---------------------------------------------------------------------------


@router.post("/text", response_model=Document)
@telemetry.track(operation_type="ingest_text", metadata_resolver=telemetry.ingest_text_metadata)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token),
) -> Document:
    """Ingest a **text** document."""
    try:
        # Free-tier usage limits (cloud mode only)
        if settings.MODE == "cloud" and auth.user_id:
            pages_est = estimate_pages_by_chars(len(request.content))
            await check_and_increment_limits(
                auth,
                "ingest",
                pages_est,
                verify_only=True,
            )

        if getattr(request, "rules", None):
            logger.warning("Legacy 'rules' field supplied to /ingest/text; ignoring payload.")

        extra_fields = getattr(request, "model_extra", {}) if hasattr(request, "model_extra") else {}
        ingestion_service._enforce_no_user_mutable_fields(
            request.metadata, request.folder_name, extra_fields, request.metadata_types, context="ingest"
        )

        return await ingestion_service.ingest_text(
            content=request.content,
            filename=request.filename,
            metadata=request.metadata,
            metadata_types=request.metadata_types,
            use_colpali=request.use_colpali,
            auth=auth,
            folder_name=request.folder_name,
            end_user_id=request.end_user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# /ingest/file
# ---------------------------------------------------------------------------


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
    """Ingest a **file** asynchronously.

    The file is uploaded to object storage, a *Document* stub is persisted
    with ``status='processing'`` and a background worker picks up the heavy
    parsing / chunking work.
    """
    try:
        # ------------------------------------------------------------------
        # Parse and validate inputs
        # ------------------------------------------------------------------
        await warn_if_legacy_rules(request, "/ingest/file", logger)
        metadata_dict = json.loads(metadata)
        metadata_types_dict = json.loads(metadata_types or "{}") if metadata_types else {}
        if metadata_types_dict is None:
            metadata_types_dict = {}
        if not isinstance(metadata_types_dict, dict):
            raise HTTPException(status_code=400, detail="metadata_types must be a JSON object")

        def str2bool(v: Union[bool, str]) -> bool:
            return v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}

        use_colpali_bool = str2bool(use_colpali)

        folder_path, folder_leaf = _normalize_folder_inputs(folder_name)

        logger.debug("Queueing file ingestion with use_colpali=%s", use_colpali_bool)

        ingestion_service._enforce_no_user_mutable_fields(
            metadata_dict, folder_name, metadata_types=metadata_types_dict, context="ingest"
        )

        # ------------------------------------------------------------------
        # Create initial Document stub (status = processing)
        # ------------------------------------------------------------------
        doc = Document(
            content_type=file.content_type,
            filename=file.filename,
            metadata=metadata_dict,
            system_metadata={"status": "processing"},
            folder_name=folder_leaf,
            folder_path=folder_path,
            end_user_id=end_user_id,
            app_id=auth.app_id,
        )
        metadata_payload = dict(metadata_dict)
        metadata_payload.setdefault("external_id", doc.external_id)
        if folder_path is not None:
            metadata_payload["folder_name"] = folder_path
        normalized_metadata, normalized_types = normalize_metadata(metadata_payload, metadata_types_dict)
        doc.metadata = normalized_metadata
        doc.metadata_types = normalized_types

        # Store stub in application database (not control-plane DB)
        app_db = ingestion_service.db
        success = await app_db.store_document(doc, auth)
        if not success:
            raise Exception("Failed to store document metadata")

        # Add the document to the requested folder immediately so folder views can show in-progress items.
        # The ingestion worker re-runs this to ensure the folder is still in sync on completion.
        if folder_path:
            try:
                folder_obj = await ingestion_service._ensure_folder_exists(folder_path, doc.external_id, auth)
                if folder_obj and folder_obj.id:
                    doc.folder_id = folder_obj.id
                    await app_db.update_document(
                        document_id=doc.external_id, updates={"folder_id": doc.folder_id}, auth=auth
                    )
            except Exception as folder_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to add document %s to folder %s immediately after ingest: %s",
                    doc.external_id,
                    folder_path,
                    folder_exc,
                )

        # ------------------------------------------------------------------
        # Read file content & pre-check storage limits
        # ------------------------------------------------------------------
        file_content = await file.read()

        if settings.MODE == "cloud" and auth.user_id:
            await check_and_increment_limits(auth, "storage_file", 1, verify_only=True)
            await check_and_increment_limits(auth, "storage_size", len(file_content), verify_only=True)

        # ------------------------------------------------------------------
        # Upload file to object storage without re-encoding
        # ------------------------------------------------------------------
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

        await app_db.update_document(
            document_id=doc.external_id,
            updates={"storage_info": doc.storage_info},
            auth=auth,
        )

        # Record storage usage now (cloud mode)
        if settings.MODE == "cloud" and auth.user_id:
            try:
                await check_and_increment_limits(auth, "storage_file", 1)
                await check_and_increment_limits(auth, "storage_size", len(file_content))
            except Exception as rec_exc:  # noqa: BLE001
                logger.error("Failed to record storage usage: %s", rec_exc)

        # ------------------------------------------------------------------
        # Push job to ingestion worker queue
        # ------------------------------------------------------------------
        auth_dict = {
            "user_id": auth.user_id,
            "app_id": auth.app_id,
        }

        job = await redis.enqueue_job(
            "process_ingestion_job",
            _job_id=f"ingest:{doc.external_id}",
            document_id=doc.external_id,
            file_key=stored_key,
            bucket=bucket,
            original_filename=file.filename,
            content_type=file.content_type,
            metadata_json=metadata,
            metadata_types_json=json.dumps(metadata_types_dict or {}),
            auth_dict=auth_dict,
            use_colpali=use_colpali_bool,
            folder_name=folder_name,
            folder_path=folder_path,
            folder_leaf=folder_leaf,
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


# ---------------------------------------------------------------------------
# /ingest/files (batch)
# ---------------------------------------------------------------------------


@router.post("/files", response_model=BatchIngestResponse)
@telemetry.track(operation_type="queue_batch_ingest", metadata_resolver=telemetry.batch_ingest_metadata)
async def batch_ingest_files(
    request: Request,
    files: List[UploadFile] = File(...),
    metadata: str = Form("{}"),
    metadata_types: str = Form("{}"),
    use_colpali: Optional[bool] = Form(None),
    folder_name: Optional[str] = Form(None),
    end_user_id: Optional[str] = Form(None),
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> BatchIngestResponse:
    """Batch ingest **multiple files** (async).

    Each file is treated the same as :func:`ingest_file` but sharing the same
    request avoids many round-trips. All heavy work is still delegated to the
    background worker pool.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch ingestion")

    try:
        await warn_if_legacy_rules(request, "/ingest/files", logger)
        metadata_value = json.loads(metadata)
        metadata_types_value = json.loads(metadata_types or "{}") if metadata_types else {}
        if metadata_types_value is None:
            metadata_types_value = {}

        def str2bool(v: Union[bool, str]) -> bool:
            return v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}

        use_colpali_bool = str2bool(use_colpali)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(exc)}")

    # Validate metadata length when list provided
    if isinstance(metadata_value, list) and len(metadata_value) != len(files):
        raise HTTPException(
            status_code=400,
            detail=(f"Number of metadata items ({len(metadata_value)}) must match number of files " f"({len(files)})"),
        )
    if isinstance(metadata_types_value, list) and len(metadata_types_value) != len(files):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Number of metadata_types items ({len(metadata_types_value)}) must match number of files "
                f"({len(files)})"
            ),
        )

    auth_dict = {
        "user_id": auth.user_id,
        "app_id": auth.app_id,
    }

    created_documents: List[Document] = []

    try:
        folder_path, folder_leaf = _normalize_folder_inputs(folder_name)
        for idx, file in enumerate(files):
            metadata_item = metadata_value[idx] if isinstance(metadata_value, list) else metadata_value
            metadata_types_item = (
                metadata_types_value[idx] if isinstance(metadata_types_value, list) else metadata_types_value
            )
            if metadata_types_item is None:
                metadata_types_item = {}
            if not isinstance(metadata_types_item, dict):
                raise HTTPException(status_code=400, detail="metadata_types entries must be JSON objects")
            # ------------------------------------------------------------------
            # Create stub Document (processing)
            # ------------------------------------------------------------------
            ingestion_service._enforce_no_user_mutable_fields(
                metadata_item, folder_name, metadata_types=metadata_types_item, context="ingest"
            )

            doc = Document(
                content_type=file.content_type,
                filename=file.filename,
                metadata=metadata_item,
                folder_name=folder_leaf,
                folder_path=folder_path,
                end_user_id=end_user_id,
                app_id=auth.app_id,
            )
            doc.system_metadata["status"] = "processing"
            metadata_payload = dict(metadata_item or {})
            metadata_payload.setdefault("external_id", doc.external_id)
            if folder_path is not None:
                metadata_payload["folder_name"] = folder_path
            normalized_metadata, normalized_types = normalize_metadata(metadata_payload, metadata_types_item)
            doc.metadata = normalized_metadata
            doc.metadata_types = normalized_types

            app_db = ingestion_service.db
            success = await app_db.store_document(doc, auth)
            if not success:
                raise Exception(f"Failed to store document metadata for {file.filename}")

            # Keep folder listings in sync immediately; worker re-runs this when processing finishes.
            if folder_path:
                try:
                    folder_obj = await ingestion_service._ensure_folder_exists(folder_path, doc.external_id, auth)
                    if folder_obj and folder_obj.id:
                        doc.folder_id = folder_obj.id
                        await app_db.update_document(
                            document_id=doc.external_id, updates={"folder_id": doc.folder_id}, auth=auth
                        )
                except Exception as folder_exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to add batch document %s to folder %s immediately after ingest: %s",
                        doc.external_id,
                        folder_path,
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
            await app_db.update_document(
                document_id=doc.external_id,
                updates={"storage_info": doc.storage_info},
                auth=auth,
            )

            if settings.MODE == "cloud" and auth.user_id:
                try:
                    await check_and_increment_limits(auth, "storage_file", 1)
                    await check_and_increment_limits(auth, "storage_size", len(file_content))
                except Exception as rec_exc:  # noqa: BLE001
                    logger.error("Failed to record storage usage: %s", rec_exc)

            metadata_json = json.dumps(metadata_item)
            metadata_types_json = json.dumps(metadata_types_item or {})

            job = await redis.enqueue_job(
                "process_ingestion_job",
                _job_id=f"ingest:{doc.external_id}",
                document_id=doc.external_id,
                file_key=stored_key,
                bucket=bucket,
                original_filename=file.filename,
                content_type=file.content_type,
                metadata_json=metadata_json,
                metadata_types_json=metadata_types_json,
                auth_dict=auth_dict,
                use_colpali=use_colpali_bool,
                folder_name=folder_name,
                folder_path=folder_path,
                folder_leaf=folder_leaf,
                end_user_id=end_user_id,
            )

            if job is None:
                logger.info("Batch ingestion already queued (doc=%s, idx=%s)", doc.external_id, idx)
            else:
                logger.info("Batch ingestion queued (job_id=%s, doc=%s, idx=%s)", job.job_id, doc.external_id, idx)
            created_documents.append(doc)

        return BatchIngestResponse(documents=created_documents, errors=[])
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error queueing batch ingestion: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error queueing batch ingestion: {str(exc)}")


# ---------------------------------------------------------------------------
# /ingest/requeue
# ---------------------------------------------------------------------------


@router.post("/requeue", response_model=RequeueIngestionResponse)
async def requeue_ingest_jobs(
    request: RequeueIngestionRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> RequeueIngestionResponse:
    """Requeue ingestion jobs for documents stuck in processing or marked as failed."""
    if not request.include_all and not request.jobs:
        raise HTTPException(status_code=400, detail="No jobs provided for requeue")

    statuses = request.statuses or ["processing", "failed"]
    auto_limit = request.limit if request.include_all and request.limit and request.limit > 0 else None
    auto_selected = 0
    colpali_overrides: Dict[str, Optional[bool]] = {job.external_id: job.use_colpali for job in request.jobs}
    processed_ids: Set[str] = set()
    results: List[RequeueIngestionResult] = []

    async def _process_document(doc: Document, override_flag: Optional[bool]) -> None:
        ext_id = doc.external_id
        if ext_id in processed_ids:
            return

        processed_ids.add(ext_id)

        try:
            auth_for_doc = AuthContext(
                user_id=auth.user_id,
                app_id=doc.app_id or auth.app_id,
            )

            bucket = doc.storage_info.get("bucket") if doc.storage_info else None
            key = doc.storage_info.get("key") if doc.storage_info else None

            if not bucket or not key:
                results.append(
                    RequeueIngestionResult(
                        external_id=ext_id,
                        status="error",
                        message="Document is missing storage location metadata",
                    )
                )
                return

            # TODO: Add storage file validation once storage.file_exists() is implemented
            # This would prevent enqueueing jobs for deleted files

            use_colpali_flag = override_flag
            if use_colpali_flag is None:
                for source in (doc.system_metadata or {}, doc.metadata or {}):
                    if isinstance(source, dict) and "use_colpali" in source:
                        raw_value = source.get("use_colpali")
                        if isinstance(raw_value, str):
                            use_colpali_flag = raw_value.lower() in {"true", "1", "yes", "y", "on"}
                        else:
                            use_colpali_flag = bool(raw_value)
                        break
            if use_colpali_flag is None:
                use_colpali_flag = True

            system_metadata = doc.system_metadata or {}
            if isinstance(system_metadata, str):
                system_metadata = json.loads(system_metadata)
            system_metadata = dict(system_metadata)
            system_metadata.pop("progress", None)
            system_metadata.pop("error", None)
            system_metadata["status"] = "processing"
            system_metadata["updated_at"] = datetime.now(UTC)

            sanitized_system_metadata = IngestionService._clean_system_metadata(system_metadata)
            await ingestion_service.db.update_document(
                document_id=ext_id,
                updates={"system_metadata": sanitized_system_metadata},
                auth=auth_for_doc,
            )

            auth_dict = {
                "user_id": auth_for_doc.user_id,
                "app_id": auth_for_doc.app_id,
            }

            doc_metadata = doc.metadata or {}
            if isinstance(doc_metadata, str):
                doc_metadata = json.loads(doc_metadata)
            doc_metadata_types = doc.metadata_types or {}

            job = await redis.enqueue_job(
                "process_ingestion_job",
                _job_id=f"ingest:{ext_id}",
                document_id=ext_id,
                file_key=key,
                bucket=bucket,
                original_filename=doc.filename,
                content_type=doc.content_type,
                metadata_json=json.dumps(doc_metadata),
                metadata_types_json=json.dumps(doc_metadata_types),
                auth_dict=auth_dict,
                use_colpali=use_colpali_flag,
                folder_name=doc.folder_name,
                end_user_id=doc.end_user_id,
            )

            if job is None:
                results.append(
                    RequeueIngestionResult(
                        external_id=ext_id,
                        status="already_queued",
                        message="An ingestion job is already pending for this document",
                    )
                )
            else:
                results.append(
                    RequeueIngestionResult(
                        external_id=ext_id,
                        status="requeued",
                        message="Ingestion job enqueued successfully",
                    )
                )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to requeue ingestion for document %s: %s", ext_id, exc, exc_info=True)
            results.append(
                RequeueIngestionResult(
                    external_id=ext_id,
                    status="error",
                    message=str(exc),
                )
            )

    async def _load_docs_by_status(target_statuses: List[str]) -> None:
        nonlocal auto_selected
        skip = 0
        limit = 200
        while True:
            if auto_limit is not None and auto_selected >= auto_limit:
                break
            batch = await ingestion_service.db.list_documents_flexible(
                auth=auth,
                skip=skip,
                limit=limit,
                status_filter=target_statuses,
                return_documents=True,
            )
            docs = batch.get("documents", [])
            if not docs:
                break
            for doc in docs:
                if auto_limit is not None and auto_selected >= auto_limit:
                    break
                if doc.external_id in processed_ids:
                    continue
                auto_selected += 1
                await _process_document(doc, colpali_overrides.get(doc.external_id))
            if len(docs) < limit:
                break
            skip += limit
            if auto_limit is not None and auto_selected >= auto_limit:
                break

    if request.include_all:
        await _load_docs_by_status(statuses)

    for job in request.jobs:
        ext_id = job.external_id
        if ext_id in processed_ids:
            continue
        try:
            doc = await ingestion_service.db.get_document(ext_id, auth)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch document %s during requeue: %s", ext_id, exc, exc_info=True)
            results.append(
                RequeueIngestionResult(
                    external_id=ext_id,
                    status="error",
                    message=str(exc),
                )
            )
            continue

        if not doc:
            results.append(
                RequeueIngestionResult(
                    external_id=ext_id,
                    status="not_found",
                    message="Document not found or access denied",
                )
            )
            continue

        await _process_document(doc, colpali_overrides.get(ext_id))

    return RequeueIngestionResponse(results=results)


# ---------------------------------------------------------------------------
# /ingest/document/ephemeral
# ---------------------------------------------------------------------------


@router.post("/document/query", response_model=DocumentQueryResponse)
@telemetry.track(operation_type="document_query", metadata_resolver=telemetry.ingest_file_metadata)
async def query_document(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    response_schema: Optional[str] = Form(None, alias="schema"),
    ingestion_options: str = Form("{}"),
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> DocumentQueryResponse:
    """
    Execute a one-off analysis for a document using Morphik On-the-Fly, optionally enforcing structured output and
    scheduling a follow-up ingestion.

    `ingestion_options` is a JSON string controlling post-analysis ingestion behaviour via keys such as `ingest`,
    `metadata`, `use_colpali`, `folder_name`, and `end_user_id`. Additional keys are ignored. A
    :class:`DocumentQueryResponse` describing the inline analysis and any queued ingestion is returned.
    """
    try:
        ingestion_options_dict = json.loads(ingestion_options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"ingestion_options must be valid JSON: {exc}") from exc

    if ingestion_options_dict is None:
        ingestion_options_dict = {}
    if not isinstance(ingestion_options_dict, dict):
        raise HTTPException(status_code=400, detail="ingestion_options must be a JSON object")

    metadata_dict = ingestion_options_dict.get("metadata", {})
    if metadata_dict is None:
        metadata_dict = {}
    if not isinstance(metadata_dict, dict):
        raise HTTPException(status_code=400, detail="ingestion_options.metadata must be a JSON object when provided")

    ingest_after_bool = _parse_bool(ingestion_options_dict.get("ingest"))
    use_colpali_bool = _parse_bool(ingestion_options_dict.get("use_colpali"))

    folder_override = ingestion_options_dict.get("folder_name")
    if folder_override in ("", None):
        folder_override = None
    elif not isinstance(folder_override, str):
        raise HTTPException(status_code=400, detail="folder_name must be a string path")

    end_user_override = ingestion_options_dict.get("end_user_id")
    if end_user_override in ("", None):
        end_user_override = None
    elif not isinstance(end_user_override, str):
        raise HTTPException(status_code=400, detail="end_user_id must be a string")

    try:
        normalized_ingestion_options = IngestionOptions(
            ingest=ingest_after_bool,
            use_colpali=use_colpali_bool,
            folder_name=folder_override,
            end_user_id=end_user_override,
            metadata=metadata_dict,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid ingestion_options: {exc}") from exc

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > MORPHIK_ON_THE_FLY_MAX_DOCUMENT_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file exceeds limit of {MORPHIK_ON_THE_FLY_MAX_DOCUMENT_BYTES // (1024 * 1024)} MB",
        )

    schema_obj: Optional[Dict[str, Any]] = None
    if response_schema:
        try:
            parsed_schema = json.loads(response_schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"schema must be valid JSON: {exc}") from exc

        if not isinstance(parsed_schema, dict):
            raise HTTPException(status_code=400, detail="schema must be a JSON object")
        schema_obj = parsed_schema

    try:
        morphik_on_the_fly_result = await generate_morphik_on_the_fly_content(
            prompt=prompt,
            schema=schema_obj,
            document_bytes=file_bytes,
            mime_type=file.content_type,
        )
    except MorphikOnTheFlyContentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    structured_output = morphik_on_the_fly_result.structured_output
    input_metadata = dict(metadata_dict)
    if structured_output is None:
        combined_metadata = input_metadata
        extracted_metadata = None
    elif isinstance(structured_output, dict):
        extracted_metadata = structured_output
        combined_metadata = {**input_metadata, **structured_output}
    else:
        extracted_metadata = None
        combined_metadata = {**input_metadata, "morphik_on_the_fly_structured_output": structured_output}

    ingestion_document: Optional[Document] = None
    if ingest_after_bool:
        filename = file.filename or "uploaded_document"

        try:
            ingestion_document = await ingestion_service.ingest_file_content(
                file_content_bytes=file_bytes,
                filename=filename,
                content_type=file.content_type,
                metadata=combined_metadata,
                metadata_types=None,
                auth=auth,
                redis=redis,
                folder_name=folder_override,
                end_user_id=end_user_override,
                use_colpali=use_colpali_bool,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except TypedMetadataError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to queue ingestion after metadata extraction for %s", filename)
            raise HTTPException(status_code=500, detail=f"Failed to queue ingestion: {exc}") from exc

    return DocumentQueryResponse(
        structured_output=structured_output,
        extracted_metadata=extracted_metadata,
        text_output=morphik_on_the_fly_result.text_output,
        ingestion_enqueued=ingest_after_bool and ingestion_document is not None,
        ingestion_document=ingestion_document,
        input_metadata=input_metadata,
        combined_metadata=combined_metadata,
        ingestion_options=normalized_ingestion_options,
    )
