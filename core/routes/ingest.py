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
from core.models.request import BatchIngestResponse, DocumentQueryResponse, IngestTextRequest, RequeueIngestionRequest
from core.models.responses import RequeueIngestionResponse, RequeueIngestionResult
from core.routes.utils import warn_if_legacy_rules
from core.services.document_service import DocumentService
from core.services.gemini_structured_output import GeminiContentError, generate_gemini_content
from core.services.telemetry import TelemetryService
from core.services_init import document_service, storage
from core.storage.utils_file_extensions import detect_file_type

# ---------------------------------------------------------------------------
# Router initialisation & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()

GEMINI_MAX_DOCUMENT_BYTES = 20 * 1024 * 1024  # 20 MB limit for Gemini inline uploads


def _parse_bool(value: Optional[Union[str, bool]]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# /ingest/text
# ---------------------------------------------------------------------------


@router.post("/text", response_model=Document)
@telemetry.track(operation_type="ingest_text", metadata_resolver=telemetry.ingest_text_metadata)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token),
) -> Document:
    """Ingest a **text** document.

    Args:
        request: IngestTextRequest payload containing:
            • content – raw text to ingest.
            • filename – optional filename to help detect MIME-type.
            • metadata – optional JSON metadata dict.
            • folder_name – optional folder scope.
            • end_user_id – optional end-user scope.
        auth: Decoded JWT context (injected).

    Returns:
        Document metadata row representing the newly-ingested text.
    """
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

        return await document_service.ingest_text(
            content=request.content,
            filename=request.filename,
            metadata=request.metadata,
            use_colpali=request.use_colpali,
            auth=auth,
            folder_name=request.folder_name,
            end_user_id=request.end_user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


# ---------------------------------------------------------------------------
# /ingest/file
# ---------------------------------------------------------------------------


@router.post("/file", response_model=Document)
@telemetry.track(operation_type="queue_ingest_file", metadata_resolver=telemetry.ingest_file_metadata)
async def ingest_file(
    request: Request,
    file: UploadFile,
    metadata: str = Form("{}"),
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

    Args:
        file: Uploaded file from multipart/form-data.
        metadata: JSON-string representing user metadata.
        auth: Caller context – must include *write* permission.
        use_colpali: Switch to multi-vector embeddings.
        folder_name: Optionally scope doc to a folder.
        end_user_id: Optionally scope doc to an end-user.
        redis: arq redis connection – used to enqueue the job.

    Returns:
        Document stub with ``status='processing'``.
    """
    try:
        # ------------------------------------------------------------------
        # Parse and validate inputs
        # ------------------------------------------------------------------
        await warn_if_legacy_rules(request, "/ingest/file", logger)
        metadata_dict = json.loads(metadata)

        def str2bool(v: Union[bool, str]) -> bool:
            return v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}

        use_colpali_bool = str2bool(use_colpali)

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        logger.debug("Queueing file ingestion with use_colpali=%s", use_colpali_bool)

        # ------------------------------------------------------------------
        # Create initial Document stub (status = processing)
        # ------------------------------------------------------------------
        doc = Document(
            content_type=file.content_type,
            filename=file.filename,
            metadata=metadata_dict,
            system_metadata={"status": "processing"},
            folder_name=folder_name,
            end_user_id=end_user_id,
            app_id=auth.app_id,
        )

        # Store stub in application database (not control-plane DB)
        app_db = document_service.db
        success = await app_db.store_document(doc, auth)
        if not success:
            raise Exception("Failed to store document metadata")

        # Add the document to the requested folder immediately so folder views can show in-progress items.
        # The ingestion worker re-runs this to ensure the folder is still in sync on completion.
        if folder_name:
            try:
                await document_service._ensure_folder_exists(folder_name, doc.external_id, auth)
            except Exception as folder_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to add document %s to folder %s immediately after ingest: %s",
                    doc.external_id,
                    folder_name,
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

        bucket_override = await document_service._get_bucket_for_app(auth.app_id)
        bucket, stored_key = await storage.upload_file(
            file_content,
            file_key,
            file.content_type,
            bucket=bucket_override or "",
        )

        doc.storage_info = {"bucket": bucket, "key": stored_key}

        # Keep storage history
        from core.models.documents import StorageFileInfo  # local import to avoid cycles

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
            metadata_json=metadata,
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

    Args:
        files: List of files to upload.
        metadata: Either a single JSON-string dict or list of dicts matching
            the number of files.
        use_colpali: Enable multi-vector embeddings.
        folder_name: Optional folder scoping for **all** files.
        end_user_id: Optional end-user scoping for **all** files.
        auth: Caller context with *write* permission.
        redis: arq redis connection to enqueue jobs.

    Returns:
        BatchIngestResponse summarising created documents & errors.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch ingestion")

    try:
        await warn_if_legacy_rules(request, "/ingest/files", logger)
        metadata_value = json.loads(metadata)

        def str2bool(v: Union[bool, str]) -> bool:
            return v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}

        use_colpali_bool = str2bool(use_colpali)

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(exc)}")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    # Validate metadata length when list provided
    if isinstance(metadata_value, list) and len(metadata_value) != len(files):
        raise HTTPException(
            status_code=400,
            detail=(f"Number of metadata items ({len(metadata_value)}) must match number of files " f"({len(files)})"),
        )

    auth_dict = {
        "entity_type": auth.entity_type.value,
        "entity_id": auth.entity_id,
        "app_id": auth.app_id,
        "permissions": list(auth.permissions),
        "user_id": auth.user_id,
    }

    created_documents: List[Document] = []

    from core.models.documents import StorageFileInfo  # local import to avoid cycles

    try:
        for idx, file in enumerate(files):
            metadata_item = metadata_value[idx] if isinstance(metadata_value, list) else metadata_value
            # ------------------------------------------------------------------
            # Create stub Document (processing)
            # ------------------------------------------------------------------
            doc = Document(
                content_type=file.content_type,
                filename=file.filename,
                metadata=metadata_item,
                folder_name=folder_name,
                end_user_id=end_user_id,
                app_id=auth.app_id,
            )
            doc.system_metadata["status"] = "processing"

            app_db = document_service.db
            success = await app_db.store_document(doc, auth)
            if not success:
                raise Exception(f"Failed to store document metadata for {file.filename}")

            # Keep folder listings in sync immediately; worker re-runs this when processing finishes.
            if folder_name:
                try:
                    await document_service._ensure_folder_exists(folder_name, doc.external_id, auth)
                except Exception as folder_exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to add batch document %s to folder %s immediately after ingest: %s",
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

            bucket_override = await document_service._get_bucket_for_app(auth.app_id)
            bucket, stored_key = await storage.upload_file(
                file_content,
                file_key,
                file.content_type,
                bucket=bucket_override or "",
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

            metadata_json = json.dumps(metadata_item)

            job = await redis.enqueue_job(
                "process_ingestion_job",
                _job_id=f"ingest:{doc.external_id}",
                document_id=doc.external_id,
                file_key=stored_key,
                bucket=bucket,
                original_filename=file.filename,
                content_type=file.content_type,
                metadata_json=metadata_json,
                auth_dict=auth_dict,
                use_colpali=use_colpali_bool,
                folder_name=folder_name,
                end_user_id=end_user_id,
            )

            if job is None:
                logger.info("Batch ingestion already queued (doc=%s, idx=%s)", doc.external_id, idx)
            else:
                logger.info("Batch ingestion queued (job_id=%s, doc=%s, idx=%s)", job.job_id, doc.external_id, idx)
            created_documents.append(doc)

        return BatchIngestResponse(documents=created_documents, errors=[])
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
    if "write" not in auth.permissions:
        raise HTTPException(status_code=403, detail="User does not have write permission")
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
                entity_type=auth.entity_type,
                entity_id=auth.entity_id,
                app_id=doc.app_id or auth.app_id,
                permissions=set(auth.permissions),
                user_id=auth.user_id,
            )

            storage_record = None
            if doc.storage_files:
                storage_record = doc.storage_files[-1]
            elif doc.storage_info:
                storage_record = doc.storage_info

            bucket = getattr(storage_record, "bucket", None) if storage_record else None
            key = getattr(storage_record, "key", None) if storage_record else None

            if isinstance(storage_record, dict):
                bucket = storage_record.get("bucket")
                key = storage_record.get("key")

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

            sanitized_system_metadata = DocumentService._clean_system_metadata(system_metadata)
            await document_service.db.update_document(
                document_id=ext_id,
                updates={"system_metadata": sanitized_system_metadata},
                auth=auth_for_doc,
            )

            auth_dict = {
                "entity_type": auth_for_doc.entity_type.value,
                "entity_id": auth_for_doc.entity_id,
                "app_id": auth_for_doc.app_id,
                "permissions": list(auth_for_doc.permissions),
                "user_id": auth_for_doc.user_id,
            }

            doc_metadata = doc.metadata or {}
            if isinstance(doc_metadata, str):
                doc_metadata = json.loads(doc_metadata)

            job = await redis.enqueue_job(
                "process_ingestion_job",
                _job_id=f"ingest:{ext_id}",
                document_id=ext_id,
                file_key=key,
                bucket=bucket,
                original_filename=doc.filename,
                content_type=doc.content_type,
                metadata_json=json.dumps(doc_metadata),
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
            batch = await document_service.db.list_documents_flexible(
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
            doc = await document_service.db.get_document(ext_id, auth)
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
    schema: Optional[str] = Form(None),
    ingestion_options: str = Form("{}"),
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> DocumentQueryResponse:
    """
    Run an on-demand Gemini document query with optional structured output and ingestion.

    Args:
        file: Uploaded source document (PDF or another supported MIME type).
        prompt: Natural-language instruction to execute against the document.
        schema: Optional JSON schema enforcing structured Gemini output.
        ingestion_options: JSON object controlling ingestion follow-up (keys: ingest, use_colpali, folder_name, end_user_id, metadata).

    Returns:
        DocumentQueryResponse containing Gemini outputs, original metadata, and ingestion status details.
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
    elif isinstance(folder_override, list):
        if not all(isinstance(item, str) for item in folder_override):
            raise HTTPException(status_code=400, detail="folder_name list must contain only strings")
    elif not isinstance(folder_override, str):
        raise HTTPException(status_code=400, detail="folder_name must be a string or list of strings")

    end_user_override = ingestion_options_dict.get("end_user_id")
    if end_user_override in ("", None):
        end_user_override = None
    elif not isinstance(end_user_override, str):
        raise HTTPException(status_code=400, detail="end_user_id must be a string")

    normalized_ingestion_options: Dict[str, Any] = {
        "ingest": ingest_after_bool,
        "use_colpali": use_colpali_bool,
    }
    if folder_override is not None:
        normalized_ingestion_options["folder_name"] = folder_override
    if end_user_override is not None:
        normalized_ingestion_options["end_user_id"] = end_user_override
    normalized_ingestion_options["metadata"] = metadata_dict

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > GEMINI_MAX_DOCUMENT_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file exceeds Gemini limit of {GEMINI_MAX_DOCUMENT_BYTES // (1024 * 1024)} MB",
        )

    schema_obj: Optional[Dict[str, Any]] = None
    if schema:
        try:
            parsed_schema = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"schema must be valid JSON: {exc}") from exc

        if not isinstance(parsed_schema, dict):
            raise HTTPException(status_code=400, detail="schema must be a JSON object")
        schema_obj = parsed_schema

    try:
        gemini_result = await generate_gemini_content(
            prompt=prompt,
            schema=schema_obj,
            document_bytes=file_bytes,
            mime_type=file.content_type,
        )
    except GeminiContentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    structured_output = gemini_result.structured_output
    input_metadata = dict(metadata_dict)
    if structured_output is None:
        combined_metadata = input_metadata
        extracted_metadata = None
    elif isinstance(structured_output, dict):
        extracted_metadata = structured_output
        combined_metadata = {**input_metadata, **structured_output}
    else:
        extracted_metadata = None
        combined_metadata = {**input_metadata, "gemini_structured_output": structured_output}

    ingestion_document: Optional[Document] = None
    if ingest_after_bool:
        if "write" not in auth.permissions:
            raise HTTPException(status_code=403, detail="User does not have write permission to ingest documents")

        filename = file.filename or "uploaded_document"

        try:
            ingestion_document = await document_service.ingest_file_content(
                file_content_bytes=file_bytes,
                filename=filename,
                content_type=file.content_type,
                metadata=combined_metadata,
                auth=auth,
                redis=redis,
                folder_name=folder_override,
                end_user_id=end_user_override,
                use_colpali=use_colpali_bool,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to queue ingestion after metadata extraction for %s", filename)
            raise HTTPException(status_code=500, detail=f"Failed to queue ingestion: {exc}") from exc

    return DocumentQueryResponse(
        structured_output=structured_output,
        extracted_metadata=extracted_metadata,
        text_output=gemini_result.text_output,
        ingestion_enqueued=ingest_after_bool and ingestion_document is not None,
        ingestion_document=ingestion_document,
        input_metadata=input_metadata,
        combined_metadata=combined_metadata,
        ingestion_options=normalized_ingestion_options,
    )
