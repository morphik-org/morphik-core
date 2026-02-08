import json
import logging
from typing import Any, Dict, List, Optional, Set

import arq
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from core.auth_utils import verify_token
from core.dependencies import get_redis_pool
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
from core.routes.utils import (
    enforce_no_user_mutable_fields,
    parse_bool,
    parse_json_dict,
    parse_json_value,
    warn_if_legacy_rules,
)
from core.services.ingestion_service import IngestionService
from core.services.morphik_on_the_fly_structured_output import (
    MorphikOnTheFlyContentError,
    generate_morphik_on_the_fly_content,
)
from core.services.telemetry import TelemetryService
from core.services_init import ingestion_service
from core.utils.typed_metadata import TypedMetadataError

# ---------------------------------------------------------------------------
# Router initialisation & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)
telemetry = TelemetryService()

MORPHIK_ON_THE_FLY_MAX_DOCUMENT_BYTES = 20 * 1024 * 1024  # 20 MB limit for inline uploads to Morphik On-the-Fly


# ---------------------------------------------------------------------------
# /ingest/text
# ---------------------------------------------------------------------------


@router.post("/text", response_model=Document)
@telemetry.track(operation_type="ingest_text", metadata_resolver=telemetry.ingest_text_metadata)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> Document:
    """Ingest a **text** document asynchronously (queued like /ingest/file)."""
    try:
        if getattr(request, "rules", None):
            logger.warning("Legacy 'rules' field supplied to /ingest/text; ignoring payload.")

        enforce_no_user_mutable_fields(
            ingestion_service,
            request.metadata,
            request.metadata_types,
            context="ingest",
            request_model=request,
        )

        filename = ingestion_service._normalize_text_filename(request.filename, request.content)
        content_bytes = request.content.encode("utf-8")

        return await ingestion_service.ingest_file_content(
            file_content_bytes=content_bytes,
            filename=filename,
            content_type=None,
            metadata=request.metadata,
            auth=auth,
            redis=redis,
            metadata_types=request.metadata_types,
            folder_name=request.folder_name,
            end_user_id=request.end_user_id,
            use_colpali=request.use_colpali,
        )
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during text ingestion: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error during text ingestion: {str(exc)}")


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
        metadata_dict = parse_json_dict(metadata, "metadata", default={})
        metadata_types_dict = parse_json_dict(metadata_types, "metadata_types", default={})
        use_colpali_bool = parse_bool(use_colpali)
        logger.debug("Queueing file ingestion with use_colpali=%s", use_colpali_bool)

        file_content = await file.read()
        filename = file.filename or "uploaded_file"

        return await ingestion_service.ingest_file_content(
            file_content_bytes=file_content,
            filename=filename,
            content_type=file.content_type,
            metadata=metadata_dict,
            auth=auth,
            redis=redis,
            metadata_types=metadata_types_dict,
            folder_name=folder_name,
            end_user_id=end_user_id,
            use_colpali=use_colpali_bool,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
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
        metadata_value = parse_json_value(metadata, "metadata", default={})
        metadata_types_value = parse_json_value(metadata_types, "metadata_types", default={})
        if metadata_types_value is None:
            metadata_types_value = {}

        use_colpali_bool = parse_bool(use_colpali)
    except HTTPException:
        raise

    if not isinstance(metadata_value, (dict, list)):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object or list of objects")
    if not isinstance(metadata_types_value, (dict, list)):
        raise HTTPException(status_code=400, detail="metadata_types must be a JSON object or list of objects")

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

    created_documents: List[Document] = []

    try:
        for idx, file in enumerate(files):
            metadata_item = metadata_value[idx] if isinstance(metadata_value, list) else metadata_value
            if metadata_item is None:
                metadata_item = {}
            if not isinstance(metadata_item, dict):
                raise HTTPException(status_code=400, detail="metadata entries must be JSON objects")
            metadata_types_item = (
                metadata_types_value[idx] if isinstance(metadata_types_value, list) else metadata_types_value
            )
            if metadata_types_item is None:
                metadata_types_item = {}
            if not isinstance(metadata_types_item, dict):
                raise HTTPException(status_code=400, detail="metadata_types entries must be JSON objects")
            file_content = await file.read()
            filename = file.filename or "uploaded_file"

            doc = await ingestion_service.ingest_file_content(
                file_content_bytes=file_content,
                filename=filename,
                content_type=file.content_type,
                metadata=metadata_item,
                auth=auth,
                redis=redis,
                metadata_types=metadata_types_item,
                folder_name=folder_name,
                end_user_id=end_user_id,
                use_colpali=use_colpali_bool,
            )

            logger.info("Batch ingestion queued (doc=%s, idx=%s)", doc.external_id, idx)
            created_documents.append(doc)

        return BatchIngestResponse(documents=created_documents, errors=[])
    except TypedMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
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
                use_colpali_flag = False

            system_metadata = doc.system_metadata or {}
            if isinstance(system_metadata, str):
                system_metadata = json.loads(system_metadata)
            sanitized_system_metadata = IngestionService._reset_processing_metadata(system_metadata)
            await ingestion_service.db.update_document(
                document_id=ext_id,
                updates={"system_metadata": sanitized_system_metadata},
                auth=auth_for_doc,
            )
            job_payload = IngestionService._build_ingestion_job_payload(
                document_id=ext_id,
                file_key=key,
                bucket=bucket,
                original_filename=doc.filename,
                content_type=doc.content_type,
                auth=auth_for_doc,
                use_colpali=use_colpali_flag,
                folder_name=doc.folder_name,
                folder_path=doc.folder_path,
                folder_leaf=doc.folder_name,
                end_user_id=doc.end_user_id,
            )
            job = await redis.enqueue_job("process_ingestion_job", **job_payload)

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
    ingestion_options_dict = parse_json_dict(ingestion_options, "ingestion_options", default={})

    metadata_dict = ingestion_options_dict.get("metadata", {})
    if metadata_dict is None:
        metadata_dict = {}
    if not isinstance(metadata_dict, dict):
        raise HTTPException(status_code=400, detail="ingestion_options.metadata must be a JSON object when provided")

    ingest_after_bool = parse_bool(ingestion_options_dict.get("ingest"))
    use_colpali_bool = parse_bool(ingestion_options_dict.get("use_colpali"))

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
        schema_obj = parse_json_dict(response_schema, "schema")

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
