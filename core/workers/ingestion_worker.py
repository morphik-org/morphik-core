import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import time
import traceback
import urllib.parse as up
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from arq.connections import RedisSettings
from opentelemetry.trace import Status, StatusCode, get_current_span
from sqlalchemy import text

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.colpali_api_embedding_model import ColpaliApiEmbeddingModel
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.auth import AuthContext
from core.parser.morphik_parser import MorphikParser
from core.services.ingestion_service import IngestionService, PdfConversionError
from core.services.telemetry import TelemetryService
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.storage.utils_file_extensions import detect_content_type, is_colpali_native_format
from core.utils.folder_utils import normalize_ingest_folder_inputs
from core.utils.storage_usage import extract_storage_bytes
from core.utils.typed_metadata import MetadataBundle
from core.vector_store.base_vector_store import BaseVectorStore
from core.vector_store.dual_multivector_store import DualMultiVectorStore
from core.vector_store.fast_multivector_store import FastMultiVectorStore
from core.vector_store.local_multivector_store import LocalMultiVectorStore
from core.vector_store.multi_vector_store import MultiVectorStore
from core.vector_store.pgvector_store import PGVectorStore

# Enterprise routing helpers
from ee.db_router import get_database_for_app, get_vector_store_for_app

logger = logging.getLogger(__name__)
for noisy_logger in ("httpx", "httpcore", "aiohttp", "turbopuffer"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
progress_logger = logging.getLogger("worker.progress")
summary_logger = logging.getLogger("worker.ingestion_summary")

# Initialize global settings once
settings = get_settings()

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set logger level based on settings (diff used INFO directly)
logger.setLevel(logging.INFO)
logger.propagate = True
progress_logger.setLevel(logging.INFO)
if not progress_logger.handlers:
    progress_handler = logging.StreamHandler()
    progress_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    progress_logger.addHandler(progress_handler)
progress_logger.propagate = False
summary_logger.setLevel(logging.INFO)
if not summary_logger.handlers:
    summary_handler = RotatingFileHandler(
        "logs/ingestion_summary.jsonl",
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    summary_handler.setFormatter(logging.Formatter("%(message)s"))
    summary_logger.addHandler(summary_handler)
summary_logger.propagate = False

_COLPALI_STORE_CACHE: Dict[tuple, BaseVectorStore] = {}
_COLPALI_STORE_LOCK = asyncio.Lock()


def _build_colpali_store_cache_key(uri: str) -> tuple:
    return (
        uri,
        settings.ENABLE_DUAL_MULTIVECTOR_INGESTION,
        settings.MULTIVECTOR_STORE_PROVIDER,
        settings.COLPALI_MODE,
    )


def _resolve_database_uri(database: PostgresDatabase) -> str:
    # Keep the password visible for psycopg and add sslmode=require in cloud mode when missing.
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    uri_raw = database.engine.url.render_as_string(hide_password=False)
    parsed = urlparse(uri_raw)
    query = parse_qs(parsed.query)
    if "sslmode" not in query and settings.MODE == "cloud":
        query["sslmode"] = ["require"]
        parsed = parsed._replace(query=urlencode(query, doseq=True))
    return urlunparse(parsed)


async def _get_worker_colpali_store(database: PostgresDatabase) -> Optional[BaseVectorStore]:
    """Return a per-worker cached ColPali vector store instance."""
    uri_final = _resolve_database_uri(database)
    cache_key = _build_colpali_store_cache_key(uri_final)

    cached = _COLPALI_STORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    async with _COLPALI_STORE_LOCK:
        cached = _COLPALI_STORE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if settings.ENABLE_DUAL_MULTIVECTOR_INGESTION:
            if not settings.TURBOPUFFER_API_KEY:
                raise ValueError("TURBOPUFFER_API_KEY is required when dual ingestion is enabled")

            fast_store = FastMultiVectorStore(
                uri=uri_final,
                tpuf_api_key=settings.TURBOPUFFER_API_KEY,
                namespace="public",
            )
            slow_store = MultiVectorStore(uri=uri_final)
            store: BaseVectorStore = DualMultiVectorStore(
                fast_store=fast_store, slow_store=slow_store, enable_dual_ingestion=True
            )
        elif settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
            if not settings.TURBOPUFFER_API_KEY:
                raise ValueError("TURBOPUFFER_API_KEY is required when using morphik multivector store provider")
            store = FastMultiVectorStore(
                uri=uri_final,
                tpuf_api_key=settings.TURBOPUFFER_API_KEY,
                namespace="public",
            )
        elif settings.MULTIVECTOR_STORE_PROVIDER == "local":
            store = LocalMultiVectorStore(
                uri=uri_final,
                chroma_host=settings.CHROMA_HOST,
                chroma_port=settings.CHROMA_PORT,
                chroma_ssl=settings.CHROMA_SSL,
                namespace="public",
            )
            logger.info(
                "Worker initialized LocalMultiVectorStore (ChromaDB) at %s:%d",
                settings.CHROMA_HOST,
                settings.CHROMA_PORT,
            )
        else:
            store = MultiVectorStore(uri=uri_final)

        await asyncio.to_thread(store.initialize)
        _COLPALI_STORE_CACHE[cache_key] = store
        return store


def _ensure_worker_logging() -> None:
    """Ensure stdout logging is configured when the worker starts."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(root_handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)


async def update_document_progress(ingestion_service, document_id, auth, current_step, total_steps, step_name):
    """
    Helper function to update document progress during ingestion.

    Args:
        ingestion_service: The ingestion service instance
        document_id: ID of the document to update
        auth: Authentication context
        current_step: Current step number (1-based)
        total_steps: Total number of steps
        step_name: Human-readable name of the current step
    """
    try:
        updates = {
            "system_metadata": {
                "status": "processing",
                "progress": {
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "step_name": step_name,
                    "percentage": round((current_step / total_steps) * 100),
                },
                "updated_at": datetime.now(UTC),
            }
        }
        await ingestion_service.db.update_document(document_id, updates, auth)
        logger.debug(f"Updated progress: {step_name} ({current_step}/{total_steps})")
    except Exception as e:
        logger.warning(f"Failed to update progress for document {document_id}: {e}")
        # Don't fail the ingestion if progress update fails


_STORE_TIME_KEYS = ("chunk_payload_upload_s", "multivector_upload_s", "vector_store_write_s", "cache_write_s")
_STORE_COUNT_KEYS = (
    "chunk_payload_objects",
    "multivector_objects",
    "vector_store_rows",
    "cache_write_objects",
    "chunk_payload_bytes",
    "multivector_bytes",
)
_STORE_BACKEND_KEYS = ("chunk_payload_backend", "multivector_backend", "vector_store_backend")


def _merge_store_metrics(total: Dict[str, Any], latest: Dict[str, Any]) -> None:
    for key in _STORE_TIME_KEYS:
        total[key] = total.get(key, 0.0) + float(latest.get(key, 0.0) or 0.0)
    for key in _STORE_COUNT_KEYS:
        total[key] = total.get(key, 0) + int(latest.get(key, 0) or 0)
    for key in _STORE_BACKEND_KEYS:
        value = latest.get(key)
        if not value:
            continue
        if key not in total:
            total[key] = value
            continue
        if total[key] == value:
            continue
        if isinstance(total[key], list):
            if value not in total[key]:
                total[key].append(value)
        else:
            total[key] = [total[key], value]


def _accumulate_store_metrics(total: Dict[str, Any], latest: Dict[str, Any]) -> None:
    if not latest:
        return
    if "fast" in latest or "slow" in latest:
        total.setdefault("mode", latest.get("mode", "dual"))
        for store_key in ("fast", "slow"):
            if store_key in latest and latest[store_key]:
                total.setdefault(store_key, {})
                _merge_store_metrics(total[store_key], latest[store_key])
        return
    _merge_store_metrics(total, latest)


def _with_throughput(metrics: Dict[str, Any], total_chunks: int) -> Dict[str, Any]:
    if not metrics:
        return {}
    result = dict(metrics)
    write_time = float(result.get("vector_store_write_s") or 0.0)
    result["throughput_chunks_s"] = round((total_chunks / write_time), 2) if write_time > 0 else 0.0
    for key in _STORE_TIME_KEYS:
        if key in result:
            result[key] = round(float(result[key]), 2)
    for key in _STORE_COUNT_KEYS:
        if key in result:
            result[key] = int(result[key])
    return result


def _strip_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


async def get_document_with_retry(ingestion_service, document_id, auth, max_retries=3, initial_delay=0.3):
    """
    Helper function to get a document with retries to handle race conditions.

    Args:
        ingestion_service: The ingestion service instance
        document_id: ID of the document to retrieve
        auth: Authentication context
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay before first attempt in seconds

    Returns:
        Document if found and accessible, None otherwise
    """
    attempt = 0
    retry_delay = initial_delay

    # Add initial delay to allow transaction to commit
    if initial_delay > 0:
        await asyncio.sleep(initial_delay)

    while attempt < max_retries:
        try:
            doc = await ingestion_service.db.get_document(document_id, auth)
            if doc:
                logger.debug(f"Successfully retrieved document {document_id} on attempt {attempt+1}")
                return doc

            # Document not found but no exception raised
            attempt += 1
            if attempt < max_retries:
                logger.warning(
                    f"Document {document_id} not found on attempt {attempt}/{max_retries}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5

        except Exception as e:
            attempt += 1
            error_msg = str(e)
            if attempt < max_retries:
                logger.warning(
                    f"Error retrieving document on attempt {attempt}/{max_retries}: {error_msg}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                logger.error(f"Failed to retrieve document after {max_retries} attempts: {error_msg}")
                return None

    return None


# ---------------------------------------------------------------------------
# Profiling helpers (worker-level)
# ---------------------------------------------------------------------------

if settings.ENABLE_PROFILING:
    try:
        import yappi  # type: ignore
    except ImportError:
        yappi = None
else:
    yappi = None


@contextlib.asynccontextmanager
async def _profile_ctx(label: str):  # type: ignore
    if yappi is None:
        yield
        return

    yappi.clear_stats()
    yappi.set_clock_type("cpu")
    yappi.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - t0
        fname = f"logs/worker_{label}_{int(t0)}.prof"
        yappi.stop()
        try:
            yappi.get_func_stats().save(fname, type="pstat")
            logger.info("Saved worker profile %s (%.2fs) to %s", label, duration, fname)
        except Exception as exc:
            logger.warning("Could not save worker profile: %s", exc)


async def process_ingestion_job(
    ctx: Dict[str, Any],
    document_id: str,
    file_key: str,
    bucket: str,
    original_filename: str,
    content_type: str,
    auth_dict: Dict[str, Any],
    use_colpali: bool,
    folder_name: Optional[str] = None,
    folder_path: Optional[str] = None,
    folder_leaf: Optional[str] = None,
    end_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background worker task that processes file ingestion jobs.

    Args:
        ctx: The ARQ context dictionary
        file_key: The storage key where the file is stored
        bucket: The storage bucket name
        original_filename: The original file name
        content_type: The file's content type/MIME type
        auth_dict: Dict representation of AuthContext
        use_colpali: Whether to use ColPali embedding model
        folder_name: Optional folder to scope the document to
        end_user_id: Optional end-user ID to scope the document to

    Returns:
        A dictionary with the document ID and processing status
    """
    telemetry = TelemetryService()

    # Normalize folder inputs for consistent storage and folder linking.
    try:
        normalized_folder = normalize_ingest_folder_inputs(
            folder_name=folder_name,
            folder_path=folder_path,
            folder_leaf=folder_leaf,
            strict=False,
        )
    except ValueError as exc:
        logger.warning("Could not normalize folder inputs (path=%s name=%s): %s", folder_path, folder_name, exc)
        normalized_folder = normalize_ingest_folder_inputs(
            folder_name=None,
            folder_path=None,
            folder_leaf=folder_leaf or folder_name,
            strict=False,
        )

    normalized_folder_path = normalized_folder.path
    normalized_folder_leaf = normalized_folder.leaf
    if (
        (folder_path or folder_name)
        and normalized_folder_path is None
        and normalized_folder_leaf
        in (
            None,
            folder_leaf,
            folder_name,
        )
    ):
        logger.warning("Folder path could not be normalized (path=%s name=%s)", folder_path, folder_name)

    # Build metadata resolver inline to capture key fields
    def _meta_resolver():  # noqa: D401
        return {
            "filename": original_filename,
            "content_type": content_type,
            "folder_name": normalized_folder_leaf or folder_name,
            "folder_path": normalized_folder_path,
            "end_user_id": end_user_id,
            "use_colpali": use_colpali,
        }

    try:
        async with telemetry.track_operation(
            operation_type="ingest_worker",
            user_id=auth_dict.get("user_id") or auth_dict.get("entity_id", "unknown"),
            app_id=auth_dict.get("app_id"),
            metadata=_meta_resolver(),
        ):
            # Start performance timer
            job_start_time = time.time()
            phase_times = {}
            embedding_time = 0.0
            embeddings_per_second = 0.0
            embedding_chunk_count = 0
            colpali_image_total_time = 0.0
            colpali_text_total_time = 0.0
            colpali_image_count = 0
            colpali_text_count = 0
            colpali_endpoints: Optional[int] = None
            colpali_embedding_time = 0.0
            store_metrics_total: Dict[str, Any] = {}
            download_size_bytes = 0
            folder_id_for_summary: Optional[str] = None
            # 1. Log the start of the job
            logger.info(f"Starting ingestion job for file: {original_filename}")
            logger.debug(f"ColPali parameter received: use_colpali={use_colpali} (type: {type(use_colpali)})")
            progress_logger.info(
                "ingest start doc_id=%s file=%s colpali=%s", document_id, original_filename, use_colpali
            )

            # Define total steps for progress tracking
            total_steps = 6

            # 2. Deserialize auth (backward compatible with old queue messages)
            deserialize_start = time.time()
            auth = AuthContext(
                user_id=auth_dict.get("user_id") or auth_dict.get("entity_id", ""),
                app_id=auth_dict.get("app_id"),
            )
            phase_times["deserialize_auth"] = time.time() - deserialize_start

            # ------------------------------------------------------------------
            # Per-app routing for database and vector store
            # ------------------------------------------------------------------

            # Resolve a dedicated database/vector-store using the JWT *app_id*.
            # When app_id is None we fall back to the control-plane resources.

            database = await get_database_for_app(auth.app_id)
            vector_store = await get_vector_store_for_app(auth.app_id)

            # Initialise a per-app MultiVectorStore for ColPali when needed
            colpali_vector_store = None
            # Check both use_colpali parameter AND global enable_colpali setting
            if use_colpali and settings.ENABLE_COLPALI:
                try:
                    colpali_vector_store = await _get_worker_colpali_store(database)
                except Exception as e:
                    logger.warning(f"Failed to initialise ColPali MultiVectorStore for app {auth.app_id}: {e}")

            # Build a fresh IngestionService scoped to this job/app so we don't
            # mutate the shared instance kept in *ctx* (avoids cross-talk between
            # concurrent jobs for different apps).
            ingestion_service = IngestionService(
                storage=ctx["storage"],
                database=database,
                vector_store=vector_store,
                embedding_model=ctx["embedding_model"],
                parser=ctx["parser"],
                colpali_embedding_model=ctx.get("colpali_embedding_model"),
                colpali_vector_store=colpali_vector_store,
            )

            # 3. Download the file from storage
            await update_document_progress(ingestion_service, document_id, auth, 1, total_steps, "Downloading file")
            logger.debug(f"Downloading file from {bucket}/{file_key}")
            download_start = time.time()
            file_content = await ingestion_service.storage.download_file(bucket, file_key)

            # Ensure file_content is bytes
            if hasattr(file_content, "read"):
                file_content = file_content.read()
            download_time = time.time() - download_start
            phase_times["download_file"] = download_time
            download_size_bytes = len(file_content)
            logger.debug(f"File download took {download_time:.2f}s for {download_size_bytes/1024/1024:.2f}MB")
            progress_logger.info(
                "ingest download doc_id=%s size_mb=%.2f time_s=%.2f",
                document_id,
                download_size_bytes / 1024 / 1024,
                download_time,
            )

            # Detect file type early for optimization decisions
            mime_type = detect_content_type(
                content=file_content,
                filename=original_filename,
                content_type_hint=content_type,
            )

            # Optional: render HTML to PDF to mimic printed output and speed up parsing
            html_conversion_start = time.time()
            html_converted = False
            if mime_type in {"text/html", "application/xhtml+xml"}:
                try:
                    from weasyprint import HTML  # type: ignore

                    html_str = file_content.decode("utf-8", errors="replace")
                    pdf_bytes = HTML(string=html_str).write_pdf()
                    if pdf_bytes:
                        file_content = pdf_bytes
                        content_type = "application/pdf"
                        mime_type = "application/pdf"
                        html_converted = True
                        logger.info("Converted HTML to PDF for ingestion (WeasyPrint)")
                except Exception as html_exc:
                    logger.warning("HTML->PDF conversion failed; falling back to raw HTML: %s", html_exc)
            phase_times["html_to_pdf"] = time.time() - html_conversion_start

            # Check if we're using ColPali
            using_colpali = (
                use_colpali and ingestion_service.colpali_embedding_model and ingestion_service.colpali_vector_store
            )
            logger.debug(
                f"ColPali decision: use_colpali={use_colpali}, "
                f"has_model={bool(ingestion_service.colpali_embedding_model)}, "
                f"has_store={bool(ingestion_service.colpali_vector_store)}, "
                f"using_colpali={using_colpali}"
            )

            colpali_native_format = is_colpali_native_format(mime_type)

            # ===== PROCESSING FLOW DECISION =====
            skip_text_parsing = using_colpali and colpali_native_format

            logger.debug(
                f"Processing decision for {mime_type or 'unknown'} file: "
                f"skip_text_parsing={skip_text_parsing} "
                f"(ColPali={using_colpali}, native_format={colpali_native_format})"
            )

            # 4. Parse file to text
            await update_document_progress(ingestion_service, document_id, auth, 2, total_steps, "Parsing file")
            # Use the filename derived from the storage key so the parser
            # receives the correct extension (.txt, .pdf, etc.).  Passing the UI
            # provided original_filename (often .pdf) can mislead the parser when
            # the stored object is a pre-extracted text file (e.g. .pdf.txt).
            parse_filename = os.path.basename(file_key) if file_key else original_filename
            if html_converted:
                base_name = os.path.splitext(original_filename or parse_filename or "document")[0]
                parse_filename = f"{base_name}.pdf"

            parse_start = time.time()

            # ===== FILE PARSING LOGIC =====
            is_xml = ingestion_service.parser.is_xml_file(parse_filename, content_type)
            xml_processing = False
            xml_chunks = []

            if is_xml:
                # XML files always need special parsing
                logger.debug(f"Detected XML file: {parse_filename}")
                xml_chunks = await ingestion_service.parser.parse_and_chunk_xml(file_content, parse_filename)
                additional_metadata = {}
                text = ""
                xml_processing = True
            elif skip_text_parsing:
                # Skip text parsing for ColPali-native formats when no text rules
                additional_metadata = {}
                text = ""
                logger.debug("Skipping text extraction - ColPali will handle this file directly")
            else:
                # Normal text parsing required
                additional_metadata, text = await ingestion_service.parser.parse_file_to_text(
                    file_content, parse_filename
                )
                # Clean the extracted text to remove NULL and other problematic control characters
                # Keep: tabs, newlines, carriage returns, and all printable characters (including Unicode)
                import re

                # Remove NULL characters
                text = re.sub(r"\x00", "", text)
                # Remove control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F) but keep tab, newline, carriage return
                text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

            logger.debug(
                f"Parsed file into {'XML chunks' if xml_processing else f'text of length {len(text)}'} (filename used: {parse_filename})"
            )
            parse_time = time.time() - parse_start
            phase_times["parse_file"] = parse_time

            # NEW -----------------------------------------------------------------
            # Estimate pages early for pre-check
            if xml_processing:
                # For XML files, estimate pages based on total content length of all chunks
                total_content_length = sum(len(chunk.content) for chunk in xml_chunks)
                num_pages_estimated = estimate_pages_by_chars(total_content_length)
            else:
                num_pages_estimated = estimate_pages_by_chars(len(text))

            # 4.b Enforce tier limits (pages ingested) for cloud/free tier users
            if settings.MODE == "cloud" and auth.user_id:
                # Calculate approximate pages using same heuristic as DocumentService
                try:
                    # Dry-run verification before heavy processing
                    await check_and_increment_limits(
                        auth,
                        "ingest",
                        num_pages_estimated,
                        document_id,
                        verify_only=True,
                    )
                except Exception as limit_exc:
                    logger.error("User %s exceeded ingest limits: %s", auth.user_id, limit_exc)
                    raise
            # ---------------------------------------------------------------------

            # 6. Retrieve the existing document
            retrieve_start = time.time()
            logger.debug(f"Retrieving document with ID: {document_id}")
            logger.debug(f"Auth context: user_id={auth.user_id}, app_id={auth.app_id}")

            # Use the retry helper function with initial delay to handle race conditions
            doc = await get_document_with_retry(ingestion_service, document_id, auth, max_retries=5, initial_delay=1.0)
            retrieve_time = time.time() - retrieve_start
            phase_times["retrieve_document"] = retrieve_time
            logger.debug(f"Document retrieval took {retrieve_time:.2f}s")

            if not doc:
                logger.error(f"Document {document_id} not found in database after multiple retries")
                logger.error(
                    f"Details - file: {original_filename}, content_type: {content_type}, bucket: {bucket}, key: {file_key}"
                )
                logger.error(f"Auth: user_id={auth.user_id}, app_id={auth.app_id}")
                raise ValueError(f"Document {document_id} not found in database after multiple retries")

            # Prepare updates for the document
            # NOTE: Metadata and metadata_types are already set correctly by the route when creating the document.
            # The worker should NOT merge/update them as that causes type inference issues with serialized values.
            # We only need to update system_metadata and additional_metadata.

            # For XML files, store the combined content of all chunks as the document content
            if xml_processing:
                combined_xml_content = "\n\n".join(chunk.content for chunk in xml_chunks)
                document_content = combined_xml_content
            else:
                document_content = text

            sanitized_system_metadata = IngestionService._clean_system_metadata(doc.system_metadata)

            updates = {
                "additional_metadata": additional_metadata,
                "system_metadata": {**sanitized_system_metadata, "content": document_content},
            }

            # Add folder info and end_user_id to updates if provided
            if normalized_folder_leaf:
                updates["folder_name"] = normalized_folder_leaf
            if normalized_folder_path:
                updates["folder_path"] = normalized_folder_path
            if doc.folder_id:
                updates["folder_id"] = doc.folder_id
            if end_user_id:
                updates["end_user_id"] = end_user_id

            # Update the document in the database
            update_start = time.time()
            success = await ingestion_service.db.update_document(document_id=document_id, updates=updates, auth=auth)
            update_time = time.time() - update_start
            phase_times["update_document_parsed"] = update_time
            logger.debug(f"Initial document update took {update_time:.2f}s")

            if not success:
                raise ValueError(f"Failed to update document {document_id}")

            # Refresh document object with updated data
            doc = await ingestion_service.db.get_document(document_id, auth)
            logger.debug("Updated document in database with parsed content")

            # 7. Split text into chunks
            await update_document_progress(
                ingestion_service, document_id, auth, 3, total_steps, "Splitting into chunks"
            )
            chunking_start = time.time()

            # ===== CHUNKING LOGIC =====
            if xml_processing:
                # XML files already have chunks from parsing
                parsed_chunks = xml_chunks
                logger.debug(f"Using pre-parsed XML chunks: {len(parsed_chunks)} chunks")
            elif skip_text_parsing:
                # ColPali-native formats without text rules - no text chunks needed
                parsed_chunks = []
                logger.debug("No text chunking needed - ColPali will create image-based chunks")
            else:
                # Normal text chunking required
                parsed_chunks = await ingestion_service.parser.split_text(text)
                if not parsed_chunks:
                    logger.warning(
                        "No text chunks extracted after parsing. Will attempt to continue "
                        "and rely on image-based chunks if available."
                    )

            chunking_time = time.time() - chunking_start
            phase_times["split_into_chunks"] = chunking_time
            logger.debug(
                f"{'XML' if xml_processing else 'Text'} chunking took {chunking_time:.2f}s to create {len(parsed_chunks)} chunks"
            )

            # Decide whether to generate image chunks; today this is driven solely by the ColPali flag.
            should_create_image_chunks = using_colpali

            # Start timer for optional image chunk creation / multivector processing
            colpali_processing_start = time.time()

            chunks_multivector = []
            if should_create_image_chunks:
                try:
                    # Use the parsed chunks to create image-friendly slices when ColPali is enabled
                    chunks_multivector = ingestion_service._create_chunks_multivector(
                        mime_type, None, file_content, parsed_chunks
                    )
                except PdfConversionError as conversion_error:
                    logger.error(
                        "PDF conversion failed for document %s (%s): %s",
                        document_id,
                        original_filename,
                        conversion_error,
                    )
                    system_metadata = dict(doc.system_metadata or {})
                    error_code = "pdf_conversion_failed"
                    error_message = str(conversion_error)
                    current_span = get_current_span()
                    current_span.set_status(Status(StatusCode.ERROR, error_message))
                    current_span.set_attribute("ingest.error_code", error_code)
                    current_span.set_attribute("ingest.error_message", error_message)
                    system_metadata.update(
                        {
                            "status": "failed",
                            "error": error_code,
                            "error_message": error_message,
                            "updated_at": datetime.now(UTC),
                            "progress": None,
                        }
                    )
                    cleaned_metadata = IngestionService._clean_system_metadata(system_metadata)
                    await ingestion_service.db.update_document(
                        document_id=document_id,
                        updates={"system_metadata": cleaned_metadata},
                        auth=auth,
                    )
                    return {
                        "document_id": document_id,
                        "status": "failed",
                        "filename": original_filename,
                        "error": error_code,
                        "error_message": error_message,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                logger.debug(
                    f"Created {len(chunks_multivector)} multivector/image chunks " f"(using_colpali={using_colpali})"
                )
            colpali_create_chunks_time = time.time() - colpali_processing_start
            if should_create_image_chunks:
                phase_times["multivector_create_chunks"] = colpali_create_chunks_time
                if using_colpali:
                    logger.debug(f"Multivector chunk creation took {colpali_create_chunks_time:.2f}s")
            else:
                phase_times["multivector_create_chunks"] = 0

            # If we still have no chunks at all (neither text nor image) abort early
            if not parsed_chunks and not chunks_multivector:
                raise ValueError("No content chunks (text or image) could be extracted from the document")

            # Determine the final page count for recording usage
            final_page_count = num_pages_estimated  # Default to estimate
            if using_colpali and chunks_multivector:
                final_page_count = len(chunks_multivector)
            final_page_count = max(1, final_page_count)  # Ensure at least 1 page
            logger.debug(
                f"Determined final page count for usage recording: {final_page_count} pages (ColPali used: {using_colpali})"
            )
            progress_logger.info(
                "ingest chunks doc_id=%s pages=%d text_chunks=%d image_chunks=%d",
                document_id,
                final_page_count,
                len(parsed_chunks),
                len(chunks_multivector),
            )

            colpali_count_for_limit_fn = len(chunks_multivector) if using_colpali else None

            processed_chunks = parsed_chunks
            processed_chunks_multivector = chunks_multivector

            # ===== REGULAR EMBEDDING GENERATION DECISION =====
            # Generate regular embeddings only if we have chunks AND not using ColPali
            chunk_objects = []

            if processed_chunks and not using_colpali:
                # Generate regular embeddings for standard flow
                await update_document_progress(
                    ingestion_service, document_id, auth, 4, total_steps, "Generating embeddings"
                )
                embedding_start = time.time()
                embeddings = await ingestion_service.embedding_model.embed_for_ingestion(processed_chunks)
                logger.debug(f"Generated {len(embeddings)} embeddings")
                embedding_time = time.time() - embedding_start
                phase_times["generate_embeddings"] = embedding_time
                embeddings_per_second = len(embeddings) / embedding_time if embedding_time > 0 else 0
                embedding_chunk_count = len(embeddings)
                logger.debug(
                    f"Embedding generation took {embedding_time:.2f}s for {len(embeddings)} embeddings "
                    f"({embeddings_per_second:.2f} embeddings/s)"
                )
                progress_logger.info(
                    "ingest embed doc_id=%s chunks=%d time_s=%.2f",
                    document_id,
                    len(embeddings),
                    embedding_time,
                )

                # Create chunk objects
                chunk_objects_start = time.time()
                chunk_objects = ingestion_service._create_chunk_objects(doc.external_id, processed_chunks, embeddings)
                logger.debug(f"Created {len(chunk_objects)} chunk objects")
                chunk_objects_time = time.time() - chunk_objects_start
                phase_times["create_chunk_objects"] = chunk_objects_time
                logger.debug(f"Creating chunk objects took {chunk_objects_time:.2f}s")
            else:
                # Skip regular embeddings
                if using_colpali:
                    logger.debug("Skipping regular embeddings - will store only in ColPali vector store")
                elif not processed_chunks:
                    logger.debug("No text chunks to embed")
                phase_times["generate_embeddings"] = 0
                phase_times["create_chunk_objects"] = 0

            # 12. Handle ColPali embeddings
            chunk_objects_multivector = []
            colpali_chunk_ids: List[str] = []
            store_batch_size: Optional[int] = None
            colpali_batches: Optional[int] = None
            if using_colpali:
                # Stream in batches to cap memory: embed -> store -> release
                store_batch_size = settings.COLPALI_STORE_BATCH_SIZE

                total = len(processed_chunks_multivector)
                colpali_batches = math.ceil(total / store_batch_size) if store_batch_size else None
                logger.debug(
                    f"Multivector streaming mode: processing {total} chunks with store batch size {store_batch_size}"
                )
                progress_logger.info(
                    "ingest batching doc_id=%s total_chunks=%d batch_size=%d batches=%d",
                    document_id,
                    total,
                    store_batch_size,
                    colpali_batches or 0,
                )
                colpali_embedding_time = 0.0
                colpali_chunk_object_time = 0.0
                colpali_store_time = 0.0
                colpali_sort_time = 0.0
                colpali_preprocess_time = 0.0
                colpali_model_time = 0.0
                colpali_convert_time = 0.0
                colpali_image_model_time = 0.0
                colpali_text_model_time = 0.0
                colpali_image_process_time = 0.0
                colpali_text_process_time = 0.0
                colpali_image_convert_time = 0.0
                colpali_text_convert_time = 0.0

                for start_idx in range(0, total, store_batch_size):
                    end_idx = min(start_idx + store_batch_size, total)
                    batch_chunks = processed_chunks_multivector[start_idx:end_idx]
                    batch_index = (start_idx // store_batch_size) + 1

                    # Embed this batch
                    batch_embed_start = time.time()
                    batch_embeddings = await ingestion_service.colpali_embedding_model.embed_for_ingestion(batch_chunks)
                    batch_embed_time = time.time() - batch_embed_start
                    colpali_embedding_time += batch_embed_time
                    timing_getter = getattr(ingestion_service.colpali_embedding_model, "latest_ingest_timing", None)
                    metrics = timing_getter() if callable(timing_getter) else {}
                    colpali_sort_time += metrics.get("sorting", 0.0)
                    colpali_preprocess_time += metrics.get("process", 0.0)
                    colpali_model_time += metrics.get("model", 0.0)
                    colpali_convert_time += metrics.get("convert", 0.0)
                    colpali_image_model_time += metrics.get("image_model", 0.0)
                    colpali_text_model_time += metrics.get("text_model", 0.0)
                    colpali_image_process_time += metrics.get("image_process", 0.0)
                    colpali_text_process_time += metrics.get("text_process", 0.0)
                    colpali_image_convert_time += metrics.get("image_convert", 0.0)
                    colpali_text_convert_time += metrics.get("text_convert", 0.0)
                    colpali_image_total_time += metrics.get("image_total", 0.0)
                    colpali_text_total_time += metrics.get("text_total", 0.0)
                    colpali_image_count += int(metrics.get("image_count", 0) or 0)
                    colpali_text_count += int(metrics.get("text_count", 0) or 0)
                    if colpali_endpoints is None:
                        endpoints = metrics.get("endpoints")
                        if endpoints:
                            colpali_endpoints = int(endpoints)
                    logger.debug(
                        f"Multivector batch embedded [{start_idx}:{end_idx}] -> {len(batch_embeddings)} embeddings"
                    )

                    # Create chunk objects for this batch with correct global indices
                    batch_chunk_objects_start = time.time()
                    batch_chunk_objects = ingestion_service._create_chunk_objects(
                        doc.external_id, batch_chunks, batch_embeddings, start_index=start_idx
                    )
                    colpali_chunk_object_time += time.time() - batch_chunk_objects_start

                    # Store this batch immediately to release memory pressure
                    batch_store_start = time.time()
                    success, stored_ids, store_metrics = await ingestion_service.colpali_vector_store.store_embeddings(
                        batch_chunk_objects, auth.app_id if auth else None
                    )
                    batch_store_time = time.time() - batch_store_start
                    colpali_store_time += batch_store_time
                    if not success:
                        raise RuntimeError("Failed to store ColPali batch embeddings")
                    colpali_chunk_ids.extend(stored_ids)
                    if store_metrics:
                        _accumulate_store_metrics(store_metrics_total, store_metrics)
                    progress_logger.info(
                        "ingest batch doc_id=%s %d/%d size=%d embed_s=%.2f store_s=%.2f",
                        document_id,
                        batch_index,
                        colpali_batches or 0,
                        len(batch_chunks),
                        batch_embed_time,
                        batch_store_time,
                    )

                # For compatibility with later summary logging
                chunk_objects_multivector = []
                colpali_pipeline_time = colpali_embedding_time + colpali_chunk_object_time + colpali_store_time
                phase_times["multivector_embedding_creation"] = colpali_embedding_time
                phase_times["multivector_embedding_sorting"] = colpali_sort_time
                phase_times["multivector_embedding_preprocess"] = colpali_preprocess_time
                phase_times["multivector_embedding_model"] = colpali_model_time
                phase_times["multivector_embedding_convert"] = colpali_convert_time
                phase_times["multivector_embedding_image_model"] = colpali_image_model_time
                phase_times["multivector_embedding_text_model"] = colpali_text_model_time
                phase_times["multivector_embedding_image_preprocess"] = colpali_image_process_time
                phase_times["multivector_embedding_text_preprocess"] = colpali_text_process_time
                phase_times["multivector_embedding_image_convert"] = colpali_image_convert_time
                phase_times["multivector_embedding_text_convert"] = colpali_text_convert_time
                phase_times["multivector_chunk_object_creation"] = colpali_chunk_object_time
                phase_times["multivector_store_embeddings"] = colpali_store_time
                phase_times["multivector_pipeline_total"] = colpali_pipeline_time
                eps = (len(colpali_chunk_ids) / colpali_pipeline_time) if colpali_pipeline_time > 0 else 0
                logger.debug(
                    "Multivector embedding: total=%.2fs (sort=%.2fs, preprocess=%.2fs, model=%.2fs, convert=%.2fs | image model=%.2fs, text model=%.2fs) "
                    "storage: chunk objects=%.2fs, vector store=%.2fs for %d chunks (%.2f chunks/s)",
                    colpali_embedding_time,
                    colpali_sort_time,
                    colpali_preprocess_time,
                    colpali_model_time,
                    colpali_convert_time,
                    colpali_image_model_time,
                    colpali_text_model_time,
                    colpali_chunk_object_time,
                    colpali_store_time,
                    len(colpali_chunk_ids),
                    eps,
                )
            else:
                phase_times["multivector_embedding_creation"] = 0
                phase_times["multivector_embedding_sorting"] = 0
                phase_times["multivector_embedding_preprocess"] = 0
                phase_times["multivector_embedding_model"] = 0
                phase_times["multivector_embedding_convert"] = 0
                phase_times["multivector_embedding_image_model"] = 0
                phase_times["multivector_embedding_text_model"] = 0
                phase_times["multivector_embedding_image_preprocess"] = 0
                phase_times["multivector_embedding_text_preprocess"] = 0
                phase_times["multivector_embedding_image_convert"] = 0
                phase_times["multivector_embedding_text_convert"] = 0
                phase_times["multivector_chunk_object_creation"] = 0
                phase_times["multivector_store_embeddings"] = 0
                phase_times["multivector_pipeline_total"] = 0

            # 11. Store chunks and update document with is_update=True
            await update_document_progress(ingestion_service, document_id, auth, 5, total_steps, "Storing chunks")
            store_start = time.time()
            if using_colpali:
                # We already stored ColPali chunks in batches; just persist doc.chunk_ids via DB update
                # Only update chunk_ids and system_metadata - everything else was set correctly by the route
                doc.chunk_ids = colpali_chunk_ids
                doc.system_metadata = IngestionService._clean_system_metadata(doc.system_metadata)
                await ingestion_service.db.update_document(
                    document_id=doc.external_id,
                    updates={
                        "chunk_ids": doc.chunk_ids,
                        "system_metadata": doc.system_metadata,
                    },
                    auth=auth,
                )
                if auth:
                    chunk_bytes, multivector_bytes = extract_storage_bytes(store_metrics_total)
                    if chunk_bytes or multivector_bytes:
                        try:
                            await ingestion_service.db.record_document_storage_deltas(
                                document_id,
                                auth.app_id,
                                chunk_bytes_delta=chunk_bytes,
                                multivector_bytes_delta=multivector_bytes,
                            )
                        except Exception as storage_err:  # noqa: BLE001
                            logger.error(
                                "Failed recording ColPali storage bytes for doc %s: %s", document_id, storage_err
                            )
            else:
                metadata_bundle = None
                if isinstance(doc.metadata, dict) and isinstance(doc.metadata_types, dict):
                    metadata_bundle = MetadataBundle(
                        values=dict(doc.metadata),
                        types=dict(doc.metadata_types),
                        is_normalized=True,
                    )
                _, store_metrics = await ingestion_service._store_chunks_and_doc(
                    chunk_objects,
                    doc,
                    use_colpali,
                    chunk_objects_multivector,
                    is_update=True,
                    auth=auth,
                    metadata_bundle=metadata_bundle,
                )
                if store_metrics:
                    _accumulate_store_metrics(store_metrics_total, store_metrics)
            store_time = time.time() - store_start
            phase_times["store_chunks_and_update_doc"] = store_time

            # ===== STORAGE SUMMARY =====
            # Log what was actually stored for clarity
            storage_summary = []
            if using_colpali:
                storage_summary.append(f"ColPali vector store: {len(doc.chunk_ids)} chunks")
            if not using_colpali and chunk_objects:
                storage_summary.append(f"Regular vector store: {len(chunk_objects)} chunks")

            logger.debug(
                f"Storage complete in {store_time:.2f}s - "
                + ("; ".join(storage_summary) if storage_summary else "No chunks stored")
            )

            logger.debug(f"Successfully completed processing for document {doc.external_id}")

            # 12. Add document to folder if requested
            if normalized_folder_path:
                try:
                    logger.info(f"Adding document {doc.external_id} to folder '{normalized_folder_path}'")
                    folder_obj = await ingestion_service._ensure_folder_exists(
                        normalized_folder_path, doc.external_id, auth
                    )
                    if folder_obj and folder_obj.id:
                        doc.folder_id = folder_obj.id
                        folder_id_for_summary = str(folder_obj.id)
                        folder_updates = ingestion_service.folder_update_fields(folder_obj)
                        await ingestion_service.db.update_document(
                            document_id=doc.external_id,
                            updates=folder_updates,
                            auth=auth,
                        )
                except Exception as folder_exc:
                    logger.error(f"Failed to add document to folder: {folder_exc}")
                    # Don't fail the entire ingestion if folder processing fails

            await update_document_progress(ingestion_service, document_id, auth, 6, total_steps, "Finalizing")
            # Update document status to completed after all processing
            doc.system_metadata["page_count"] = final_page_count
            doc.system_metadata["status"] = "completed"
            doc.system_metadata["updated_at"] = datetime.now(UTC)
            # Clear progress info on completion
            doc.system_metadata.pop("progress", None)

            # Final update to mark as completed
            doc.system_metadata = IngestionService._clean_system_metadata(doc.system_metadata)
            await ingestion_service.db.update_document(
                document_id=document_id, updates={"system_metadata": doc.system_metadata}, auth=auth
            )

            # 13. Log successful completion
            logger.info(f"Successfully completed ingestion for {original_filename}, document ID: {doc.external_id}")
            # Performance summary
            total_time = time.time() - job_start_time
            progress_logger.info("ingest done doc_id=%s status=completed total_s=%.2f", document_id, total_time)

            if not folder_id_for_summary and doc.folder_id:
                folder_id_for_summary = str(doc.folder_id)
            folder_path_summary = (
                normalized_folder_path or getattr(doc, "folder_path", None) or getattr(doc, "folder_name", None)
            )

            text_chunk_count = len(parsed_chunks)
            image_chunk_count = len(chunks_multivector)
            if using_colpali:
                total_chunks = len(colpali_chunk_ids) if colpali_chunk_ids else image_chunk_count
            else:
                total_chunks = text_chunk_count

            content_type_summary = mime_type or content_type
            input_summary = _strip_none(
                {
                    "size_bytes": download_size_bytes,
                    "size_mb": round(download_size_bytes / 1024 / 1024, 2) if download_size_bytes else 0.0,
                    "content_type": content_type_summary,
                    "skip_text_parsing": skip_text_parsing,
                    "colpali": bool(using_colpali),
                    "colpali_mode": settings.COLPALI_MODE if using_colpali else None,
                }
            )

            counts_summary = _strip_none(
                {
                    "pages_estimated": num_pages_estimated,
                    "pages_final": final_page_count,
                    "text_chunks": text_chunk_count,
                    "image_chunks": image_chunk_count,
                    "total_chunks": total_chunks,
                    "batch_size": store_batch_size if using_colpali else None,
                    "batches": colpali_batches if using_colpali else None,
                }
            )

            contextual_chunking_model = None
            if settings.USE_CONTEXTUAL_CHUNKING:
                contextual_chunking_model = getattr(
                    getattr(ingestion_service.parser, "chunker", None), "model_key", None
                )
            config_summary = _strip_none(
                {
                    "mode": settings.MODE,
                    "storage_provider": settings.STORAGE_PROVIDER,
                    "storage_bucket": settings.S3_BUCKET if settings.STORAGE_PROVIDER == "aws-s3" else None,
                    "storage_path": settings.STORAGE_PATH if settings.STORAGE_PROVIDER == "local" else None,
                    "cache_enabled": settings.CACHE_ENABLED,
                    "cache_max_bytes": settings.CACHE_MAX_BYTES,
                    "cache_path": settings.CACHE_PATH if settings.CACHE_ENABLED else None,
                    "s3_upload_concurrency": settings.S3_UPLOAD_CONCURRENCY,
                    "multivector_store_provider": settings.MULTIVECTOR_STORE_PROVIDER,
                    "dual_multivector_ingestion": settings.ENABLE_DUAL_MULTIVECTOR_INGESTION,
                    "vector_store_provider": settings.VECTOR_STORE_PROVIDER,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "embedding_dimensions": settings.VECTOR_DIMENSIONS,
                    "embedding_similarity_metric": settings.EMBEDDING_SIMILARITY_METRIC,
                    "parser_chunk_size": settings.CHUNK_SIZE,
                    "parser_chunk_overlap": settings.CHUNK_OVERLAP,
                    "contextual_chunking": settings.USE_CONTEXTUAL_CHUNKING,
                    "contextual_chunking_model": contextual_chunking_model,
                    "enable_colpali": settings.ENABLE_COLPALI,
                    "colpali_mode": settings.COLPALI_MODE,
                    "colpali_api_endpoints": (
                        len(settings.MORPHIK_EMBEDDING_API_DOMAIN) if settings.COLPALI_MODE == "api" else None
                    ),
                    "colpali_pdf_dpi": settings.COLPALI_PDF_DPI if using_colpali else None,
                    "colpali_store_batch_size": settings.COLPALI_STORE_BATCH_SIZE,
                    "arq_max_jobs": settings.ARQ_MAX_JOBS,
                }
            )

            if xml_processing:
                config_summary["parser_xml"] = {
                    "max_tokens": settings.PARSER_XML.max_tokens,
                    "preferred_unit_tags": settings.PARSER_XML.preferred_unit_tags,
                    "ignore_tags": settings.PARSER_XML.ignore_tags,
                }

            flags_summary = _strip_none(
                {
                    "xml_processing": xml_processing,
                    "html_to_pdf": html_converted,
                }
            )

            embedding_summary: Dict[str, Any] = {}
            if using_colpali:
                is_api = isinstance(ingestion_service.colpali_embedding_model, ColpaliApiEmbeddingModel)
                if colpali_endpoints is None and is_api:
                    colpali_endpoints = len(getattr(ingestion_service.colpali_embedding_model, "endpoints", []))
                embedding_summary = _strip_none(
                    {
                        "backend": "colpali_api" if is_api else "colpali_local",
                        "endpoints": colpali_endpoints,
                        "total_s": round(colpali_embedding_time, 2),
                        "image_s": round(colpali_image_total_time, 2),
                        "text_s": round(colpali_text_total_time, 2),
                        "image_count": colpali_image_count,
                        "text_count": colpali_text_count,
                        "throughput_chunks_s": (
                            round((total_chunks / colpali_embedding_time), 2) if colpali_embedding_time > 0 else 0.0
                        ),
                    }
                )
            elif embedding_chunk_count:
                embedding_summary = _strip_none(
                    {
                        "backend": "lite_llm",
                        "model_key": settings.EMBEDDING_MODEL,
                        "total_s": round(embedding_time, 2),
                        "throughput_chunks_s": round(embeddings_per_second, 2),
                    }
                )

            storage_summary: Dict[str, Any] = {}
            if store_metrics_total:
                if "fast" in store_metrics_total or "slow" in store_metrics_total:
                    storage_summary["mode"] = store_metrics_total.get("mode", "dual")
                    if "fast" in store_metrics_total:
                        storage_summary["fast"] = _with_throughput(store_metrics_total["fast"], total_chunks)
                    if "slow" in store_metrics_total:
                        storage_summary["slow"] = _with_throughput(store_metrics_total["slow"], total_chunks)
                else:
                    storage_summary = _with_throughput(store_metrics_total, total_chunks)

            phases_summary = {
                phase: round(duration, 2)
                for phase, duration in sorted(phase_times.items(), key=lambda x: x[1], reverse=True)
                if duration > 0
            }

            summary = {
                "event": "ingestion_performance_summary",
                "document_id": document_id,
                "filename": original_filename,
                "folder_path": folder_path_summary,
                "folder_id": folder_id_for_summary,
                "input": input_summary,
                "counts": counts_summary,
                "flags": flags_summary,
                "config": config_summary,
                "embedding": embedding_summary,
                "storage": storage_summary,
                "phases_s": phases_summary,
                "total_s": round(total_time, 2),
            }
            summary_logger.info(json.dumps(summary, separators=(",", ":"), default=str))

            # Record ingest usage *after* successful completion using the final page count
            if settings.MODE == "cloud" and auth.user_id:
                try:
                    await check_and_increment_limits(
                        auth,
                        "ingest",
                        final_page_count,
                        document_id,
                        use_colpali=using_colpali,
                        colpali_chunks_count=colpali_count_for_limit_fn,
                    )
                except Exception as rec_exc:
                    logger.error("Failed to record ingest usage after completion: %s", rec_exc)

            # 14. Return document ID
            return {
                "document_id": document_id,
                "status": "completed",
                "filename": original_filename,
                "content_type": content_type,
                "timestamp": datetime.now(UTC).isoformat(),
            }
    except Exception as e:
        logger.error(f"Error processing ingestion job for file {original_filename}: {str(e)}")
        logger.error(traceback.format_exc())
        progress_logger.error("ingest failed doc_id=%s file=%s error=%s", document_id, original_filename, e)

        # ------------------------------------------------------------------
        # Ensure we update the *per-app* database where the document lives.
        # Falling back to the control-plane DB (ctx["database"]) can silently
        # fail because the row doesn't exist there.
        # ------------------------------------------------------------------

        # Reconstruct auth from auth_dict in case exception occurred before auth was defined
        try:
            auth
        except NameError:
            auth = AuthContext(
                user_id=auth_dict.get("user_id") or auth_dict.get("entity_id", ""),
                app_id=auth_dict.get("app_id"),
            )

        try:
            database: Optional[PostgresDatabase] = None

            # Prefer the tenant-specific database
            if auth.app_id is not None:
                try:
                    database = await get_database_for_app(auth.app_id)
                    await database.initialize()
                except Exception as db_err:
                    logger.warning(
                        "Failed to obtain per-app database in error handler: %s. Falling back to default.",
                        db_err,
                    )

            # Fallback to the default database kept in the worker context
            if database is None:
                database = ctx.get("database")

            # Proceed only if we have a database object
            if database:
                # Try to get the document
                doc = await database.get_document(document_id, auth)

                if doc:
                    # Update the document status to failed
                    await database.update_document(
                        document_id=document_id,
                        updates={
                            "system_metadata": {
                                **doc.system_metadata,
                                "status": "failed",
                                "error": str(e),
                                "updated_at": datetime.now(UTC),
                                # Clear progress info on failure
                                "progress": None,
                            }
                        },
                        auth=auth,
                    )
                    logger.info(f"Updated document {document_id} status to failed")
        except Exception as inner_e:
            logger.error(f"Failed to update document status: {inner_e}")

        # Note: TelemetryService will persist an error log entry automatically

        # 14. Return error information
        return {
            "status": "failed",
            "filename": original_filename,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


async def startup(ctx):
    """
    Worker startup: Initialize all necessary services that will be reused across jobs.

    This initialization is similar to what happens in core/api.py during app startup,
    but adapted for the worker context.
    """
    _ensure_worker_logging()
    logger.info("Worker starting up. Initializing services...")

    # Initialize database
    logger.info("Initializing database...")
    database = PostgresDatabase(uri=settings.POSTGRES_URI)
    # database = PostgresDatabase(uri="postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
    success = await database.initialize()
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")
    ctx["database"] = database

    # Initialize vector store
    logger.info("Initializing primary vector store...")
    vector_store = PGVectorStore(uri=settings.POSTGRES_URI)
    # vector_store = PGVectorStore(uri="postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
    success = await vector_store.initialize()
    if success:
        logger.info("Primary vector store initialization successful")
    else:
        logger.error("Primary vector store initialization failed")
    ctx["vector_store"] = vector_store

    # Initialize storage
    if settings.STORAGE_PROVIDER == "local":
        storage = LocalStorage(storage_path=settings.STORAGE_PATH)
    elif settings.STORAGE_PROVIDER == "aws-s3":
        storage = S3Storage(
            aws_access_key=settings.AWS_ACCESS_KEY,
            aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            default_bucket=settings.S3_BUCKET,
            upload_concurrency=settings.S3_UPLOAD_CONCURRENCY,
        )
    else:
        raise ValueError(f"Unsupported storage provider: {settings.STORAGE_PROVIDER}")
    ctx["storage"] = storage

    # Initialize parser
    parser = MorphikParser(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        use_contextual_chunking=settings.USE_CONTEXTUAL_CHUNKING,
    )
    ctx["parser"] = parser

    # Initialize embedding model
    embedding_model = LiteLLMEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
    logger.info(f"Initialized LiteLLM embedding model with model key: {settings.EMBEDDING_MODEL}")
    ctx["embedding_model"] = embedding_model

    # Skip initializing completion model and reranker since they're not needed for ingestion

    # Initialize ColPali embedding model and vector store per mode
    colpali_embedding_model = None
    colpali_vector_store = None

    # Check enable_colpali first - if disabled, skip all ColPali initialization
    if not settings.ENABLE_COLPALI:
        logger.info("ColPali disabled by configuration (enable_colpali=false)")
    elif settings.COLPALI_MODE != "off":
        logger.info(f"Initializing ColPali components (mode={settings.COLPALI_MODE}) ...")
        # Choose embedding implementation
        match settings.COLPALI_MODE:
            case "local":
                colpali_embedding_model = ColpaliEmbeddingModel()
            case "api":
                colpali_embedding_model = ColpaliApiEmbeddingModel()
            case _:
                raise ValueError(f"Unsupported COLPALI_MODE: {settings.COLPALI_MODE}")

        # Vector store is needed for both local and api modes
        # Choose multivector store implementation based on provider and dual ingestion setting
        if settings.ENABLE_DUAL_MULTIVECTOR_INGESTION:
            # Dual ingestion mode: create both stores and wrap them
            if not settings.TURBOPUFFER_API_KEY:
                raise ValueError("TURBOPUFFER_API_KEY is required when dual ingestion is enabled")

            fast_store = FastMultiVectorStore(
                uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
            )
            slow_store = MultiVectorStore(uri=settings.POSTGRES_URI)
            colpali_vector_store = DualMultiVectorStore(
                fast_store=fast_store, slow_store=slow_store, enable_dual_ingestion=True
            )
        elif settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
            if not settings.TURBOPUFFER_API_KEY:
                raise ValueError("TURBOPUFFER_API_KEY is required when using morphik multivector store provider")
            colpali_vector_store = FastMultiVectorStore(
                uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
            )
        elif settings.MULTIVECTOR_STORE_PROVIDER == "local":
            colpali_vector_store = LocalMultiVectorStore(
                uri=settings.POSTGRES_URI,
                chroma_host=settings.CHROMA_HOST,
                chroma_port=settings.CHROMA_PORT,
                chroma_ssl=settings.CHROMA_SSL,
                namespace="public",
            )
            logger.info(
                "Initialized LocalMultiVectorStore (ChromaDB) at %s:%d",
                settings.CHROMA_HOST,
                settings.CHROMA_PORT,
            )
        else:
            colpali_vector_store = MultiVectorStore(uri=settings.POSTGRES_URI)
        success = await asyncio.to_thread(colpali_vector_store.initialize)
        if success:
            logger.info("ColPali vector store initialization successful")
        else:
            logger.error("ColPali vector store initialization failed")
    ctx["colpali_embedding_model"] = colpali_embedding_model
    ctx["colpali_vector_store"] = colpali_vector_store

    # Initialize telemetry service
    telemetry = TelemetryService()
    ctx["telemetry"] = telemetry

    logger.info("Worker startup complete. Core ingestion components initialized.")


async def shutdown(ctx):
    """
    Worker shutdown: Clean up resources.

    Properly close connections and cleanup resources to prevent leaks.
    """
    logger.info("Worker shutting down. Cleaning up resources...")

    # Close database connections
    if "database" in ctx and hasattr(ctx["database"], "engine"):
        logger.info("Closing database connections...")
        await ctx["database"].engine.dispose()

    async def _shutdown_store(store_key: str) -> None:
        store = ctx.get(store_key)
        if not store:
            return

        close_candidate = getattr(store, "close", None)
        if callable(close_candidate):
            logger.info("Closing %s via close()...", store_key)
            try:
                if inspect.iscoroutinefunction(close_candidate):
                    await close_candidate()
                else:
                    maybe_coro = close_candidate()
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close %s cleanly: %s", store_key, exc)
            return

        engine = getattr(store, "engine", None)
        if engine is not None and hasattr(engine, "dispose"):
            logger.info("Disposing engine for %s...", store_key)
            await engine.dispose()

    await _shutdown_store("vector_store")
    await _shutdown_store("colpali_vector_store")

    # Close any other open connections or resources that need cleanup
    logger.info("Worker shutdown complete.")


def redis_settings_from_env() -> RedisSettings:
    """
    Create RedisSettings from environment variables for ARQ worker.

    Returns:
        RedisSettings configured for Redis connection with optimized performance
    """
    url = up.urlparse(settings.REDIS_URL)

    # Use ARQ's supported parameters with optimized values for stability
    # For high-volume ingestion (100+ documents), these settings help prevent timeouts
    return RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=int(url.path.lstrip("/") or 0),
        conn_timeout=5,  # Increased connection timeout (seconds)
        conn_retries=15,  # More retries for transient connection issues
        conn_retry_delay=1,  # Quick retry delay (seconds)
    )


# ARQ Worker Settings
class WorkerSettings:
    """
    ARQ Worker settings for the ingestion worker.

    This defines the functions available to the worker, startup and shutdown handlers,
    and any specific Redis settings.
    """

    functions = [process_ingestion_job]
    on_startup = startup
    on_shutdown = shutdown

    # Use robust Redis settings that handle connection issues
    redis_settings = redis_settings_from_env()

    # Result storage settings
    keep_result_ms = 15 * 60 * 1000  # Keep results for 15 minutes

    # Concurrency settings - keep low by default to avoid OOM on small EC2s.
    # Override with [worker].arq_max_jobs in morphik.toml if you have sufficient memory.
    max_jobs = settings.ARQ_MAX_JOBS

    # Resource management
    health_check_interval = 600  # Extended to 10 minutes to reduce Redis overhead
    job_timeout = 7200  # Extended to 2 hours for large document processing
    max_tries = 5  # Retry failed jobs up to 5 times
    poll_delay = 2.0  # Increased poll delay to prevent Redis connection saturation

    # High reliability settings
    allow_abort_jobs = False  # Don't abort jobs on worker shutdown
    retry_jobs = True  # Always retry failed jobs

    # Prevent queue blocking on error
    skip_queue_when_queues_read_fails = True  # Continue processing other queues if one fails

    # Log Redis and connection pool information for debugging
    @staticmethod
    async def health_check(ctx):
        """
        Enhanced periodic health check to log connection status and job stats.
        Monitors Redis memory, database connections, and job processing metrics.
        """
        database = ctx.get("database")
        vector_store = ctx.get("vector_store")
        job_stats = ctx.get("job_stats", {})

        # Get detailed Redis info
        try:
            redis_info = await ctx["redis"].info(section=["Server", "Memory", "Clients", "Stats"])

            # Server and resource usage info
            redis_version = redis_info.get("redis_version", "unknown")
            used_memory = redis_info.get("used_memory_human", "unknown")
            used_memory_peak = redis_info.get("used_memory_peak_human", "unknown")
            clients_connected = redis_info.get("connected_clients", "unknown")
            rejected_connections = redis_info.get("rejected_connections", 0)
            total_commands = redis_info.get("total_commands_processed", 0)

            # DB keys
            db_info = redis_info.get("db0", {})
            keys_count = db_info.get("keys", 0) if isinstance(db_info, dict) else 0

            # Log comprehensive server status
            logger.info(
                f"Redis Status: v{redis_version} | "
                f"Memory: {used_memory} (peak: {used_memory_peak}) | "
                f"Clients: {clients_connected} (rejected: {rejected_connections}) | "
                f"DB Keys: {keys_count} | Commands: {total_commands}"
            )

            # Check for memory warning thresholds
            if isinstance(used_memory, str) and used_memory.endswith("G"):
                memory_value = float(used_memory[:-1])
                if memory_value > 1.0:  # More than 1GB used
                    logger.warning(f"Redis memory usage is high: {used_memory}")

            # Check for connection issues
            if rejected_connections and int(rejected_connections) > 0:
                logger.warning(f"Redis has rejected {rejected_connections} connections")
        except Exception as e:
            logger.error(f"Failed to get Redis info: {str(e)}")

        # Log job statistics with detailed processing metrics
        ongoing = job_stats.get("ongoing", 0)
        queued = job_stats.get("queued", 0)

        logger.info(
            f"Job Stats: completed={job_stats.get('complete', 0)} | "
            f"failed={job_stats.get('failed', 0)} | "
            f"retried={job_stats.get('retried', 0)} | "
            f"ongoing={ongoing} | queued={queued}"
        )

        # Warn if too many jobs are queued/backed up
        if queued > 50:
            logger.warning(f"Large job queue backlog: {queued} jobs waiting")

        # Test database connectivity with extended timeout
        if database and hasattr(database, "async_session"):
            try:
                async with database.async_session() as session:
                    await session.execute(text("SELECT 1"))
                    logger.debug("Database connection is healthy")
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")

        # Test vector store connectivity if available
        if vector_store and hasattr(vector_store, "async_session"):
            try:
                async with vector_store.get_session_with_retry() as session:
                    logger.debug("Vector store connection is healthy")
            except Exception as e:
                logger.error(f"Vector store connection test failed: {str(e)}")
