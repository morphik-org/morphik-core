import json
import logging
import secrets
import time  # Add time import for profiling
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import arq
import jwt
import sentry_sdk
import tomli
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.sessions import SessionMiddleware

from core.app_factory import lifespan
from core.auth_utils import verify_token
from core.config import get_settings
from core.database.postgres_database import InvalidMetadataFilterError
from core.dependencies import get_redis_pool
from core.limits_utils import check_and_increment_limits
from core.logging_config import setup_logging
from core.middleware.profiling import ProfilingMiddleware
from core.models.auth import AuthContext
from core.models.chat import ChatMessage
from core.models.completion import CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentResult, GroupedChunkResponse
from core.models.prompts import validate_prompt_overrides_with_http_exception
from core.models.request import (
    BatchChunksRequest,
    BatchDocumentsRequest,
    CompletionQueryRequest,
    GenerateUriRequest,
    RetrieveRequest,
    SearchDocumentsRequest,
)
from core.models.responses import ChatTitleResponse, ModelsResponse
from core.routes.documents import router as documents_router
from core.routes.folders import router as folders_router
from core.routes.graph import router as graph_router
from core.routes.health import router as health_router
from core.routes.ingest import router as ingest_router
from core.routes.logs import router as logs_router  # noqa: E402 – import after FastAPI app
from core.routes.models import router as models_router
from core.routes.usage import router as usage_router
from core.services.telemetry import TelemetryService
from core.services_init import document_service, ingestion_service
from core.utils.folder_utils import normalize_folder_selector

# Set up logging configuration for Docker environment
setup_logging()


def decode_query_image(query_image: Optional[str]) -> Optional[bytes]:
    """Decode a base64-encoded query image to bytes.

    Handles data URI format (e.g., "data:image/png;base64,...") by stripping the prefix.
    Raises HTTPException with 400 status if the base64 encoding is invalid.
    """
    if not query_image:
        return None

    import base64
    import binascii

    # Handle data URI format if present
    image_data = query_image
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    try:
        return base64.b64decode(image_data)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64-encoded image: {e}")


# Initialize FastAPI app
logger = logging.getLogger(__name__)


# Performance tracking class
class PerformanceTracker:
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.phases = {}
        self.current_phase = None
        self.sub_operations = {}  # Track sub-operations for hierarchical display
        self.phase_start = None

    def start_phase(self, phase_name: str):
        # End current phase if one is running
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start

        # Start new phase
        self.current_phase = phase_name
        self.phase_start = time.time()

    def end_phase(self):
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start
            self.current_phase = None
            self.phase_start = None

    def add_suboperation(self, name: str, duration: float, parent_phase: Optional[str] = None):
        """Add a sub-operation timing that will be displayed under its parent phase"""
        if parent_phase:
            if parent_phase not in self.sub_operations:
                self.sub_operations[parent_phase] = {}
            self.sub_operations[parent_phase][name] = duration
        else:
            # If no parent specified, add as a regular phase
            self.phases[name] = duration

    def log_summary(self, additional_info: str = ""):
        total_time = time.time() - self.start_time

        # End current phase if still running
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start

        logger.info(f"=== {self.operation_name} Performance Summary ===")
        logger.info(f"Total time: {total_time:.2f}s")

        # Sort phases by duration (longest first) and include sub-operations under each phase
        for phase, duration in sorted(self.phases.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")

            # Display sub-operations for this phase if any exist
            if phase in self.sub_operations:
                for sub_name, sub_duration in sorted(
                    self.sub_operations[phase].items(), key=lambda x: x[1], reverse=True
                ):
                    sub_percentage = (sub_duration / total_time) * 100 if total_time > 0 else 0
                    logger.info(f"    - {sub_name}: {sub_duration:.2f}s ({sub_percentage:.1f}%)")

        if additional_info:
            logger.info(additional_info)
        logger.info("=" * (len(self.operation_name) + 31))


# Global settings object
settings = get_settings()

# ---------------------------------------------------------------------------
# Initialize Sentry
# ---------------------------------------------------------------------------

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=True,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for tracing.
        traces_sample_rate=1.0,
        # Set profile_session_sample_rate to 1.0 to profile 100%
        # of profile sessions.
        profile_session_sample_rate=1.0,
        # Set profile_lifecycle to "trace" to automatically
        # run the profiler on when there is an active transaction
        profile_lifecycle="trace",
    )
else:
    logger.warning("SENTRY_DSN is not set, skipping Sentry initialization")

# ---------------------------------------------------------------------------
# Application instance & core initialisation (moved lifespan, rest unchanged)
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

# --------------------------------------------------------
# Optional per-request profiler (ENABLE_PROFILING=1)
# --------------------------------------------------------

app.add_middleware(ProfilingMiddleware)

# Add CORS middleware (same behaviour as before refactor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise telemetry service
telemetry = TelemetryService()

# OpenTelemetry instrumentation – exclude noisy spans/headers
FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls="health,health/.*",
    exclude_spans=["send", "receive"],
    http_capture_headers_server_request=None,
    http_capture_headers_server_response=None,
    tracer_provider=None,
)

# ---------------------------------------------------------------------------
# Session cookie behaviour differs between cloud / self-hosted
# ---------------------------------------------------------------------------

if settings.MODE == "cloud":
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.SESSION_SECRET_KEY,
        same_site="none",
        https_only=True,
    )
else:
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY)


def _validate_admin_secret(admin_secret: Optional[str]) -> bool:
    """Return True if the provided admin secret is valid, otherwise raise."""
    if not admin_secret:
        return False
    if not settings.ADMIN_SERVICE_SECRET:
        raise HTTPException(status_code=403, detail="Admin secret authentication is not configured")
    if not secrets.compare_digest(admin_secret, settings.ADMIN_SERVICE_SECRET):
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    return True


@app.get("/models", response_model=ModelsResponse)
async def get_available_models(auth: AuthContext = Depends(verify_token)):
    """
    Get list of available models from configuration.

    Returns models grouped by type (chat, embedding, etc.) with their metadata.
    """
    try:
        # Load the morphik.toml file to get registered models
        with open("morphik.toml", "rb") as f:
            config = tomli.load(f)

        registered_models = config.get("registered_models", {})

        # Group models by their purpose
        chat_models = []
        embedding_models = []

        for model_key, model_config in registered_models.items():
            model_info = {
                "id": model_key,
                "model": model_config.get("model_name", model_key),
                "provider": _extract_provider(model_config.get("model_name", "")),
                "config": model_config,
            }

            # Categorize models based on their names or configuration
            if "embedding" in model_key.lower():
                embedding_models.append(model_info)
            else:
                chat_models.append(model_info)

        # Also add the default configured models
        default_models = {
            "completion": config.get("completion", {}).get("model"),
            "embedding": config.get("embedding", {}).get("model"),
        }

        return {
            "chat_models": chat_models,
            "embedding_models": embedding_models,
            "default_models": default_models,
            "providers": ["openai", "anthropic", "google", "azure", "ollama", "custom"],
        }
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Failed to load available models")


def _extract_provider(model_name: str) -> str:
    """Extract provider from model name."""
    if model_name.startswith("gpt"):
        return "openai"
    elif model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith("gemini"):
        return "google"
    elif model_name.startswith("ollama"):
        return "ollama"
    elif "azure" in model_name:
        return "azure"
    else:
        return "custom"


# ---------------------------------------------------------------------------
# Core singletons (database, vector store, storage, parser, models …)
# ---------------------------------------------------------------------------


# Store on app.state for later access
app.state.document_service = document_service
app.state.ingestion_service = ingestion_service
logger.info("Document and ingestion services initialized and stored on app.state")

# Register health router
app.include_router(health_router)

# Register ingest router
app.include_router(ingest_router)

# Register documents router
app.include_router(documents_router)

# Register folders router
app.include_router(folders_router)

# Register models router
app.include_router(models_router)

# Register logs router
app.include_router(logs_router)

# Register usage router
app.include_router(usage_router)

# Register graph router
app.include_router(graph_router)


# Enterprise-only routes (optional)
try:
    from ee.routers import init_app as _init_ee_app  # type: ignore  # noqa: E402

    _init_ee_app(app)  # noqa: SLF001 – runtime extension
except ModuleNotFoundError as exc:
    logger.debug("Enterprise package not found – running in community mode.")
    logger.error("ModuleNotFoundError: %s", exc, exc_info=True)
except ImportError as exc:
    logger.error("Failed to import init_app from ee.routers: %s", exc, exc_info=True)
except Exception as exc:  # noqa: BLE001
    logger.error("An unexpected error occurred during EE app initialization: %s", exc, exc_info=True)


@app.post("/retrieve/chunks", response_model=List[ChunkResult])
@telemetry.track(operation_type="retrieve_chunks", metadata_resolver=telemetry.retrieve_chunks_metadata)
async def retrieve_chunks(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant chunks.

    The optional `request.filters` payload accepts equality checks (which also match scalars inside JSON arrays)
    plus the logical operators `$and`, `$or`, `$nor`, and `$not`. Field-level predicates include `$eq`, `$ne`,
    `$in`, `$nin`, `$exists`, `$type`, `$regex`, `$contains`, and the comparison operators `$gt`, `$gte`, `$lt`,
    and `$lte`. Comparison clauses evaluate typed metadata (`number`, `decimal`, `datetime`, or `date`) and
    raise detailed validation errors when operands cannot be coerced. Regex filters allow the optional `i` flag
    for case-insensitive matching, while `$contains` performs substring checks (case-insensitive by default,
    configurable via `case_sensitive`). Filters can be nested freely, for example:

    ```json
    {
      "$and": [
        {"category": "policy"},
        {"$or": [{"region": "emea"}, {"priority": {"$in": ["p0", "p1"]}}]}
      ]
    }
    ```
    Returns a list of `ChunkResult` objects ordered by relevance.
    """
    # Initialize performance tracker
    query_preview = (request.query[:50] + "...") if request.query else "[image query]"
    perf = PerformanceTracker(f"Retrieve Chunks: '{query_preview}'")

    # Decode query_image if provided (base64 -> bytes)
    query_image_bytes = decode_query_image(request.query_image)

    try:
        # Main retrieval operation
        perf.start_phase("document_service_retrieve_chunks")
        results = await document_service.retrieve_chunks(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.use_reranking,
            request.use_colpali,
            request.folder_name,
            request.folder_depth,
            request.end_user_id,
            perf,  # Pass performance tracker
            request.padding,  # Pass padding parameter
            request.output_format or "base64",
            query_image=query_image_bytes,
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)} chunks")

        return results
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/retrieve/chunks/grouped", response_model=GroupedChunkResponse)
@telemetry.track(operation_type="retrieve_chunks_grouped", metadata_resolver=telemetry.retrieve_chunks_metadata)
async def retrieve_chunks_grouped(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant chunks with grouped response format.

    Uses the same filter operators as `/retrieve/chunks` (equality, `$eq/$ne`, `$gt/$gte/$lt/$lte`, `$in/$nin`,
    `$exists`, `$type`, `$regex`, `$contains`, and the logical `$and/$or/$nor/$not`), with arbitrary nesting
    supported inside `request.filters`.

    Returns both flat results (for backward compatibility) and grouped results (for UI).
    When padding > 0, groups chunks by main matches and their padding chunks.
    """
    # Initialize performance tracker
    query_preview = (request.query[:50] + "...") if request.query else "[image query]"
    perf = PerformanceTracker(f"Retrieve Chunks Grouped: '{query_preview}'")

    # Decode query_image if provided (base64 -> bytes)
    query_image_bytes = decode_query_image(request.query_image)

    try:
        # Main retrieval operation
        perf.start_phase("document_service_retrieve_chunks_grouped")
        result = await document_service.retrieve_chunks_grouped(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.use_reranking,
            request.use_colpali,
            request.folder_name,
            request.folder_depth,
            request.end_user_id,
            perf,  # Pass performance tracker
            request.padding,  # Pass padding parameter
            request.output_format or "base64",
            query_image=query_image_bytes,
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(result.chunks)} total chunks in {len(result.groups)} groups")

        return result
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/retrieve/docs", response_model=List[DocumentResult])
@telemetry.track(operation_type="retrieve_docs", metadata_resolver=telemetry.retrieve_docs_metadata)
async def retrieve_documents(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant documents.

    `request.filters` supports equality checks (including scalar-to-array matches) and the same operator set as
    `/retrieve/chunks`: logical composition via `$and`, `$or`, `$nor`, `$not`, plus field predicates `$eq`, `$ne`,
    `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$type`, `$regex`, and `$contains`. Use the same JSON
    structure as `/retrieve/chunks` when expressing complex logic. Comparison operators require metadata typed as
    `number`, `decimal`, `datetime`, or `date`.
    """
    # Image queries not supported for document retrieval
    if request.query_image:
        raise HTTPException(
            status_code=400,
            detail="Image queries are not supported for document retrieval. Use /retrieve/chunks instead.",
        )

    # Text query is required for document retrieval
    if not request.query:
        raise HTTPException(
            status_code=400,
            detail="A text query is required for document retrieval.",
        )

    # Initialize performance tracker
    perf = PerformanceTracker(f"Retrieve Docs: '{request.query[:50]}...'")

    try:
        # Main retrieval operation
        perf.start_phase("document_service_retrieve_docs")
        results = await document_service.retrieve_docs(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.use_reranking,
            request.use_colpali,
            request.folder_name,
            request.folder_depth,
            request.end_user_id,
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)} documents")

        return results
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/search/documents", response_model=List[Document])
@telemetry.track(operation_type="search_documents", metadata_resolver=telemetry.search_documents_metadata)
async def search_documents_by_name(
    request: SearchDocumentsRequest,
    auth: AuthContext = Depends(verify_token),
):
    """
    Search documents by filename using full-text search.

    `request.filters` accepts the same operator set as `/retrieve/chunks`: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`,
    `$lte`, `$in`, `$nin`, `$exists`, `$type`, `$regex` (with optional `i` flag), `$contains`, and the logical
    operators `$and`, `$or`, `$nor`, `$not`. Comparison clauses honor typed metadata (`number`, `decimal`,
    `datetime`, `date`).
    """
    try:
        results = await document_service.search_documents_by_name(
            query=request.query,
            auth=auth,
            limit=request.limit,
            filters=request.filters,
            folder_name=request.folder_name,
            folder_depth=request.folder_depth,
            end_user_id=request.end_user_id,
        )

        logger.info(f"Document name search for '{request.query}' returned {len(results)} results")
        return results

    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching documents by name: {e}")
        raise HTTPException(status_code=500, detail="Failed to search documents")


@app.post("/batch/documents", response_model=List[Document])
@telemetry.track(operation_type="batch_get_documents", metadata_resolver=telemetry.batch_documents_metadata)
async def batch_get_documents(request: BatchDocumentsRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve multiple documents by their IDs in a single batch operation.
    """
    # Initialize performance tracker
    perf = PerformanceTracker("Batch Get Documents")

    try:
        perf.start_phase("request_extraction")
        if not request.document_ids:
            perf.log_summary("No document IDs provided")
            return []

        normalized_folder = None
        if request.folder_name is not None:
            try:
                normalized_folder = normalize_folder_selector(request.folder_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_documents")
        results = await document_service.batch_retrieve_documents(
            document_ids=request.document_ids,
            auth=auth,
            folder_name=normalized_folder,
            folder_depth=None,
            end_user_id=request.end_user_id,
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(request.document_ids)} documents")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/chunks", response_model=List[ChunkResult])
@telemetry.track(operation_type="batch_get_chunks", metadata_resolver=telemetry.batch_chunks_metadata)
async def batch_get_chunks(request: BatchChunksRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve specific chunks by their document ID and chunk number in a single batch operation.
    """
    # Initialize performance tracker
    perf = PerformanceTracker("Batch Get Chunks")

    try:
        perf.start_phase("request_extraction")
        if not request.sources:
            perf.log_summary("No sources provided")
            return []

        normalized_folder_name = None
        if request.folder_name is not None:
            try:
                normalized_folder_name = normalize_folder_selector(request.folder_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_chunks")
        try:
            results = await document_service.batch_retrieve_chunks(
                chunk_ids=request.sources,
                auth=auth,
                folder_name=normalized_folder_name,
                folder_depth=None,
                end_user_id=request.end_user_id,
                use_colpali=request.use_colpali,
                output_format=request.output_format or "base64",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(request.sources)} chunks")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """
    Generate completion using relevant chunks as context.

    When graph_name is provided, the query will leverage the knowledge graph
    to enhance retrieval by finding relevant entities and their connected documents.
    """
    # Image queries not supported for completion
    if request.query_image:
        raise HTTPException(
            status_code=400,
            detail="Image queries are not supported for completion. Use /retrieve/chunks instead.",
        )

    # Text query is required for completion
    if not request.query:
        raise HTTPException(
            status_code=400,
            detail="A text query is required for completion.",
        )

    # Initialize performance tracker
    perf = PerformanceTracker(f"Query: '{request.query[:50]}...'")

    # Prepare telemetry metadata
    meta = telemetry.query_metadata(None, request=request)  # type: ignore[arg-type]
    token_est = len(request.query.split()) if isinstance(request.query, str) else 0

    try:
        # Validate prompt overrides before proceeding
        perf.start_phase("prompt_validation")
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="query")

        # Chat history retrieval
        perf.start_phase("chat_history_retrieval")
        history_key = None
        history: List[Dict[str, Any]] = []
        if request.chat_id:
            history_key = f"chat:{request.chat_id}"
            stored = await redis.get(history_key)
            if stored:
                try:
                    history = json.loads(stored)
                except Exception:
                    history = []
            else:
                db_hist = await document_service.db.get_chat_history(request.chat_id, auth.user_id, auth.app_id)
                if db_hist:
                    history = db_hist

            history.append(
                {
                    "role": "user",
                    "content": request.query,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Check query limits if in cloud mode
        perf.start_phase("limits_check")
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "query", 1)

        # Main query processing
        perf.start_phase("document_service_query")

        # Debug log for inline citations
        logger.debug(f"Query request - inline_citations: {request.inline_citations}")

        try:
            result = await document_service.query(
                request.query,
                auth,
                request.filters,
                request.k,
                request.min_score,
                request.max_tokens,
                request.temperature,
                request.use_reranking,
                request.use_colpali,
                request.graph_name,
                request.hop_depth,
                request.include_paths,
                request.prompt_overrides,
                request.folder_name,
                request.folder_depth,
                request.end_user_id,
                request.response_schema,
                history,
                perf,
                request.stream_response,
                request.llm_config,
                request.padding,
                request.inline_citations,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Handle streaming vs non-streaming responses
        if request.stream_response:
            # For streaming responses, unpack the tuple
            response_stream, sources = result

            async def generate_stream():
                full_content = ""
                first_token_time = None

                async for chunk in response_stream:
                    # Track time to first token
                    if first_token_time is None:
                        first_token_time = time.time()
                        completion_start_to_first_token = first_token_time - perf.start_time
                        perf.add_suboperation("completion_start_to_first_token", completion_start_to_first_token)
                        logger.info(f"Completion start to first token: {completion_start_to_first_token:.2f}s")

                    full_content += chunk
                    yield f"data: {json.dumps({'type': 'assistant', 'content': chunk})}\n\n"

                # Convert sources to the format expected by frontend
                sources_info = [
                    {"document_id": source.document_id, "chunk_number": source.chunk_number, "score": source.score}
                    for source in sources
                ]

                # Send completion signal with sources
                yield f"data: {json.dumps({'type': 'done', 'sources': sources_info})}\n\n"

                # Handle chat history after streaming is complete
                if history_key:
                    history.append(
                        {
                            "role": "assistant",
                            "content": full_content,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    await redis.set(history_key, json.dumps(history))
                    await document_service.db.upsert_chat_history(
                        request.chat_id,
                        auth.user_id,
                        auth.app_id,
                        history,
                    )

                # Log consolidated performance summary for streaming
                streaming_time = time.time() - first_token_time if first_token_time else 0
                perf.add_suboperation("streaming_duration", streaming_time)
                perf.log_summary(f"Generated streaming completion with {len(sources)} sources")

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }

            # Wrap original generator with telemetry span so formatting and history logic are preserved
            async def wrapped():
                async with telemetry.track_operation(
                    operation_type="query",
                    user_id=auth.user_id,
                    app_id=auth.app_id,
                    tokens_used=token_est,
                    metadata=meta,
                ):
                    async for item in generate_stream():
                        yield item

            return StreamingResponse(wrapped(), media_type="text/event-stream", headers=headers)
        else:
            # For non-streaming responses, we record telemetry around result construction
            async with telemetry.track_operation(
                operation_type="query",
                user_id=auth.user_id,
                app_id=auth.app_id,
                tokens_used=token_est,
                metadata=meta,
            ):
                response = result

            # Chat history storage for non-streaming responses
            perf.start_phase("chat_history_storage")
            if history_key:
                # Handle structured completions (Pydantic models) for chat history storage
                # Convert to JSON string since chat_history.content must be a string
                completion_content = response.completion
                if hasattr(completion_content, "model_dump"):
                    completion_content = json.dumps(completion_content.model_dump())
                history.append(
                    {
                        "role": "assistant",
                        "content": completion_content,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
                await redis.set(history_key, json.dumps(history))
                await document_service.db.upsert_chat_history(
                    request.chat_id,
                    auth.user_id,
                    auth.app_id,
                    history,
                )

            # Log consolidated performance summary
            perf.log_summary(f"Generated completion with {len(response.sources) if response.sources else 0} sources")

            return response
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="query", error=e)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/chat/{chat_id}", response_model=List[ChatMessage])
async def get_chat_history(
    chat_id: str,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Retrieve the message history for a chat conversation."""
    history_key = f"chat:{chat_id}"
    stored = await redis.get(history_key)
    if not stored:
        db_hist = await document_service.db.get_chat_history(chat_id, auth.user_id, auth.app_id)
        if not db_hist:
            return []
        return [ChatMessage(**m) for m in db_hist]
    try:
        data = json.loads(stored)
        return [ChatMessage(**m) for m in data]
    except Exception:
        return []


@app.get("/models/available")
async def get_available_models_for_selection(auth: AuthContext = Depends(verify_token)):
    """Get list of available models for UI selection.

    Returns a list of models that can be used for queries. Each model includes:
    - id: Model identifier to use in llm_config
    - name: Display name for the model
    - provider: The LLM provider (e.g., openai, anthropic, ollama)
    - description: Optional description of the model
    """
    # For now, return some common models that work with LiteLLM
    # In the future, this could be configurable or dynamically determined
    models = [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "provider": "openai",
            "description": "OpenAI's most capable model with vision support",
        },
        {
            "id": "gpt-4o-mini",
            "name": "GPT-4o Mini",
            "provider": "openai",
            "description": "Faster, more affordable GPT-4o variant",
        },
        {
            "id": "claude-3-5-sonnet-20241022",
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "description": "Anthropic's most intelligent model",
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "name": "Claude 3.5 Haiku",
            "provider": "anthropic",
            "description": "Fast and affordable Claude model",
        },
        {
            "id": "gemini/gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "provider": "google",
            "description": "Google's advanced model with long context",
        },
        {
            "id": "gemini/gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "provider": "google",
            "description": "Fast and efficient Gemini model",
        },
        {
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek Chat",
            "provider": "deepseek",
            "description": "DeepSeek's conversational AI model",
        },
        {
            "id": "groq/llama-3.3-70b-versatile",
            "name": "Llama 3.3 70B",
            "provider": "groq",
            "description": "Fast inference with Groq",
        },
        {
            "id": "groq/llama-3.1-8b-instant",
            "name": "Llama 3.1 8B",
            "provider": "groq",
            "description": "Ultra-fast small model on Groq",
        },
    ]

    # Check if there's a configured model in settings to add to the list
    if hasattr(settings, "COMPLETION_MODEL") and hasattr(settings, "REGISTERED_MODELS"):
        configured_model = settings.COMPLETION_MODEL
        if configured_model in settings.REGISTERED_MODELS:
            config = settings.REGISTERED_MODELS[configured_model]
            model_name = config.get("model_name", configured_model)
            # Add the configured model if it's not already in the list
            if not any(m["id"] == model_name for m in models):
                models.insert(
                    0,
                    {
                        "id": model_name,
                        "name": f"{configured_model} (Configured)",
                        "provider": "configured",
                        "description": "Currently configured model in morphik.toml",
                    },
                )

    return {"models": models}


# @app.post("/local/generate_uri", include_in_schema=True)
# async def generate_local_uri(
#     name: str = Form("admin"),
#     expiry_days: int = Form(30),
#     password_token: str = Form(...),
#     server_mode: bool = Form(False),
# ) -> Dict[str, str]:
#     """Generate a development URI for running Morphik locally."""
#     try:
#         # Authenticate with LOCAL_URI_PASSWORD
#         if not settings.LOCAL_URI_PASSWORD:
#             raise HTTPException(status_code=500, detail="LOCAL_URI_PASSWORD not configured")

#         if password_token != settings.LOCAL_URI_PASSWORD:
#             raise HTTPException(status_code=401, detail="Invalid authentication token")

#         # Clean name
#         name = name.replace(" ", "_").lower()

#         # Create payload (keep entity_id for backward compatibility with old clients)
#         payload = {
#             "user_id": name,
#             "entity_id": name,  # backward compat
#             "app_id": str(uuid.uuid4()),
#             "exp": datetime.now(UTC) + timedelta(days=expiry_days),
#         }

#         # Generate token
#         token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

#         # Read config for host/port
#         with open("morphik.toml", "rb") as f:
#             config = tomli.load(f)

#         # Determine base URL based on server_mode
#         if server_mode:
#             # Get external IP address
#             try:
#                 response = requests.get("http://checkip.amazonaws.com", timeout=10)
#                 if response.status_code == 200:
#                     external_ip = response.text.strip()
#                     base_url = f"{external_ip}:{config['api']['port']}"
#                 else:
#                     # Fallback to localhost if request fails
#                     logger.warning("Failed to get external IP, falling back to localhost")
#                     base_url = f"{config['api']['host']}:{config['api']['port']}".replace("localhost", "127.0.0.1")
#             except requests.RequestException as e:
#                 logger.warning(f"Failed to get external IP: {e}, falling back to localhost")
#                 base_url = f"{config['api']['host']}:{config['api']['port']}".replace("localhost", "127.0.0.1")
#         else:
#             # Use localhost as before
#             base_url = f"{config['api']['host']}:{config['api']['port']}".replace("localhost", "127.0.0.1")

#         # Generate URI
#         uri = f"morphik://{name}:{token}@{base_url}"
#         return {"uri": uri}
#     except HTTPException:
#         # Re-raise HTTP exceptions
#         raise
#     except Exception as e:
#         logger.error(f"Error generating local URI: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/cloud/generate_uri", include_in_schema=True)
async def generate_cloud_uri(
    request: GenerateUriRequest,
    authorization: str = Header(None),
    admin_secret: Optional[str] = Header(default=None, alias="X-Morphik-Admin-Secret"),
) -> Dict[str, str]:
    """Generate an authenticated URI for a cloud-hosted Morphik application."""
    try:
        app_id = request.app_id
        name = request.name
        user_id = request.user_id
        expiry_days = request.expiry_days
        is_user_token_flow = False  # Will be set to True if user is using existing app token
        source_app_id: Optional[str] = None  # Set when using existing token to inherit org_id

        is_admin_call = _validate_admin_secret(admin_secret)

        if not is_admin_call:
            # Verify authorization header before proceeding
            if not authorization:
                logger.warning("Missing authorization header")
                raise HTTPException(
                    status_code=401,
                    detail="Missing authorization header",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Invalid authorization header")

            token = authorization[7:]  # Remove "Bearer "

            try:
                # Decode the token to ensure it's valid
                payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

                token_user_id = payload.get("user_id")
                token_app_id = payload.get("app_id")
                token_permissions = payload.get("permissions", [])

                if not token_user_id:
                    raise HTTPException(status_code=401, detail="Token is missing user_id")

                if token_app_id:
                    # Token has app_id - this is a user with an existing token creating a new app
                    # They can ONLY provide name; everything else is derived or auto-generated
                    is_user_token_flow = True
                    source_app_id = token_app_id  # Store to look up org_id later
                    user_id = token_user_id
                    app_id = str(uuid.uuid4())  # Always generate new app_id
                    expiry_days = 5475  # 15 years - default expiry for user-created apps
                else:
                    is_user_token_flow = False
                    # Token has no app_id - this is cloud-ui backend with a fresh token
                    # Allow all fields from request
                    if not user_id:
                        user_id = token_user_id
                    if not (token_user_id == user_id or "admin" in token_permissions):
                        raise HTTPException(
                            status_code=403,
                            detail="You can only create apps for your own account unless you have admin permissions",
                        )
                    if not app_id:
                        app_id = str(uuid.uuid4())
            except jwt.InvalidTokenError as e:
                raise HTTPException(status_code=401, detail=str(e))
        elif not user_id:
            raise HTTPException(status_code=400, detail="user_id is required when using admin secret")
        else:
            # Admin call - generate app_id if not provided
            if not app_id:
                app_id = str(uuid.uuid4())

        logger.debug(
            "Generating cloud URI for app_id=%s, name=%s, user_id=%s (admin_header=%s)",
            app_id,
            name,
            user_id,
            bool(admin_secret),
        )
        # Import UserService here to avoid circular imports
        from core.services.user_service import UserService

        user_service = UserService()

        # Initialize user service if needed
        await user_service.initialize()

        # Clean name
        name = name.replace(" ", "_").lower()

        # Determine org_id and created_by_user_id
        inherited_org_id: Optional[str] = None
        inherited_created_by: Optional[str] = None

        if is_user_token_flow:
            # Look up the source app to inherit org_id
            source_app = await user_service.get_app_by_id(source_app_id)
            if source_app:
                inherited_org_id = source_app.get("org_id")
                inherited_created_by = source_app.get("created_by_user_id")
                logger.debug(
                    "Inheriting org_id=%s from source app %s for new app %s",
                    inherited_org_id,
                    source_app_id,
                    app_id,
                )
            else:
                logger.warning(
                    "Source app %s not found in apps table - new app %s will have no org association",
                    source_app_id,
                    app_id,
                )

        # Check if the user is within app limit and generate URI
        uri = await user_service.generate_cloud_uri(
            user_id,
            app_id,
            name,
            expiry_days,
            org_id=inherited_org_id if is_user_token_flow else request.org_id,
            created_by_user_id=inherited_created_by if is_user_token_flow else request.created_by_user_id,
            is_admin_call=is_admin_call,
        )

        if not uri:
            logger.warning(
                "URI generation returned None for user_id=%s, app_id=%s (likely limit reached)", user_id, app_id
            )
            raise HTTPException(status_code=403, detail="Application limit reached for this account tier")

        return {"uri": uri, "app_id": app_id}
    except ValueError as e:
        # Handle duplicate name or validation errors
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating cloud URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/apps", include_in_schema=True)
async def list_cloud_apps(
    org_id: Optional[str] = Query(default=None, description="Filter apps by organization ID"),
    user_id: Optional[str] = Query(default=None, description="Filter apps by creator"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    authorization: Optional[str] = Header(default=None),
    admin_secret: Optional[str] = Header(default=None, alias="X-Morphik-Admin-Secret"),
):
    """List provisioned apps for the specified organization/user."""

    try:
        is_admin_call = _validate_admin_secret(admin_secret)
    except HTTPException:
        # Invalid admin secret provided
        raise

    token_user_id: Optional[str] = None
    token_permissions: List[str] = []

    if not is_admin_call:
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        except jwt.InvalidTokenError as exc:  # pragma: no cover - propagated error
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        token_user_id = payload.get("user_id")
        token_permissions = payload.get("permissions", []) or []

    if not is_admin_call:
        if user_id and user_id != token_user_id and "admin" not in token_permissions:
            raise HTTPException(status_code=403, detail="Cannot list apps for another user")
        if not user_id:
            user_id = token_user_id

    try:
        from core.services.user_service import UserService

        user_service = UserService()
        await user_service.initialize()
        apps = await user_service.list_apps(
            org_id=org_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
            strict_org_scope=not is_admin_call,
        )
        return {"apps": apps, "count": len(apps)}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - server failure
        logger.error("Failed to list apps: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list applications")


# ---------------------------------------------------------------------------
# Cloud – delete application (control-plane only)
# ---------------------------------------------------------------------------


@app.delete("/apps")
async def delete_cloud_app(
    app_name: str = Query(..., description="Name of the application to delete"),
    authorization: Optional[str] = Header(default=None),
    admin_secret: Optional[str] = Header(default=None, alias="X-Morphik-Admin-Secret"),
) -> Dict[str, Any]:
    """Delete all resources associated with a given cloud application."""

    # Check if admin secret is provided and valid
    is_admin_call = False
    try:
        is_admin_call = _validate_admin_secret(admin_secret)
    except HTTPException:
        # Invalid admin secret provided
        raise

    user_id: Optional[str] = None

    if not is_admin_call:
        # Require Bearer token if no admin secret
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            user_id = payload.get("user_id")
        except jwt.InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    logger.info(f"Deleting app {app_name} for user {user_id} (admin_call={is_admin_call})")

    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select

    from core.models.apps import AppModel
    from core.services.user_service import UserService

    # 1) Resolve app_id from apps table ----------------------------------
    async with document_service.db.async_session() as session:
        if is_admin_call:
            # Admin call: look up by name only
            stmt = select(AppModel).where(AppModel.name == app_name)
        else:
            # User call: look up by user_id and name
            stmt = select(AppModel).where(AppModel.user_id == user_id, AppModel.name == app_name)
        res = await session.execute(stmt)
        app_row = res.scalar_one_or_none()

    if app_row is None:
        raise HTTPException(status_code=404, detail="Application not found")

    app_id = app_row.app_id
    # For admin calls, get user_id from the app record
    effective_user_id = user_id if user_id else str(app_row.user_id)

    # ------------------------------------------------------------------
    # Create an AuthContext scoped to *this* application so that the
    # underlying access-control filters in the database layer allow us to
    # see and delete resources that belong to the app – even if the JWT
    # used to call this endpoint was scoped to a *different* app.
    # ------------------------------------------------------------------

    app_auth = AuthContext(
        user_id=effective_user_id,
        app_id=app_id,
    )

    # 2) Delete all documents for this app ------------------------------
    # ------------------------------------------------------------------
    # Fetch ALL documents for *this* app using the app-scoped auth.
    # ------------------------------------------------------------------
    doc_ids = await document_service.db.find_authorized_and_filtered_documents(app_auth)

    deleted = 0
    doc_failures: List[str] = []
    for doc_id in doc_ids:
        try:
            await document_service.delete_document(doc_id, app_auth)
            deleted += 1
        except Exception as exc:
            logger.warning("Failed to delete document %s for app %s: %s", doc_id, app_id, exc)
            doc_failures.append(str(doc_id))

    # 3) Delete folders associated with this app -----------------------
    # ------------------------------------------------------------------
    # Fetch ALL folders for *this* app using the same app-scoped auth.
    # ------------------------------------------------------------------
    folder_ids_deleted = 0
    folder_failures: List[str] = []
    folders = await document_service.db.list_folders(app_auth)

    for folder in folders:
        try:
            await document_service.db.delete_folder(folder.id, app_auth)
            folder_ids_deleted += 1
        except Exception as f_exc:  # noqa: BLE001
            logger.warning("Failed to delete folder %s for app %s: %s", folder.id, app_id, f_exc)
            folder_failures.append(folder.id)

    if doc_failures or folder_failures:
        failure_detail = {
            "documents_failed": doc_failures,
            "folders_failed": folder_failures,
        }
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to delete all application resources; aborting app deletion",
                **failure_detail,
            },
        )

    # 4) Remove apps table entry ---------------------------------------
    async with document_service.db.async_session() as session:
        await session.execute(sa_delete(AppModel).where(AppModel.app_id == app_id))
        await session.commit()

    # 5) Update user_limits --------------------------------------------
    user_service = UserService()
    await user_service.initialize()
    await user_service.unregister_app(effective_user_id, app_id)

    return {
        "app_name": app_name,
        "status": "deleted",
        "documents_deleted": deleted,
        "folders_deleted": folder_ids_deleted,
    }


@app.get("/chats", response_model=List[Dict[str, Any]])
async def list_chat_conversations(
    auth: AuthContext = Depends(verify_token),
    limit: int = Query(100, ge=1, le=500),
):
    """List chat conversations available to the current user."""
    try:
        convos = await document_service.db.list_chat_conversations(
            user_id=auth.user_id,
            app_id=auth.app_id,
            limit=limit,
        )
        return convos
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing chat conversations: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list chat conversations")


@app.patch("/chats/{chat_id}/title", response_model=ChatTitleResponse)
async def update_chat_title(
    chat_id: str,
    title: str = Query(..., description="New title for the chat"),
    auth: AuthContext = Depends(verify_token),
):
    """Update the title of a chat conversation."""
    try:
        success = await document_service.db.update_chat_title(
            conversation_id=chat_id,
            title=title,
            user_id=auth.user_id,
            app_id=auth.app_id,
        )
        if success:
            return {"status": "success", "message": "Chat title updated successfully", "title": title}
        else:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Error updating chat title: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update chat title")
