import asyncio
import json
import logging
import time  # Add time import for profiling
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import arq
import jwt
import tomli
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.sessions import SessionMiddleware

from core.agent import MorphikAgent
from core.app_factory import lifespan
from core.auth_utils import verify_token
from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.dependencies import get_redis_pool
from core.limits_utils import check_and_increment_limits
from core.logging_config import setup_logging
from core.middleware.profiling import ProfilingMiddleware
from core.models.auth import AuthContext, EntityType
from core.models.chat import ChatMessage
from core.models.completion import ChunkSource, CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentResult
from core.models.folders import Folder, FolderCreate, FolderSummary
from core.models.graph import Graph
from core.models.prompts import validate_prompt_overrides_with_http_exception
from core.models.request import (
    AgentQueryRequest,
    CompletionQueryRequest,
    CreateGraphRequest,
    GenerateUriRequest,
    IngestTextRequest,
    RetrieveRequest,
    SetFolderRuleRequest,
    UpdateGraphRequest,
)
from core.routes.document import router as document_router
from core.routes.ingest import router as ingest_router
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# Set up logging configuration for Docker environment
setup_logging()

# Initialize FastAPI app
logger = logging.getLogger(__name__)


# Performance tracking class
class PerformanceTracker:
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.phases = {}
        self.current_phase = None
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

    def add_suboperation(self, name: str, duration: float):
        """Add a sub-operation timing"""
        self.phases[name] = duration

    def log_summary(self, additional_info: str = ""):
        total_time = time.time() - self.start_time

        # End current phase if still running
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start

        logger.info(f"=== {self.operation_name} Performance Summary ===")
        logger.info(f"Total time: {total_time:.2f}s")

        # Sort phases by duration (longest first)
        for phase, duration in sorted(self.phases.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")

        if additional_info:
            logger.info(additional_info)
        logger.info("=" * (len(self.operation_name) + 31))


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

# Global settings object
settings = get_settings()

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


# Simple health check endpoint
@app.get("/ping")
async def ping_health():
    """Simple health check endpoint that returns 200 OK."""
    return {"status": "ok", "message": "Server is running"}


# ---------------------------------------------------------------------------
# Core singletons (database, vector store, storage, parser, models …)
# ---------------------------------------------------------------------------


# Store on app.state for later access
app.state.document_service = document_service
logger.info("Document service initialized and stored on app.state")

# Register ingest router
app.include_router(ingest_router)

# Register document router
app.include_router(document_router)

# Single MorphikAgent instance (tool definitions cached)
morphik_agent = MorphikAgent(document_service=document_service)


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

    Args:
        request: RetrieveRequest containing:
            - query: Search query text
            - filters: Optional metadata filters
            - k: Number of results (default: 4)
            - min_score: Minimum similarity threshold (default: 0.0)
            - use_reranking: Whether to use reranking
            - use_colpali: Whether to use ColPali-style embedding model
            - folder_name: Optional folder to scope the search to
            - end_user_id: Optional end-user ID to scope the search to
        auth: Authentication context

    Returns:
        List[ChunkResult]: List of relevant chunks
    """
    # Initialize performance tracker
    perf = PerformanceTracker(f"Retrieve Chunks: '{request.query[:50]}...'")

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
            request.end_user_id,
            perf,  # Pass performance tracker
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)} chunks")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/retrieve/docs", response_model=List[DocumentResult])
@telemetry.track(operation_type="retrieve_docs", metadata_resolver=telemetry.retrieve_docs_metadata)
async def retrieve_documents(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant documents.

    Args:
        request: RetrieveRequest containing:
            - query: Search query text
            - filters: Optional metadata filters
            - k: Number of results (default: 4)
            - min_score: Minimum similarity threshold (default: 0.0)
            - use_reranking: Whether to use reranking
            - use_colpali: Whether to use ColPali-style embedding model
            - folder_name: Optional folder to scope the search to
            - end_user_id: Optional end-user ID to scope the search to
        auth: Authentication context

    Returns:
        List[DocumentResult]: List of relevant documents
    """
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
            request.end_user_id,
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)} documents")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/documents", response_model=List[Document])
@telemetry.track(operation_type="batch_get_documents", metadata_resolver=telemetry.batch_documents_metadata)
async def batch_get_documents(request: Dict[str, Any], auth: AuthContext = Depends(verify_token)):
    """
    Retrieve multiple documents by their IDs in a single batch operation.

    Args:
        request: Dictionary containing:
            - document_ids: List of document IDs to retrieve
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
        auth: Authentication context

    Returns:
        List[Document]: List of documents matching the IDs
    """
    # Initialize performance tracker
    perf = PerformanceTracker("Batch Get Documents")

    try:
        # Extract document_ids from request
        perf.start_phase("request_extraction")
        document_ids = request.get("document_ids", [])
        folder_name = request.get("folder_name")
        end_user_id = request.get("end_user_id")

        if not document_ids:
            perf.log_summary("No document IDs provided")
            return []

        # Create system filters for folder and user scoping
        perf.start_phase("filter_creation")
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_documents")
        results = await document_service.batch_retrieve_documents(document_ids, auth, folder_name, end_user_id)

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(document_ids)} documents")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/chunks", response_model=List[ChunkResult])
@telemetry.track(operation_type="batch_get_chunks", metadata_resolver=telemetry.batch_chunks_metadata)
async def batch_get_chunks(request: Dict[str, Any], auth: AuthContext = Depends(verify_token)):
    """
    Retrieve specific chunks by their document ID and chunk number in a single batch operation.

    Args:
        request: Dictionary containing:
            - sources: List of ChunkSource objects (with document_id and chunk_number)
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
            - use_colpali: Whether to use ColPali-style embedding
        auth: Authentication context

    Returns:
        List[ChunkResult]: List of chunk results
    """
    # Initialize performance tracker
    perf = PerformanceTracker("Batch Get Chunks")

    try:
        # Extract sources from request
        perf.start_phase("request_extraction")
        sources = request.get("sources", [])
        folder_name = request.get("folder_name")
        end_user_id = request.get("end_user_id")
        use_colpali = request.get("use_colpali")

        if not sources:
            perf.log_summary("No sources provided")
            return []

        # Convert sources to ChunkSource objects if needed
        perf.start_phase("source_conversion")
        chunk_sources = []
        for source in sources:
            if isinstance(source, dict):
                chunk_sources.append(ChunkSource(**source))
            else:
                chunk_sources.append(source)

        # Create system filters for folder and user scoping
        perf.start_phase("filter_creation")
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_chunks")
        results = await document_service.batch_retrieve_chunks(
            chunk_sources, auth, folder_name, end_user_id, use_colpali
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(sources)} chunks")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
@telemetry.track(operation_type="query", metadata_resolver=telemetry.query_metadata)
async def query_completion(
    request: CompletionQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """
    Generate completion using relevant chunks as context.

    When graph_name is provided, the query will leverage the knowledge graph
    to enhance retrieval by finding relevant entities and their connected documents.

    Args:
        request: CompletionQueryRequest containing:
            - query: Query text
            - filters: Optional metadata filters
            - k: Number of chunks to use as context (default: 4)
            - min_score: Minimum similarity threshold (default: 0.0)
            - max_tokens: Maximum tokens in completion
            - temperature: Model temperature
            - use_reranking: Whether to use reranking
            - use_colpali: Whether to use ColPali-style embedding model
            - graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            - hop_depth: Number of relationship hops to traverse in the graph (1-3)
            - include_paths: Whether to include relationship paths in the response
            - prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
            - schema: Optional schema for structured output
            - chat_id: Optional chat conversation identifier for maintaining history
        auth: Authentication context

    Returns:
        CompletionResponse: Generated text completion or structured output
    """
    # Initialize performance tracker
    perf = PerformanceTracker(f"Query: '{request.query[:50]}...'")

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
            request.end_user_id,
            request.schema,
            history,
            perf,  # Pass performance tracker
            request.stream_response,
        )

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
            return StreamingResponse(generate_stream(), media_type="text/event-stream", headers=headers)
        else:
            # For non-streaming responses, result is just the CompletionResponse
            response = result

            # Chat history storage for non-streaming responses
            perf.start_phase("chat_history_storage")
            if history_key:
                history.append(
                    {
                        "role": "assistant",
                        "content": response.completion,
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
    """Retrieve the message history for a chat conversation.

    Args:
        chat_id: Identifier of the conversation whose history should be loaded.
        auth: Authentication context used to verify access to the conversation.
        redis: Redis connection where chat messages are stored.

    Returns:
        A list of :class:`ChatMessage` objects or an empty list if no history
        exists.
    """
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


@app.post("/agent", response_model=Dict[str, Any])
@telemetry.track(operation_type="agent_query")
async def agent_query(
    request: AgentQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Execute an agent-style query using the :class:`MorphikAgent`.

    Args:
        request: The query payload containing the natural language question and optional chat_id.
        auth: Authentication context used to enforce limits and access control.
        redis: Redis connection for chat history storage.

    Returns:
        A dictionary with the agent's full response.
    """
    # Chat history retrieval
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

    # Check free-tier agent call limits in cloud mode
    if settings.MODE == "cloud" and auth.user_id:
        await check_and_increment_limits(auth, "agent", 1)

    # Use the shared MorphikAgent instance; per-run state is now isolated internally
    response = await morphik_agent.run(request.query, auth, history)

    # Chat history storage
    if history_key:
        # Store the full agent response with structured data
        agent_message = {
            "role": "assistant",
            "content": response.get("response", ""),
            "timestamp": datetime.now(UTC).isoformat(),
            # Store agent-specific structured data
            "agent_data": {
                "display_objects": response.get("display_objects", []),
                "tool_history": response.get("tool_history", []),
                "sources": response.get("sources", []),
            },
        }
        history.append(agent_message)
        await redis.set(history_key, json.dumps(history))
        await document_service.db.upsert_chat_history(
            request.chat_id,
            auth.user_id,
            auth.app_id,
            history,
        )

    # Return the complete response dictionary
    return response


@app.post("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 10000,
    filters: Optional[Dict[str, Any]] = None,
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
):
    """
    List accessible documents.

    Args:
        auth: Authentication context
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        filters: Optional metadata filters
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
    if auth.app_id:
        system_filters["app_id"] = auth.app_id

    return await document_service.db.get_documents(auth, skip, limit, filters, system_filters)


@app.get("/documents/{document_id}", response_model=Document)
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


@app.get("/documents/{document_id}/status", response_model=Dict[str, Any])
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

        # Add error information if failed
        if status == "failed":
            response["error"] = doc.system_metadata.get("error", "Unknown error")

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document status: {str(e)}")


@app.delete("/documents/{document_id}")
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


@app.get("/documents/filename/{filename}", response_model=Document)
async def get_document_by_filename(
    filename: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = None,
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


@app.get("/documents/{document_id}/download_url")
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


@app.get("/documents/{document_id}/file")
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


@app.post("/documents/{document_id}/update_text", response_model=Document)
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


@app.post("/documents/{document_id}/update_file", response_model=Document)
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


@app.post("/documents/{document_id}/update_metadata", response_model=Document)
@telemetry.track(
    operation_type="update_document_metadata",
    metadata_resolver=telemetry.document_update_metadata_resolver,
)
async def update_document_metadata(
    document_id: str, metadata: Dict[str, Any], auth: AuthContext = Depends(verify_token)
):
    """
    Update only a document's metadata.

    Args:
        document_id: ID of the document to update
        metadata: New metadata to merge with existing metadata
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
            metadata=metadata,
            rules=[],
            update_strategy="add",
            use_colpali=None,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


# Usage tracking endpoints
@app.get("/usage/stats")
@telemetry.track(operation_type="get_usage_stats", metadata_resolver=telemetry.usage_stats_metadata)
async def get_usage_stats(auth: AuthContext = Depends(verify_token)) -> Dict[str, int]:
    """Get usage statistics for the authenticated user.

    Args:
        auth: Authentication context identifying the caller.

    Returns:
        A mapping of operation types to token usage counts.
    """
    if not auth.permissions or "admin" not in auth.permissions:
        return telemetry.get_user_usage(auth.entity_id)
    return telemetry.get_user_usage(auth.entity_id)


@app.get("/usage/recent")
@telemetry.track(operation_type="get_recent_usage", metadata_resolver=telemetry.recent_usage_metadata)
async def get_recent_usage(
    auth: AuthContext = Depends(verify_token),
    operation_type: Optional[str] = None,
    since: Optional[datetime] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """Retrieve recent telemetry records for the user or application.

    Args:
        auth: Authentication context; admin users receive global records.
        operation_type: Optional operation type to filter by.
        since: Only return records newer than this timestamp.
        status: Optional status filter (e.g. ``success`` or ``error``).

    Returns:
        A list of usage entries sorted by timestamp, each represented as a
        dictionary.
    """
    if not auth.permissions or "admin" not in auth.permissions:
        records = telemetry.get_recent_usage(
            user_id=auth.entity_id, operation_type=operation_type, since=since, status=status
        )
    else:
        records = telemetry.get_recent_usage(operation_type=operation_type, since=since, status=status)

    return [
        {
            "timestamp": record.timestamp,
            "operation_type": record.operation_type,
            "tokens_used": record.tokens_used,
            "user_id": record.user_id,
            "duration_ms": record.duration_ms,
            "status": record.status,
            "metadata": record.metadata,
        }
        for record in records
    ]


# Cache endpoints
@app.post("/cache/create")
@telemetry.track(operation_type="create_cache", metadata_resolver=telemetry.cache_create_metadata)
async def create_cache(
    name: str,
    model: str,
    gguf_file: str,
    filters: Optional[Dict[str, Any]] = None,
    docs: Optional[List[str]] = None,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Create a persistent cache for low-latency completions.

    Args:
        name: Unique identifier for the cache.
        model: The model name to use when generating completions.
        gguf_file: Path to the ``gguf`` weights file to load.
        filters: Optional metadata filters used to select documents.
        docs: Explicit list of document IDs to include in the cache.
        auth: Authentication context used for permission checks.

    Returns:
        A dictionary describing the created cache.
    """
    try:
        # Check cache creation limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "cache", 1)

        filter_docs = set(await document_service.db.get_documents(auth, filters=filters))
        additional_docs = (
            {await document_service.db.get_document(document_id=doc_id, auth=auth) for doc_id in docs}
            if docs
            else set()
        )
        docs_to_add = list(filter_docs.union(additional_docs))
        if not docs_to_add:
            raise HTTPException(status_code=400, detail="No documents to add to cache")
        response = await document_service.create_cache(name, model, gguf_file, docs_to_add, filters)
        return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/cache/{name}")
@telemetry.track(operation_type="get_cache", metadata_resolver=telemetry.cache_get_metadata)
async def get_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, Any]:
    """Retrieve information about a specific cache.

    Args:
        name: Name of the cache to inspect.
        auth: Authentication context used to authorize the request.

    Returns:
        A dictionary with a boolean ``exists`` field indicating whether the
        cache is loaded.
    """
    try:
        exists = await document_service.load_cache(name)
        return {"exists": exists}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/update")
@telemetry.track(operation_type="update_cache", metadata_resolver=telemetry.cache_update_metadata)
async def update_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, bool]:
    """Refresh an existing cache with newly available documents.

    Args:
        name: Identifier of the cache to update.
        auth: Authentication context used for permission checks.

    Returns:
        A dictionary indicating whether any documents were added.
    """
    try:
        if name not in document_service.active_caches:
            exists = await document_service.load_cache(name)
            if not exists:
                raise HTTPException(status_code=404, detail=f"Cache '{name}' not found")
        cache = document_service.active_caches[name]
        docs = await document_service.db.get_documents(auth, filters=cache.filters)
        docs_to_add = [doc for doc in docs if doc.id not in cache.docs]
        return cache.add_docs(docs_to_add)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/add_docs")
@telemetry.track(operation_type="add_docs_to_cache", metadata_resolver=telemetry.cache_add_docs_metadata)
async def add_docs_to_cache(name: str, docs: List[str], auth: AuthContext = Depends(verify_token)) -> Dict[str, bool]:
    """Manually add documents to an existing cache.

    Args:
        name: Name of the target cache.
        docs: List of document IDs to insert.
        auth: Authentication context used for authorization.

    Returns:
        A dictionary indicating whether the documents were queued for addition.
    """
    try:
        cache = document_service.active_caches[name]
        docs_to_add = [
            await document_service.db.get_document(doc_id, auth) for doc_id in docs if doc_id not in cache.docs
        ]
        return cache.add_docs(docs_to_add)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/query")
@telemetry.track(operation_type="query_cache", metadata_resolver=telemetry.cache_query_metadata)
async def query_cache(
    name: str,
    query: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    auth: AuthContext = Depends(verify_token),
) -> CompletionResponse:
    """Generate a completion using a pre-populated cache.

    Args:
        name: Name of the cache to query.
        query: Prompt text to send to the model.
        max_tokens: Optional maximum number of tokens to generate.
        temperature: Optional sampling temperature for the model.
        auth: Authentication context for permission checks.

    Returns:
        A :class:`CompletionResponse` object containing the model output.
    """
    try:
        # Check cache query limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "cache_query", 1)

        cache = document_service.active_caches[name]
        logger.info(f"Cache state: {cache.state.n_tokens}")
        return cache.query(query)  # , max_tokens, temperature)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/graph/create", response_model=Graph)
@telemetry.track(operation_type="create_graph", metadata_resolver=telemetry.create_graph_metadata)
async def create_graph(
    request: CreateGraphRequest,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """Create a new graph based on document contents.

    The graph is created asynchronously. A stub graph record is returned with
    ``status = "processing"`` while a background task extracts entities and
    relationships.

    Args:
        request: Graph creation parameters including name and optional filters.
        auth: Authentication context authorizing the operation.

    Returns:
        The placeholder :class:`Graph` object which clients can poll for status.
    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="graph")

        # Enforce usage limits (cloud mode)
        if settings.MODE == "cloud" and auth.user_id:
            await check_and_increment_limits(auth, "graph", 1)

        # --------------------
        # Build system filters
        # --------------------
        system_filters: Dict[str, Any] = {}
        if request.folder_name is not None:
            normalized_folder_name = normalize_folder_name(request.folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if request.end_user_id:
            system_filters["end_user_id"] = request.end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        # --------------------
        # Create stub graph
        # --------------------
        import uuid
        from datetime import UTC, datetime

        from core.models.graph import Graph

        access_control = {
            "readers": [auth.entity_id],
            "writers": [auth.entity_id],
            "admins": [auth.entity_id],
        }
        if auth.user_id:
            access_control["user_id"] = [auth.user_id]

        graph_stub = Graph(
            id=str(uuid.uuid4()),
            name=request.name,
            filters=request.filters,
            owner={"type": auth.entity_type.value, "id": auth.entity_id},
            access_control=access_control,
        )

        # Persist scoping info in system metadata
        if system_filters.get("folder_name"):
            graph_stub.system_metadata["folder_name"] = system_filters["folder_name"]
        if system_filters.get("end_user_id"):
            graph_stub.system_metadata["end_user_id"] = system_filters["end_user_id"]
        if auth.app_id:
            graph_stub.system_metadata["app_id"] = auth.app_id

        # Mark graph as processing
        graph_stub.system_metadata["status"] = "processing"
        graph_stub.system_metadata["created_at"] = datetime.now(UTC)
        graph_stub.system_metadata["updated_at"] = datetime.now(UTC)

        # Store the stub graph so clients can poll for status
        success = await document_service.db.store_graph(graph_stub)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create graph stub")

        # --------------------
        # Background processing
        # --------------------
        async def _build_graph_async():
            try:
                await document_service.update_graph(
                    name=request.name,
                    auth=auth,
                    additional_filters=None,  # original filters already on stub
                    additional_documents=request.documents,
                    prompt_overrides=request.prompt_overrides,
                    system_filters=system_filters,
                    is_initial_build=True,  # Indicate this is the initial build
                )
            except Exception as e:
                logger.error(f"Graph creation failed for {request.name}: {e}")
                # Update graph status to failed
                existing = await document_service.db.get_graph(request.name, auth, system_filters=system_filters)
                if existing:
                    existing.system_metadata["status"] = "failed"
                    existing.system_metadata["error"] = str(e)
                    existing.system_metadata["updated_at"] = datetime.now(UTC)
                    await document_service.db.update_graph(existing)

        import asyncio

        asyncio.create_task(_build_graph_async())

        # Return the stub graph immediately
        return graph_stub
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="graph", error=e)


@app.post("/folders", response_model=Folder)
@telemetry.track(operation_type="create_folder", metadata_resolver=telemetry.create_folder_metadata)
async def create_folder(
    folder_create: FolderCreate,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Create a new folder.

    Args:
        folder_create: Folder creation request containing name and optional description
        auth: Authentication context

    Returns:
        Folder: Created folder
    """
    try:
        # Create a folder object with explicit ID
        import uuid

        folder_id = str(uuid.uuid4())
        logger.info(f"Creating folder with ID: {folder_id}, auth.user_id: {auth.user_id}")

        # Set up access control with user_id
        access_control = {
            "readers": [auth.entity_id],
            "writers": [auth.entity_id],
            "admins": [auth.entity_id],
        }

        if auth.user_id:
            access_control["user_id"] = [auth.user_id]
            logger.info(f"Adding user_id {auth.user_id} to folder access control")

        folder = Folder(
            id=folder_id,
            name=folder_create.name,
            description=folder_create.description,
            owner={
                "type": auth.entity_type.value,
                "id": auth.entity_id,
            },
            access_control=access_control,
        )

        # Scope folder to the application ID for developer tokens
        if auth.app_id:
            folder.system_metadata["app_id"] = auth.app_id

        # Store in database
        success = await document_service.db.create_folder(folder)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create folder")

        return folder
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders", response_model=List[Folder])
@telemetry.track(operation_type="list_folders", metadata_resolver=telemetry.list_folders_metadata)
async def list_folders(
    auth: AuthContext = Depends(verify_token),
) -> List[Folder]:
    """
    List all folders the user has access to.

    Args:
        auth: Authentication context

    Returns:
        List[Folder]: List of folders
    """
    try:
        folders = await document_service.db.list_folders(auth)
        return folders
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders/summary", response_model=List[FolderSummary])
@telemetry.track(operation_type="list_folders_summary")
async def list_folder_summaries(auth: AuthContext = Depends(verify_token)) -> List[FolderSummary]:
    """Return compact folder list (id, name, doc_count, updated_at)."""

    try:
        summaries = await document_service.db.list_folders_summary(auth)
        return summaries  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing folder summaries: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/folders/{folder_id}", response_model=Folder)
@telemetry.track(operation_type="get_folder", metadata_resolver=telemetry.get_folder_metadata)
async def get_folder(
    folder_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Get a folder by ID.

    Args:
        folder_id: ID of the folder
        auth: Authentication context

    Returns:
        Folder: Folder if found and accessible
    """
    try:
        folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        return folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folders/{folder_name}")
@telemetry.track(operation_type="delete_folder", metadata_resolver=telemetry.delete_folder_metadata)
async def delete_folder(
    folder_name: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Delete a folder and all associated documents.

    Args:
        folder_name: Name of the folder to delete
        auth: Authentication context (must have write access to the folder)

    Returns:
        Deletion status
    """
    try:
        folder = await document_service.db.get_folder_by_name(folder_name, auth)
        folder_id = folder.id
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        document_ids = folder.document_ids
        tasks = [remove_document_from_folder(folder_id, document_id, auth) for document_id in document_ids]
        results = await asyncio.gather(*tasks)
        stati = [res.get("status", False) for res in results]
        if not all(stati):
            failed = [doc for doc, stat in zip(document_ids, stati) if not stat]
            msg = "Failed to remove the following documents from folder: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # folder is empty now
        delete_tasks = [document_service.db.delete_document(document_id, auth) for document_id in document_ids]
        stati = await asyncio.gather(*delete_tasks)
        if not all(stati):
            failed = [doc for doc, stat in zip(document_ids, stati) if not stat]
            msg = "Failed to delete the following documents: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        db: PostgresDatabase = document_service.db
        # just remove the folder too now.
        status = await db.delete_folder(folder_id, auth)
        if not status:
            logger.error(f"Failed to delete folder {folder_id}")
            raise HTTPException(status_code=500, detail=f"Failed to delete folder {folder_id}")
        return {"status": "success", "message": f"Folder {folder_id} deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folders/{folder_id}/documents/{document_id}")
@telemetry.track(operation_type="add_document_to_folder", metadata_resolver=telemetry.add_document_to_folder_metadata)
async def add_document_to_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Add a document to a folder.

    Args:
        folder_id: ID of the folder
        document_id: ID of the document
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.add_document_to_folder(folder_id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to folder")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding document to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folders/{folder_id}/documents/{document_id}")
@telemetry.track(
    operation_type="remove_document_from_folder", metadata_resolver=telemetry.remove_document_from_folder_metadata
)
async def remove_document_from_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Remove a document from a folder.

    Args:
        folder_id: ID of the folder
        document_id: ID of the document
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.remove_document_from_folder(folder_id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove document from folder")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error removing document from folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/{name}", response_model=Graph)
@telemetry.track(operation_type="get_graph", metadata_resolver=telemetry.get_graph_metadata)
async def get_graph(
    name: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = None,
    end_user_id: Optional[str] = None,
) -> Graph:
    """
    Get a graph by name.

    This endpoint retrieves a graph by its name if the user has access to it.

    Args:
        name: Name of the graph to retrieve
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        Graph: The requested graph object
    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        graph = await document_service.db.get_graph(name, auth, system_filters)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Graph '{name}' not found")
        return graph
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graphs", response_model=List[Graph])
@telemetry.track(operation_type="list_graphs", metadata_resolver=telemetry.list_graphs_metadata)
async def list_graphs(
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = None,
    end_user_id: Optional[str] = None,
) -> List[Graph]:
    """
    List all graphs the user has access to.

    This endpoint retrieves all graphs the user has access to.

    Args:
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        List[Graph]: List of graph objects
    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        return await document_service.db.list_graphs(auth, system_filters)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/{name}/visualization", response_model=Dict[str, Any])
@telemetry.track(operation_type="get_graph_visualization", metadata_resolver=telemetry.get_graph_metadata)
async def get_graph_visualization(
    name: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = None,
    end_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get graph visualization data.

    This endpoint retrieves the nodes and links data needed for graph visualization.
    It works with both local and API-based graph services.

    Args:
        name: Name of the graph to visualize
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        Dict: Visualization data containing nodes and links arrays
    """
    try:
        return await document_service.get_graph_visualization_data(
            name=name,
            auth=auth,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting graph visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/{name}/update", response_model=Graph)
@telemetry.track(operation_type="update_graph", metadata_resolver=telemetry.update_graph_metadata)
async def update_graph(
    name: str,
    request: UpdateGraphRequest,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """
    Update an existing graph with new documents.

    This endpoint processes additional documents based on the original graph filters
    and/or new filters/document IDs, extracts entities and relationships, and
    updates the graph with new information.

    Args:
        name: Name of the graph to update
        request: UpdateGraphRequest containing:
            - additional_filters: Optional additional metadata filters to determine which new documents to include
            - additional_documents: Optional list of additional document IDs to include
            - prompt_overrides: Optional customizations for entity extraction and resolution prompts
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
        auth: Authentication context

    Returns:
        Graph: The updated graph object
    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="graph")

        # Create system filters for folder and user scoping
        system_filters = {}
        if request.folder_name:
            system_filters["folder_name"] = request.folder_name
        if request.end_user_id:
            system_filters["end_user_id"] = request.end_user_id

        # Developer tokens: always scope by app_id to prevent cross-app leakage
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        return await document_service.update_graph(
            name=name,
            auth=auth,
            additional_filters=request.additional_filters,
            additional_documents=request.additional_documents,
            prompt_overrides=request.prompt_overrides,
            system_filters=system_filters,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="graph", error=e)
    except Exception as e:
        logger.error(f"Error updating graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/workflow/{workflow_id}/status", response_model=Dict[str, Any])
@telemetry.track(operation_type="check_workflow_status", metadata_resolver=telemetry.workflow_status_metadata)
async def check_workflow_status(
    workflow_id: str,
    run_id: Optional[str] = None,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Check the status of a graph build/update workflow.

    This endpoint polls the external graph API to check the status of an async operation.

    Args:
        workflow_id: The workflow ID returned from build/update operations
        run_id: Optional run ID for the specific workflow run
        auth: Authentication context

    Returns:
        Dict containing status ('running', 'completed', or 'failed') and optional result
    """
    try:
        # Get the graph service (either local or API-based)
        graph_service = document_service.graph_service

        # Check if it's the MorphikGraphService
        from core.services.morphik_graph_service import MorphikGraphService

        if isinstance(graph_service, MorphikGraphService):
            # Use the new check_workflow_status method
            result = await graph_service.check_workflow_status(workflow_id=workflow_id, run_id=run_id, auth=auth)

            # If the workflow is completed, update the corresponding graph status
            if result.get("status") == "completed":
                # Extract graph_id from workflow_id (format: "build-update-{graph_name}-...")
                # This is a simple heuristic, adjust based on actual workflow_id format
                parts = workflow_id.split("-")
                if len(parts) >= 3:
                    graph_name = parts[2]
                    try:
                        # Find and update the graph
                        graphs = await document_service.db.list_graphs(auth)
                        for graph in graphs:
                            if graph.name == graph_name or workflow_id in graph.system_metadata.get("workflow_id", ""):
                                graph.system_metadata["status"] = "completed"
                                # Clear workflow tracking data
                                graph.system_metadata.pop("workflow_id", None)
                                graph.system_metadata.pop("run_id", None)
                                await document_service.db.update_graph(graph)
                                break
                    except Exception as e:
                        logger.warning(f"Failed to update graph status after workflow completion: {e}")

            return result
        else:
            # For local graph service, workflows complete synchronously
            return {"status": "completed", "result": {"message": "Local graph operations complete synchronously"}}

    except Exception as e:
        logger.error(f"Error checking workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/local/generate_uri", include_in_schema=True)
async def generate_local_uri(
    name: str = Form("admin"),
    expiry_days: int = Form(30),
) -> Dict[str, str]:
    """Generate a development URI for running Morphik locally.

    Args:
        name: Developer name to embed in the token payload.
        expiry_days: Number of days the generated token should remain valid.

    Returns:
        A dictionary containing the ``uri`` that can be used to connect to the
        local instance.
    """
    try:
        # Clean name
        name = name.replace(" ", "_").lower()

        # Create payload
        payload = {
            "type": "developer",
            "entity_id": name,
            "permissions": ["read", "write", "admin"],
            "exp": datetime.now(UTC) + timedelta(days=expiry_days),
        }

        # Generate token
        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

        # Read config for host/port
        with open("morphik.toml", "rb") as f:
            config = tomli.load(f)
        base_url = f"{config['api']['host']}:{config['api']['port']}".replace("localhost", "127.0.0.1")

        # Generate URI
        uri = f"morphik://{name}:{token}@{base_url}"
        return {"uri": uri}
    except Exception as e:
        logger.error(f"Error generating local URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cloud/generate_uri", include_in_schema=True)
async def generate_cloud_uri(
    request: GenerateUriRequest,
    authorization: str = Header(None),
) -> Dict[str, str]:
    """Generate an authenticated URI for a cloud-hosted Morphik application.

    Args:
        request: Parameters for URI generation including ``app_id`` and ``name``.
        authorization: Bearer token of the user requesting the URI.

    Returns:
        A dictionary with the generated ``uri`` and associated ``app_id``.
    """
    try:
        app_id = request.app_id
        name = request.name
        user_id = request.user_id
        expiry_days = request.expiry_days

        logger.debug(f"Generating cloud URI for app_id={app_id}, name={name}, user_id={user_id}")

        # Verify authorization header before proceeding
        if not authorization:
            logger.warning("Missing authorization header")
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify the token is valid
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]  # Remove "Bearer "

        try:
            # Decode the token to ensure it's valid
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

            # Only allow users to create apps for themselves (or admin)
            token_user_id = payload.get("user_id")
            logger.debug(f"Token user ID: {token_user_id}")
            logger.debug(f"User ID: {user_id}")
            if not (token_user_id == user_id or "admin" in payload.get("permissions", [])):
                raise HTTPException(
                    status_code=403,
                    detail="You can only create apps for your own account unless you have admin permissions",
                )
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=str(e))

        # Import UserService here to avoid circular imports
        from core.services.user_service import UserService

        user_service = UserService()

        # Initialize user service if needed
        await user_service.initialize()

        # Clean name
        name = name.replace(" ", "_").lower()

        # Check if the user is within app limit and generate URI
        uri = await user_service.generate_cloud_uri(user_id, app_id, name, expiry_days)

        if not uri:
            logger.debug("Application limit reached for this account tier with user_id: %s", user_id)
            raise HTTPException(status_code=403, detail="Application limit reached for this account tier")

        return {"uri": uri, "app_id": app_id}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating cloud URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folders/{folder_id}/set_rule")
@telemetry.track(operation_type="set_folder_rule", metadata_resolver=telemetry.set_folder_rule_metadata)
async def set_folder_rule(
    folder_id: str,
    request: SetFolderRuleRequest,
    auth: AuthContext = Depends(verify_token),
    apply_to_existing: bool = True,
):
    """
    Set extraction rules for a folder.

    Args:
        folder_id: ID of the folder to set rules for
        request: SetFolderRuleRequest containing metadata extraction rules
        auth: Authentication context
        apply_to_existing: Whether to apply rules to existing documents in the folder

    Returns:
        Success status with processing results
    """
    # Import text here to ensure it's available in this function's scope
    from sqlalchemy import text

    try:
        # Log detailed information about the rules
        logger.debug(f"Setting rules for folder {folder_id}")
        logger.debug(f"Number of rules: {len(request.rules)}")

        for i, rule in enumerate(request.rules):
            logger.debug(f"\nRule {i + 1}:")
            logger.debug(f"Type: {rule.type}")
            logger.debug("Schema:")
            for field_name, field_config in rule.schema.items():
                logger.debug(f"  Field: {field_name}")
                logger.debug(f"    Type: {field_config.get('type', 'unknown')}")
                logger.debug(f"    Description: {field_config.get('description', 'No description')}")
                if "schema" in field_config:
                    logger.debug("    Has JSON schema: Yes")
                    logger.debug(f"    Schema: {field_config['schema']}")

        # Get the folder
        folder = await document_service.db.get_folder(folder_id, auth)
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        # Check if user has write access to the folder
        if not document_service.db._check_folder_access(folder, auth, "write"):
            raise HTTPException(status_code=403, detail="You don't have write access to this folder")

        # Update folder with rules
        # Convert rules to dicts for JSON serialization
        rules_dicts = [rule.model_dump() for rule in request.rules]

        # Update the folder in the database
        async with document_service.db.async_session() as session:
            # Execute update query
            await session.execute(
                text(
                    """
                    UPDATE folders
                    SET rules = :rules
                    WHERE id = :folder_id
                    """
                ),
                {"folder_id": folder_id, "rules": json.dumps(rules_dicts)},
            )
            await session.commit()

        logger.info(f"Successfully updated folder {folder_id} with {len(request.rules)} rules")

        # Get updated folder
        updated_folder = await document_service.db.get_folder(folder_id, auth)

        # If apply_to_existing is True, apply these rules to all existing documents in the folder
        processing_results = {"processed": 0, "errors": []}

        if apply_to_existing and folder.document_ids:
            logger.info(f"Applying rules to {len(folder.document_ids)} existing documents in folder")

            # Import rules processor

            # Get all documents in the folder
            documents = await document_service.db.get_documents_by_id(folder.document_ids, auth)

            # Process each document
            for doc in documents:
                try:
                    # Get document content
                    logger.info(f"Processing document {doc.external_id}")

                    # For each document, apply the rules from the folder
                    doc_content = None

                    # Get content from system_metadata if available
                    if doc.system_metadata and "content" in doc.system_metadata:
                        doc_content = doc.system_metadata["content"]
                        logger.info(f"Retrieved content from system_metadata for document {doc.external_id}")

                    # If we still have no content, log error and continue
                    if not doc_content:
                        error_msg = f"No content found in system_metadata for document {doc.external_id}"
                        logger.error(error_msg)
                        processing_results["errors"].append({"document_id": doc.external_id, "error": error_msg})
                        continue

                    # Process document with rules
                    try:
                        # Convert request rules to actual rule models and apply them
                        from core.models.rules import MetadataExtractionRule

                        for rule_request in request.rules:
                            if rule_request.type == "metadata_extraction":
                                # Create the actual rule model
                                rule = MetadataExtractionRule(type=rule_request.type, schema=rule_request.schema)

                                # Apply the rule with retries
                                max_retries = 3
                                base_delay = 1  # seconds
                                extracted_metadata = None
                                last_error = None

                                for retry_count in range(max_retries):
                                    try:
                                        if retry_count > 0:
                                            # Exponential backoff
                                            delay = base_delay * (2 ** (retry_count - 1))
                                            logger.info(f"Retry {retry_count}/{max_retries} after {delay}s delay")
                                            await asyncio.sleep(delay)

                                        extracted_metadata, _ = await rule.apply(doc_content, {})
                                        logger.info(
                                            f"Successfully extracted metadata on attempt {retry_count + 1}: "
                                            f"{extracted_metadata}"
                                        )
                                        break  # Success, exit retry loop

                                    except Exception as rule_apply_error:
                                        last_error = rule_apply_error
                                        logger.warning(
                                            f"Metadata extraction attempt {retry_count + 1} failed: "
                                            f"{rule_apply_error}"
                                        )
                                        if retry_count == max_retries - 1:  # Last attempt
                                            logger.error(f"All {max_retries} metadata extraction attempts failed")
                                            processing_results["errors"].append(
                                                {
                                                    "document_id": doc.external_id,
                                                    "error": f"Failed to extract metadata after {max_retries} "
                                                    f"attempts: {str(last_error)}",
                                                }
                                            )
                                            continue  # Skip to next document

                                # Update document metadata if extraction succeeded
                                if extracted_metadata:
                                    # Merge new metadata with existing
                                    doc.metadata.update(extracted_metadata)

                                    # Create an updates dict that only updates metadata
                                    # We need to create system_metadata with all preserved fields
                                    # Note: In the database, metadata is stored as 'doc_metadata', not 'metadata'
                                    updates = {
                                        "doc_metadata": doc.metadata,  # Use doc_metadata for the database
                                        "system_metadata": {},  # Will be merged with existing in update_document
                                    }

                                    # Explicitly preserve the content field in system_metadata
                                    if "content" in doc.system_metadata:
                                        updates["system_metadata"]["content"] = doc.system_metadata["content"]

                                    # Log the updates we're making
                                    logger.info(
                                        f"Updating document {doc.external_id} with metadata: {extracted_metadata}"
                                    )
                                    logger.info(f"Full metadata being updated: {doc.metadata}")
                                    logger.info(f"Update object being sent to database: {updates}")
                                    logger.info(
                                        f"Preserving content in system_metadata: {'content' in doc.system_metadata}"
                                    )

                                    # Update document in database
                                    app_db = document_service.db
                                    success = await app_db.update_document(doc.external_id, updates, auth)

                                    if success:
                                        logger.info(f"Updated metadata for document {doc.external_id}")
                                        processing_results["processed"] += 1
                                    else:
                                        logger.error(f"Failed to update metadata for document {doc.external_id}")
                                        processing_results["errors"].append(
                                            {
                                                "document_id": doc.external_id,
                                                "error": "Failed to update document metadata",
                                            }
                                        )
                    except Exception as rule_error:
                        logger.error(f"Error processing rules for document {doc.external_id}: {rule_error}")
                        processing_results["errors"].append(
                            {
                                "document_id": doc.external_id,
                                "error": f"Error processing rules: {str(rule_error)}",
                            }
                        )

                except Exception as doc_error:
                    logger.error(f"Error processing document {doc.external_id}: {doc_error}")
                    processing_results["errors"].append({"document_id": doc.external_id, "error": str(doc_error)})

            return {
                "status": "success",
                "message": "Rules set successfully",
                "folder_id": folder_id,
                "rules": updated_folder.rules,
                "processing_results": processing_results,
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error setting folder rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Cloud – delete application (control-plane only)
# ---------------------------------------------------------------------------


@app.delete("/cloud/apps")
async def delete_cloud_app(
    app_name: str = Query(..., description="Name of the application to delete"),
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Delete all resources associated with a given cloud application.

    Args:
        app_name: Name of the application whose data should be removed.
        auth: Authentication context of the requesting user.

    Returns:
        A summary describing how many documents and folders were removed.
    """

    user_id = auth.user_id or auth.entity_id
    logger.info(f"Deleting app {app_name} for user {user_id}")

    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select

    from core.models.apps import AppModel
    from core.services.user_service import UserService

    # 1) Resolve app_id from apps table ----------------------------------
    async with document_service.db.async_session() as session:
        stmt = select(AppModel).where(AppModel.user_id == user_id, AppModel.name == app_name)
        res = await session.execute(stmt)
        app_row = res.scalar_one_or_none()

    if app_row is None:
        raise HTTPException(status_code=404, detail="Application not found")

    app_id = app_row.app_id

    # ------------------------------------------------------------------
    # Create an AuthContext scoped to *this* application so that the
    # underlying access-control filters in the database layer allow us to
    # see and delete resources that belong to the app – even if the JWT
    # used to call this endpoint was scoped to a *different* app.
    # ------------------------------------------------------------------

    if auth.entity_type == EntityType.DEVELOPER:
        app_auth = AuthContext(
            entity_type=auth.entity_type,
            entity_id=auth.entity_id,
            app_id=app_id,
            permissions=auth.permissions or {"read", "write", "admin"},
            user_id=auth.user_id,
        )
    else:
        app_auth = auth

    # 2) Delete all documents for this app ------------------------------
    # ------------------------------------------------------------------
    # Fetch ALL documents for *this* app using the app-scoped auth.
    # ------------------------------------------------------------------
    doc_ids = await document_service.db.find_authorized_and_filtered_documents(app_auth)

    deleted = 0
    for doc_id in doc_ids:
        try:
            await document_service.delete_document(doc_id, app_auth)
            deleted += 1
        except Exception as exc:
            logger.warning("Failed to delete document %s for app %s: %s", doc_id, app_id, exc)

    # 3) Delete folders associated with this app -----------------------
    # ------------------------------------------------------------------
    # Fetch ALL folders for *this* app using the same app-scoped auth.
    # ------------------------------------------------------------------
    folder_ids_deleted = 0
    folders = await document_service.db.list_folders(app_auth)

    for folder in folders:
        try:
            await document_service.db.delete_folder(folder.id, app_auth)
            folder_ids_deleted += 1
        except Exception as f_exc:  # noqa: BLE001
            logger.warning("Failed to delete folder %s for app %s: %s", folder.id, app_id, f_exc)

    # 4) Remove apps table entry ---------------------------------------
    async with document_service.db.async_session() as session:
        await session.execute(sa_delete(AppModel).where(AppModel.app_id == app_id))
        await session.commit()

    # 5) Update user_limits --------------------------------------------
    user_service = UserService()
    await user_service.initialize()
    await user_service.unregister_app(user_id, app_id)

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
    """List chat conversations available to the current user.

    Args:
        auth: Authentication context containing user and app identifiers.
        limit: Maximum number of conversations to return (1-500)

    Returns:
        A list of dictionaries describing each conversation, ordered by most
        recent activity.
    """
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
