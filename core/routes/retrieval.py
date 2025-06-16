import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

import arq
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from core.agent import MorphikAgent
from core.auth_utils import verify_token
from core.config import get_settings
from core.dependencies import get_redis_pool
from core.limits_utils import check_and_increment_limits
from core.models.auth import AuthContext
from core.models.chunk import ChunkResult
from core.models.completion import ChunkSource, CompletionResponse
from core.models.documents import Document, DocumentResult
from core.models.prompts import (
    validate_prompt_overrides_with_http_exception,
)
from core.models.request import (
    AgentQueryRequest,
    CompletionQueryRequest,
    RetrieveRequest,
)
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# ---------------------------------------------------------------------------
# Router initialisation & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Retrieval"])
logger = logging.getLogger(__name__)
settings = get_settings()
telemetry = TelemetryService()

# Single MorphikAgent instance (tool definitions cached)
morphik_agent = MorphikAgent(document_service=document_service)


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


# Helper function to normalize folder_name parameter
def normalize_folder_name(
    folder_name: Optional[Union[str, List[str]]]
) -> Optional[Union[str, List[str]]]:
    """Convert string 'null' to None for folder_name parameter."""
    if folder_name is None:
        return None
    if isinstance(folder_name, str):
        return None if folder_name.lower() == "null" else folder_name
    if isinstance(folder_name, list):
        return [None if f.lower() == "null" else f for f in folder_name]
    return folder_name


# ---------------------------------------------------------------------------
# Retrieval endpoints
# ---------------------------------------------------------------------------


@router.post("/chunks", response_model=List[ChunkResult])
@telemetry.track(
    operation_type="retrieve_chunks",
    metadata_resolver=telemetry.retrieve_chunks_metadata
)
async def retrieve_chunks(
    request: RetrieveRequest,
    auth: AuthContext = Depends(verify_token)
):
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


@router.post("/docs", response_model=List[DocumentResult])
@telemetry.track(
    operation_type="retrieve_docs",
    metadata_resolver=telemetry.retrieve_docs_metadata
)
async def retrieve_documents(
    request: RetrieveRequest,
    auth: AuthContext = Depends(verify_token)
):
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


# ---------------------------------------------------------------------------
# Batch retrieval endpoints
# ---------------------------------------------------------------------------


@router.post("/batch/documents", response_model=List[Document])
@telemetry.track(
    operation_type="batch_get_documents",
    metadata_resolver=telemetry.batch_documents_metadata
)
async def batch_get_documents(
    request: Dict[str, Any],
    auth: AuthContext = Depends(verify_token)
):
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
        results = await document_service.batch_retrieve_documents(
            document_ids, auth, folder_name, end_user_id
        )

        # Log consolidated performance summary
        perf.log_summary(
            f"Retrieved {len(results)}/{len(document_ids)} documents"
        )

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/batch/chunks", response_model=List[ChunkResult])
@telemetry.track(
    operation_type="batch_get_chunks",
    metadata_resolver=telemetry.batch_chunks_metadata
)
async def batch_get_chunks(
    request: Dict[str, Any],
    auth: AuthContext = Depends(verify_token)
):
    """
    Retrieve specific chunks by their document ID and chunk number.

    Args:
        request: Dictionary containing:
            - sources: List of ChunkSource objects
              (with document_id and chunk_number)
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


@router.post("/query", response_model=CompletionResponse)
@telemetry.track(
    operation_type="query", metadata_resolver=telemetry.query_metadata
)
async def query_completion(
    request: CompletionQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """
    Generate completion using relevant chunks as context.

    When graph_name is provided, the query will leverage the knowledge graph
    to enhance retrieval by finding relevant entities and their connected
    documents.

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
            - graph_name: Optional name of the graph to use for knowledge
              graph-enhanced retrieval
            - hop_depth: Number of relationship hops to traverse in the
              graph (1-3)
            - include_paths: Whether to include relationship paths in the
              response
            - prompt_overrides: Optional customizations for entity extraction,
              resolution, and query prompts
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
            - schema: Optional schema for structured output
            - chat_id: Optional chat conversation identifier for maintaining
              history
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
            validate_prompt_overrides_with_http_exception(
                request.prompt_overrides, operation_type="query"
            )

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
                db_hist = await document_service.db.get_chat_history(
                    request.chat_id, auth.user_id, auth.app_id
                )
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
                        time_to_first_token = (
                            first_token_time - perf.start_time
                        )
                        perf.add_suboperation(
                            "completion_start_to_first_token",
                            time_to_first_token
                        )
                        logger.info(
                            f"Completion start to first token: "
                            f"{time_to_first_token:.2f}s"
                        )

                    full_content += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                # Convert sources to the format expected by frontend
                sources_info = [
                    {
                        "document_id": source.document_id,
                        "chunk_number": source.chunk_number,
                        "score": source.score,
                    }
                    for source in sources
                ]

                # Send completion signal with sources
                sources_data = {"done": True, "sources": sources_info}
                yield f"data: {json.dumps(sources_data)}\n\n"

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
                streaming_time = (
                    time.time() - first_token_time
                    if first_token_time else 0
                )
                perf.add_suboperation("streaming_duration", streaming_time)
                perf.log_summary(
                    f"Generated streaming completion with "
                    f"{len(sources)} sources"
                )

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            # For non-streaming responses,
            # result is just the CompletionResponse
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
            sources_count = len(response.sources) if response.sources else 0
            perf.log_summary(
                f"Generated completion with {sources_count} sources"
            )

            return response
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(
            operation_type="query", error=e
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/agent", response_model=Dict[str, Any])
@telemetry.track(operation_type="agent_query")
async def agent_query(
    request: AgentQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Execute an agent-style query using the :class:`MorphikAgent`.

    Args:
        request: The query payload containing the natural language question
            and optional chat_id.
        auth: Authentication context used to enforce limits and
            access control.
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
            db_hist = await document_service.db.get_chat_history(
                request.chat_id, auth.user_id, auth.app_id
            )
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

    # Use the shared MorphikAgent instance;
    # per-run state is now isolated internally
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