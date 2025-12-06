import logging
import secrets
import time  # Add time import for profiling
from typing import Dict, List, Optional

import jwt
import sentry_sdk
from fastapi import Depends, FastAPI, Header, HTTPException
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from core.app_factory import lifespan
from core.auth_utils import verify_token
from core.config import get_settings
from core.logging_config import setup_logging
from core.models.auth import AuthContext
from core.models.documents import ChunkResult, Document
from core.models.request import BatchChunksRequest, GenerateUriRequest, RetrieveRequest, SearchDocumentsRequest
from core.routes.documents import router as documents_router
from core.routes.folders import router as folders_router
from core.routes.health import router as health_router
from core.routes.ingest import router as ingest_router
from core.services.telemetry import TelemetryService
from core.services_init import document_service, ingestion_service
from core.utils.folder_utils import normalize_folder_name

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

# NOTE FOR AI AND OTHER HUMANS:
# THIS IS SUPPOSED TO COMMENTED OUT
# - REQUESTS FROM TYPESCRIPT SDK
#  WILL FAIL IF YOU CHANGE THIS

# app.add_middleware(ProfilingMiddleware)

# # Add CORS middleware (same behaviour as before refactor)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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

# if settings.MODE == "cloud":
#     app.add_middleware(
#         SessionMiddleware,
#         secret_key=settings.SESSION_SECRET_KEY,
#         same_site="none",
#         https_only=True,
#     )
# else:
#     app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY)


def _validate_admin_secret(admin_secret: Optional[str]) -> bool:
    """Return True if the provided admin secret is valid, otherwise raise."""
    if not admin_secret:
        return False
    if not settings.ADMIN_SERVICE_SECRET:
        raise HTTPException(status_code=403, detail="Admin secret authentication is not configured")
    if not secrets.compare_digest(admin_secret, settings.ADMIN_SERVICE_SECRET):
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    return True


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
        # Normalize folder_name if needed
        normalized_folder_name = normalize_folder_name(request.folder_name) if request.folder_name else None

        results = await document_service.search_documents_by_name(
            query=request.query,
            auth=auth,
            limit=request.limit,
            filters=request.filters,
            folder_name=normalized_folder_name,
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

        normalized_folder_name = normalize_folder_name(request.folder_name) if request.folder_name is not None else None

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_chunks")
        results = await document_service.batch_retrieve_chunks(
            request.sources,
            auth,
            normalized_folder_name,
            request.end_user_id,
            request.use_colpali,
            request.output_format or "base64",
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(request.sources)} chunks")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


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

        logger.debug(
            "Generating cloud URI for app_id=%s, name=%s, user_id=%s (admin_header=%s)",
            app_id,
            name,
            user_id,
            bool(admin_secret),
        )

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
        uri = await user_service.generate_cloud_uri(
            user_id,
            app_id,
            name,
            expiry_days,
            org_id=request.org_id,
            created_by_user_id=request.created_by_user_id,
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
