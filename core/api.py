import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.sessions import SessionMiddleware

from core.app_factory import lifespan
from core.config import get_settings
from core.logging_config import setup_logging
from core.middleware.profiling import ProfilingMiddleware
from core.routes.admin import router as admin_router
from core.routes.cache import router as cache_router
from core.routes.chat import router as chat_router
from core.routes.document import router as document_router
from core.routes.folders import router as folders_router
from core.routes.graph import router as graph_router
from core.routes.ingest import router as ingest_router
from core.routes.retrieval import router as retrieval_router
from core.services_init import document_service

# Set up logging configuration for Docker environment
setup_logging()

# Initialize FastAPI app
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application instance & core initialisation
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

# --------------------------------------------------------
# Optional per-request profiler (ENABLE_PROFILING=1)
# --------------------------------------------------------

app.add_middleware(ProfilingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Core singletons
# ---------------------------------------------------------------------------

# Store on app.state for later access
app.state.document_service = document_service
logger.info("Document service initialized and stored on app.state")

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

# Register ingest router
app.include_router(ingest_router)

# Register document router
app.include_router(document_router)

# Register retrieval router
app.include_router(retrieval_router)

# Register folders router
app.include_router(folders_router)

# Register graph router
app.include_router(graph_router)

# Register cache router
app.include_router(cache_router)

# Register chat router
app.include_router(chat_router)

# Register admin router
app.include_router(admin_router)

# ---------------------------------------------------------------------------
# Enterprise-only routes (optional)
# ---------------------------------------------------------------------------

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
