import asyncio
import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arq
import jwt
import tomli
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.sessions import SessionMiddleware

from core.agent import MorphikAgent
from core.auth_utils import verify_token
from core.cache.llama_cache_factory import LlamaCacheFactory
from core.completion.litellm_completion import LiteLLMCompletionModel
from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.dependencies import get_redis_pool
from core.embedding.colpali_api_embedding_model import ColpaliApiEmbeddingModel
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.limits_utils import check_and_increment_limits
from core.models.auth import AuthContext
from core.models.completion import ChunkSource, CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentResult
from core.models.folders import Folder, FolderCreate
from core.models.graph import Graph
from core.models.prompts import validate_prompt_overrides_with_http_exception
from core.models.request import (
    AgentQueryRequest,
    BatchIngestResponse,
    CompletionQueryRequest,
    CreateGraphRequest,
    GenerateUriRequest,
    IngestTextRequest,
    RetrieveRequest,
    SetFolderRuleRequest,
    UpdateGraphRequest,
)
from core.parser.morphik_parser import MorphikParser
from core.reranker.flag_reranker import FlagReranker
from core.services.document_service import DocumentService
from core.services.telemetry import TelemetryService
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.vector_store.multi_vector_store import MultiVectorStore
from core.vector_store.pgvector_store import PGVectorStore

# Initialize FastAPI app
logger = logging.getLogger(__name__)

# Global variable for redis_pool, primarily for shutdown if app.state access fails.
_global_redis_pool: Optional[arq.ArqRedis] = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # --- BEGIN MOVED STARTUP LOGIC ---
    logger.info("Lifespan: Initializing Database...")
    try:
        success = await database.initialize()
        if success:
            logger.info("Lifespan: Database initialization successful")
        else:
            logger.error("Lifespan: Database initialization failed")
    except Exception as e:
        logger.error(f"Lifespan: CRITICAL - Failed to initialize Database: {e}", exc_info=True)
        raise  # Or handle more gracefully if appropriate

    logger.info("Lifespan: Initializing Vector Store...")
    try:
        global vector_store
        if hasattr(vector_store, "initialize"):
            await vector_store.initialize()
        logger.info("Lifespan: Vector Store initialization successful (or not applicable).")
    except Exception as e:
        logger.error(f"Lifespan: CRITICAL - Failed to initialize Vector Store: {e}", exc_info=True)
        # Decide if this is fatal
        # raise

    logger.info("Lifespan: Attempting to initialize Redis connection pool...")
    global _global_redis_pool  # Ensure we're using the global for assignment
    try:
        redis_settings_obj = arq.connections.RedisSettings(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
        )
        logger.info(f"Lifespan: Redis settings for pool: host={settings.REDIS_HOST}, port={settings.REDIS_PORT}")
        current_redis_pool = await arq.create_pool(redis_settings_obj)
        if current_redis_pool:
            app_instance.state.redis_pool = current_redis_pool  # Use app_instance from lifespan
            _global_redis_pool = current_redis_pool
            logger.info("Lifespan: Successfully initialized Redis connection pool and stored on app.state.")
        else:
            logger.error("Lifespan: arq.create_pool returned None or a falsey value for Redis pool.")
            raise RuntimeError("Lifespan: Failed to create Redis pool - arq.create_pool returned non-truthy value.")
    except Exception as e:
        logger.error(f"Lifespan: CRITICAL - Failed to initialize Redis connection pool: {e}", exc_info=True)
        raise RuntimeError(f"Lifespan: CRITICAL - Failed to initialize Redis connection pool: {e}") from e
    # --- END MOVED STARTUP LOGIC ---

    logger.info("Lifespan: Core startup logic executed.")
    yield
    # Shutdown logic
    logger.info("Lifespan: Shutdown initiated.")
    # Use app_instance.state to get the pool for shutdown too
    pool_to_close = getattr(app_instance.state, "redis_pool", _global_redis_pool)
    if pool_to_close:
        logger.info("Closing Redis connection pool from lifespan...")
        pool_to_close.close()
        await pool_to_close.wait_closed()
        logger.info("Redis connection pool closed from lifespan.")
    logger.info("Lifespan: Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# Add CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize telemetry
telemetry = TelemetryService()

# Add OpenTelemetry instrumentation - exclude HTTP send/receive spans
FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls="health,health/.*",  # Exclude health check endpoints
    exclude_spans=["send", "receive"],  # Exclude HTTP send/receive spans to reduce telemetry volume
    http_capture_headers_server_request=None,  # Don't capture request headers
    http_capture_headers_server_response=None,  # Don't capture response headers
    tracer_provider=None,  # Use the global tracer provider
)

# Initialize service
settings = get_settings()

# Add SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY)

# Initialize database
if not settings.POSTGRES_URI:
    raise ValueError("PostgreSQL URI is required for PostgreSQL database")
database = PostgresDatabase(uri=settings.POSTGRES_URI)

# Initialize vector store
if not settings.POSTGRES_URI:
    raise ValueError("PostgreSQL URI is required for pgvector store")

vector_store = PGVectorStore(
    uri=settings.POSTGRES_URI,
)

# Initialize storage
match settings.STORAGE_PROVIDER:
    case "local":
        storage = LocalStorage(storage_path=settings.STORAGE_PATH)
    case "aws-s3":
        if not settings.AWS_ACCESS_KEY or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS credentials are required for S3 storage")
        storage = S3Storage(
            aws_access_key=settings.AWS_ACCESS_KEY,
            aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            default_bucket=settings.S3_BUCKET,
        )
    case _:
        raise ValueError(f"Unsupported storage provider: {settings.STORAGE_PROVIDER}")

# Initialize parser
parser = MorphikParser(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    use_unstructured_api=settings.USE_UNSTRUCTURED_API,
    unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
    assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
    anthropic_api_key=settings.ANTHROPIC_API_KEY,
    use_contextual_chunking=settings.USE_CONTEXTUAL_CHUNKING,
)

# Initialize embedding model
# Create a LiteLLM model using the registered model config
embedding_model = LiteLLMEmbeddingModel(
    model_key=settings.EMBEDDING_MODEL,
)
logger.info(f"Initialized LiteLLM embedding model with model key: {settings.EMBEDDING_MODEL}")

# Initialize completion model
# Create a LiteLLM model using the registered model config
completion_model = LiteLLMCompletionModel(
    model_key=settings.COMPLETION_MODEL,
)
logger.info(f"Initialized LiteLLM completion model with model key: {settings.COMPLETION_MODEL}")

# Initialize reranker
reranker = None
if settings.USE_RERANKING:
    match settings.RERANKER_PROVIDER:
        case "flag":
            reranker = FlagReranker(
                model_name=settings.RERANKER_MODEL,
                device=settings.RERANKER_DEVICE,
                use_fp16=settings.RERANKER_USE_FP16,
                query_max_length=settings.RERANKER_QUERY_MAX_LENGTH,
                passage_max_length=settings.RERANKER_PASSAGE_MAX_LENGTH,
            )
        case _:
            raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")

# Initialize cache factory
cache_factory = LlamaCacheFactory(Path(settings.STORAGE_PATH))

# Initialize ColPali embedding model per mode (off/local/api)
match settings.COLPALI_MODE:
    case "off":
        colpali_embedding_model = None
        colpali_vector_store = None
    case "local":
        colpali_embedding_model = ColpaliEmbeddingModel()
        colpali_vector_store = MultiVectorStore(uri=settings.POSTGRES_URI)
    case "api":
        colpali_embedding_model = ColpaliApiEmbeddingModel()
        colpali_vector_store = MultiVectorStore(uri=settings.POSTGRES_URI)
    case _:
        raise ValueError(f"Unsupported COLPALI_MODE: {settings.COLPALI_MODE}")

# Initialize document service with configured components
document_service = DocumentService(
    database=database,
    vector_store=vector_store,
    storage=storage,
    parser=parser,
    embedding_model=embedding_model,
    completion_model=completion_model,
    cache_factory=cache_factory,
    reranker=reranker,
    enable_colpali=settings.ENABLE_COLPALI,
    colpali_embedding_model=colpali_embedding_model,
    colpali_vector_store=colpali_vector_store,
)
# Store document_service on app.state immediately after it's created.
# This must happen before _init_ee_app(app) is called.
app.state.document_service = document_service
logger.info("Document service initialized and stored on app.state")

# Initialize the MorphikAgent once to load tool definitions and avoid repeated I/O
morphik_agent = MorphikAgent(document_service=document_service)

# ---------------------------------------------------------------------------
# Mount enterprise-only routes when the proprietary ``ee`` package
# is present.
# ---------------------------------------------------------------------------

try:
    from ee.routers import init_app as _init_ee_app  # type: ignore

    _init_ee_app(app)  # noqa: SLF001 – runtime extension
except ModuleNotFoundError:
    # Expected in OSS builds – silently ignore.
    logger.debug("Enterprise package not found – running in community mode.")
except ImportError as e:
    logger.error(f"Failed to import init_app from ee.routers: {e}", exc_info=True)
except Exception as e:
    logger.error(f"An unexpected error occurred during EE app initialization: {e}", exc_info=True)


@app.post("/ingest/text", response_model=Document)
@telemetry.track(operation_type="ingest_text", metadata_resolver=telemetry.ingest_text_metadata)
async def ingest_text(
    request: IngestTextRequest,
    auth: AuthContext = Depends(verify_token),
) -> Document:
    """
    Ingest a text document.

    Args:
        request: IngestTextRequest containing:
            - content: Text content to ingest
            - filename: Optional filename to help determine content type
            - metadata: Optional metadata dictionary
            - rules: Optional list of rules. Each rule should be either:
                   - MetadataExtractionRule: {"type": "metadata_extraction", "schema": {...}}
                   - NaturalLanguageRule: {"type": "natural_language", "prompt": "..."}
            - folder_name: Optional folder to scope the document to
            - end_user_id: Optional end-user ID to scope the document to
        auth: Authentication context

    Returns:
        Document: Metadata of ingested document
    """
    try:
        return await document_service.ingest_text(
            content=request.content,
            filename=request.filename,
            metadata=request.metadata,
            rules=request.rules,
            use_colpali=request.use_colpali,
            auth=auth,
            folder_name=request.folder_name,
            end_user_id=request.end_user_id,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/ingest/file", response_model=Document)
@telemetry.track(operation_type="queue_ingest_file", metadata_resolver=telemetry.ingest_file_metadata)
async def ingest_file(
    file: UploadFile,
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    auth: AuthContext = Depends(verify_token),
    use_colpali: Optional[bool] = None,
    folder_name: Optional[str] = Form(None),
    end_user_id: Optional[str] = Form(None),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> Document:
    """
    Ingest a file document asynchronously.

    Args:
        file: File to ingest
        metadata: JSON string of metadata
        rules: JSON string of rules list. Each rule should be either:
               - MetadataExtractionRule: {"type": "metadata_extraction", "schema": {...}}
               - NaturalLanguageRule: {"type": "natural_language", "prompt": "..."}
        auth: Authentication context
        use_colpali: Whether to use ColPali embedding model
        folder_name: Optional folder to scope the document to
        end_user_id: Optional end-user ID to scope the document to
        redis: Redis connection pool for background tasks

    Returns:
        Document with processing status that can be used to check progress
    """
    try:
        # Parse metadata and rules
        metadata_dict = json.loads(metadata)
        rules_list = json.loads(rules)

        # Fix bool conversion: ensure string "false" is properly converted to False
        def str2bool(v):
            return v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}

        use_colpali = str2bool(use_colpali)

        # Ensure user has write permission
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        logger.debug(f"API: Queueing file ingestion with use_colpali: {use_colpali}")

        # Create a document with processing status
        doc = Document(
            content_type=file.content_type,
            filename=file.filename,
            metadata=metadata_dict,
            owner={"type": auth.entity_type.value, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
                "user_id": [auth.user_id if auth.user_id else []],
                "app_access": ([auth.app_id] if auth.app_id else []),
            },
            system_metadata={"status": "processing"},
        )

        # Add folder_name and end_user_id to system_metadata if provided
        if folder_name:
            doc.system_metadata["folder_name"] = folder_name
        if end_user_id:
            doc.system_metadata["end_user_id"] = end_user_id
        if auth.app_id:
            doc.system_metadata["app_id"] = auth.app_id

        # Set processing status
        doc.system_metadata["status"] = "processing"

        # Store the document in the *per-app* database that verify_token has
        # already routed to (document_service.db).  Using the global
        # *database* here would put the row into the control-plane DB and the
        # ingestion worker – which connects to the per-app DB – would never
        # find it.
        app_db = document_service.db

        success = await app_db.store_document(doc)
        if not success:
            raise Exception("Failed to store document metadata")

        # If folder_name is provided, ensure the folder exists and add document to it
        if folder_name:
            try:
                await document_service._ensure_folder_exists(folder_name, doc.external_id, auth)
                logger.debug(f"Ensured folder '{folder_name}' exists and contains document {doc.external_id}")
            except Exception as e:
                # Log error but don't raise - we want document ingestion to continue even if folder operation fails
                logger.error(f"Error ensuring folder exists: {e}")

        # Read file content
        file_content = await file.read()

        # Generate a unique key for the file
        file_key = f"ingest_uploads/{uuid.uuid4()}/{file.filename}"

        # Store the file in the dedicated bucket for this app (if any)
        file_content_base64 = base64.b64encode(file_content).decode()

        bucket_override = await document_service._get_bucket_for_app(auth.app_id)

        bucket, stored_key = await storage.upload_from_base64(
            file_content_base64,
            file_key,
            file.content_type,
            bucket=bucket_override or "",
        )
        logger.debug(f"Stored file in bucket {bucket} with key {stored_key}")

        # Update document with storage info
        doc.storage_info = {"bucket": bucket, "key": stored_key}

        # Initialize storage_files array with the first file
        from datetime import UTC, datetime

        from core.models.documents import StorageFileInfo

        # Create a StorageFileInfo for the initial file
        initial_file_info = StorageFileInfo(
            bucket=bucket,
            key=stored_key,
            version=1,
            filename=file.filename,
            content_type=file.content_type,
            timestamp=datetime.now(UTC),
        )
        doc.storage_files = [initial_file_info]

        # Log storage files
        logger.debug(f"Initial storage_files for {doc.external_id}: {doc.storage_files}")

        # Update both storage_info and storage_files
        await app_db.update_document(
            document_id=doc.external_id,
            updates={"storage_info": doc.storage_info, "storage_files": doc.storage_files},
            auth=auth,
        )

        # Convert auth context to a dictionary for serialization
        auth_dict = {
            "entity_type": auth.entity_type.value,
            "entity_id": auth.entity_id,
            "app_id": auth.app_id,
            "permissions": list(auth.permissions),
            "user_id": auth.user_id,
        }

        # Enqueue the background job
        job = await redis.enqueue_job(
            "process_ingestion_job",
            document_id=doc.external_id,
            file_key=stored_key,
            bucket=bucket,
            original_filename=file.filename,
            content_type=file.content_type,
            metadata_json=metadata,
            auth_dict=auth_dict,
            rules_list=rules_list,
            use_colpali=use_colpali,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        logger.info(f"File ingestion job queued with ID: {job.job_id} for document: {doc.external_id}")

        return doc
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error during file ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during file ingestion: {str(e)}")


@app.post("/ingest/files", response_model=BatchIngestResponse)
@telemetry.track(operation_type="queue_batch_ingest", metadata_resolver=telemetry.batch_ingest_metadata)
async def batch_ingest_files(
    files: List[UploadFile] = File(...),
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    use_colpali: Optional[bool] = Form(None),
    parallel: Optional[bool] = Form(True),
    folder_name: Optional[str] = Form(None),
    end_user_id: Optional[str] = Form(None),
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
) -> BatchIngestResponse:
    """
    Batch ingest multiple files using the task queue.

    Args:
        files: List of files to ingest
        metadata: JSON string of metadata (either a single dict or list of dicts)
        rules: JSON string of rules list. Can be either:
               - A single list of rules to apply to all files
               - A list of rule lists, one per file
        use_colpali: Whether to use ColPali-style embedding
        folder_name: Optional folder to scope the documents to
        end_user_id: Optional end-user ID to scope the documents to
        auth: Authentication context
        redis: Redis connection pool for background tasks

    Returns:
        BatchIngestResponse containing:
            - documents: List of created documents with processing status
            - errors: List of errors that occurred during the batch operation
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch ingestion")

    try:
        metadata_value = json.loads(metadata)
        rules_list = json.loads(rules)

        # Fix bool conversion: ensure string "false" is properly converted to False
        def str2bool(v):
            return str(v).lower() in {"true", "1", "yes"}

        use_colpali = str2bool(use_colpali)

        # Ensure user has write permission
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # Validate metadata if it's a list
    if isinstance(metadata_value, list) and len(metadata_value) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Number of metadata items ({len(metadata_value)}) must match number of files ({len(files)})",
        )

    # Validate rules if it's a list of lists
    if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list):
        if len(rules_list) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of rule lists ({len(rules_list)}) must match number of files ({len(files)})",
            )

    # Convert auth context to a dictionary for serialization
    auth_dict = {
        "entity_type": auth.entity_type.value,
        "entity_id": auth.entity_id,
        "app_id": auth.app_id,
        "permissions": list(auth.permissions),
        "user_id": auth.user_id,
    }

    created_documents = []

    try:
        for i, file in enumerate(files):
            # Get the metadata and rules for this file
            metadata_item = metadata_value[i] if isinstance(metadata_value, list) else metadata_value
            file_rules = (
                rules_list[i]
                if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list)
                else rules_list
            )

            # Create a document with processing status
            doc = Document(
                content_type=file.content_type,
                filename=file.filename,
                metadata=metadata_item,
                owner={"type": auth.entity_type.value, "id": auth.entity_id},
                access_control={
                    "readers": [auth.entity_id],
                    "writers": [auth.entity_id],
                    "admins": [auth.entity_id],
                    "user_id": [auth.user_id] if auth.user_id else [],
                    "app_access": ([auth.app_id] if auth.app_id else []),
                },
            )

            # Add folder_name and end_user_id to system_metadata if provided
            if folder_name:
                doc.system_metadata["folder_name"] = folder_name
            if end_user_id:
                doc.system_metadata["end_user_id"] = end_user_id
            if auth.app_id:
                doc.system_metadata["app_id"] = auth.app_id

            # Set processing status
            doc.system_metadata["status"] = "processing"

            # Store the document in the *per-app* database that verify_token has
            # already routed to (document_service.db).  Using the global
            # *database* here would put the row into the control-plane DB and the
            # ingestion worker – which connects to the per-app DB – would never
            # find it.
            app_db = document_service.db

            success = await app_db.store_document(doc)
            if not success:
                raise Exception(f"Failed to store document metadata for {file.filename}")

            # If folder_name is provided, ensure the folder exists and add document to it
            if folder_name:
                try:
                    await document_service._ensure_folder_exists(folder_name, doc.external_id, auth)
                    logger.debug(f"Ensured folder '{folder_name}' exists and contains document {doc.external_id}")
                except Exception as e:
                    # Log error but don't raise - we want document ingestion to continue even if folder operation fails
                    logger.error(f"Error ensuring folder exists: {e}")

            # Read file content
            file_content = await file.read()

            # Generate a unique key for the file
            file_key = f"ingest_uploads/{uuid.uuid4()}/{file.filename}"

            # Store the file in the dedicated bucket for this app (if any)
            file_content_base64 = base64.b64encode(file_content).decode()

            bucket_override = await document_service._get_bucket_for_app(auth.app_id)

            bucket, stored_key = await storage.upload_from_base64(
                file_content_base64,
                file_key,
                file.content_type,
                bucket=bucket_override or "",
            )
            logger.debug(f"Stored file in bucket {bucket} with key {stored_key}")

            # Update document with storage info
            doc.storage_info = {"bucket": bucket, "key": stored_key}
            await app_db.update_document(
                document_id=doc.external_id, updates={"storage_info": doc.storage_info}, auth=auth
            )

            # Convert metadata to JSON string for job
            metadata_json = json.dumps(metadata_item)

            # Enqueue the background job
            job = await redis.enqueue_job(
                "process_ingestion_job",
                document_id=doc.external_id,
                file_key=stored_key,
                bucket=bucket,
                original_filename=file.filename,
                content_type=file.content_type,
                metadata_json=metadata_json,
                auth_dict=auth_dict,
                rules_list=file_rules,
                use_colpali=use_colpali,
                folder_name=folder_name,
                end_user_id=end_user_id,
            )

            logger.info(f"File ingestion job queued with ID: {job.job_id} for document: {doc.external_id}")

            # Add document to the list
            created_documents.append(doc)

        # Return information about created documents
        return BatchIngestResponse(documents=created_documents, errors=[])

    except Exception as e:
        logger.error(f"Error queueing batch file ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error queueing batch file ingestion: {str(e)}")


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
    try:
        return await document_service.retrieve_chunks(
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
    try:
        return await document_service.retrieve_docs(
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
    try:
        # Extract document_ids from request
        document_ids = request.get("document_ids", [])
        folder_name = request.get("folder_name")
        end_user_id = request.get("end_user_id")

        if not document_ids:
            return []

        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        return await document_service.batch_retrieve_documents(document_ids, auth, folder_name, end_user_id)
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
        auth: Authentication context

    Returns:
        List[ChunkResult]: List of chunk results
    """
    try:
        # Extract sources from request
        sources = request.get("sources", [])
        folder_name = request.get("folder_name")
        end_user_id = request.get("end_user_id")
        use_colpali = request.get("use_colpali")

        if not sources:
            return []

        # Convert sources to ChunkSource objects if needed
        chunk_sources = []
        for source in sources:
            if isinstance(source, dict):
                chunk_sources.append(ChunkSource(**source))
            else:
                chunk_sources.append(source)

        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        if auth.app_id:
            system_filters["app_id"] = auth.app_id

        return await document_service.batch_retrieve_chunks(chunk_sources, auth, folder_name, end_user_id, use_colpali)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
@telemetry.track(operation_type="query", metadata_resolver=telemetry.query_metadata)
async def query_completion(request: CompletionQueryRequest, auth: AuthContext = Depends(verify_token)):
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
        auth: Authentication context

    Returns:
        CompletionResponse: Generated text completion or structured output
    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="query")

        # Check query limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "query", 1)

        return await document_service.query(
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
        )
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="query", error=e)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/agent", response_model=Dict[str, Any])
@telemetry.track(operation_type="agent_query")
async def agent_query(request: AgentQueryRequest, auth: AuthContext = Depends(verify_token)):
    """
    Process a natural language query using the MorphikAgent and return the response.
    """
    # Check free-tier agent call limits in cloud mode
    if settings.MODE == "cloud" and auth.user_id:
        await check_and_increment_limits(auth, "agent", 1)
    # Use the shared MorphikAgent instance; per-run state is now isolated internally
    response = await morphik_agent.run(request.query, auth)
    # Return the complete response dictionary
    return response


@app.post("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 10000,
    filters: Optional[Dict[str, Any]] = None,
    folder_name: Optional[Union[str, List[str]]] = None,
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
    if folder_name:
        system_filters["folder_name"] = folder_name
    if end_user_id:
        system_filters["end_user_id"] = end_user_id
    if auth.app_id:
        system_filters["app_id"] = auth.app_id

    return await document_service.db.get_documents(auth, skip, limit, filters, system_filters)


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """Get document by ID."""
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
        if folder_name:
            system_filters["folder_name"] = folder_name
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
    use_colpali: Optional[bool] = None,
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
    """Get usage statistics for the authenticated user."""
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
    """Get recent usage records."""
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
    """Create a new cache with specified configuration."""
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
    """Get cache configuration by name."""
    try:
        exists = await document_service.load_cache(name)
        return {"exists": exists}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/update")
@telemetry.track(operation_type="update_cache", metadata_resolver=telemetry.cache_update_metadata)
async def update_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, bool]:
    """Update cache with new documents matching its filter."""
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
    """Add specific documents to the cache."""
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
    """Query the cache with a prompt."""
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
    """
    Create a graph from documents **asynchronously**.

    Instead of blocking on the potentially slow entity/relationship extraction, we immediately
    create a placeholder graph with `status = "processing"`. A background task then fills in
    entities/relationships and marks the graph as completed.
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
        if request.folder_name:
            system_filters["folder_name"] = request.folder_name
        if request.end_user_id:
            system_filters["end_user_id"] = request.end_user_id

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
        async with telemetry.track_operation(
            operation_type="create_folder",
            user_id=auth.entity_id,
            metadata={
                "name": folder_create.name,
            },
        ):
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
        async with telemetry.track_operation(
            operation_type="list_folders",
            user_id=auth.entity_id,
        ):
            folders = await document_service.db.list_folders(auth)
            return folders
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders/{folder_id}", response_model=Folder)
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
        async with telemetry.track_operation(
            operation_type="get_folder",
            user_id=auth.entity_id,
            metadata={
                "folder_id": folder_id,
            },
        ):
            folder = await document_service.db.get_folder(folder_id, auth)

            if not folder:
                raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

            return folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folders/{folder_id}/documents/{document_id}")
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
        async with telemetry.track_operation(
            operation_type="add_document_to_folder",
            user_id=auth.entity_id,
            metadata={
                "folder_id": folder_id,
                "document_id": document_id,
            },
        ):
            success = await document_service.db.add_document_to_folder(folder_id, document_id, auth)

            if not success:
                raise HTTPException(status_code=500, detail="Failed to add document to folder")

            return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding document to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folders/{folder_id}/documents/{document_id}")
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
        async with telemetry.track_operation(
            operation_type="remove_document_from_folder",
            user_id=auth.entity_id,
            metadata={
                "folder_id": folder_id,
                "document_id": document_id,
            },
        ):
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
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

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
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        return await document_service.db.list_graphs(auth, system_filters)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
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


@app.post("/local/generate_uri", include_in_schema=True)
async def generate_local_uri(
    name: str = Form("admin"),
    expiry_days: int = Form(30),
) -> Dict[str, str]:
    """Generate a local URI for development. This endpoint is unprotected."""
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
    """Generate a URI for cloud hosted applications."""
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
