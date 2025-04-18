import asyncio
import json
import base64
import uuid
from datetime import datetime, UTC, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import jwt
import logging
import arq
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from core.limits_utils import check_and_increment_limits
from core.models.request import GenerateUriRequest, RetrieveRequest, CompletionQueryRequest, IngestTextRequest, CreateGraphRequest, UpdateGraphRequest, BatchIngestResponse, SetFolderRuleRequest
from core.models.completion import ChunkSource, CompletionResponse
from core.models.documents import Document, DocumentResult, ChunkResult
from core.models.graph import Graph
from core.models.auth import AuthContext, EntityType
from core.models.prompts import validate_prompt_overrides_with_http_exception
from core.models.folders import Folder, FolderCreate
from core.parser.morphik_parser import MorphikParser
from core.services.document_service import DocumentService
from core.services.telemetry import TelemetryService
from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.vector_store.multi_vector_store import MultiVectorStore
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.storage.s3_storage import S3Storage
from core.storage.local_storage import LocalStorage
from core.reranker.flag_reranker import FlagReranker
from core.cache.llama_cache_factory import LlamaCacheFactory
import tomli
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Morphik API")
logger = logging.getLogger(__name__)


# Add health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check that verifies the application is initialized."""
    return {
        "status": "ready",
        "components": {
            "database": settings.DATABASE_PROVIDER,
            "vector_store": settings.VECTOR_STORE_PROVIDER,
            "embedding": settings.EMBEDDING_PROVIDER,
            "completion": settings.COMPLETION_PROVIDER,
            "storage": settings.STORAGE_PROVIDER,
        },
    }


# Initialize telemetry
telemetry = TelemetryService()

# Add OpenTelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
settings = get_settings()

# Initialize database
if not settings.POSTGRES_URI:
    raise ValueError("PostgreSQL URI is required for PostgreSQL database")
database = PostgresDatabase(uri=settings.POSTGRES_URI)

# Redis settings already imported at top of file


@app.on_event("startup")
async def initialize_database():
    """Initialize database tables and indexes on application startup."""
    logger.info("Initializing database...")
    success = await database.initialize()
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")
        # We don't raise an exception here to allow the app to continue starting
        # even if there are initialization errors

@app.on_event("startup")
async def initialize_vector_store():
    """Initialize vector store tables and indexes on application startup."""
    # First initialize the primary vector store (PGVectorStore if using pgvector)
    logger.info("Initializing primary vector store...")
    if hasattr(vector_store, 'initialize'):
        success = await vector_store.initialize()
        if success:
            logger.info("Primary vector store initialization successful")
        else:
            logger.error("Primary vector store initialization failed")
    else:
        logger.warning("Primary vector store does not have an initialize method")
    
    # Then initialize the multivector store if enabled
    if settings.ENABLE_COLPALI and colpali_vector_store:
        logger.info("Initializing multivector store...")
        # Handle both synchronous and asynchronous initialize methods
        if hasattr(colpali_vector_store.initialize, '__awaitable__'):
            success = await colpali_vector_store.initialize()
        else:
            success = colpali_vector_store.initialize()
            
        if success:
            logger.info("Multivector store initialization successful")
        else:
            logger.error("Multivector store initialization failed")

@app.on_event("startup")
async def initialize_user_limits_database():
    """Initialize user service on application startup."""
    logger.info("Initializing user service...")
    if settings.MODE == "cloud":
        from core.database.user_limits_db import UserLimitsDatabase
        user_limits_db = UserLimitsDatabase(uri=settings.POSTGRES_URI)
        await user_limits_db.initialize()

@app.on_event("startup")
async def initialize_redis_pool():
    """Initialize the Redis connection pool for background tasks."""
    global redis_pool
    logger.info("Initializing Redis connection pool...")
    
    # Get Redis settings from configuration
    redis_host = settings.REDIS_HOST
    redis_port = settings.REDIS_PORT
    
    # Log the Redis connection details
    logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
    
    redis_settings = arq.connections.RedisSettings(
        host=redis_host,
        port=redis_port,
    )

    redis_pool = await arq.create_pool(redis_settings)
    logger.info("Redis connection pool initialized successfully")

@app.on_event("shutdown")
async def close_redis_pool():
    """Close the Redis connection pool on application shutdown."""
    global redis_pool
    if redis_pool:
        logger.info("Closing Redis connection pool...")
        redis_pool.close()
        await redis_pool.wait_closed()
        logger.info("Redis connection pool closed")

# Initialize vector store
if not settings.POSTGRES_URI:
    raise ValueError("PostgreSQL URI is required for pgvector store")
from core.vector_store.pgvector_store import PGVectorStore

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
# Import here to avoid circular imports
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel

# Create a LiteLLM model using the registered model config
embedding_model = LiteLLMEmbeddingModel(
    model_key=settings.EMBEDDING_MODEL,
)
logger.info(f"Initialized LiteLLM embedding model with model key: {settings.EMBEDDING_MODEL}")

# Initialize completion model
# Import here to avoid circular imports
from core.completion.litellm_completion import LiteLLMCompletionModel

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

# Initialize ColPali embedding model if enabled
colpali_embedding_model = ColpaliEmbeddingModel() if settings.ENABLE_COLPALI else None
colpali_vector_store = (
    MultiVectorStore(uri=settings.POSTGRES_URI) if settings.ENABLE_COLPALI else None
)

# Initialize document service with configured components
document_service = DocumentService(
    storage=storage,
    database=database,
    vector_store=vector_store,
    embedding_model=embedding_model,
    completion_model=completion_model,
    parser=parser,
    reranker=reranker,
    cache_factory=cache_factory,
    enable_colpali=settings.ENABLE_COLPALI,
    colpali_embedding_model=colpali_embedding_model,
    colpali_vector_store=colpali_vector_store,
)


async def verify_token(authorization: str = Header(None)) -> AuthContext:
    """Verify JWT Bearer token or return dev context if dev_mode is enabled."""
    # Check if dev mode is enabled
    if settings.dev_mode:
        return AuthContext(
            entity_type=EntityType(settings.dev_entity_type),
            entity_id=settings.dev_entity_id,
            permissions=set(settings.dev_permissions),
            user_id=settings.dev_entity_id,  # In dev mode, entity_id is also the user_id
        )

    # Normal token verification flow
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]  # Remove "Bearer "
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

        if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
            raise HTTPException(status_code=401, detail="Token expired")

        # Support both "type" and "entity_type" fields for compatibility
        entity_type_field = payload.get("type") or payload.get("entity_type")
        if not entity_type_field:
            raise HTTPException(status_code=401, detail="Missing entity type in token")
            
        return AuthContext(
            entity_type=EntityType(entity_type_field),
            entity_id=payload["entity_id"],
            app_id=payload.get("app_id"),
            permissions=set(payload.get("permissions", ["read"])),
            user_id=payload.get("user_id", payload["entity_id"]),  # Use user_id if available, fallback to entity_id
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/ingest/text", response_model=Document)
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
        async with telemetry.track_operation(
            operation_type="ingest_text",
            user_id=auth.entity_id,
            tokens_used=len(request.content.split()),  # Approximate token count
            metadata={
                "metadata": request.metadata,
                "rules": request.rules,
                "use_colpali": request.use_colpali,
                "folder_name": request.folder_name,
                "end_user_id": request.end_user_id,
            },
        ):
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


# Redis pool for background tasks
redis_pool = None

def get_redis_pool():
    """Get the global Redis connection pool for background tasks."""
    return redis_pool

@app.post("/ingest/file", response_model=Document)
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
        def str2bool(v): return v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}
        use_colpali = str2bool(use_colpali)
        
        # Ensure user has write permission
        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        async with telemetry.track_operation(
            operation_type="queue_ingest_file",
            user_id=auth.entity_id,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "metadata": metadata_dict,
                "rules": rules_list,
                "use_colpali": use_colpali,
                "folder_name": folder_name,
                "end_user_id": end_user_id,
            },
        ):
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
                    "user_id": [auth.user_id] if auth.user_id else [],
                },
                system_metadata={"status": "processing"}
            )
            
            # Add folder_name and end_user_id to system_metadata if provided
            if folder_name:
                doc.system_metadata["folder_name"] = folder_name
            if end_user_id:
                doc.system_metadata["end_user_id"] = end_user_id
                
            # Set processing status
            doc.system_metadata["status"] = "processing"
            
            # Store the document in the database
            success = await database.store_document(doc)
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
            
            # Store the file in the configured storage
            file_content_base64 = base64.b64encode(file_content).decode()
            bucket, stored_key = await storage.upload_from_base64(
                file_content_base64, 
                file_key, 
                file.content_type
            )
            logger.debug(f"Stored file in bucket {bucket} with key {stored_key}")
            
            # Update document with storage info
            doc.storage_info = {"bucket": bucket, "key": stored_key}
            await database.update_document(
                document_id=doc.external_id,
                updates={"storage_info": doc.storage_info},
                auth=auth
            )
            
            # Convert auth context to a dictionary for serialization
            auth_dict = {
                "entity_type": auth.entity_type.value,
                "entity_id": auth.entity_id,
                "app_id": auth.app_id,
                "permissions": list(auth.permissions),
                "user_id": auth.user_id
            }
            
            # Enqueue the background job
            job = await redis.enqueue_job(
                'process_ingestion_job',
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
                end_user_id=end_user_id
            )
            
            logger.info(f"File ingestion job queued with ID: {job.job_id} for document: {doc.external_id}")
            
            # Return the document with processing status
            return doc
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error queueing file ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error queueing file ingestion: {str(e)}")


@app.post("/ingest/files", response_model=BatchIngestResponse)
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
        raise HTTPException(
            status_code=400,
            detail="No files provided for batch ingestion"
        )

    try:
        metadata_value = json.loads(metadata)
        rules_list = json.loads(rules)
        # Fix bool conversion: ensure string "false" is properly converted to False
        def str2bool(v): return str(v).lower() in {"true", "1", "yes"}
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
            detail=f"Number of metadata items ({len(metadata_value)}) must match number of files ({len(files)})"
        )

    # Validate rules if it's a list of lists
    if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list):
        if len(rules_list) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of rule lists ({len(rules_list)}) must match number of files ({len(files)})"
            )

    # Convert auth context to a dictionary for serialization
    auth_dict = {
        "entity_type": auth.entity_type.value,
        "entity_id": auth.entity_id,
        "app_id": auth.app_id,
        "permissions": list(auth.permissions),
        "user_id": auth.user_id
    }

    created_documents = []

    async with telemetry.track_operation(
        operation_type="queue_batch_ingest",
        user_id=auth.entity_id,
        metadata={
            "file_count": len(files),
            "metadata_type": "list" if isinstance(metadata_value, list) else "single",
            "rules_type": "per_file" if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list) else "shared",
            "folder_name": folder_name,
            "end_user_id": end_user_id,
        },
    ):
        try:
            for i, file in enumerate(files):
                # Get the metadata and rules for this file
                metadata_item = metadata_value[i] if isinstance(metadata_value, list) else metadata_value
                file_rules = rules_list[i] if isinstance(rules_list, list) and rules_list and isinstance(rules_list[0], list) else rules_list
                
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
                    },
                )
                
                # Add folder_name and end_user_id to system_metadata if provided
                if folder_name:
                    doc.system_metadata["folder_name"] = folder_name
                if end_user_id:
                    doc.system_metadata["end_user_id"] = end_user_id
                    
                # Set processing status
                doc.system_metadata["status"] = "processing"
                
                # Store the document in the database
                success = await database.store_document(doc)
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
                
                # Store the file in the configured storage
                file_content_base64 = base64.b64encode(file_content).decode()
                bucket, stored_key = await storage.upload_from_base64(
                    file_content_base64, 
                    file_key, 
                    file.content_type
                )
                logger.debug(f"Stored file in bucket {bucket} with key {stored_key}")
                
                # Update document with storage info
                doc.storage_info = {"bucket": bucket, "key": stored_key}
                await database.update_document(
                    document_id=doc.external_id,
                    updates={"storage_info": doc.storage_info},
                    auth=auth
                )
                
                # Convert metadata to JSON string for job
                metadata_json = json.dumps(metadata_item)
                
                # Enqueue the background job
                job = await redis.enqueue_job(
                    'process_ingestion_job',
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
                    end_user_id=end_user_id
                )
                
                logger.info(f"File ingestion job queued with ID: {job.job_id} for document: {doc.external_id}")
                
                # Add document to the list
                created_documents.append(doc)
                
            # Return information about created documents
            return BatchIngestResponse(
                documents=created_documents,
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error queueing batch file ingestion: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error queueing batch file ingestion: {str(e)}")




@app.post("/retrieve/chunks", response_model=List[ChunkResult])
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
        async with telemetry.track_operation(
            operation_type="retrieve_chunks",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "use_reranking": request.use_reranking,
                "use_colpali": request.use_colpali,
                "folder_name": request.folder_name,
                "end_user_id": request.end_user_id,
            },
        ):
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
        async with telemetry.track_operation(
            operation_type="retrieve_docs",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "use_reranking": request.use_reranking,
                "use_colpali": request.use_colpali,
                "folder_name": request.folder_name,
                "end_user_id": request.end_user_id,
            },
        ):
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
    try:
        # Extract document_ids from request
        document_ids = request.get("document_ids", [])
        folder_name = request.get("folder_name")
        end_user_id = request.get("end_user_id")
        
        if not document_ids:
            return []
            
        async with telemetry.track_operation(
            operation_type="batch_get_documents",
            user_id=auth.entity_id,
            metadata={
                "document_count": len(document_ids),
                "folder_name": folder_name,
                "end_user_id": end_user_id,
            },
        ):
            return await document_service.batch_retrieve_documents(document_ids, auth, folder_name, end_user_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/chunks", response_model=List[ChunkResult])
async def batch_get_chunks(
    request: Dict[str, Any],
    auth: AuthContext = Depends(verify_token)
):
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
        
        if not sources:
            return []
            
        async with telemetry.track_operation(
            operation_type="batch_get_chunks",
            user_id=auth.entity_id,
            metadata={
                "chunk_count": len(sources),
                "folder_name": folder_name,
                "end_user_id": end_user_id,
            },
        ):
            # Convert sources to ChunkSource objects if needed
            chunk_sources = []
            for source in sources:
                if isinstance(source, dict):
                    chunk_sources.append(ChunkSource(**source))
                else:
                    chunk_sources.append(source)
                    
            return await document_service.batch_retrieve_chunks(chunk_sources, auth, folder_name, end_user_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest, auth: AuthContext = Depends(verify_token)
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
        auth: Authentication context
        
    Returns:
        CompletionResponse: Generated completion
    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="query")
        
        # Check query limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "query", 1)
            
        async with telemetry.track_operation(
            operation_type="query",
            user_id=auth.entity_id,
            metadata={
                "k": request.k,
                "min_score": request.min_score,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "use_reranking": request.use_reranking,
                "use_colpali": request.use_colpali,
                "graph_name": request.graph_name,
                "hop_depth": request.hop_depth,
                "include_paths": request.include_paths,
                "folder_name": request.folder_name,
                "end_user_id": request.end_user_id,
            },
        ):
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
            )
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="query", error=e)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/documents", response_model=List[Document])
async def list_documents(
    auth: AuthContext = Depends(verify_token),
    skip: int = 0,
    limit: int = 10000,
    filters: Optional[Dict[str, Any]] = None,
    folder_name: Optional[str] = None,
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
        async with telemetry.track_operation(
            operation_type="delete_document",
            user_id=auth.entity_id,
            metadata={"document_id": document_id},
        ):
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
    folder_name: Optional[str] = None,
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
async def update_document_text(
    document_id: str,
    request: IngestTextRequest,
    update_strategy: str = "add",
    auth: AuthContext = Depends(verify_token)
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
        async with telemetry.track_operation(
            operation_type="update_document_text",
            user_id=auth.entity_id,
            metadata={
                "document_id": document_id,
                "update_strategy": update_strategy,
                "use_colpali": request.use_colpali,
                "has_filename": request.filename is not None,
            },
        ):
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
async def update_document_file(
    document_id: str,
    file: UploadFile,
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    update_strategy: str = Form("add"),
    use_colpali: Optional[bool] = None,
    auth: AuthContext = Depends(verify_token)
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

        async with telemetry.track_operation(
            operation_type="update_document_file",
            user_id=auth.entity_id,
            metadata={
                "document_id": document_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "update_strategy": update_strategy,
                "use_colpali": use_colpali,
            },
        ):
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
async def update_document_metadata(
    document_id: str,
    metadata: Dict[str, Any],
    auth: AuthContext = Depends(verify_token)
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
        async with telemetry.track_operation(
            operation_type="update_document_metadata",
            user_id=auth.entity_id,
            metadata={
                "document_id": document_id,
            },
        ):
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
async def get_usage_stats(auth: AuthContext = Depends(verify_token)) -> Dict[str, int]:
    """Get usage statistics for the authenticated user."""
    async with telemetry.track_operation(operation_type="get_usage_stats", user_id=auth.entity_id):
        if not auth.permissions or "admin" not in auth.permissions:
            return telemetry.get_user_usage(auth.entity_id)
        return telemetry.get_user_usage(auth.entity_id)


@app.get("/usage/recent")
async def get_recent_usage(
    auth: AuthContext = Depends(verify_token),
    operation_type: Optional[str] = None,
    since: Optional[datetime] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """Get recent usage records."""
    async with telemetry.track_operation(
        operation_type="get_recent_usage",
        user_id=auth.entity_id,
        metadata={
            "operation_type": operation_type,
            "since": since.isoformat() if since else None,
            "status": status,
        },
    ):
        if not auth.permissions or "admin" not in auth.permissions:
            records = telemetry.get_recent_usage(
                user_id=auth.entity_id, operation_type=operation_type, since=since, status=status
            )
        else:
            records = telemetry.get_recent_usage(
                operation_type=operation_type, since=since, status=status
            )

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
            
        async with telemetry.track_operation(
            operation_type="create_cache",
            user_id=auth.entity_id,
            metadata={
                "name": name,
                "model": model,
                "gguf_file": gguf_file,
                "filters": filters,
                "docs": docs,
            },
        ):
            filter_docs = set(await document_service.db.get_documents(auth, filters=filters))
            additional_docs = (
                {
                    await document_service.db.get_document(document_id=doc_id, auth=auth)
                    for doc_id in docs
                }
                if docs
                else set()
            )
            docs_to_add = list(filter_docs.union(additional_docs))
            if not docs_to_add:
                raise HTTPException(status_code=400, detail="No documents to add to cache")
            response = await document_service.create_cache(
                name, model, gguf_file, docs_to_add, filters
            )
            return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/cache/{name}")
async def get_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, Any]:
    """Get cache configuration by name."""
    try:
        async with telemetry.track_operation(
            operation_type="get_cache",
            user_id=auth.entity_id,
            metadata={"name": name},
        ):
            exists = await document_service.load_cache(name)
            return {"exists": exists}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/update")
async def update_cache(name: str, auth: AuthContext = Depends(verify_token)) -> Dict[str, bool]:
    """Update cache with new documents matching its filter."""
    try:
        async with telemetry.track_operation(
            operation_type="update_cache",
            user_id=auth.entity_id,
            metadata={"name": name},
        ):
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
async def add_docs_to_cache(
    name: str, docs: List[str], auth: AuthContext = Depends(verify_token)
) -> Dict[str, bool]:
    """Add specific documents to the cache."""
    try:
        async with telemetry.track_operation(
            operation_type="add_docs_to_cache",
            user_id=auth.entity_id,
            metadata={"name": name, "docs": docs},
        ):
            cache = document_service.active_caches[name]
            docs_to_add = [
                await document_service.db.get_document(doc_id, auth)
                for doc_id in docs
                if doc_id not in cache.docs
            ]
            return cache.add_docs(docs_to_add)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/cache/{name}/query")
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
            
        async with telemetry.track_operation(
            operation_type="query_cache",
            user_id=auth.entity_id,
            metadata={
                "name": name,
                "query": query,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        ):
            cache = document_service.active_caches[name]
            print(f"Cache state: {cache.state.n_tokens}", file=sys.stderr)
            return cache.query(query)  # , max_tokens, temperature)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/graph/create", response_model=Graph)
async def create_graph(
    request: CreateGraphRequest,
    auth: AuthContext = Depends(verify_token),
) -> Graph:
    """
    Create a graph from documents.

    This endpoint extracts entities and relationships from documents
    matching the specified filters or document IDs and creates a graph.

    Args:
        request: CreateGraphRequest containing:
            - name: Name of the graph to create
            - filters: Optional metadata filters to determine which documents to include
            - documents: Optional list of specific document IDs to include
            - prompt_overrides: Optional customizations for entity extraction and resolution prompts
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
        auth: Authentication context

    Returns:
        Graph: The created graph object
    """
    try:
        # Validate prompt overrides before proceeding
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="graph")
        
        # Check graph creation limits if in cloud mode
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "graph", 1)
            
        async with telemetry.track_operation(
            operation_type="create_graph",
            user_id=auth.entity_id,
            metadata={
                "name": request.name,
                "filters": request.filters,
                "documents": request.documents,
                "folder_name": request.folder_name,
                "end_user_id": request.end_user_id,
            },
        ):
            # Create system filters for folder and user scoping
            system_filters = {}
            if request.folder_name:
                system_filters["folder_name"] = request.folder_name
            if request.end_user_id:
                system_filters["end_user_id"] = request.end_user_id
                
            return await document_service.create_graph(
                name=request.name,
                auth=auth,
                filters=request.filters,
                documents=request.documents,
                prompt_overrides=request.prompt_overrides,
                system_filters=system_filters,
            )
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
async def get_graph(
    name: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[str] = None,
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
        async with telemetry.track_operation(
            operation_type="get_graph",
            user_id=auth.entity_id,
            metadata={
                "name": name,
                "folder_name": folder_name,
                "end_user_id": end_user_id
            },
        ):
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
async def list_graphs(
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[str] = None,
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
        async with telemetry.track_operation(
            operation_type="list_graphs",
            user_id=auth.entity_id,
            metadata={
                "folder_name": folder_name,
                "end_user_id": end_user_id
            },
        ):
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
        
        async with telemetry.track_operation(
            operation_type="update_graph",
            user_id=auth.entity_id,
            metadata={
                "name": name,
                "additional_filters": request.additional_filters,
                "additional_documents": request.additional_documents,
                "folder_name": request.folder_name,
                "end_user_id": request.end_user_id,
            },
        ):
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
        base_url = f"{config['api']['host']}:{config['api']['port']}".replace(
            "localhost", "127.0.0.1"
        )

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
                    detail="You can only create apps for your own account unless you have admin permissions"
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
            raise HTTPException(
                status_code=403,
                detail="Application limit reached for this account tier"
            )

        return {"uri": uri, "app_id": app_id}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating cloud URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folders/{folder_id}/set_rule")
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
        async with telemetry.track_operation(
            operation_type="set_folder_rule",
            user_id=auth.entity_id,
            metadata={
                "folder_id": folder_id,
                "rule_count": len(request.rules),
                "apply_to_existing": apply_to_existing,
            },
        ):
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
                    if 'schema' in field_config:
                        logger.debug(f"    Has JSON schema: Yes")
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
                from core.services.rules_processor import RulesProcessor
                rules_processor = RulesProcessor()
                
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
                            processing_results["errors"].append({
                                "document_id": doc.external_id,
                                "error": error_msg
                            })
                            continue
                            
                        # Process document with rules
                        try:
                            # Convert request rules to actual rule models and apply them
                            from core.models.rules import MetadataExtractionRule
                            
                            for rule_request in request.rules:
                                if rule_request.type == "metadata_extraction":
                                    # Create the actual rule model
                                    rule = MetadataExtractionRule(
                                        type=rule_request.type,
                                        schema=rule_request.schema
                                    )
                                    
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
                                            
                                            extracted_metadata, _ = await rule.apply(doc_content)
                                            logger.info(f"Successfully extracted metadata on attempt {retry_count + 1}: {extracted_metadata}")
                                            break  # Success, exit retry loop
                                            
                                        except Exception as rule_apply_error:
                                            last_error = rule_apply_error
                                            logger.warning(f"Metadata extraction attempt {retry_count + 1} failed: {rule_apply_error}")
                                            if retry_count == max_retries - 1:  # Last attempt
                                                logger.error(f"All {max_retries} metadata extraction attempts failed")
                                                processing_results["errors"].append({
                                                    "document_id": doc.external_id,
                                                    "error": f"Failed to extract metadata after {max_retries} attempts: {str(last_error)}"
                                                })
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
                                            "system_metadata": {}  # Will be merged with existing in update_document
                                        }
                                        
                                        # Explicitly preserve the content field in system_metadata
                                        if "content" in doc.system_metadata:
                                            updates["system_metadata"]["content"] = doc.system_metadata["content"]
                                        
                                        # Log the updates we're making
                                        logger.info(f"Updating document {doc.external_id} with metadata: {extracted_metadata}")
                                        logger.info(f"Full metadata being updated: {doc.metadata}")
                                        logger.info(f"Update object being sent to database: {updates}")
                                        logger.info(f"Preserving content in system_metadata: {'content' in doc.system_metadata}")
                                        
                                        # Update document in database
                                        success = await document_service.db.update_document(
                                            doc.external_id,
                                            updates,
                                            auth
                                        )
                                        
                                        if success:
                                            logger.info(f"Updated metadata for document {doc.external_id}")
                                            processing_results["processed"] += 1
                                        else:
                                            logger.error(f"Failed to update metadata for document {doc.external_id}")
                                            processing_results["errors"].append({
                                                "document_id": doc.external_id,
                                                "error": "Failed to update document metadata"
                                            })
                        except Exception as rule_error:
                            logger.error(f"Error processing rules for document {doc.external_id}: {rule_error}")
                            processing_results["errors"].append({
                                "document_id": doc.external_id,
                                "error": f"Error processing rules: {str(rule_error)}"
                            })
                            
                    except Exception as doc_error:
                        logger.error(f"Error processing document {doc.external_id}: {doc_error}")
                        processing_results["errors"].append({
                            "document_id": doc.external_id,
                            "error": str(doc_error)
                        })
            
            return {
                "status": "success",
                "message": "Rules set successfully",
                "folder_id": folder_id,
                "rules": updated_folder.rules,
                "processing_results": processing_results
            }
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error setting folder rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))
