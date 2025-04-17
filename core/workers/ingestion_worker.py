import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
from pathlib import Path
import asyncio
import os
import urllib.parse as up

import arq
from arq.connections import RedisSettings
from core.models.auth import AuthContext, EntityType
from core.models.documents import Document
from core.database.postgres_database import PostgresDatabase
from core.vector_store.pgvector_store import PGVectorStore
from core.parser.morphik_parser import MorphikParser
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.completion.litellm_completion import LiteLLMCompletionModel
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.vector_store.multi_vector_store import MultiVectorStore
from core.services.document_service import DocumentService
from core.services.telemetry import TelemetryService
from core.services.rules_processor import RulesProcessor
from core.config import get_settings
from sqlalchemy import text

logger = logging.getLogger(__name__)

async def get_document_with_retry(document_service, document_id, auth, max_retries=3, initial_delay=0.3):
    """
    Helper function to get a document with retries to handle race conditions.
    
    Args:
        document_service: The document service instance
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
            doc = await document_service.db.get_document(document_id, auth)
            if doc:
                logger.debug(f"Successfully retrieved document {document_id} on attempt {attempt+1}")
                return doc
                
            # Document not found but no exception raised
            attempt += 1
            if attempt < max_retries:
                logger.warning(f"Document {document_id} not found on attempt {attempt}/{max_retries}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            
        except Exception as e:
            attempt += 1
            error_msg = str(e)
            if attempt < max_retries:
                logger.warning(f"Error retrieving document on attempt {attempt}/{max_retries}: {error_msg}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                logger.error(f"Failed to retrieve document after {max_retries} attempts: {error_msg}")
                return None
    
    return None

async def process_ingestion_job(
    ctx: Dict[str, Any],
    document_id: str,
    file_key: str,
    bucket: str,
    original_filename: str,
    content_type: str,
    metadata_json: str,
    auth_dict: Dict[str, Any],
    rules_list: List[Dict[str, Any]],
    use_colpali: bool,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Background worker task that processes file ingestion jobs.
    
    Args:
        ctx: The ARQ context dictionary
        file_key: The storage key where the file is stored
        bucket: The storage bucket name
        original_filename: The original file name
        content_type: The file's content type/MIME type
        metadata_json: JSON string of metadata
        auth_dict: Dict representation of AuthContext
        rules_list: List of rules to apply (already converted to dictionaries)
        use_colpali: Whether to use ColPali embedding model
        folder_name: Optional folder to scope the document to
        end_user_id: Optional end-user ID to scope the document to
        
    Returns:
        A dictionary with the document ID and processing status
    """
    try:
        # 1. Log the start of the job
        logger.info(f"Starting ingestion job for file: {original_filename}")
        
        # 2. Deserialize metadata and auth
        metadata = json.loads(metadata_json) if metadata_json else {}
        auth = AuthContext(
            entity_type=EntityType(auth_dict.get("entity_type", "unknown")),
            entity_id=auth_dict.get("entity_id", ""),
            app_id=auth_dict.get("app_id"),
            permissions=set(auth_dict.get("permissions", ["read"])),
            user_id=auth_dict.get("user_id", auth_dict.get("entity_id", ""))
        )
        
        # Get document service from the context
        document_service : DocumentService = ctx['document_service']
        
        # 3. Download the file from storage
        logger.info(f"Downloading file from {bucket}/{file_key}")
        file_content = await document_service.storage.download_file(bucket, file_key)
        
        # Ensure file_content is bytes
        if hasattr(file_content, 'read'):
            file_content = file_content.read()
        
        # 4. Parse file to text
        additional_metadata, text = await document_service.parser.parse_file_to_text(
            file_content, original_filename
        )
        logger.debug(f"Parsed file into text of length {len(text)}")
        
        # 5. Apply rules if provided
        if rules_list:
            rule_metadata, modified_text = await document_service.rules_processor.process_rules(text, rules_list)
            # Update document metadata with extracted metadata from rules
            metadata.update(rule_metadata)
            
            if modified_text:
                text = modified_text
                logger.info("Updated text with modified content from rules")
        
        # 6. Retrieve the existing document
        logger.debug(f"Retrieving document with ID: {document_id}")
        logger.debug(f"Auth context: entity_type={auth.entity_type}, entity_id={auth.entity_id}, permissions={auth.permissions}")
        
        # Use the retry helper function with initial delay to handle race conditions
        doc = await get_document_with_retry(document_service, document_id, auth, max_retries=5, initial_delay=1.0)
        
        if not doc:
            logger.error(f"Document {document_id} not found in database after multiple retries")
            logger.error(f"Details - file: {original_filename}, content_type: {content_type}, bucket: {bucket}, key: {file_key}")
            logger.error(f"Auth: entity_type={auth.entity_type}, entity_id={auth.entity_id}, permissions={auth.permissions}")
            # Try to get all accessible documents to debug
            try:
                all_docs = await document_service.db.get_documents(auth, 0, 100)
                logger.debug(f"User has access to {len(all_docs)} documents: {[d.external_id for d in all_docs]}")
            except Exception as list_err:
                logger.error(f"Failed to list user documents: {str(list_err)}")
            
            raise ValueError(f"Document {document_id} not found in database after multiple retries")
            
        # Prepare updates for the document
        updates = {
            "metadata": metadata,
            "additional_metadata": additional_metadata,
            "system_metadata": {**doc.system_metadata, "content": text}
        }
        
        # Add folder_name and end_user_id to system_metadata if provided
        if folder_name:
            updates["system_metadata"]["folder_name"] = folder_name
        if end_user_id:
            updates["system_metadata"]["end_user_id"] = end_user_id
        
        # Update the document in the database
        success = await document_service.db.update_document(
            document_id=document_id,
            updates=updates,
            auth=auth
        )
        
        if not success:
            raise ValueError(f"Failed to update document {document_id}")
        
        # Refresh document object with updated data
        doc = await document_service.db.get_document(document_id, auth)
        logger.debug(f"Updated document in database with parsed content")
        
        # 7. Split text into chunks
        chunks = await document_service.parser.split_text(text)
        if not chunks:
            raise ValueError("No content chunks extracted")
        logger.debug(f"Split processed text into {len(chunks)} chunks")
        
        # 8. Generate embeddings for chunks
        embeddings = await document_service.embedding_model.embed_for_ingestion(chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        
        # 9. Create chunk objects
        chunk_objects = document_service._create_chunk_objects(doc.external_id, chunks, embeddings)
        logger.debug(f"Created {len(chunk_objects)} chunk objects")
        
        # 10. Handle ColPali embeddings if enabled
        chunk_objects_multivector = []
        if use_colpali and document_service.colpali_embedding_model and document_service.colpali_vector_store:
            import filetype
            file_type = filetype.guess(file_content)
            
            # For ColPali we need the base64 encoding of the file
            import base64
            file_content_base64 = base64.b64encode(file_content).decode()
            
            chunks_multivector = document_service._create_chunks_multivector(
                file_type, file_content_base64, file_content, chunks
            )
            logger.debug(f"Created {len(chunks_multivector)} chunks for multivector embedding")
            
            colpali_embeddings = await document_service.colpali_embedding_model.embed_for_ingestion(
                chunks_multivector
            )
            logger.debug(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")
            
            chunk_objects_multivector = document_service._create_chunk_objects(
                doc.external_id, chunks_multivector, colpali_embeddings
            )
        
        # Update document status to completed before storing
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        
        # 11. Store chunks and update document with is_update=True
        chunk_ids = await document_service._store_chunks_and_doc(
            chunk_objects, doc, use_colpali, chunk_objects_multivector,
            is_update=True, auth=auth
        )
            
        logger.debug(f"Successfully completed processing for document {doc.external_id}")
        
        # 13. Log successful completion
        logger.info(f"Successfully completed ingestion for {original_filename}, document ID: {doc.external_id}")
        
        # 14. Return document ID
        return {
            "document_id": doc.external_id,
            "status": "completed",
            "filename": original_filename,
            "content_type": content_type,
            "timestamp": datetime.now(UTC).isoformat()
        }
            
    except Exception as e:
        logger.error(f"Error processing ingestion job for file {original_filename}: {str(e)}")
        
        # Update document status to failed if the document exists
        try:
            # Create AuthContext for database operations
            auth_context = AuthContext(
                entity_type=EntityType(auth_dict.get("entity_type", "unknown")),
                entity_id=auth_dict.get("entity_id", ""),
                app_id=auth_dict.get("app_id"),
                permissions=set(auth_dict.get("permissions", ["read"])),
                user_id=auth_dict.get("user_id", auth_dict.get("entity_id", ""))
            )
            
            # Get database from context
            database = ctx.get('database')
            
            if database:
                # Try to get the document
                doc = await database.get_document(document_id, auth_context)
                
                if doc:
                    # Update the document status to failed
                    await database.update_document(
                        document_id=document_id,
                        updates={
                            "system_metadata": {
                                **doc.system_metadata,
                                "status": "failed",
                                "error": str(e),
                                "updated_at": datetime.now(UTC)
                            }
                        },
                        auth=auth_context
                    )
                    logger.info(f"Updated document {document_id} status to failed")
        except Exception as inner_e:
            logger.error(f"Failed to update document status: {str(inner_e)}")
        
        # Return error information
        return {
            "status": "failed",
            "filename": original_filename,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }

async def startup(ctx):
    """
    Worker startup: Initialize all necessary services that will be reused across jobs.
    
    This initialization is similar to what happens in core/api.py during app startup,
    but adapted for the worker context.
    """
    logger.info("Worker starting up. Initializing services...")
    
    # Get settings
    settings = get_settings()
    
    # Initialize database
    logger.info("Initializing database...")
    database = PostgresDatabase(uri=settings.POSTGRES_URI)
    success = await database.initialize()
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")
    ctx['database'] = database
    
    # Initialize vector store
    logger.info("Initializing primary vector store...")
    vector_store = PGVectorStore(uri=settings.POSTGRES_URI)
    success = await vector_store.initialize()
    if success:
        logger.info("Primary vector store initialization successful")
    else:
        logger.error("Primary vector store initialization failed")
    ctx['vector_store'] = vector_store
    
    # Initialize storage
    if settings.STORAGE_PROVIDER == "local":
        storage = LocalStorage(storage_path=settings.STORAGE_PATH)
    elif settings.STORAGE_PROVIDER == "aws-s3":
        storage = S3Storage(
            aws_access_key=settings.AWS_ACCESS_KEY,
            aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            default_bucket=settings.S3_BUCKET,
        )
    else:
        raise ValueError(f"Unsupported storage provider: {settings.STORAGE_PROVIDER}")
    ctx['storage'] = storage
    
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
    ctx['parser'] = parser
    
    # Initialize embedding model
    embedding_model = LiteLLMEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
    logger.info(f"Initialized LiteLLM embedding model with model key: {settings.EMBEDDING_MODEL}")
    ctx['embedding_model'] = embedding_model
    
    # Initialize completion model
    completion_model = LiteLLMCompletionModel(model_key=settings.COMPLETION_MODEL)
    logger.info(f"Initialized LiteLLM completion model with model key: {settings.COMPLETION_MODEL}")
    ctx['completion_model'] = completion_model
    
    # Initialize reranker
    reranker = None
    if settings.USE_RERANKING:
        if settings.RERANKER_PROVIDER == "flag":
            from core.reranker.flag_reranker import FlagReranker
            reranker = FlagReranker(
                model_name=settings.RERANKER_MODEL,
                device=settings.RERANKER_DEVICE,
                use_fp16=settings.RERANKER_USE_FP16,
                query_max_length=settings.RERANKER_QUERY_MAX_LENGTH,
                passage_max_length=settings.RERANKER_PASSAGE_MAX_LENGTH,
            )
        else:
            logger.warning(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")
    ctx['reranker'] = reranker
    
    # Initialize ColPali embedding model and vector store if enabled
    colpali_embedding_model = None
    colpali_vector_store = None
    if settings.ENABLE_COLPALI:
        logger.info("Initializing ColPali components...")
        colpali_embedding_model = ColpaliEmbeddingModel()
        colpali_vector_store = MultiVectorStore(uri=settings.POSTGRES_URI)
        # Properly await the initialization to ensure indexes are ready
        # MultiVectorStore.initialize is synchronous, so we need to run it in a thread
        success = await asyncio.to_thread(colpali_vector_store.initialize)
        if success:
            logger.info("ColPali vector store initialization successful")
        else:
            logger.error("ColPali vector store initialization failed")
    ctx['colpali_embedding_model'] = colpali_embedding_model
    ctx['colpali_vector_store'] = colpali_vector_store
    
    # Initialize cache factory for DocumentService (may not be used for ingestion)
    from core.cache.llama_cache_factory import LlamaCacheFactory
    cache_factory = LlamaCacheFactory(Path(settings.STORAGE_PATH))
    ctx['cache_factory'] = cache_factory
    
    # Initialize rules processor
    rules_processor = RulesProcessor()
    ctx['rules_processor'] = rules_processor
    
    # Initialize telemetry service
    telemetry = TelemetryService()
    ctx['telemetry'] = telemetry
    
    # Create the document service using all initialized components
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
    ctx['document_service'] = document_service
    
    logger.info("Worker startup complete. All services initialized.")

async def shutdown(ctx):
    """
    Worker shutdown: Clean up resources.
    
    Properly close connections and cleanup resources to prevent leaks.
    """
    logger.info("Worker shutting down. Cleaning up resources...")
    
    # Close database connections
    if 'database' in ctx and hasattr(ctx['database'], 'engine'):
        logger.info("Closing database connections...")
        await ctx['database'].engine.dispose()
    
    # Close vector store connections if they exist
    if 'vector_store' in ctx and hasattr(ctx['vector_store'], 'engine'):
        logger.info("Closing vector store connections...")
        await ctx['vector_store'].engine.dispose()
    
    # Close colpali vector store connections if they exist
    if 'colpali_vector_store' in ctx and hasattr(ctx['colpali_vector_store'], 'engine'):
        logger.info("Closing colpali vector store connections...")
        await ctx['colpali_vector_store'].engine.dispose()
    
    # Close any other open connections or resources that need cleanup
    logger.info("Worker shutdown complete.")

def redis_settings_from_env() -> RedisSettings:
    """
    Create RedisSettings from environment variables for ARQ worker.
    
    Returns:
        RedisSettings configured for Redis connection with optimized performance
    """
    url = up.urlparse(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
    
    # Use ARQ's supported parameters with optimized values for stability
    # For high-volume ingestion (100+ documents), these settings help prevent timeouts
    return RedisSettings(
        host=url.hostname or os.getenv("REDIS_HOST", "127.0.0.1"),
        port=url.port or int(os.getenv("REDIS_PORT", "6379")),
        database=int(url.path.lstrip("/") or 0),
        conn_timeout=5,        # Increased connection timeout (seconds)
        conn_retries=15,       # More retries for transient connection issues
        conn_retry_delay=1     # Quick retry delay (seconds)
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
    keep_result_ms = 24 * 60 * 60 * 1000  # Keep results for 24 hours (24 * 60 * 60 * 1000 ms)
    
    # Concurrency settings - optimized for high-volume ingestion
    max_jobs = 3  # Reduced to prevent resource contention during batch processing
    
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
        database = ctx.get('database')
        vector_store = ctx.get('vector_store')
        job_stats = ctx.get('job_stats', {})
        
        # Get detailed Redis info
        try:
            redis_info = await ctx['redis'].info(section=['Server', 'Memory', 'Clients', 'Stats'])
            
            # Server and resource usage info
            redis_version = redis_info.get('redis_version', 'unknown')
            used_memory = redis_info.get('used_memory_human', 'unknown')
            used_memory_peak = redis_info.get('used_memory_peak_human', 'unknown')
            clients_connected = redis_info.get('connected_clients', 'unknown')
            rejected_connections = redis_info.get('rejected_connections', 0)
            total_commands = redis_info.get('total_commands_processed', 0)
            
            # DB keys
            db_info = redis_info.get('db0', {})
            keys_count = db_info.get('keys', 0) if isinstance(db_info, dict) else 0
            
            # Log comprehensive server status
            logger.info(
                f"Redis Status: v{redis_version} | "
                f"Memory: {used_memory} (peak: {used_memory_peak}) | "
                f"Clients: {clients_connected} (rejected: {rejected_connections}) | "
                f"DB Keys: {keys_count} | Commands: {total_commands}"
            )
            
            # Check for memory warning thresholds
            if isinstance(used_memory, str) and used_memory.endswith('G'):
                memory_value = float(used_memory[:-1])
                if memory_value > 1.0:  # More than 1GB used
                    logger.warning(f"Redis memory usage is high: {used_memory}")
            
            # Check for connection issues
            if rejected_connections and int(rejected_connections) > 0:
                logger.warning(f"Redis has rejected {rejected_connections} connections")
        except Exception as e:
            logger.error(f"Failed to get Redis info: {str(e)}")
        
        # Log job statistics with detailed processing metrics
        ongoing = job_stats.get('ongoing', 0)
        queued = job_stats.get('queued', 0)
        
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
        if database and hasattr(database, 'async_session'):
            try:
                async with database.async_session() as session:
                    await session.execute(text("SELECT 1"))
                    logger.debug("Database connection is healthy")
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")
                
        # Test vector store connectivity if available
        if vector_store and hasattr(vector_store, 'async_session'):
            try:
                async with vector_store.get_session_with_retry() as session:
                    logger.debug("Vector store connection is healthy")
            except Exception as e:
                logger.error(f"Vector store connection test failed: {str(e)}")