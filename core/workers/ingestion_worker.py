import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
from pathlib import Path
import asyncio

import arq
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
        logger.info(f"[TRACE-JOB] Starting ingestion job for file: {original_filename}, document_id: {document_id}")
        logger.info(f"[TRACE-JOB] Job parameters: bucket={bucket}, key={file_key}, content_type={content_type}, use_colpali={use_colpali}, folder_name={folder_name}, end_user_id={end_user_id}")
        
        # 2. Deserialize metadata and auth
        metadata = json.loads(metadata_json) if metadata_json else {}
        auth = AuthContext(
            entity_type=EntityType(auth_dict.get("entity_type", "unknown")),
            entity_id=auth_dict.get("entity_id", ""),
            app_id=auth_dict.get("app_id"),
            permissions=set(auth_dict.get("permissions", ["read"])),
            user_id=auth_dict.get("user_id", auth_dict.get("entity_id", ""))
        )
        logger.info(f"[TRACE-JOB] Auth context for document {document_id}: entity_type={auth.entity_type}, entity_id={auth.entity_id}, permissions={auth.permissions}")
        
        # Get document service from the context
        document_service : DocumentService = ctx['document_service']
        
        # 3. Download the file from storage
        logger.info(f"[TRACE-JOB] Downloading file from {bucket}/{file_key} for document {document_id}")
        try:
            file_content = await document_service.storage.download_file(bucket, file_key)
            
            # Ensure file_content is bytes
            if hasattr(file_content, 'read'):
                file_content = file_content.read()
                
            logger.info(f"[TRACE-JOB] Successfully downloaded file of size {len(file_content)} bytes for document {document_id}")
        except Exception as download_err:
            logger.error(f"[TRACE-JOB] Error downloading file {bucket}/{file_key} for document {document_id}: {str(download_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] File download error traceback: {traceback.format_exc()}")
            raise
        
        # 4. Parse file to text
        logger.info(f"[TRACE-JOB] Parsing file {original_filename} to text for document {document_id}")
        try:
            additional_metadata, text = await document_service.parser.parse_file_to_text(
                file_content, original_filename
            )
            logger.info(f"[TRACE-JOB] Successfully parsed file into text of length {len(text)} for document {document_id}")
        except Exception as parse_err:
            logger.error(f"[TRACE-JOB] Error parsing file {original_filename} for document {document_id}: {str(parse_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] File parsing error traceback: {traceback.format_exc()}")
            raise
        
        # 5. Apply rules if provided
        if rules_list:
            logger.info(f"[TRACE-JOB] Applying {len(rules_list)} rules to document {document_id}")
            try:
                rule_metadata, modified_text = await document_service.rules_processor.process_rules(text, rules_list)
                # Update document metadata with extracted metadata from rules
                metadata.update(rule_metadata)
                
                if modified_text:
                    text = modified_text
                    logger.info(f"[TRACE-JOB] Updated text with modified content from rules for document {document_id}")
                else:
                    logger.info(f"[TRACE-JOB] Rules applied but text was not modified for document {document_id}")
            except Exception as rules_err:
                logger.error(f"[TRACE-JOB] Error applying rules to document {document_id}: {str(rules_err)}")
                # Log the full traceback
                import traceback
                logger.error(f"[TRACE-JOB] Rules processing error traceback: {traceback.format_exc()}")
                raise
        
        # 6. Retrieve the existing document
        logger.info(f"[TRACE-JOB] Retrieving document metadata for document {document_id}")
        
        # Add retry logic for database operations
        max_retries = 3
        retry_delay = 1.0
        attempt = 0
        doc = None
        
        while attempt < max_retries:
            try:
                logger.info(f"[TRACE-JOB] Document retrieval attempt {attempt+1}/{max_retries} for document {document_id}")
                doc = await document_service.db.get_document(document_id, auth)
                # If successful, break out of the retry loop
                logger.info(f"[TRACE-JOB] Successfully retrieved document metadata for document {document_id}")
                break
            except Exception as e:
                attempt += 1
                error_msg = str(e)
                logger.error(f"[TRACE-JOB] Error retrieving document {document_id} on attempt {attempt}: {error_msg}")
                
                if "connection was closed" in error_msg or "ConnectionDoesNotExistError" in error_msg:
                    if attempt < max_retries:
                        logger.warning(f"[TRACE-JOB] Database connection error during document retrieval (attempt {attempt}/{max_retries}) for document {document_id}: {error_msg}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        # Increase delay for next retry (exponential backoff)
                        retry_delay *= 2
                    else:
                        logger.error(f"[TRACE-JOB] All database connection attempts failed after {max_retries} retries for document {document_id}")
                        # Log the full traceback
                        import traceback
                        logger.error(f"[TRACE-JOB] Document retrieval error traceback: {traceback.format_exc()}")
                        
                        # Try to log database connection pool stats
                        try:
                            if hasattr(document_service.db, 'engine') and hasattr(document_service.db.engine, '_pool'):
                                pool = document_service.db.engine._pool
                                logger.error(f"[TRACE-JOB] Pool stats: size={pool.size}, overflow={pool.overflow}, checkedin={pool.checkedin}, checkedout={pool.checkedout}")
                        except Exception as pool_err:
                            logger.error(f"[TRACE-JOB] Error getting pool stats: {str(pool_err)}")
                        
                        raise
                else:
                    # For other exceptions, don't retry
                    logger.error(f"[TRACE-JOB] Error retrieving document {document_id}: {error_msg}")
                    # Log the full traceback
                    import traceback
                    logger.error(f"[TRACE-JOB] Document retrieval error traceback: {traceback.format_exc()}")
                    raise
        
        if not doc:
            logger.error(f"[TRACE-JOB] Document {document_id} not found in database")
            logger.error(f"[TRACE-JOB] Details - file: {original_filename}, content_type: {content_type}, bucket: {bucket}, key: {file_key}")
            logger.error(f"[TRACE-JOB] Auth: entity_type={auth.entity_type}, entity_id={auth.entity_id}, permissions={auth.permissions}")
            # Try to get all accessible documents to debug
            try:
                all_docs = await document_service.db.get_documents(auth, 0, 100)
                logger.debug(f"[TRACE-JOB] User has access to {len(all_docs)} documents: {[d.external_id for d in all_docs]}")
            except Exception as list_err:
                logger.error(f"[TRACE-JOB] Failed to list user documents: {str(list_err)}")
            
            raise ValueError(f"Document {document_id} not found in database")
            
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
        logger.info(f"[TRACE-JOB] Updating document metadata in database for document {document_id}")
        try:
            success = await document_service.db.update_document(
                document_id=document_id,
                updates=updates,
                auth=auth
            )
            
            if not success:
                logger.error(f"[TRACE-JOB] Failed to update document {document_id} metadata in database")
                raise ValueError(f"Failed to update document {document_id}")
                
            logger.info(f"[TRACE-JOB] Successfully updated document metadata in database for document {document_id}")
        except Exception as update_err:
            logger.error(f"[TRACE-JOB] Error updating document {document_id} metadata: {str(update_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Document update error traceback: {traceback.format_exc()}")
            raise
        
        # Refresh document object with updated data
        try:
            doc = await document_service.db.get_document(document_id, auth)
            logger.info(f"[TRACE-JOB] Refreshed document object with updated data for document {document_id}")
        except Exception as refresh_err:
            logger.error(f"[TRACE-JOB] Error refreshing document {document_id} data: {str(refresh_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Document refresh error traceback: {traceback.format_exc()}")
            raise
        
        # 7. Split text into chunks
        logger.info(f"[TRACE-JOB] Splitting text into chunks for document {document_id}")
        try:
            chunks = await document_service.parser.split_text(text)
            if not chunks:
                logger.error(f"[TRACE-JOB] No content chunks extracted for document {document_id}")
                raise ValueError("No content chunks extracted")
            logger.info(f"[TRACE-JOB] Split processed text into {len(chunks)} chunks for document {document_id}")
        except Exception as chunk_err:
            logger.error(f"[TRACE-JOB] Error splitting text into chunks for document {document_id}: {str(chunk_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Text chunking error traceback: {traceback.format_exc()}")
            raise
        
        # 8. Generate embeddings for chunks
        logger.info(f"[TRACE-JOB] Generating embeddings for {len(chunks)} chunks for document {document_id}")
        try:
            embeddings = await document_service.embedding_model.embed_for_ingestion(chunks)
            logger.info(f"[TRACE-JOB] Generated {len(embeddings)} embeddings for document {document_id}")
        except Exception as embed_err:
            logger.error(f"[TRACE-JOB] Error generating embeddings for document {document_id}: {str(embed_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Embedding generation error traceback: {traceback.format_exc()}")
            raise
        
        # 9. Create chunk objects
        logger.info(f"[TRACE-JOB] Creating chunk objects for document {document_id}")
        try:
            chunk_objects = document_service._create_chunk_objects(doc.external_id, chunks, embeddings)
            logger.info(f"[TRACE-JOB] Created {len(chunk_objects)} chunk objects for document {document_id}")
        except Exception as chunk_obj_err:
            logger.error(f"[TRACE-JOB] Error creating chunk objects for document {document_id}: {str(chunk_obj_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Chunk objects creation error traceback: {traceback.format_exc()}")
            raise
        
        # 10. Handle ColPali embeddings if enabled
        chunk_objects_multivector = []
        if use_colpali and document_service.colpali_embedding_model and document_service.colpali_vector_store:
            logger.info(f"[TRACE-JOB] Processing colpali embeddings for document {document_id}")
            try:
                import filetype
                file_type = filetype.guess(file_content)
                logger.info(f"[TRACE-JOB] File type detected as {file_type.mime if file_type else 'unknown'} for document {document_id}")
                
                # For ColPali we need the base64 encoding of the file
                import base64
                file_content_base64 = base64.b64encode(file_content).decode()
                
                chunks_multivector = document_service._create_chunks_multivector(
                    file_type, file_content_base64, file_content, chunks
                )
                logger.info(f"[TRACE-JOB] Created {len(chunks_multivector)} chunks for multivector embedding for document {document_id}")
                
                colpali_embeddings = await document_service.colpali_embedding_model.embed_for_ingestion(
                    chunks_multivector
                )
                logger.info(f"[TRACE-JOB] Generated {len(colpali_embeddings)} embeddings for multivector embedding for document {document_id}")
                
                chunk_objects_multivector = document_service._create_chunk_objects(
                    doc.external_id, chunks_multivector, colpali_embeddings
                )
                logger.info(f"[TRACE-JOB] Created {len(chunk_objects_multivector)} colpali chunk objects for document {document_id}")
            except Exception as colpali_err:
                logger.error(f"[TRACE-JOB] Error processing colpali embeddings for document {document_id}: {str(colpali_err)}")
                # Log the full traceback
                import traceback
                logger.error(f"[TRACE-JOB] Colpali processing error traceback: {traceback.format_exc()}")
                raise
        
        # Update document status to completed before storing
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        
        # 11. Store chunks and update document with is_update=True
        logger.info(f"[TRACE-JOB] Storing {len(chunk_objects)} chunks and updating document {document_id}")
        try:
            chunk_ids = await document_service._store_chunks_and_doc(
                chunk_objects, doc, use_colpali, chunk_objects_multivector,
                is_update=True, auth=auth
            )
            logger.info(f"[TRACE-JOB] Successfully stored {len(chunk_ids)} chunks for document {document_id}")
        except Exception as store_err:
            logger.error(f"[TRACE-JOB] Error storing chunks and document for document {document_id}: {str(store_err)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Chunk storage error traceback: {traceback.format_exc()}")
            raise
            
        logger.info(f"[TRACE-JOB] Successfully completed processing for document {doc.external_id}")
        
        # 13. Log successful completion
        logger.info(f"[TRACE-JOB] Successfully completed ingestion for {original_filename}, document ID: {doc.external_id}")
        
        # 14. Return document ID
        return {
            "document_id": doc.external_id,
            "status": "completed",
            "filename": original_filename,
            "content_type": content_type,
            "timestamp": datetime.now(UTC).isoformat()
        }
            
    except Exception as e:
        logger.error(f"[TRACE-JOB] Error processing ingestion job for file {original_filename}: {str(e)}")
        # Log the full traceback
        import traceback
        logger.error(f"[TRACE-JOB] Complete ingestion job error traceback: {traceback.format_exc()}")
        
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
                logger.info(f"[TRACE-JOB] Updating document {document_id} status to 'failed' after error")
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
                    logger.info(f"[TRACE-JOB] Successfully updated document {document_id} status to failed")
                else:
                    logger.error(f"[TRACE-JOB] Could not find document {document_id} to mark as failed")
        except Exception as inner_e:
            logger.error(f"[TRACE-JOB] Failed to update document status: {str(inner_e)}")
            # Log the full traceback
            import traceback
            logger.error(f"[TRACE-JOB] Status update error traceback: {traceback.format_exc()}")
        
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
        _ = colpali_vector_store.initialize()
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
    # Redis settings will be loaded from environment variables by default
    # Other optional settings:
    # redis_settings = arq.connections.RedisSettings(host='localhost', port=6379)
    keep_result_ms = 24 * 60 * 60 * 1000  # Keep results for 24 hours (24 * 60 * 60 * 1000 ms)
    max_jobs = 5  # Reduce concurrent jobs to prevent connection pool exhaustion
    health_check_interval = 30  # Check worker health every 30 seconds
    job_timeout = 3600  # 1 hour timeout for jobs
    max_tries = 3  # Retry failed jobs up to 3 times
    poll_delay = 0.5  # Poll delay to prevent excessive Redis queries
    
    # Log Redis and connection pool information for debugging
    @staticmethod
    async def health_check(ctx):
        """Periodic health check to log connection status and job stats."""
        database = ctx.get('database')
        vector_store = ctx.get('vector_store')
        job_stats = ctx.get('job_stats', {})
        redis_info = await ctx['redis'].info()
        
        logger.info(f"Health check: Redis v{redis_info.get('redis_version', 'unknown')} "
                   f"mem_usage={redis_info.get('used_memory_human', 'unknown')} "
                   f"clients_connected={redis_info.get('connected_clients', 'unknown')} "
                   f"db_keys={redis_info.get('db0', {}).get('keys', 0)}"
        )
        
        # Log job statistics
        logger.info(f"Job stats: completed={job_stats.get('complete', 0)} "
                  f"failed={job_stats.get('failed', 0)} "
                  f"retried={job_stats.get('retried', 0)} "
                  f"ongoing={job_stats.get('ongoing', 0)} "
                  f"queued={job_stats.get('queued', 0)}"
        )
        
        # Test database connectivity
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