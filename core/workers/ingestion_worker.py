import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
from pathlib import Path

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

logger = logging.getLogger(__name__)

async def process_ingestion_job(
    ctx: Dict[str, Any],
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
        document_service = ctx['document_service']
        
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
        
        # 6. Create document object
        doc = Document(
            content_type=content_type,
            filename=original_filename,
            metadata=metadata,
            owner={"type": auth.entity_type, "id": auth.entity_id},
            access_control={
                "readers": [auth.entity_id],
                "writers": [auth.entity_id],
                "admins": [auth.entity_id],
                "user_id": [auth.user_id] if auth.user_id else [],
            },
            additional_metadata=additional_metadata,
        )
        
        # Add folder_name and end_user_id to system_metadata if provided
        if folder_name:
            doc.system_metadata["folder_name"] = folder_name
        if end_user_id:
            doc.system_metadata["end_user_id"] = end_user_id
        
        # Store full content
        doc.system_metadata["content"] = text
        
        # The file is already stored, just set the storage info
        doc.storage_info = {"bucket": bucket, "key": file_key}
        logger.debug(f"File already stored in bucket `{bucket}` with key `{file_key}`")
        
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
        
        # 11. Store chunks and document
        doc.chunk_ids = await document_service._store_chunks_and_doc(
            chunk_objects, doc, use_colpali, chunk_objects_multivector
        )
        logger.debug(f"Successfully stored file document {doc.external_id}")
        
        # 12. Log successful completion
        logger.info(f"Successfully completed ingestion for {original_filename}, document ID: {doc.external_id}")
        
        # 13. Return document ID
        return {
            "document_id": doc.external_id,
            "status": "completed",
            "filename": original_filename,
            "content_type": content_type,
            "timestamp": datetime.now(UTC).isoformat()
        }
            
    except Exception as e:
        logger.error(f"Error processing ingestion job for file {original_filename}: {str(e)}")
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
    max_jobs = 10  # Maximum number of jobs to run concurrently