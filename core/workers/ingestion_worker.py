import asyncio
import json
import logging
import os
import time
import urllib.parse as up
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from arq.connections import RedisSettings
from sqlalchemy import text

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.colpali_api_embedding_model import ColpaliApiEmbeddingModel
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.limits_utils import check_and_increment_limits
from core.models.auth import AuthContext, EntityType
from core.models.rules import MetadataExtractionRule
from core.parser.morphik_parser import MorphikParser
from core.services.document_service import CHARS_PER_TOKEN, TOKENS_PER_PAGE, DocumentService
from core.services.rules_processor import RulesProcessor
from core.services.telemetry import TelemetryService
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.vector_store.multi_vector_store import MultiVectorStore
from core.vector_store.pgvector_store import PGVectorStore

# Enterprise routing helpers
from ee.db_router import get_database_for_app, get_vector_store_for_app

logger = logging.getLogger(__name__)

# Initialize global settings once
settings = get_settings()

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up file handler for worker_ingestion.log
file_handler = logging.FileHandler("logs/worker_ingestion.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
# Set logger level based on settings (diff used INFO directly)
logger.setLevel(logging.INFO)


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
                logger.warning(
                    f"Document {document_id} not found on attempt {attempt}/{max_retries}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5

        except Exception as e:
            attempt += 1
            error_msg = str(e)
            if attempt < max_retries:
                logger.warning(
                    f"Error retrieving document on attempt {attempt}/{max_retries}: {error_msg}. "
                    f"Retrying in {retry_delay}s..."
                )
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
    end_user_id: Optional[str] = None,
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
        # Start performance timer
        job_start_time = time.time()
        phase_times = {}
        # 1. Log the start of the job
        logger.info(f"Starting ingestion job for file: {original_filename}")

        # 2. Deserialize metadata and auth
        deserialize_start = time.time()
        metadata = json.loads(metadata_json) if metadata_json else {}
        auth = AuthContext(
            entity_type=EntityType(auth_dict.get("entity_type", "unknown")),
            entity_id=auth_dict.get("entity_id", ""),
            app_id=auth_dict.get("app_id"),
            permissions=set(auth_dict.get("permissions", ["read"])),
            user_id=auth_dict.get("user_id", auth_dict.get("entity_id", "")),
        )
        phase_times["deserialize_auth"] = time.time() - deserialize_start

        # ------------------------------------------------------------------
        # Per-app routing for database and vector store
        # ------------------------------------------------------------------

        # Resolve a dedicated database/vector-store using the JWT *app_id*.
        # When app_id is None we fall back to the control-plane resources.

        database = await get_database_for_app(auth.app_id)
        await database.initialize()

        vector_store = await get_vector_store_for_app(auth.app_id)
        if vector_store and hasattr(vector_store, "initialize"):
            # PGVectorStore.initialize is *async*
            try:
                await vector_store.initialize()
            except Exception as init_err:
                logger.warning(f"Vector store initialization failed for app {auth.app_id}: {init_err}")

        # Initialise a per-app MultiVectorStore for ColPali when needed
        colpali_vector_store = None
        if use_colpali:
            try:
                # Use render_as_string(hide_password=False) so the URI keeps the
                # password – str(engine.url) masks it with "***" which breaks
                # authentication for psycopg.  Also append sslmode=require when
                # missing to satisfy Neon.
                from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

                uri_raw = database.engine.url.render_as_string(hide_password=False)

                parsed = urlparse(uri_raw)
                query = parse_qs(parsed.query)
                if "sslmode" not in query and settings.MODE == "cloud":
                    query["sslmode"] = ["require"]
                    parsed = parsed._replace(query=urlencode(query, doseq=True))

                uri_final = urlunparse(parsed)

                colpali_vector_store = MultiVectorStore(uri=uri_final)
                await asyncio.to_thread(colpali_vector_store.initialize)
            except Exception as e:
                logger.warning(f"Failed to initialise ColPali MultiVectorStore for app {auth.app_id}: {e}")

        # Build a fresh DocumentService scoped to this job/app so we don't
        # mutate the shared instance kept in *ctx* (avoids cross-talk between
        # concurrent jobs for different apps).
        document_service = DocumentService(
            storage=ctx["storage"],
            database=database,
            vector_store=vector_store,
            embedding_model=ctx["embedding_model"],
            parser=ctx["parser"],
            cache_factory=None,
            enable_colpali=use_colpali,
            colpali_embedding_model=ctx.get("colpali_embedding_model"),
            colpali_vector_store=colpali_vector_store,
        )

        # 3. Download the file from storage
        logger.info(f"Downloading file from {bucket}/{file_key}")
        download_start = time.time()
        file_content = await document_service.storage.download_file(bucket, file_key)

        # Ensure file_content is bytes
        if hasattr(file_content, "read"):
            file_content = file_content.read()
        download_time = time.time() - download_start
        phase_times["download_file"] = download_time
        logger.info(f"File download took {download_time:.2f}s for {len(file_content)/1024/1024:.2f}MB")

        # 4. Parse file to text
        parse_start = time.time()
        additional_metadata, text = await document_service.parser.parse_file_to_text(file_content, original_filename)
        logger.debug(f"Parsed file into text of length {len(text)}")
        parse_time = time.time() - parse_start
        phase_times["parse_file"] = parse_time

        # NEW -----------------------------------------------------------------
        # 4.b Enforce tier limits (pages ingested) for cloud/free tier users
        if settings.MODE == "cloud" and auth.user_id:
            # Calculate approximate pages using same heuristic as DocumentService
            num_pages = int(len(text) / (CHARS_PER_TOKEN * TOKENS_PER_PAGE)) or 1
            try:
                await check_and_increment_limits(auth, "ingest", num_pages, document_id)
            except Exception as limit_exc:
                logger.error("User %s exceeded ingest limits: %s", auth.user_id, limit_exc)
                raise
        # ---------------------------------------------------------------------

        # === Apply post_parsing rules ===
        rules_start = time.time()
        document_rule_metadata = {}
        if rules_list:
            logger.info("Applying post-parsing rules...")
            document_rule_metadata, text = await document_service.rules_processor.process_document_rules(
                text, rules_list
            )
            metadata.update(document_rule_metadata)  # Merge metadata into main doc metadata
            logger.info(f"Document metadata after post-parsing rules: {metadata}")
            logger.info(f"Content length after post-parsing rules: {len(text)}")
        rules_time = time.time() - rules_start
        phase_times["apply_post_parsing_rules"] = rules_time
        if rules_list:
            logger.info(f"Post-parsing rules processing took {rules_time:.2f}s")

        # 6. Retrieve the existing document
        retrieve_start = time.time()
        logger.debug(f"Retrieving document with ID: {document_id}")
        logger.debug(
            f"Auth context: entity_type={auth.entity_type}, entity_id={auth.entity_id}, permissions={auth.permissions}"
        )

        # Use the retry helper function with initial delay to handle race conditions
        doc = await get_document_with_retry(document_service, document_id, auth, max_retries=5, initial_delay=1.0)
        retrieve_time = time.time() - retrieve_start
        phase_times["retrieve_document"] = retrieve_time
        logger.info(f"Document retrieval took {retrieve_time:.2f}s")

        if not doc:
            logger.error(f"Document {document_id} not found in database after multiple retries")
            logger.error(
                f"Details - file: {original_filename}, content_type: {content_type}, bucket: {bucket}, key: {file_key}"
            )
            logger.error(
                f"Auth: entity_type={auth.entity_type}, entity_id={auth.entity_id}, permissions={auth.permissions}"
            )
            raise ValueError(f"Document {document_id} not found in database after multiple retries")

        # Prepare updates for the document
        # Merge new metadata with existing metadata to preserve external_id
        merged_metadata = {**doc.metadata, **metadata}
        # Make sure external_id is preserved in the metadata
        merged_metadata["external_id"] = doc.external_id

        updates = {
            "metadata": merged_metadata,
            "additional_metadata": additional_metadata,
            "system_metadata": {**doc.system_metadata, "content": text},
        }

        # Add folder_name and end_user_id to system_metadata if provided
        if folder_name:
            updates["system_metadata"]["folder_name"] = folder_name
        if end_user_id:
            updates["system_metadata"]["end_user_id"] = end_user_id

        # Update the document in the database
        update_start = time.time()
        success = await document_service.db.update_document(document_id=document_id, updates=updates, auth=auth)
        update_time = time.time() - update_start
        phase_times["update_document_parsed"] = update_time
        logger.info(f"Initial document update took {update_time:.2f}s")

        if not success:
            raise ValueError(f"Failed to update document {document_id}")

        # Refresh document object with updated data
        doc = await document_service.db.get_document(document_id, auth)
        logger.debug("Updated document in database with parsed content")

        # 7. Split text into chunks
        chunking_start = time.time()
        parsed_chunks = await document_service.parser.split_text(text)
        if not parsed_chunks:
            # No text was extracted from the file.  In many cases (e.g. pure images)
            # we can still proceed if ColPali multivector chunks are produced later.
            # Therefore we defer the fatal check until after ColPali chunk creation.
            logger.warning(
                "No text chunks extracted after parsing. Will attempt to continue "
                "and rely on image-based chunks if available."
            )
        chunking_time = time.time() - chunking_start
        phase_times["split_into_chunks"] = chunking_time
        logger.info(f"Text chunking took {chunking_time:.2f}s to create {len(parsed_chunks)} chunks")

        # Decide whether we need image chunks either for ColPali embedding or because
        # there are image-based rules (use_images=True) that must process them.
        has_image_rules = any(
            r.get("stage", "post_parsing") == "post_chunking"
            and r.get("type") == "metadata_extraction"
            and r.get("use_images", False)
            for r in rules_list or []
        )

        using_colpali = (
            use_colpali and document_service.colpali_embedding_model and document_service.colpali_vector_store
        )

        should_create_image_chunks = has_image_rules or using_colpali

        # Start timer for optional image chunk creation / multivector processing
        colpali_processing_start = time.time()

        chunks_multivector = []
        if should_create_image_chunks:
            import base64

            import filetype

            file_type = filetype.guess(file_content)
            file_content_base64 = base64.b64encode(file_content).decode()

            # Use the parsed chunks for ColPali/image rules – this will create image chunks if appropriate
            chunks_multivector = document_service._create_chunks_multivector(
                file_type, file_content_base64, file_content, parsed_chunks
            )
            logger.debug(
                f"Created {len(chunks_multivector)} multivector/image chunks "
                f"(has_image_rules={has_image_rules}, using_colpali={using_colpali})"
            )
        colpali_create_chunks_time = time.time() - colpali_processing_start
        phase_times["colpali_create_chunks"] = colpali_create_chunks_time
        if using_colpali:
            logger.info(f"Colpali chunk creation took {colpali_create_chunks_time:.2f}s")

        # If we still have no chunks at all (neither text nor image) abort early
        if not parsed_chunks and not chunks_multivector:
            raise ValueError("No content chunks (text or image) could be extracted from the document")

        # 9. Apply post_chunking rules and aggregate metadata
        processed_chunks = []
        processed_chunks_multivector = []
        aggregated_chunk_metadata: Dict[str, Any] = {}  # Initialize dict for aggregated metadata
        chunk_contents = []  # Initialize list to collect chunk contents as we process them

        if rules_list:
            logger.info("Applying post-chunking rules...")

            # Partition rules by type
            text_rules = []
            image_rules = []

            for rule_dict in rules_list:
                rule = document_service.rules_processor._parse_rule(rule_dict)
                if rule.stage == "post_chunking":
                    if isinstance(rule, MetadataExtractionRule) and rule.use_images:
                        image_rules.append(rule_dict)
                    else:
                        text_rules.append(rule_dict)

            logger.info(f"Partitioned rules: {len(text_rules)} text rules, {len(image_rules)} image rules")

            # Process regular text chunks with text rules only
            if text_rules:
                logger.info(f"Applying {len(text_rules)} text rules to text chunks...")
                for chunk_obj in parsed_chunks:
                    # Get metadata *and* the potentially modified chunk
                    chunk_rule_metadata, processed_chunk = await document_service.rules_processor.process_chunk_rules(
                        chunk_obj, text_rules
                    )
                    processed_chunks.append(processed_chunk)
                    chunk_contents.append(processed_chunk.content)  # Collect content as we process
                    # Aggregate the metadata extracted from this chunk
                    aggregated_chunk_metadata.update(chunk_rule_metadata)
            else:
                processed_chunks = parsed_chunks  # No text rules, use original chunks

            # Process colpali image chunks with image rules if they exist
            if chunks_multivector and image_rules:
                logger.info(f"Applying {len(image_rules)} image rules to image chunks...")
                for chunk_obj in chunks_multivector:
                    # Only process if it's an image chunk - pass the image content to the rule
                    if chunk_obj.metadata.get("is_image", False):
                        # Get metadata *and* the potentially modified chunk
                        chunk_rule_metadata, processed_chunk = (
                            await document_service.rules_processor.process_chunk_rules(chunk_obj, image_rules)
                        )
                        processed_chunks_multivector.append(processed_chunk)
                        # Aggregate the metadata extracted from this chunk
                        aggregated_chunk_metadata.update(chunk_rule_metadata)
                    else:
                        # Non-image chunks from multivector don't need further processing
                        processed_chunks_multivector.append(chunk_obj)

                logger.info(f"Finished applying image rules to {len(processed_chunks_multivector)} image chunks.")
            elif chunks_multivector:
                # No image rules, use original multivector chunks
                processed_chunks_multivector = chunks_multivector

            logger.info(f"Finished applying post-chunking rules to {len(processed_chunks)} regular chunks.")
            logger.info(f"Aggregated metadata from all chunks: {aggregated_chunk_metadata}")

            # Update the document content with the stitched content from processed chunks
            if processed_chunks:
                logger.info("Updating document content with processed chunks...")
                stitched_content = "\n".join(chunk_contents)
                doc.system_metadata["content"] = stitched_content
                logger.info(f"Updated document content with stitched chunks (length: {len(stitched_content)})")
        else:
            processed_chunks = parsed_chunks  # No rules, use original chunks
            processed_chunks_multivector = chunks_multivector  # No rules, use original multivector chunks

        # 10. Generate embeddings for processed chunks
        embedding_start = time.time()
        embeddings = await document_service.embedding_model.embed_for_ingestion(processed_chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        embedding_time = time.time() - embedding_start
        phase_times["generate_embeddings"] = embedding_time
        embeddings_per_second = len(embeddings) / embedding_time if embedding_time > 0 else 0
        logger.info(
            f"Embedding generation took {embedding_time:.2f}s for {len(embeddings)} embeddings "
            f"({embeddings_per_second:.2f} embeddings/s)"
        )

        # 11. Create chunk objects with potentially modified chunk content and metadata
        chunk_objects_start = time.time()
        chunk_objects = document_service._create_chunk_objects(doc.external_id, processed_chunks, embeddings)
        logger.debug(f"Created {len(chunk_objects)} chunk objects")
        chunk_objects_time = time.time() - chunk_objects_start
        phase_times["create_chunk_objects"] = chunk_objects_time
        logger.debug(f"Creating chunk objects took {chunk_objects_time:.2f}s")

        # 12. Handle ColPali embeddings
        colpali_embed_start = time.time()
        chunk_objects_multivector = []
        if using_colpali:
            colpali_embeddings = await document_service.colpali_embedding_model.embed_for_ingestion(
                processed_chunks_multivector
            )
            logger.debug(f"Generated {len(colpali_embeddings)} embeddings for multivector embedding")

            chunk_objects_multivector = document_service._create_chunk_objects(
                doc.external_id, processed_chunks_multivector, colpali_embeddings
            )
        colpali_embed_time = time.time() - colpali_embed_start
        phase_times["colpali_generate_embeddings"] = colpali_embed_time
        if using_colpali:
            embeddings_per_second = len(colpali_embeddings) / colpali_embed_time if colpali_embed_time > 0 else 0
            logger.info(
                f"Colpali embedding took {colpali_embed_time:.2f}s for {len(colpali_embeddings)} embeddings "
                f"({embeddings_per_second:.2f} embeddings/s)"
            )

        # === Merge aggregated chunk metadata into document metadata ===
        if aggregated_chunk_metadata:
            logger.info("Merging aggregated chunk metadata into document metadata...")
            # Make sure doc.metadata exists
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(aggregated_chunk_metadata)
            logger.info(f"Final document metadata after merge: {doc.metadata}")
        # ===========================================================

        # Update document status to completed before storing
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)

        # 11. Store chunks and update document with is_update=True
        store_start = time.time()
        await document_service._store_chunks_and_doc(
            chunk_objects, doc, use_colpali, chunk_objects_multivector, is_update=True, auth=auth
        )
        store_time = time.time() - store_start
        phase_times["store_chunks_and_update_doc"] = store_time
        logger.info(f"Storing chunks and final document update took {store_time:.2f}s for {len(chunk_objects)} chunks")

        logger.debug(f"Successfully completed processing for document {doc.external_id}")

        # 13. Log successful completion
        logger.info(f"Successfully completed ingestion for {original_filename}, document ID: {doc.external_id}")
        # Performance summary
        total_time = time.time() - job_start_time

        # Log performance summary
        logger.info("=== Ingestion Performance Summary ===")
        logger.info(f"Total processing time: {total_time:.2f}s")
        for phase, duration in sorted(phase_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")
        logger.info("=====================================")

        # 14. Return document ID
        return {
            "document_id": doc.external_id,
            "status": "completed",
            "filename": original_filename,
            "content_type": content_type,
            "timestamp": datetime.now(UTC).isoformat(),
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
                user_id=auth_dict.get("user_id", auth_dict.get("entity_id", "")),
            )

            # Get database from context
            database = ctx.get("database")

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
                                "updated_at": datetime.now(UTC),
                            }
                        },
                        auth=auth_context,
                    )
                    logger.info(f"Updated document {document_id} status to failed")
        except Exception as inner_e:
            logger.error(f"Failed to update document status: {str(inner_e)}")

        # Return error information
        return {
            "status": "failed",
            "filename": original_filename,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


async def startup(ctx):
    """
    Worker startup: Initialize all necessary services that will be reused across jobs.

    This initialization is similar to what happens in core/api.py during app startup,
    but adapted for the worker context.
    """
    logger.info("Worker starting up. Initializing services...")

    # Initialize database
    logger.info("Initializing database...")
    database = PostgresDatabase(uri=settings.POSTGRES_URI)
    success = await database.initialize()
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")
    ctx["database"] = database

    # Initialize vector store
    logger.info("Initializing primary vector store...")
    vector_store = PGVectorStore(uri=settings.POSTGRES_URI)
    success = await vector_store.initialize()
    if success:
        logger.info("Primary vector store initialization successful")
    else:
        logger.error("Primary vector store initialization failed")
    ctx["vector_store"] = vector_store

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
    ctx["storage"] = storage

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
    ctx["parser"] = parser

    # Initialize embedding model
    embedding_model = LiteLLMEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
    logger.info(f"Initialized LiteLLM embedding model with model key: {settings.EMBEDDING_MODEL}")
    ctx["embedding_model"] = embedding_model

    # Skip initializing completion model and reranker since they're not needed for ingestion

    # Initialize ColPali embedding model and vector store per mode
    colpali_embedding_model = None
    colpali_vector_store = None

    if settings.COLPALI_MODE != "off":
        logger.info(f"Initializing ColPali components (mode={settings.COLPALI_MODE}) ...")
        # Choose embedding implementation
        match settings.COLPALI_MODE:
            case "local":
                colpali_embedding_model = ColpaliEmbeddingModel()
            case "api":
                colpali_embedding_model = ColpaliApiEmbeddingModel()
            case _:
                raise ValueError(f"Unsupported COLPALI_MODE: {settings.COLPALI_MODE}")

        # Vector store is needed for both local and api modes
        colpali_vector_store = MultiVectorStore(uri=settings.POSTGRES_URI)
        success = await asyncio.to_thread(colpali_vector_store.initialize)
        if success:
            logger.info("ColPali vector store initialization successful")
        else:
            logger.error("ColPali vector store initialization failed")
    ctx["colpali_embedding_model"] = colpali_embedding_model
    ctx["colpali_vector_store"] = colpali_vector_store
    ctx["cache_factory"] = None

    # Initialize rules processor
    rules_processor = RulesProcessor()
    ctx["rules_processor"] = rules_processor

    # Initialize telemetry service
    telemetry = TelemetryService()
    ctx["telemetry"] = telemetry

    # Create the document service using only the components needed for ingestion
    document_service = DocumentService(
        storage=storage,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        parser=parser,
        cache_factory=None,
        enable_colpali=(settings.COLPALI_MODE != "off"),
        colpali_embedding_model=colpali_embedding_model,
        colpali_vector_store=colpali_vector_store,
    )
    ctx["document_service"] = document_service

    logger.info("Worker startup complete. All services initialized.")


async def shutdown(ctx):
    """
    Worker shutdown: Clean up resources.

    Properly close connections and cleanup resources to prevent leaks.
    """
    logger.info("Worker shutting down. Cleaning up resources...")

    # Close database connections
    if "database" in ctx and hasattr(ctx["database"], "engine"):
        logger.info("Closing database connections...")
        await ctx["database"].engine.dispose()

    # Close vector store connections if they exist
    if "vector_store" in ctx and hasattr(ctx["vector_store"], "engine"):
        logger.info("Closing vector store connections...")
        await ctx["vector_store"].engine.dispose()

    # Close colpali vector store connections if they exist
    if "colpali_vector_store" in ctx and hasattr(ctx["colpali_vector_store"], "engine"):
        logger.info("Closing colpali vector store connections...")
        await ctx["colpali_vector_store"].engine.dispose()

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
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=int(url.path.lstrip("/") or 0),
        conn_timeout=5,  # Increased connection timeout (seconds)
        conn_retries=15,  # More retries for transient connection issues
        conn_retry_delay=1,  # Quick retry delay (seconds)
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
        database = ctx.get("database")
        vector_store = ctx.get("vector_store")
        job_stats = ctx.get("job_stats", {})

        # Get detailed Redis info
        try:
            redis_info = await ctx["redis"].info(section=["Server", "Memory", "Clients", "Stats"])

            # Server and resource usage info
            redis_version = redis_info.get("redis_version", "unknown")
            used_memory = redis_info.get("used_memory_human", "unknown")
            used_memory_peak = redis_info.get("used_memory_peak_human", "unknown")
            clients_connected = redis_info.get("connected_clients", "unknown")
            rejected_connections = redis_info.get("rejected_connections", 0)
            total_commands = redis_info.get("total_commands_processed", 0)

            # DB keys
            db_info = redis_info.get("db0", {})
            keys_count = db_info.get("keys", 0) if isinstance(db_info, dict) else 0

            # Log comprehensive server status
            logger.info(
                f"Redis Status: v{redis_version} | "
                f"Memory: {used_memory} (peak: {used_memory_peak}) | "
                f"Clients: {clients_connected} (rejected: {rejected_connections}) | "
                f"DB Keys: {keys_count} | Commands: {total_commands}"
            )

            # Check for memory warning thresholds
            if isinstance(used_memory, str) and used_memory.endswith("G"):
                memory_value = float(used_memory[:-1])
                if memory_value > 1.0:  # More than 1GB used
                    logger.warning(f"Redis memory usage is high: {used_memory}")

            # Check for connection issues
            if rejected_connections and int(rejected_connections) > 0:
                logger.warning(f"Redis has rejected {rejected_connections} connections")
        except Exception as e:
            logger.error(f"Failed to get Redis info: {str(e)}")

        # Log job statistics with detailed processing metrics
        ongoing = job_stats.get("ongoing", 0)
        queued = job_stats.get("queued", 0)

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
        if database and hasattr(database, "async_session"):
            try:
                async with database.async_session() as session:
                    await session.execute(text("SELECT 1"))
                    logger.debug("Database connection is healthy")
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")

        # Test vector store connectivity if available
        if vector_store and hasattr(vector_store, "async_session"):
            try:
                async with vector_store.get_session_with_retry() as session:
                    logger.debug("Vector store connection is healthy")
            except Exception as e:
                logger.error(f"Vector store connection test failed: {str(e)}")
