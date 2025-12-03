"""Centralised initialisation of core services.

This file was introduced during the refactor of `core/api.py` to keep the
monolithic API file small.  It performs *exactly* the same initialisation
logic that previously lived in `core/api.py` (lines ~90-210) and exposes the
created singletons so that other modules can simply import them:

    from core.services_init import document_service, settings

No behaviour has changed – only the physical location of the code.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.completion.litellm_completion import LiteLLMCompletionModel
from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.colpali_api_embedding_model import ColpaliApiEmbeddingModel
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.parser.morphik_parser import MorphikParser
from core.reranker.flag_reranker import FlagReranker
from core.services.document_service import DocumentService
from core.services.ingestion_service import IngestionService
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.vector_store.dual_multivector_store import DualMultiVectorStore
from core.vector_store.fast_multivector_store import FastMultiVectorStore
from core.vector_store.multi_vector_store import MultiVectorStore
from core.vector_store.pgvector_store import PGVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

settings = get_settings()

# ---------------------------------------------------------------------------
# Database & vector store
# ---------------------------------------------------------------------------

if not settings.POSTGRES_URI:
    raise ValueError("PostgreSQL URI is required for PostgreSQL database")

database = PostgresDatabase(uri=settings.POSTGRES_URI)
logger.debug("Created PostgresDatabase singleton")

vector_store = PGVectorStore(uri=settings.POSTGRES_URI)
logger.debug("Created PGVectorStore singleton")

# ---------------------------------------------------------------------------
# Object storage
# ---------------------------------------------------------------------------

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
logger.debug("Initialised Storage layer: %s", settings.STORAGE_PROVIDER)

# ---------------------------------------------------------------------------
# Parser & models
# ---------------------------------------------------------------------------

parser = MorphikParser(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
    anthropic_api_key=settings.ANTHROPIC_API_KEY,
    use_contextual_chunking=settings.USE_CONTEXTUAL_CHUNKING,
    settings=settings,
)

embedding_model = LiteLLMEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
logger.info("Initialized LiteLLM embedding model with model key: %s", settings.EMBEDDING_MODEL)

completion_model = LiteLLMCompletionModel(model_key=settings.COMPLETION_MODEL)
logger.info("Initialized LiteLLM completion model with model key: %s", settings.COMPLETION_MODEL)

# ---------------------------------------------------------------------------
# Optional reranker
# ---------------------------------------------------------------------------

reranker: Optional[FlagReranker] = None
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
logger.debug("Reranker enabled: %s", bool(reranker))

# ---------------------------------------------------------------------------
# ColPali multi-vector support
# ---------------------------------------------------------------------------

# Check enable_colpali first - if disabled, skip all ColPali initialization
if not settings.ENABLE_COLPALI:
    logger.info("ColPali disabled by configuration (enable_colpali=false)")
    colpali_embedding_model = None
    colpali_vector_store = None
else:
    # Only initialize ColPali if enabled AND mode is not "off"
    match settings.COLPALI_MODE:
        case "off":
            logger.info("ColPali mode set to 'off'")
            colpali_embedding_model = None
            colpali_vector_store = None
        case "local":
            logger.info("Initializing ColPali in local mode")
            from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel

            colpali_embedding_model = ColpaliEmbeddingModel()
            # Choose multivector store implementation based on provider and dual ingestion setting
            if settings.ENABLE_DUAL_MULTIVECTOR_INGESTION:
                # Dual ingestion mode: create both stores and wrap them
                if not settings.TURBOPUFFER_API_KEY:
                    raise ValueError("TURBOPUFFER_API_KEY is required when dual ingestion is enabled")

                fast_store = FastMultiVectorStore(
                    uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
                )
                slow_store = MultiVectorStore(
                    uri=settings.POSTGRES_URI, enable_external_storage=True, auto_initialize=False
                )
                colpali_vector_store = DualMultiVectorStore(
                    fast_store=fast_store, slow_store=slow_store, enable_dual_ingestion=True
                )
                logger.info("Initialized DualMultiVectorStore for migration (dual ingestion enabled)")
            elif settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
                if not settings.TURBOPUFFER_API_KEY:
                    raise ValueError("TURBOPUFFER_API_KEY is required when using morphik multivector store provider")
                colpali_vector_store = FastMultiVectorStore(
                    uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
                )
            else:
                colpali_vector_store = MultiVectorStore(
                    uri=settings.POSTGRES_URI, enable_external_storage=True, auto_initialize=False
                )
        case "api":
            logger.info("Initializing ColPali in API mode")
            colpali_embedding_model = ColpaliApiEmbeddingModel()
            # Choose multivector store implementation based on provider and dual ingestion setting
            if settings.ENABLE_DUAL_MULTIVECTOR_INGESTION:
                # Dual ingestion mode: create both stores and wrap them
                if not settings.TURBOPUFFER_API_KEY:
                    raise ValueError("TURBOPUFFER_API_KEY is required when dual ingestion is enabled")

                fast_store = FastMultiVectorStore(
                    uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
                )
                slow_store = MultiVectorStore(
                    uri=settings.POSTGRES_URI, enable_external_storage=True, auto_initialize=False
                )
                colpali_vector_store = DualMultiVectorStore(
                    fast_store=fast_store, slow_store=slow_store, enable_dual_ingestion=True
                )
                logger.info("Initialized DualMultiVectorStore for migration (dual ingestion enabled)")
            elif settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
                if not settings.TURBOPUFFER_API_KEY:
                    raise ValueError("TURBOPUFFER_API_KEY is required when using morphik multivector store provider")
                colpali_vector_store = FastMultiVectorStore(
                    uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
                )
            else:
                colpali_vector_store = MultiVectorStore(
                    uri=settings.POSTGRES_URI, enable_external_storage=True, auto_initialize=False
                )
        case _:
            raise ValueError(f"Unsupported COLPALI_MODE: {settings.COLPALI_MODE}")

# ---------------------------------------------------------------------------
# Document service (ties everything together)
# ---------------------------------------------------------------------------

document_service = DocumentService(
    database=database,
    vector_store=vector_store,
    storage=storage,
    parser=parser,
    embedding_model=embedding_model,
    completion_model=completion_model,
    reranker=reranker,
    enable_colpali=settings.ENABLE_COLPALI,
    colpali_embedding_model=colpali_embedding_model,
    colpali_vector_store=colpali_vector_store,
)
logger.info("Document service initialised")

# ---------------------------------------------------------------------------
# Ingestion service (handles document ingestion operations)
# ---------------------------------------------------------------------------

ingestion_service = IngestionService(
    database=database,
    vector_store=vector_store,
    storage=storage,
    parser=parser,
    embedding_model=embedding_model,
    colpali_embedding_model=colpali_embedding_model,
    colpali_vector_store=colpali_vector_store,
)
logger.info("Ingestion service initialised")

__all__ = [
    "settings",
    "database",
    "vector_store",
    "storage",
    "embedding_model",
    "completion_model",
    "document_service",
    "ingestion_service",
]
