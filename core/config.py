import os
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

import tomli
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from utils.env_loader import load_local_env

# Default to loading from .env unless a secret manager (e.g., Infisical) is
# injecting variables.
load_local_env(override=True)


class ParserXMLSettings(BaseModel):
    max_tokens: int = 350
    preferred_unit_tags: List[str] = ["SECTION", "Section", "Article", "clause"]
    ignore_tags: List[str] = ["TOC", "INDEX"]


class Settings(BaseSettings):
    """Morphik configuration settings."""

    # Environment variables
    JWT_SECRET_KEY: str
    SESSION_SECRET_KEY: str
    POSTGRES_URI: Optional[str] = None
    AWS_ACCESS_KEY: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ASSEMBLYAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    TURBOPUFFER_API_KEY: Optional[str] = None
    GEMINI_API_BASE_URL: str = "https://generativelanguage.googleapis.com"
    GEMINI_METADATA_MODEL: str = "gemini-2.5-flash"

    # API configuration
    HOST: str
    PORT: int
    RELOAD: bool
    SENTRY_DSN: Optional[str] = None
    PROJECT_NAME: Optional[str] = None
    # Morphik Embedding API server configuration
    MORPHIK_EMBEDDING_API_KEY: Optional[str] = None
    MORPHIK_EMBEDDING_API_DOMAIN: str

    # Auth configuration
    JWT_ALGORITHM: str
    bypass_auth_mode: bool = False
    dev_user_id: str = "dev_user"
    ADMIN_SERVICE_SECRET: Optional[str] = None

    # Registered models configuration
    REGISTERED_MODELS: Dict[str, Dict[str, Any]] = {}

    # Completion configuration
    COMPLETION_PROVIDER: Literal["litellm"] = "litellm"
    COMPLETION_MODEL: str

    # Document analysis configuration
    DOCUMENT_ANALYSIS_MODEL: str

    # Database configuration
    DATABASE_PROVIDER: Literal["postgres"]
    DATABASE_NAME: Optional[str] = None
    # Database connection pool settings
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_POOL_TIMEOUT: int = 10
    DB_POOL_PRE_PING: bool = True
    DB_MAX_RETRIES: int = 3
    DB_RETRY_DELAY: float = 1.0

    # Embedding configuration
    EMBEDDING_PROVIDER: Literal["litellm"] = "litellm"
    EMBEDDING_MODEL: str
    VECTOR_DIMENSIONS: int
    EMBEDDING_SIMILARITY_METRIC: Literal["cosine", "dotProduct"]

    # Parser configuration
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    FRAME_SAMPLE_RATE: Optional[int] = None
    USE_CONTEXTUAL_CHUNKING: bool = False
    PARSER_XML: ParserXMLSettings = ParserXMLSettings()

    # Graph configuration
    GRAPH_MODE: Literal["local", "api"] = "local"
    GRAPH_PROVIDER: Literal["litellm"] = "litellm"
    GRAPH_MODEL: Optional[str] = None
    ENABLE_ENTITY_RESOLUTION: Optional[bool] = None
    # Graph API configuration
    MORPHIK_GRAPH_API_KEY: Optional[str] = None
    MORPHIK_GRAPH_BASE_URL: Optional[str] = None

    # Reranker configuration
    USE_RERANKING: bool
    RERANKER_PROVIDER: Optional[Literal["flag"]] = None
    RERANKER_MODEL: Optional[str] = None
    RERANKER_QUERY_MAX_LENGTH: Optional[int] = None
    RERANKER_PASSAGE_MAX_LENGTH: Optional[int] = None
    RERANKER_USE_FP16: Optional[bool] = None
    RERANKER_DEVICE: Optional[str] = None

    # Storage configuration
    STORAGE_PROVIDER: Literal["local", "aws-s3"]
    STORAGE_PATH: Optional[str] = None
    AWS_REGION: Optional[str] = None
    S3_BUCKET: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    CACHE_ENABLED: bool = False
    CACHE_MAX_BYTES: int = 10 * 1024 * 1024 * 1024
    CACHE_CHUNK_MAX_BYTES: int = 10 * 1024 * 1024 * 1024
    CACHE_PATH: str = "./storage/cache"

    # Vector store configuration
    VECTOR_STORE_PROVIDER: Literal["pgvector"]
    VECTOR_STORE_DATABASE_NAME: Optional[str] = None

    # Multivector store configuration
    MULTIVECTOR_STORE_PROVIDER: Literal["postgres", "morphik"] = "postgres"
    # Enable dual ingestion to both fast and slow multivector stores during migration
    ENABLE_DUAL_MULTIVECTOR_INGESTION: bool = False

    # Colpali configuration
    ENABLE_COLPALI: bool
    # Colpali embedding mode: off, local, or api
    COLPALI_MODE: Literal["off", "local", "api"] = "local"

    # Mode configuration
    MODE: Literal["cloud", "self_hosted"] = "cloud"
    SECRET_MANAGER: Literal["env", "infisical"] = "env"

    # API configuration
    API_DOMAIN: str = "api.morphik.ai"

    # PDF Viewer configuration
    PDF_VIEWER_FRONTEND_URL: Optional[str] = "https://morphik.ai/api/pdf"

    # Service configuration
    ENVIRONMENT: str = "development"
    VERSION: str = "unknown"
    ENABLE_PROFILING: bool = False

    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Worker configuration
    ARQ_MAX_JOBS: int = 1
    COLPALI_STORE_BATCH_SIZE: int = 16

    # PDF processing configuration
    COLPALI_PDF_DPI: int = 150

    # Telemetry configuration
    TELEMETRY_ENABLED: bool = True
    SERVICE_NAME: str = "databridge-core"
    PROJECT_NAME: Optional[str] = None
    TELEMETRY_UPLOAD_INTERVAL_HOURS: float = 4.0
    TELEMETRY_MAX_LOCAL_BYTES: int = 1073741824

    # LiteLLM configuration
    LITELLM_DUMMY_API_KEY: str = "ollama"

    # Local URI password for authentication
    LOCAL_URI_PASSWORD: Optional[str] = None

    @property
    def dev_mode(self) -> bool:  # pragma: no cover - compatibility shim
        """Backward-compatible alias for bypass_auth_mode."""
        return self.bypass_auth_mode


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    load_local_env(override=True)

    # Load config.toml
    with open("morphik.toml", "rb") as f:
        config = tomli.load(f)

    em = "'{missing_value}' needed if '{field}' is set to '{value}'"
    settings_dict = {}

    # Load API config
    settings_dict.update(
        {
            "HOST": config["api"]["host"],
            "PORT": int(config["api"]["port"]),
            "RELOAD": bool(config["api"]["reload"]),
            "SENTRY_DSN": os.getenv("SENTRY_DSN", None),
        }
    )

    # Load service config
    if "service" in config:
        service_cfg = config["service"]
        settings_dict.update(
            {
                "ENVIRONMENT": service_cfg.get("environment", "development"),
                "VERSION": service_cfg.get("version", "unknown"),
                "ENABLE_PROFILING": service_cfg.get("enable_profiling", False),
            }
        )

    # Load auth config
    settings_dict.update(
        {
            "JWT_ALGORITHM": config["auth"]["jwt_algorithm"],
            "JWT_SECRET_KEY": os.environ.get("JWT_SECRET_KEY", "dev-secret-key"),  # Default for bypass mode
            "SESSION_SECRET_KEY": os.environ.get("SESSION_SECRET_KEY", "super-secret-dev-session-key"),
            "bypass_auth_mode": config["auth"].get("bypass_auth_mode", config["auth"].get("dev_mode", False)),
            "dev_user_id": config["auth"].get("dev_user_id", config["auth"].get("dev_entity_id", "dev_user")),
        }
    )

    # Only require JWT_SECRET_KEY in non-dev mode
    if not settings_dict["bypass_auth_mode"] and "JWT_SECRET_KEY" not in os.environ:
        raise ValueError("JWT_SECRET_KEY is required when bypass_auth_mode is disabled")

    # Load registered models if available
    if "registered_models" in config:
        settings_dict["REGISTERED_MODELS"] = config["registered_models"]

    # Load completion config
    settings_dict["COMPLETION_PROVIDER"] = "litellm"
    if "model" not in config["completion"]:
        raise ValueError("'model' is required in the completion configuration")
    settings_dict["COMPLETION_MODEL"] = config["completion"]["model"]

    # Load database config
    settings_dict.update(
        {
            "DATABASE_PROVIDER": config["database"]["provider"],
            "DATABASE_NAME": config["database"].get("name", None),
            "DB_POOL_SIZE": config["database"].get("pool_size", 20),
            "DB_MAX_OVERFLOW": config["database"].get("max_overflow", 30),
            "DB_POOL_RECYCLE": config["database"].get("pool_recycle", 3600),
            "DB_POOL_TIMEOUT": config["database"].get("pool_timeout", 10),
            "DB_POOL_PRE_PING": config["database"].get("pool_pre_ping", True),
            "DB_MAX_RETRIES": config["database"].get("max_retries", 3),
            "DB_RETRY_DELAY": config["database"].get("retry_delay", 1.0),
        }
    )

    if settings_dict["DATABASE_PROVIDER"] != "postgres":
        raise ValueError(f"Unknown database provider selected: '{settings_dict['DATABASE_PROVIDER']}'")

    if "POSTGRES_URI" in os.environ:
        settings_dict["POSTGRES_URI"] = os.environ["POSTGRES_URI"]
    else:
        raise ValueError(em.format(missing_value="POSTGRES_URI", field="database.provider", value="postgres"))

    # Load embedding config
    settings_dict.update(
        {
            "EMBEDDING_PROVIDER": "litellm",
            "VECTOR_DIMENSIONS": config["embedding"]["dimensions"],
            "EMBEDDING_SIMILARITY_METRIC": config["embedding"]["similarity_metric"],
        }
    )

    if "model" not in config["embedding"]:
        raise ValueError("'model' is required in the embedding configuration")
    settings_dict["EMBEDDING_MODEL"] = config["embedding"]["model"]

    # Load parser config
    settings_dict.update(
        {
            "CHUNK_SIZE": config["parser"]["chunk_size"],
            "CHUNK_OVERLAP": config["parser"]["chunk_overlap"],
            "USE_CONTEXTUAL_CHUNKING": config["parser"].get("use_contextual_chunking", False),
        }
    )

    # Load parser XML config
    if "xml" in config["parser"]:
        xml_config = config["parser"]["xml"]
        settings_dict["PARSER_XML"] = ParserXMLSettings(
            max_tokens=xml_config.get("max_tokens", 350),
            preferred_unit_tags=xml_config.get("preferred_unit_tags", ["SECTION", "Section", "Article", "clause"]),
            ignore_tags=xml_config.get("ignore_tags", ["TOC", "INDEX"]),
        )

    # Load reranker config
    settings_dict["USE_RERANKING"] = config["reranker"]["use_reranker"]
    if settings_dict["USE_RERANKING"]:
        settings_dict.update(
            {
                "RERANKER_PROVIDER": config["reranker"]["provider"],
                "RERANKER_MODEL": config["reranker"]["model_name"],
                "RERANKER_QUERY_MAX_LENGTH": config["reranker"]["query_max_length"],
                "RERANKER_PASSAGE_MAX_LENGTH": config["reranker"]["passage_max_length"],
                "RERANKER_USE_FP16": config["reranker"]["use_fp16"],
                "RERANKER_DEVICE": config["reranker"]["device"],
            }
        )

    # Load storage config
    settings_dict.update(
        {
            "STORAGE_PROVIDER": config["storage"]["provider"],
            "STORAGE_PATH": config["storage"]["storage_path"],
            "S3_ENDPOINT_URL": config["storage"]["endpoint_url"],
        }
    )

    match settings_dict["STORAGE_PROVIDER"]:
        case "local":
            settings_dict["STORAGE_PATH"] = config["storage"]["storage_path"]
        case "aws-s3" if all(key in os.environ for key in ["AWS_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"]):
            settings_dict.update(
                {
                    "AWS_REGION": config["storage"]["region"],
                    "S3_BUCKET": config["storage"]["bucket_name"],
                    "AWS_ACCESS_KEY": os.environ["AWS_ACCESS_KEY"],
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "S3_ENDPOINT_URL": config["storage"]["endpoint_url"],
                }
            )
        case "aws-s3":
            raise ValueError(em.format(missing_value="AWS credentials", field="storage.provider", value="aws-s3"))
        case _:
            raise ValueError(f"Unknown storage provider selected: '{settings_dict['STORAGE_PROVIDER']}'")

    cache_base = config["storage"].get("storage_path") or "./storage"
    cache_enabled = config["storage"].get("cache_enabled", False)
    settings_dict["CACHE_ENABLED"] = bool(cache_enabled)
    cache_path_override = config["storage"].get("cache_path")
    settings_dict["CACHE_PATH"] = cache_path_override or os.path.join(cache_base, "cache")
    max_size_gb = (
        config["storage"].get("cache_max_size_gb")
        if "cache_max_size_gb" in config["storage"]
        else config["storage"].get("max_size_gb", 10)
    )
    try:
        cache_bytes = int(float(max_size_gb) * 1024 * 1024 * 1024)
    except (TypeError, ValueError):
        cache_bytes = 10 * 1024 * 1024 * 1024
    settings_dict["CACHE_MAX_BYTES"] = max(cache_bytes, 0)

    chunk_max_size_gb = config["storage"].get("cache_chunk_max_size_gb")
    if chunk_max_size_gb is None:
        chunk_max_size_gb = config["storage"].get("chunk_max_size_gb", 10)
    try:
        chunk_cache_bytes = int(float(chunk_max_size_gb) * 1024 * 1024 * 1024)
    except (TypeError, ValueError):
        chunk_cache_bytes = 10 * 1024 * 1024 * 1024
    settings_dict["CACHE_CHUNK_MAX_BYTES"] = max(chunk_cache_bytes, 0)

    # Load vector store config
    settings_dict["VECTOR_STORE_PROVIDER"] = config["vector_store"]["provider"]
    if settings_dict["VECTOR_STORE_PROVIDER"] != "pgvector":
        raise ValueError(f"Unknown vector store provider selected: '{settings_dict['VECTOR_STORE_PROVIDER']}'")

    if "POSTGRES_URI" not in os.environ:
        raise ValueError(em.format(missing_value="POSTGRES_URI", field="vector_store.provider", value="pgvector"))

    # Load morphik config
    api_domain = config["morphik"].get("api_domain", "api.morphik.ai")
    embedding_api_domain = config["morphik"].get("morphik_embedding_api_domain") or api_domain
    secret_manager = config["morphik"].get("secret_manager", "env")

    settings_dict.update(
        {
            "ENABLE_COLPALI": config["morphik"]["enable_colpali"],
            "COLPALI_MODE": config["morphik"].get("colpali_mode", "local"),
            "MODE": config["morphik"].get("mode", "cloud"),
            "SECRET_MANAGER": secret_manager,
            "API_DOMAIN": api_domain,
            "MORPHIK_EMBEDDING_API_DOMAIN": embedding_api_domain,
        }
    )

    # Load pdf viewer config
    if "pdf_viewer" in config:
        settings_dict["PDF_VIEWER_FRONTEND_URL"] = config["pdf_viewer"].get(
            "frontend_url", "https://morphik.ai/api/pdf"
        )

    # Load graph config
    if config["graph"].get("mode", "local") == "local":
        settings_dict.update(
            {
                "GRAPH_MODE": "local",
                "GRAPH_PROVIDER": "litellm",
                "ENABLE_ENTITY_RESOLUTION": config["graph"].get("enable_entity_resolution", True),
            }
        )
    else:
        settings_dict.update(
            {
                "GRAPH_MODE": "api",
                "MORPHIK_GRAPH_BASE_URL": config["graph"].get("base_url", "https://graph-api.morphik.ai"),
                "MORPHIK_GRAPH_API_KEY": os.environ.get("MORPHIK_GRAPH_API_KEY", None),
            }
        )

    if "model" not in config["graph"]:
        raise ValueError("'model' is required in the graph configuration")
    settings_dict["GRAPH_MODEL"] = config["graph"]["model"]

    # Load document analysis config
    if "document_analysis" in config:
        settings_dict["DOCUMENT_ANALYSIS_MODEL"] = config["document_analysis"]["model"]

    # Load redis config
    if "redis" in config:
        redis_cfg = config["redis"]
        settings_dict.update(
            {
                "REDIS_URL": redis_cfg.get("url", "redis://localhost:6379/0"),
                "REDIS_HOST": redis_cfg.get("host", "localhost"),
                "REDIS_PORT": redis_cfg.get("port", 6379),
            }
        )

    # Load worker config
    if "worker" in config:
        worker_cfg = config["worker"]
        settings_dict.update(
            {
                "ARQ_MAX_JOBS": worker_cfg.get("arq_max_jobs", 1),
                "COLPALI_STORE_BATCH_SIZE": worker_cfg.get("colpali_store_batch_size", 16),
            }
        )

    # Load pdf config
    if "pdf" in config:
        pdf_cfg = config["pdf"]
        settings_dict["COLPALI_PDF_DPI"] = pdf_cfg.get("colpali_pdf_dpi", 150)

    # Load telemetry config
    if "telemetry" in config:
        telemetry_cfg = config["telemetry"]
        settings_dict.update(
            {
                "SERVICE_NAME": telemetry_cfg.get("service_name", "databridge-core"),
                "PROJECT_NAME": telemetry_cfg.get("project_name") or None,
                "TELEMETRY_UPLOAD_INTERVAL_HOURS": telemetry_cfg.get("upload_interval_hours", 4.0),
                "TELEMETRY_MAX_LOCAL_BYTES": telemetry_cfg.get("max_local_bytes", 1073741824),
            }
        )

    settings_dict["TELEMETRY_ENABLED"] = os.getenv("TELEMETRY", "").strip().lower() != "false"

    # Load LOCAL_URI_PASSWORD from environment
    settings_dict["LOCAL_URI_PASSWORD"] = os.environ.get("LOCAL_URI_PASSWORD")

    # Load LiteLLM config (dummy API key for providers that don't need auth)
    settings_dict["LITELLM_DUMMY_API_KEY"] = os.environ.get("LITELLM_DUMMY_API_KEY", "ollama")

    # Load multivector store config
    if "multivector_store" in config:
        settings_dict["MULTIVECTOR_STORE_PROVIDER"] = config["multivector_store"].get("provider", "postgres")

        # Check for Turbopuffer API key if using morphik provider
        if settings_dict["MULTIVECTOR_STORE_PROVIDER"] == "morphik":
            if "TURBOPUFFER_API_KEY" not in os.environ:
                raise ValueError(
                    em.format(missing_value="TURBOPUFFER_API_KEY", field="multivector_store.provider", value="morphik")
                )
            settings_dict["TURBOPUFFER_API_KEY"] = os.environ["TURBOPUFFER_API_KEY"]

    return Settings(**settings_dict)
