from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import tomli
from dotenv import load_dotenv


class Settings(BaseSettings):
    """DataBridge configuration settings."""

    # Required environment variables (referenced in config.toml)
    MONGODB_URI: str = Field(..., env="MONGODB_URI")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    UNSTRUCTURED_API_KEY: str = Field(..., env="UNSTRUCTURED_API_KEY")
    ASSEMBLYAI_API_KEY: str = Field(..., env="ASSEMBLYAI_API_KEY")
    AWS_ACCESS_KEY: str = Field(..., env="AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")

    # Service settings
    HOST: str = "localhost"
    PORT: int = 8000
    RELOAD: bool = False

    # Component selection
    STORAGE_PROVIDER: str = "aws-s3"
    DATABASE_PROVIDER: str = "mongodb"
    VECTOR_STORE_PROVIDER: str = "mongodb"
    EMBEDDING_PROVIDER: str = "openai"
    COMPLETION_PROVIDER: str = "ollama"
    PARSER_PROVIDER: str = "combined"

    # Storage settings
    AWS_REGION: str = "us-east-2"
    S3_BUCKET: str = "databridge-s3-storage"

    # Database settings
    DATABRIDGE_DB: str = "DataBridgeTest"
    DOCUMENTS_COLLECTION: str = "documents"
    CHUNKS_COLLECTION: str = "document_chunks"

    # Vector store settings
    VECTOR_INDEX_NAME: str = "vector_index"
    VECTOR_DIMENSIONS: int = 1536

    # Model settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    COMPLETION_MODEL: str = "llama3.1"
    COMPLETION_MAX_TOKENS: int = 1000
    COMPLETION_TEMPERATURE: float = 0.7

    # Processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DEFAULT_K: int = 4
    FRAME_SAMPLE_RATE: int = 120

    # Auth settings
    JWT_ALGORITHM: str = "HS256"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    load_dotenv()

    # Load config.toml
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    # Map config.toml values to settings
    settings_dict = {
        # Service settings
        "HOST": config["service"]["host"],
        "PORT": config["service"]["port"],
        "RELOAD": config["service"]["reload"],
        # Component selection
        "STORAGE_PROVIDER": config["service"]["components"]["storage"],
        "DATABASE_PROVIDER": config["service"]["components"]["database"],
        "VECTOR_STORE_PROVIDER": config["service"]["components"]["vector_store"],
        "EMBEDDING_PROVIDER": config["service"]["components"]["embedding"],
        "COMPLETION_PROVIDER": config["service"]["components"]["completion"],
        "PARSER_PROVIDER": config["service"]["components"]["parser"],
        # Storage settings
        "AWS_REGION": config["storage"]["aws"]["region"],
        "S3_BUCKET": config["storage"]["aws"]["bucket_name"],
        # Database settings
        "DATABRIDGE_DB": config["database"]["mongodb"]["database_name"],
        "DOCUMENTS_COLLECTION": config["database"]["mongodb"]["documents_collection"],
        "CHUNKS_COLLECTION": config["database"]["mongodb"]["chunks_collection"],
        # Vector store settings
        "VECTOR_INDEX_NAME": config["vector_store"]["mongodb"]["index_name"],
        "VECTOR_DIMENSIONS": config["vector_store"]["mongodb"]["dimensions"],
        # Model settings
        "EMBEDDING_MODEL": config["models"]["embedding"]["model_name"],
        "COMPLETION_MODEL": config["models"]["completion"]["model_name"],
        "COMPLETION_MAX_TOKENS": config["models"]["completion"]["default_max_tokens"],
        "COMPLETION_TEMPERATURE": config["models"]["completion"]["default_temperature"],
        # Processing settings
        "CHUNK_SIZE": config["processing"]["text"]["chunk_size"],
        "CHUNK_OVERLAP": config["processing"]["text"]["chunk_overlap"],
        "DEFAULT_K": config["processing"]["text"]["default_k"],
        "FRAME_SAMPLE_RATE": config["processing"]["video"]["frame_sample_rate"],
        # Auth settings
        "JWT_ALGORITHM": config["auth"]["jwt_algorithm"],
    }

    return Settings(**settings_dict)
