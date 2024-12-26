from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import tomli
from dotenv import load_dotenv


class Settings(BaseSettings):
    """DataBridge configuration settings."""

    # Required environment variables
    MONGODB_URI: str = Field(..., env="MONGODB_URI")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    UNSTRUCTURED_API_KEY: str = Field(..., env="UNSTRUCTURED_API_KEY")
    ASSEMBLYAI_API_KEY: str = Field(..., env="ASSEMBLYAI_API_KEY")
    AWS_ACCESS_KEY: str = Field(..., env="AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")

    # Values from config.toml with defaults
    AWS_REGION: str = "us-east-2"
    S3_BUCKET: str = "databridge-s3-storage"
    DATABRIDGE_DB: str = "databridge"
    DOCUMENTS_COLLECTION: str = "documents"
    CHUNKS_COLLECTION: str = "document_chunks"
    VECTOR_INDEX_NAME: str = "vector_index"
    VECTOR_DIMENSIONS: int = 1536
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    COMPLETION_MODEL: str = "gpt-3.5-turbo"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DEFAULT_K: int = 4
    HOST: str = "localhost"
    PORT: int = 8000
    RELOAD: bool = False
    JWT_ALGORITHM: str = "HS256"
    FRAME_SAMPLE_RATE: int = 120


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    load_dotenv()

    # Load config.toml
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    # Map config.toml values to settings
    settings_dict = {
        "AWS_REGION": config["aws"]["default_region"],
        "S3_BUCKET": config["aws"]["default_bucket_name"],
        "DATABRIDGE_DB": config["mongodb"]["database_name"],
        "DOCUMENTS_COLLECTION": config["mongodb"]["documents_collection"],
        "CHUNKS_COLLECTION": config["mongodb"]["chunks_collection"],
        "VECTOR_INDEX_NAME": config["mongodb"]["vector"]["index_name"],
        "VECTOR_DIMENSIONS": config["mongodb"]["vector"]["dimensions"],
        "EMBEDDING_MODEL": config["model"]["embedding_model"],
        "COMPLETION_MODEL": config["model"]["completion_model"],
        "CHUNK_SIZE": config["document_processing"]["chunk_size"],
        "CHUNK_OVERLAP": config["document_processing"]["chunk_overlap"],
        "DEFAULT_K": config["document_processing"]["default_k"],
        "HOST": config["server"]["host"],
        "PORT": config["server"]["port"],
        "RELOAD": config["server"]["reload"],
        "JWT_ALGORITHM": config["auth"]["jwt_algorithm"],
        "FRAME_SAMPLE_RATE": config["video_processing"]["frame_sample_rate"],
    }

    return Settings(**settings_dict)
