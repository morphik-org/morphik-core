from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import tomli
from dotenv import load_dotenv, find_dotenv

def load_toml_config() -> Dict[Any, Any]:
    """Load configuration from config.toml file."""
    with open("config.toml", "rb") as f:
        return tomli.load(f)

class Settings(BaseSettings):
    """DataBridge configuration settings."""

    # MongoDB settings
    MONGODB_URI: str = Field(..., env="MONGODB_URI")
    DATABRIDGE_DB: str = Field(None)
    
    # Collection names
    DOCUMENTS_COLLECTION: str = Field(None)
    CHUNKS_COLLECTION: str = Field(None)
    
    # Vector search settings
    VECTOR_INDEX_NAME: str = Field(None)
    
    # API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    UNSTRUCTURED_API_KEY: str = Field(..., env="UNSTRUCTURED_API_KEY")

    # Optional API keys for alternative models
    ANTHROPIC_API_KEY: str | None = Field(None, env="ANTHROPIC_API_KEY")
    COHERE_API_KEY: str | None = Field(None, env="COHERE_API_KEY")
    VOYAGE_API_KEY: str | None = Field(None, env="VOYAGE_API_KEY")
    
    # Model settings
    EMBEDDING_MODEL: str = Field("text-embedding-3-small")
    
    # Document processing settings
    CHUNK_SIZE: int = Field(1000)
    CHUNK_OVERLAP: int = Field(200)
    DEFAULT_K: int = Field(4)
    
    # Storage settings
    AWS_ACCESS_KEY: str = Field(..., env="AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(None)
    S3_BUCKET: str = Field(None)
    
    # Auth settings
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field("HS256")
    
    # Server settings
    HOST: str = Field("localhost")
    PORT: int = Field(8000)
    RELOAD: bool = Field(False)

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def __init__(self, **kwargs):
        # Force reload of environment variables
        load_dotenv(find_dotenv(), override=True)
        
        config = load_toml_config()
        
        # Set values from config.toml
        kwargs.update({
            # MongoDB settings
            "DATABRIDGE_DB": config["mongodb"]["database_name"],
            "DOCUMENTS_COLLECTION": config["mongodb"]["documents_collection"],
            "CHUNKS_COLLECTION": config["mongodb"]["chunks_collection"],
            "VECTOR_INDEX_NAME": config["mongodb"]["vector"]["index_name"],
            
            # AWS settings
            "AWS_REGION": config["aws"]["default_region"],
            "S3_BUCKET": config["aws"]["default_bucket_name"],
            
            # Model settings
            "EMBEDDING_MODEL": config["model"]["embedding_model"],
            
            # Document processing settings
            "CHUNK_SIZE": config["document_processing"]["chunk_size"],
            "CHUNK_OVERLAP": config["document_processing"]["chunk_overlap"],
            "DEFAULT_K": config["document_processing"]["default_k"],
            
            # Server settings
            "HOST": config["server"]["host"],
            "PORT": config["server"]["port"],
            "RELOAD": config["server"]["reload"],
            
            # Auth settings
            "JWT_ALGORITHM": config["auth"]["jwt_algorithm"],
        })
        
        super().__init__(**kwargs)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
