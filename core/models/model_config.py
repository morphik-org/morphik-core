from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Model configuration for user-specific AI model settings."""
    
    id: str
    user_id: str
    app_id: str
    provider: str  # e.g., "openai", "anthropic", "google", "groq", "deepseek"
    config_data: Dict[str, Any]  # API keys, base URLs, model settings
    created_at: datetime
    updated_at: datetime


class ModelConfigCreate(BaseModel):
    """Request model for creating a model configuration."""
    
    provider: str
    config_data: Dict[str, Any]


class ModelConfigUpdate(BaseModel):
    """Request model for updating a model configuration."""
    
    config_data: Dict[str, Any]


class ModelConfigResponse(BaseModel):
    """Response model for model configuration."""
    
    id: str
    provider: str
    config_data: Dict[str, Any]  # Will exclude sensitive data like API keys in responses
    created_at: datetime
    updated_at: datetime


class CustomModel(BaseModel):
    """Custom model definition."""
    
    id: str
    name: str
    provider: str
    model_name: str
    config: Dict[str, Any]


class CustomModelCreate(BaseModel):
    """Request model for creating a custom model."""
    
    name: str
    provider: str
    model_name: str
    config: Dict[str, Any]