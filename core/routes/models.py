import logging
import uuid
from datetime import UTC, datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.models.model_config import ModelConfig
from core.services_init import document_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


class SaveModelRequest(BaseModel):
    """Request to save a custom model."""
    name: str
    provider: str
    config: Dict  # LiteLLM config including model name, temperature, etc.


class SaveApiKeyRequest(BaseModel):
    """Request to save API keys."""
    provider: str
    api_key: str
    base_url: Optional[str] = None


class ModelResponse(BaseModel):
    """Response for a saved model."""
    id: str
    name: str
    provider: str
    config: Dict
    created_at: str
    updated_at: str


@router.post("/models", response_model=ModelResponse)
async def save_model(
    request: SaveModelRequest,
    auth: AuthContext = Depends(verify_token),
) -> ModelResponse:
    """Save a custom model configuration."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Create a unique ID for this model
        model_id = str(uuid.uuid4())
        
        # Create the model config
        config = ModelConfig(
            id=model_id,
            user_id=auth.user_id,
            app_id=auth.app_id,
            provider="custom",  # All user-defined models are "custom"
            config_data={
                "models": [{
                    "id": model_id,
                    "name": request.name,
                    "provider": request.provider,
                    "config": request.config
                }]
            },
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        
        # Check if there's already a custom model config for this user/app
        existing_configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        custom_config = None
        for existing in existing_configs:
            if existing.provider == "custom":
                custom_config = existing
                break
        
        if custom_config:
            # Append to existing custom models
            models = custom_config.config_data.get("models", [])
            models.append({
                "id": model_id,
                "name": request.name,
                "provider": request.provider,
                "config": request.config
            })
            
            success = await document_service.db.update_model_config(
                config_id=custom_config.id,
                user_id=auth.user_id,
                app_id=auth.app_id,
                updates={
                    "config_data": {
                        **custom_config.config_data,
                        "models": models
                    }
                }
            )
        else:
            # Create new custom models config
            success = await document_service.db.store_model_config(config)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save model"
            )
        
        return ModelResponse(
            id=model_id,
            name=request.name,
            provider=request.provider,
            config=request.config,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/custom", response_model=List[ModelResponse])
async def list_custom_models(
    auth: AuthContext = Depends(verify_token),
) -> List[ModelResponse]:
    """List all custom models for the authenticated user."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Get all model configs
        configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        # Extract custom models
        custom_models = []
        for config in configs:
            if config.provider == "custom" and "models" in config.config_data:
                for model in config.config_data["models"]:
                    custom_models.append(ModelResponse(
                        id=model.get("id", ""),
                        name=model.get("name", ""),
                        provider=model.get("provider", ""),
                        config=model.get("config", {}),
                        created_at=config.created_at.isoformat(),
                        updated_at=config.updated_at.isoformat(),
                    ))
        
        return custom_models
        
    except Exception as e:
        logger.error(f"Error listing custom models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, str]:
    """Delete a custom model."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Get all model configs
        configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        # Find the config containing this model
        for config in configs:
            if config.provider == "custom" and "models" in config.config_data:
                models = config.config_data["models"]
                updated_models = [m for m in models if m.get("id") != model_id]
                
                if len(models) != len(updated_models):
                    # Model was found and removed
                    if updated_models:
                        # Update the config with remaining models
                        success = await document_service.db.update_model_config(
                            config_id=config.id,
                            user_id=auth.user_id,
                            app_id=auth.app_id,
                            updates={
                                "config_data": {
                                    **config.config_data,
                                    "models": updated_models
                                }
                            }
                        )
                    else:
                        # No models left, delete the entire config
                        success = await document_service.db.delete_model_config(
                            config_id=config.id,
                            user_id=auth.user_id,
                            app_id=auth.app_id
                        )
                    
                    if success:
                        return {"message": "Model deleted successfully"}
        
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api-keys")
async def save_api_key(
    request: SaveApiKeyRequest,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, str]:
    """Save API key for a provider."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Create or update the config for this provider
        config_data = {
            "apiKey": request.api_key,
        }
        if request.base_url:
            config_data["baseUrl"] = request.base_url
        
        # Check if config already exists for this provider
        existing_configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        provider_config = None
        for existing in existing_configs:
            if existing.provider == request.provider:
                provider_config = existing
                break
        
        if provider_config:
            # Update existing config
            success = await document_service.db.update_model_config(
                config_id=provider_config.id,
                user_id=auth.user_id,
                app_id=auth.app_id,
                updates={
                    "config_data": config_data
                }
            )
        else:
            # Create new config
            config = ModelConfig(
                id=str(uuid.uuid4()),
                user_id=auth.user_id,
                app_id=auth.app_id,
                provider=request.provider,
                config_data=config_data,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            success = await document_service.db.store_model_config(config)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save API key"
            )
        
        return {"message": "API key saved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-keys")
async def list_api_keys(
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Dict]:
    """List all configured API keys (sanitized)."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Get all model configs
        configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        # Build response with sanitized API keys
        api_keys = {}
        for config in configs:
            if config.provider != "custom":  # Skip custom models config
                api_keys[config.provider] = {
                    "apiKey": "***" if config.config_data.get("apiKey") else None,
                    "baseUrl": config.config_data.get("baseUrl"),
                    "configured": bool(config.config_data.get("apiKey"))
                }
        
        return api_keys
        
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))