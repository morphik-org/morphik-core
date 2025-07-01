import logging
import uuid
from datetime import UTC, datetime
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.models.model_config import (
    CustomModel,
    CustomModelCreate,
    ModelConfig,
    ModelConfigCreate,
    ModelConfigResponse,
    ModelConfigUpdate,
)
from core.services_init import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model-config", tags=["model-config"])


def sanitize_config_for_response(config_data: Dict) -> Dict:
    """Remove sensitive data like API keys from config response."""
    sanitized = config_data.copy()
    
    # Remove API keys but keep other settings
    sensitive_keys = ["apiKey", "api_key", "API_KEY", "api-key"]
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "***"  # Mask the value
    
    return sanitized


@router.get("/", response_model=List[ModelConfigResponse])
async def list_model_configs(
    auth: AuthContext = Depends(verify_token),
) -> List[ModelConfigResponse]:
    """List all model configurations for the authenticated user and app."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        # Convert to response models with sanitized config
        response_configs = []
        for config in configs:
            response_configs.append(ModelConfigResponse(
                id=config.id,
                provider=config.provider,
                config_data=sanitize_config_for_response(config.config_data),
                created_at=config.created_at,
                updated_at=config.updated_at,
            ))
        
        return response_configs
        
    except Exception as e:
        logger.error(f"Error listing model configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{config_id}", response_model=ModelConfigResponse)
async def get_model_config(
    config_id: str,
    auth: AuthContext = Depends(verify_token),
) -> ModelConfigResponse:
    """Get a specific model configuration."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        config = await document_service.db.get_model_config(
            config_id=config_id,
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        if not config:
            raise HTTPException(
                status_code=404,
                detail="Model configuration not found"
            )
        
        return ModelConfigResponse(
            id=config.id,
            provider=config.provider,
            config_data=sanitize_config_for_response(config.config_data),
            created_at=config.created_at,
            updated_at=config.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ModelConfigResponse)
async def create_model_config(
    config_create: ModelConfigCreate,
    auth: AuthContext = Depends(verify_token),
) -> ModelConfigResponse:
    """Create a new model configuration."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Create the model config object
        config = ModelConfig(
            id=str(uuid.uuid4()),
            user_id=auth.user_id,
            app_id=auth.app_id,
            provider=config_create.provider,
            config_data=config_create.config_data,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        
        # Store in database
        success = await document_service.db.store_model_config(config)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store model configuration"
            )
        
        return ModelConfigResponse(
            id=config.id,
            provider=config.provider,
            config_data=sanitize_config_for_response(config.config_data),
            created_at=config.created_at,
            updated_at=config.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{config_id}", response_model=ModelConfigResponse)
async def update_model_config(
    config_id: str,
    config_update: ModelConfigUpdate,
    auth: AuthContext = Depends(verify_token),
) -> ModelConfigResponse:
    """Update an existing model configuration."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Check if config exists
        existing_config = await document_service.db.get_model_config(
            config_id=config_id,
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        if not existing_config:
            raise HTTPException(
                status_code=404,
                detail="Model configuration not found"
            )
        
        # Update the config
        updates = {
            "config_data": config_update.config_data,
        }
        
        success = await document_service.db.update_model_config(
            config_id=config_id,
            user_id=auth.user_id,
            app_id=auth.app_id,
            updates=updates
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update model configuration"
            )
        
        # Get updated config
        updated_config = await document_service.db.get_model_config(
            config_id=config_id,
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        return ModelConfigResponse(
            id=updated_config.id,
            provider=updated_config.provider,
            config_data=sanitize_config_for_response(updated_config.config_data),
            created_at=updated_config.created_at,
            updated_at=updated_config.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{config_id}")
async def delete_model_config(
    config_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, str]:
    """Delete a model configuration."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        success = await document_service.db.delete_model_config(
            config_id=config_id,
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Model configuration not found or could not be deleted"
            )
        
        return {"message": "Model configuration deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Custom models endpoints (stored within model configs)

@router.get("/custom-models/list", response_model=List[CustomModel])
async def list_custom_models(
    auth: AuthContext = Depends(verify_token),
) -> List[CustomModel]:
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
        
        # Extract custom models from configs
        custom_models = []
        for config in configs:
            if config.provider == "custom" and "models" in config.config_data:
                for model in config.config_data["models"]:
                    custom_models.append(CustomModel(**model))
        
        return custom_models
        
    except Exception as e:
        logger.error(f"Error listing custom models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-models", response_model=CustomModel)
async def create_custom_model(
    model_create: CustomModelCreate,
    auth: AuthContext = Depends(verify_token),
) -> CustomModel:
    """Create a new custom model."""
    try:
        if not auth.user_id or not auth.app_id:
            raise HTTPException(
                status_code=400,
                detail="User ID and App ID are required"
            )
        
        # Get or create custom models config
        configs = await document_service.db.get_model_configs(
            user_id=auth.user_id,
            app_id=auth.app_id
        )
        
        custom_config = None
        for config in configs:
            if config.provider == "custom":
                custom_config = config
                break
        
        # Create custom model object
        custom_model = CustomModel(
            id=str(uuid.uuid4()),
            name=model_create.name,
            provider=model_create.provider,
            model_name=model_create.model_name,
            config=model_create.config,
        )
        
        if custom_config:
            # Update existing config
            models = custom_config.config_data.get("models", [])
            models.append(custom_model.model_dump())
            
            updates = {
                "config_data": {
                    **custom_config.config_data,
                    "models": models
                }
            }
            
            success = await document_service.db.update_model_config(
                config_id=custom_config.id,
                user_id=auth.user_id,
                app_id=auth.app_id,
                updates=updates
            )
        else:
            # Create new config for custom models
            config = ModelConfig(
                id=str(uuid.uuid4()),
                user_id=auth.user_id,
                app_id=auth.app_id,
                provider="custom",
                config_data={
                    "models": [custom_model.model_dump()]
                },
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            
            success = await document_service.db.store_model_config(config)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store custom model"
            )
        
        return custom_model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))