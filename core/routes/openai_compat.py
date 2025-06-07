"""
OpenAI API compatibility router for Morphik.
Provides OpenAI SDK compatibility while leveraging Morphik's RAG and LiteLLM capabilities.
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from core.auth_utils import AuthContext, verify_token
from core.completion.litellm_completion import LiteLLMCompletionModel
from core.config import get_settings
from core.dependencies import get_document_service
from core.limits_utils import check_and_increment_limits
from core.models.chat import ChatMessage
from core.models.completion import CompletionRequest
from core.models.openai_compat import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunk,
    OpenAIChoice,
    OpenAIStreamChoice,
    OpenAIMessage,
    OpenAIUsage,
    OpenAIModel,
    OpenAIModelList,
    OpenAIError,
    OpenAIErrorResponse,
)
from core.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])


def create_error_response(message: str, error_type: str = "invalid_request_error", param: Optional[str] = None, code: Optional[str] = None) -> OpenAIErrorResponse:
    """Create an OpenAI-compatible error response."""
    return OpenAIErrorResponse(
        error=OpenAIError(
            message=message,
            type=error_type,
            param=param,
            code=code
        )
    )


def convert_morphik_to_openai_messages(messages: List[OpenAIMessage]) -> List[ChatMessage]:
    """Convert OpenAI messages to Morphik ChatMessage format."""
    chat_messages = []
    for msg in messages:
        # Convert content to string if it's a list (multimodal content)
        content = msg.content
        if isinstance(content, list):
            # Extract text content from multimodal messages
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts) if text_parts else ""
        
        chat_messages.append(ChatMessage(
            role=msg.role,
            content=content or "",
            timestamp=time.time()
        ))
    
    return chat_messages


def extract_user_query_from_messages(messages: List[OpenAIMessage]) -> str:
    """Extract the user query from OpenAI messages."""
    # Get the last user message as the query
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, list):
                # Extract text from multimodal content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return "\n".join(text_parts) if text_parts else ""
            return content or ""
    return ""


@router.get("/models")
async def list_models(auth_context: AuthContext = Depends(verify_token)) -> OpenAIModelList:
    """List available models in OpenAI format."""
    try:
        settings = get_settings()
        
        # Apply rate limiting (consistent with query endpoints)
        if settings.MODE == "cloud" and auth_context.user_id:
            await check_and_increment_limits(auth_context, "query", 1)
        models = []
        
        # Add registered models
        for model_key, model_config in settings.REGISTERED_MODELS.items():
            models.append(OpenAIModel(
                id=model_key,
                created=int(time.time()),
                owned_by="morphik"
            ))
        
        # Add default models if no registered models
        if not models:
            default_models = [
                settings.COMPLETION_MODEL,
                settings.AGENT_MODEL,
            ]
            for model_name in default_models:
                if model_name:
                    models.append(OpenAIModel(
                        id=model_name,
                        created=int(time.time()),
                        owned_by="morphik"
                    ))
        
        return OpenAIModelList(data=models)
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                message=f"Failed to list models: {str(e)}",
                error_type="internal_server_error"
            ).dict()
        )


@router.post("/chat/completions")
async def create_chat_completion(
    request: OpenAIChatCompletionRequest,
    auth_context: AuthContext = Depends(verify_token),
    document_service: DocumentService = Depends(get_document_service),
) -> OpenAIChatCompletionResponse:
    """Create a chat completion in OpenAI format."""
    try:
        settings = get_settings()
        
        # Apply rate limiting (consistent with query endpoints)
        if settings.MODE == "cloud" and auth_context.user_id:
            await check_and_increment_limits(auth_context, "query", 1)
        
        # Validate model
        if request.model not in settings.REGISTERED_MODELS:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    message=f"Model '{request.model}' not found",
                    error_type="invalid_request_error",
                    param="model"
                ).dict()
            )
        
        # Extract user query
        user_query = extract_user_query_from_messages(request.messages)
        if not user_query:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    message="No user message found in request",
                    error_type="invalid_request_error",
                    param="messages"
                ).dict()
            )
        
        # Retrieve relevant context if RAG is enabled
        context_chunks = []
        if request.use_rag:
            try:
                # Use document service to retrieve relevant chunks
                retrieve_result = await document_service.retrieve_chunks(
                    query=user_query,
                    app_id=auth_context.app_id,
                    entity_type=auth_context.entity_type,
                    entity_id=auth_context.entity_id,
                    top_k=request.top_k or 5,
                    folder_name=request.folder_name,
                )
                context_chunks = [chunk.content for chunk in retrieve_result.chunks]
                logger.info(f"Retrieved {len(context_chunks)} context chunks for OpenAI completion")
            except Exception as e:
                logger.warning(f"Failed to retrieve context chunks: {e}. Proceeding without RAG.")
        
        # Convert messages to chat history (excluding system messages and last user message)
        chat_history = []
        for msg in request.messages[:-1]:  # Exclude the last message (current query)
            if msg.role != "system":  # System messages are handled separately
                chat_history.append(ChatMessage(
                    role=msg.role,
                    content=msg.content if isinstance(msg.content, str) else str(msg.content),
                    timestamp=time.time()
                ))
        
        # Handle structured output
        schema = None
        if request.response_format and request.response_format.type in ["json_object", "json_schema"]:
            if request.response_format.type == "json_schema" and request.response_format.json_schema:
                schema = request.response_format.json_schema
            else:
                # For json_object, we'll let the model handle it naturally
                pass
        
        # Create completion request
        completion_request = CompletionRequest(
            query=user_query,
            context_chunks=context_chunks,
            max_tokens=request.max_tokens or request.max_completion_tokens,
            temperature=request.temperature,
            chat_history=chat_history,
            stream_response=request.stream,
            schema=schema,
            end_user_id=request.user,
            folder_name=request.folder_name,
        )
        
        # Initialize completion model
        completion_model = LiteLLMCompletionModel(request.model)
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(completion_request, completion_model, request),
                media_type="text/plain",
                headers={"X-Accel-Buffering": "no"}  # Disable nginx buffering
            )
        
        # Non-streaming completion
        response = await completion_model.complete(completion_request)
        
        # Convert to OpenAI format
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())
        
        choice = OpenAIChoice(
            index=0,
            message=OpenAIMessage(
                role="assistant",
                content=str(response.completion)
            ),
            finish_reason=response.finish_reason or "stop"
        )
        
        usage = OpenAIUsage(
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            total_tokens=response.usage.get("total_tokens", 0)
        )
        
        return OpenAIChatCompletionResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                message=f"Internal server error: {str(e)}",
                error_type="internal_server_error"
            ).dict()
        )


async def stream_chat_completion(
    completion_request: CompletionRequest,
    completion_model: LiteLLMCompletionModel,
    openai_request: OpenAIChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream chat completion in OpenAI format."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    
    try:
        # Get the streaming generator
        stream = await completion_model.complete(completion_request)
        
        # Stream the chunks
        async for chunk_content in stream:
            chunk = OpenAIChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=openai_request.model,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIMessage(
                            role="assistant",
                            content=chunk_content
                        ),
                        finish_reason=None
                    )
                ]
            )
            
            yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Send final chunk with finish_reason
        final_chunk = OpenAIChatCompletionChunk(
            id=completion_id,
            created=created_time,
            model=openai_request.model,
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIMessage(role="assistant"),
                    finish_reason="stop"
                )
            ]
        )
        
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in streaming completion: {e}")
        error_chunk = {
            "error": {
                "message": f"Stream error: {str(e)}",
                "type": "internal_server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


# Chat session endpoints for persistent chat
@router.get("/chat/sessions/{chat_id}")
async def get_chat_session(
    chat_id: str,
    auth_context: AuthContext = Depends(verify_token)
) -> Dict:
    """Get chat session history (Morphik extension)."""
    try:
        # This would integrate with Morphik's existing chat functionality
        # For now, return a placeholder
        return {
            "id": chat_id,
            "messages": [],
            "created": int(time.time()),
            "model": "morphik"
        }
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                message=f"Failed to get chat session: {str(e)}",
                error_type="internal_server_error"
            ).dict()
        )


@router.delete("/chat/sessions/{chat_id}")
async def delete_chat_session(
    chat_id: str,
    auth_context: AuthContext = Depends(verify_token)
) -> Dict:
    """Delete chat session (Morphik extension)."""
    try:
        # This would integrate with Morphik's existing chat functionality
        return {"deleted": True}
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                message=f"Failed to delete chat session: {str(e)}",
                error_type="internal_server_error"
            ).dict()
        )