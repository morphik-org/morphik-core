import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import arq
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.completion.litellm_completion import LiteLLMCompletionModel
from core.dependencies import get_redis_pool
from core.models.auth import AuthContext
from core.models.chat import ChatMessage
from core.models.completion import CompletionRequest
from core.services_init import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["document"])


class DocumentChatRequest(BaseModel):
    """Request model for document chat completion."""

    message: str
    document_id: Optional[str] = None


@router.get("/chat/{chat_id}", response_model=List[ChatMessage])
async def get_document_chat_history(
    chat_id: str,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Retrieve the message history for a document chat conversation.

    Args:
        chat_id: Identifier of the document chat conversation.
        auth: Authentication context used to verify access to the conversation.
        redis: Redis connection where chat messages are stored.

    Returns:
        A list of ChatMessage objects or an empty list if no history exists.
    """
    history_key = f"document_chat:{chat_id}"
    stored = await redis.get(history_key)

    if not stored:
        # Try to get from database
        db_hist = await document_service.db.get_chat_history(chat_id, auth.user_id, auth.app_id)
        if not db_hist:
            return []
        return [ChatMessage(**m) for m in db_hist]

    try:
        data = json.loads(stored)
        return [ChatMessage(**m) for m in data]
    except Exception as e:
        logger.error(f"Error parsing chat history from Redis: {e}")
        return []


@router.post("/chat/{chat_id}/complete")
async def complete_document_chat(
    chat_id: str,
    request: DocumentChatRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Stream a chat completion response for a document chat conversation.

    Args:
        chat_id: Identifier of the document chat conversation.
        request: The chat request containing the user message.
        auth: Authentication context.
        redis: Redis connection for chat history storage.

    Returns:
        StreamingResponse with the assistant's response.
    """
    try:
        # Get chat history
        history_key = f"document_chat:{chat_id}"
        history: List[Dict[str, Any]] = []

        stored = await redis.get(history_key)
        if stored:
            try:
                history = json.loads(stored)
            except Exception as e:
                logger.error(f"Error parsing chat history: {e}")
                history = []
        else:
            # Try to get from database
            db_hist = await document_service.db.get_chat_history(chat_id, auth.user_id, auth.app_id)
            if db_hist:
                history = db_hist

        # Add user message to history
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        history.append(user_message)

        # Initialize the completion model with Gemini 2.5-flash
        completion_model = LiteLLMCompletionModel("gemini_flash")

        # Create completion request
        completion_request = CompletionRequest(
            query=request.message,
            context_chunks=[],  # For now, no document context
            max_tokens=1000,
            temperature=0.7,
            stream_response=True,
            chat_history=[ChatMessage(**msg) for msg in history],
        )

        # Generate streaming response
        async def generate_stream():
            full_response = ""
            try:
                stream = await completion_model.complete(completion_request)

                async for chunk in stream:
                    if chunk:
                        full_response += chunk
                        # Format as SSE (Server-Sent Events)
                        yield f"data: {json.dumps({'content': chunk})}\n\n"

                # Add assistant message to history
                assistant_message = {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                history.append(assistant_message)

                # Store updated history
                await redis.set(history_key, json.dumps(history))
                await document_service.db.upsert_chat_history(
                    chat_id,
                    auth.user_id,
                    auth.app_id,
                    history,
                )

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming completion: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        logger.error(f"Error in document chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
