import json
import logging
from typing import Any, Dict, List

import arq
from fastapi import APIRouter, Depends, HTTPException, Query

from core.auth_utils import verify_token
from core.dependencies import get_redis_pool
from core.models.auth import AuthContext
from core.models.chat import ChatMessage
from core.services.telemetry import TelemetryService
from core.services_init import document_service

# ---------------------------------------------------------------------------
# Router initialisation & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)
telemetry = TelemetryService()


# ---------------------------------------------------------------------------
# Chat history endpoints
# ---------------------------------------------------------------------------


@router.get("/chat/{chat_id}", response_model=List[ChatMessage])
async def get_chat_history(
    chat_id: str,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Get chat history for a specific conversation.

    Args:
        chat_id: Unique identifier for the chat conversation
        auth: Authentication context
        redis: Redis connection for chat storage

    Returns:
        List[ChatMessage]: List of messages in the conversation
    """
    history_key = f"chat:{chat_id}"
    stored = await redis.get(history_key)

    if not stored:
        # Try to get from database
        db_history = await document_service.db.get_chat_history(
            chat_id, auth.user_id, auth.app_id
        )
        if db_history:
            return [ChatMessage(**msg) for msg in db_history]
        return []

    try:
        history = json.loads(stored)
        return [ChatMessage(**msg) for msg in history]
    except Exception as e:
        logger.error(f"Error parsing chat history: {e}")
        return []


@router.get("/chats", response_model=List[Dict[str, Any]])
async def list_chat_conversations(
    auth: AuthContext = Depends(verify_token),
    limit: int = Query(100, ge=1, le=500),
):
    """List recent chat conversations for the authenticated user.

    Args:
        auth: Authentication context
        limit: Maximum number of conversations to return (1-500)

    Returns:
        List of chat conversation summaries with:
            - chat_id: Unique conversation identifier
            - last_message: Most recent message content
            - updated_at: Timestamp of last activity
            - message_count: Total messages in conversation
    """
    try:
        # Get recent conversations from database
        conversations = await document_service.db.list_chat_conversations(
            auth.user_id, auth.app_id, limit=limit
        )

        # Format response
        result = []
        for conv in conversations:
            result.append({
                "chat_id": conv.get("chat_id"),
                "last_message": conv.get("last_message", ""),
                "updated_at": conv.get("updated_at"),
                "message_count": conv.get("message_count", 0),
                "created_at": conv.get("created_at"),
            })

        return result
    except Exception as e:
        logger.error(f"Error listing chat conversations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chat conversations"
        )