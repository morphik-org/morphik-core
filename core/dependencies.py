from typing import TYPE_CHECKING, Optional

import arq
from fastapi import Request

if TYPE_CHECKING:
    from core.services.document_service import DocumentService
    from core.services.ingestion_service import IngestionService


async def get_redis_pool(request: Request) -> arq.ArqRedis:
    if not hasattr(request.app.state, "redis_pool") or request.app.state.redis_pool is None:
        raise RuntimeError("Redis pool not initialized or not available on app.state")
    return request.app.state.redis_pool


async def get_optional_redis_pool(request: Request) -> Optional[arq.ArqRedis]:
    return getattr(request.app.state, "redis_pool", None)


async def get_document_service(request: Request) -> "DocumentService":
    if not hasattr(request.app.state, "document_service") or request.app.state.document_service is None:
        raise RuntimeError("Document service not initialized or not available on app.state")
    return request.app.state.document_service


async def get_ingestion_service(request: Request) -> "IngestionService":
    if not hasattr(request.app.state, "ingestion_service") or request.app.state.ingestion_service is None:
        raise RuntimeError("Ingestion service not initialized or not available on app.state")
    return request.app.state.ingestion_service
