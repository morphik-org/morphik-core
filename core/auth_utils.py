from datetime import UTC, datetime
from logging import getLogger

import jwt
from fastapi import Header, HTTPException

from core.config import get_settings
from core.models.auth import AuthContext

logger = getLogger(__name__)

__all__ = ["verify_token"]

# Load settings once at import time
settings = get_settings()


async def verify_token(authorization: str = Header(None)) -> AuthContext:  # noqa: D401 – FastAPI dependency
    """Return an :class:`AuthContext` for a valid JWT bearer *authorization* header.

    When *bypass_auth_mode* is enabled we skip cryptographic checks and
    fabricate a permissive context so that local development environments
    can quickly spin up without real tokens.
    """

    # ------------------------------------------------------------------
    # 1. Development shortcut – trust everyone when auth-bypass mode is active.
    # ------------------------------------------------------------------
    if settings.bypass_auth_mode:
        return AuthContext(
            user_id=settings.dev_user_id,
            app_id=None,
        )

    # ------------------------------------------------------------------
    # 2. Normal token verification flow
    # ------------------------------------------------------------------
    if not authorization:
        logger.info("Missing authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[7:]  # Strip "Bearer " prefix

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    # Check expiry manually – jwt.decode does *not* enforce expiry on psycopg2.
    if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
        raise HTTPException(status_code=401, detail="Token expired")

    # Extract user_id - support legacy "entity_id" for backward compatibility
    user_id = payload.get("user_id") or payload.get("entity_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user_id in token")

    ctx = AuthContext(
        user_id=user_id,
        app_id=payload.get("app_id"),
    )

    # ------------------------------------------------------------------
    # Enterprise enhancement – swap database & vector store based on app_id
    # ------------------------------------------------------------------
    try:
        from core import api as core_api  # type: ignore
        from ee.db_router import (  # noqa: WPS433 – runtime import
            get_database_for_app,
            get_multi_vector_store_for_app,
            get_vector_store_for_app,
        )

        # Replace DB connection pool
        core_api.document_service.db = await get_database_for_app(ctx.app_id)  # noqa: SLF001

        # Replace vector store (if available)
        vstore = await get_vector_store_for_app(ctx.app_id)
        if vstore is not None:
            core_api.vector_store = vstore  # noqa: SLF001 – monkey-patch
            core_api.document_service.vector_store = vstore  # noqa: SLF001 – monkey-patch

        # Route ColPali multi-vector store (if service uses one)
        try:
            mv_store = await get_multi_vector_store_for_app(ctx.app_id)
            if mv_store is not None:
                core_api.document_service.colpali_vector_store = mv_store  # noqa: SLF001 – monkey-patch
        except Exception as mv_exc:  # pragma: no cover – log, but don't block request
            logger.debug("MultiVector store routing skipped: %s", mv_exc)
    except ModuleNotFoundError:
        # Enterprise package not installed – nothing to do.
        pass

    return ctx
