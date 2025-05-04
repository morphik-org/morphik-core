from datetime import UTC, datetime

import jwt
from fastapi import Header, HTTPException

from core.config import get_settings
from core.models.auth import AuthContext, EntityType

__all__ = ["verify_token"]

# Load settings once at import time
settings = get_settings()


async def verify_token(authorization: str = Header(None)) -> AuthContext:  # noqa: D401 – FastAPI dependency
    """Return an :class:`AuthContext` for a valid JWT bearer *authorization* header.

    In *dev_mode* we skip cryptographic checks and fabricate a permissive
    context so that local development environments can quickly spin up
    without real tokens.
    """

    # ------------------------------------------------------------------
    # 1. Development shortcut – trust everyone when *dev_mode* is active.
    # ------------------------------------------------------------------
    if settings.dev_mode:
        return AuthContext(
            entity_type=EntityType(settings.dev_entity_type),
            entity_id=settings.dev_entity_id,
            permissions=set(settings.dev_permissions),
            user_id=settings.dev_entity_id,  # In dev mode, entity_id == user_id
        )

    # ------------------------------------------------------------------
    # 2. Normal token verification flow
    # ------------------------------------------------------------------
    if not authorization:
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

    # Support both legacy "type" and new "entity_type" fields
    entity_type_field = payload.get("type") or payload.get("entity_type")
    if entity_type_field is None:
        raise HTTPException(status_code=401, detail="Missing entity type in token")

    return AuthContext(
        entity_type=EntityType(entity_type_field),
        entity_id=payload["entity_id"],
        app_id=payload.get("app_id"),
        permissions=set(payload.get("permissions", ["read"])),
        user_id=payload.get("user_id", payload["entity_id"]),
    )
