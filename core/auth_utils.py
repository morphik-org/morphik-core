from datetime import UTC, datetime
from logging import getLogger
from typing import Any, Optional

import jwt
from fastapi import Header, HTTPException, Request

from core.config import get_settings
from core.models.auth import AuthContext

logger = getLogger(__name__)

__all__ = [
    "clear_app_active_cache",
    "ensure_app_is_active",
    "mark_app_active",
    "mark_app_revoked",
    "verify_token",
]

# Load settings once at import time
settings = get_settings()

_ACTIVE_CACHE_PREFIX = "auth:app_active:"
_REVOKED_CACHE_PREFIX = "auth:app_revoked:"


def _active_cache_key(app_id: str) -> str:
    return f"{_ACTIVE_CACHE_PREFIX}{app_id}"


def _revoked_cache_key(app_id: str) -> str:
    return f"{_REVOKED_CACHE_PREFIX}{app_id}"


def _get_redis_pool(request: Request) -> Optional[Any]:
    return getattr(request.app.state, "redis_pool", None)


def _normalize_token_version(token_version: Optional[Any]) -> int:
    if token_version is None:
        return 0
    try:
        return int(token_version)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token version") from exc


async def mark_app_active(app_id: Optional[str], token_version: Optional[Any], redis_pool: Optional[Any]) -> None:
    if not app_id or redis_pool is None:
        return

    try:
        normalized_version = _normalize_token_version(token_version)
        await redis_pool.set(
            _active_cache_key(app_id),
            str(normalized_version),
            ex=settings.APP_AUTH_ACTIVE_TTL_SECONDS,
        )
        await redis_pool.delete(_revoked_cache_key(app_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to mark app %s as active in cache: %s", app_id, exc)


async def mark_app_revoked(app_id: Optional[str], redis_pool: Optional[Any]) -> None:
    if not app_id or redis_pool is None:
        return

    try:
        await redis_pool.set(
            _revoked_cache_key(app_id),
            "1",
            ex=settings.APP_AUTH_REVOKED_TTL_SECONDS,
        )
        await redis_pool.delete(_active_cache_key(app_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to mark app %s as revoked in cache: %s", app_id, exc)


async def clear_app_active_cache(app_id: Optional[str], redis_pool: Optional[Any]) -> None:
    if not app_id or redis_pool is None:
        return

    try:
        await redis_pool.delete(_active_cache_key(app_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to clear app cache for %s: %s", app_id, exc)


async def ensure_app_is_active(
    app_id: Optional[str],
    token_version: Optional[Any] = None,
    redis_pool: Optional[Any] = None,
) -> None:
    """Ensure the app_id still exists; reject tokens for deleted apps."""
    if settings.bypass_auth_mode or not app_id:
        return

    normalized_version = _normalize_token_version(token_version)

    if redis_pool is not None:
        try:
            revoked = await redis_pool.get(_revoked_cache_key(app_id))
            if revoked is not None:
                raise HTTPException(status_code=401, detail="Invalid or revoked token")

            active = await redis_pool.get(_active_cache_key(app_id))
            if active is not None:
                try:
                    active_version = int(active)
                except (TypeError, ValueError):
                    logger.debug("Invalid active token version cache for app %s", app_id)
                else:
                    if active_version == normalized_version:
                        return
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis app cache unavailable; falling back to DB: %s", exc)
            redis_pool = None

    try:
        from core.services_init import database
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load database for app validation: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to validate token") from exc

    try:
        app_record = await database.get_app_record(app_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to validate app_id %s: %s", app_id, exc)
        raise HTTPException(status_code=500, detail="Failed to validate token") from exc

    if app_record is None:
        await mark_app_revoked(app_id, redis_pool)
        raise HTTPException(status_code=401, detail="Invalid or revoked token")

    app_version = app_record.get("token_version", 0) or 0
    if normalized_version != app_version:
        await mark_app_active(app_id, app_version, redis_pool)
        raise HTTPException(status_code=401, detail="Invalid or revoked token")

    await mark_app_active(app_id, app_version, redis_pool)


async def verify_token(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> AuthContext:  # noqa: D401 – FastAPI dependency
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

    app_id = payload.get("app_id")
    token_version = payload.get("token_version")
    await ensure_app_is_active(app_id, token_version=token_version, redis_pool=_get_redis_pool(request))

    ctx = AuthContext(
        user_id=user_id,
        app_id=app_id,
    )

    return ctx
