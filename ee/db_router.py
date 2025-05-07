from __future__ import annotations

from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.models.app_metadata import AppMetadataModel

"""Database router for enterprise deployments.

This helper abstracts *per-app* database routing.  For requests that originate
from a *developer* token (i.e. the JWT contains an ``app_id``) we look up the
connection URI in the **app_metadata** catalogue and return a dedicated
:class:`core.database.postgres_database.PostgresDatabase` instance that is
backed by the Neon project provisioned for that application.

If no *app_id* is present the control-plane database (configured via
``settings.POSTGRES_URI``) is returned.

The router keeps an in-memory cache so that each unique connection URI only
creates **one** connection pool.
"""

try:
    from core.vector_store.pgvector_store import PGVectorStore
except ImportError:  # When vector store module missing (tests)
    PGVectorStore = None  # type: ignore

__all__ = ["get_database_for_app"]

# ---------------------------------------------------------------------------
# Internal caches (process-wide)
# ---------------------------------------------------------------------------

_CONTROL_PLANE_DB: Optional[PostgresDatabase] = None
_DB_CACHE: Dict[str, PostgresDatabase] = {}
# PGVectorStore cache keyed by connection URI to avoid duplicate pools
_VSTORE_CACHE: Dict[str, "PGVectorStore"] = {}


async def _resolve_connection_uri(app_id: str) -> Optional[str]:
    """Return ``connection_uri`` for *app_id* from **app_metadata** (async lookup)."""
    settings = get_settings()

    engine = create_async_engine(settings.POSTGRES_URI, pool_size=2, max_overflow=4)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as sess:
        result = await sess.execute(select(AppMetadataModel).where(AppMetadataModel.id == app_id))
        meta: AppMetadataModel | None = result.scalars().first()
        return meta.connection_uri if meta else None


async def get_database_for_app(app_id: str | None) -> PostgresDatabase:
    """Return a :class:`PostgresDatabase` instance for *app_id*.

    For *None* we always return the control-plane database instance.
    Connection pools are cached per app so repeated calls are cheap.
    """

    global _CONTROL_PLANE_DB  # noqa: PLW0603 – module-level cache

    # ------------------------------------------------------------------
    # 1) No app –> control-plane DB
    # ------------------------------------------------------------------
    if not app_id:
        if _CONTROL_PLANE_DB is None:
            settings = get_settings()
            _CONTROL_PLANE_DB = PostgresDatabase(uri=settings.POSTGRES_URI)
        return _CONTROL_PLANE_DB

    # ------------------------------------------------------------------
    # 2) Cached? –> quick return
    # ------------------------------------------------------------------
    if app_id in _DB_CACHE:
        return _DB_CACHE[app_id]

    # ------------------------------------------------------------------
    # 3) Resolve via catalogue and create new pool
    # ------------------------------------------------------------------
    connection_uri = await _resolve_connection_uri(app_id)

    # Fallback to control-plane DB when catalogue entry is missing (shouldn't
    # happen in normal operation but avoids 500s due to mis-configuration).
    if not connection_uri:
        return await get_database_for_app(None)  # type: ignore[return-value]

    db = PostgresDatabase(uri=connection_uri)
    _DB_CACHE[app_id] = db
    return db


async def get_vector_store_for_app(app_id: str | None):
    """Return a :class:`PGVectorStore` bound to the connection URI of *app_id*."""
    if PGVectorStore is None:
        return None

    if not app_id:
        settings = get_settings()
        return PGVectorStore(uri=settings.POSTGRES_URI)

    db = await get_database_for_app(app_id)
    uri = str(db.engine.url)  # type: ignore[arg-type]

    if uri in _VSTORE_CACHE:
        return _VSTORE_CACHE[uri]

    store = PGVectorStore(uri=uri)
    _VSTORE_CACHE[uri] = store
    return store
