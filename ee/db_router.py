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

try:
    from core.vector_store.multi_vector_store import MultiVectorStore
except ImportError:  # pragma: no cover – module optional in some deployments
    MultiVectorStore = None  # type: ignore

__all__ = ["get_database_for_app"]

# ---------------------------------------------------------------------------
# Internal caches (process-wide)
# ---------------------------------------------------------------------------

_CONTROL_PLANE_DB: Optional[PostgresDatabase] = None
_DB_CACHE: Dict[str, PostgresDatabase] = {}
# PGVectorStore cache keyed by connection URI to avoid duplicate pools
_VSTORE_CACHE: Dict[str, "PGVectorStore"] = {}
_MVSTORE_CACHE: Dict[str, "MultiVectorStore"] = {}


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

    # Fetch the raw connection URI directly from the catalogue – this string
    # already contains the password.  We augment it to make sure it has
    # sslmode=require and uses the asyncpg driver.

    uri = await _resolve_connection_uri(app_id)

    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    if uri is None:
        # Should not happen – fallback to control-plane vector store
        settings = get_settings()
        return PGVectorStore(uri=settings.POSTGRES_URI)

    parsed = urlparse(uri)

    # Ensure asyncpg driver in scheme so we get an async pool
    scheme = parsed.scheme
    if scheme == "postgres" or scheme == "postgresql":
        parsed = parsed._replace(scheme="postgresql+asyncpg")
    elif not scheme.endswith("+asyncpg"):
        parsed = parsed._replace(scheme=scheme + "+asyncpg")

    # asyncpg raises "unexpected keyword argument 'sslmode'" if it sees that
    # parameter, so we must NOT propagate it via the query string.  Strip it
    # instead of adding it.
    query_params = parse_qs(parsed.query)
    query_params.pop("sslmode", None)
    parsed = parsed._replace(query=urlencode(query_params, doseq=True))
    uri = urlunparse(parsed)

    if uri in _VSTORE_CACHE:
        return _VSTORE_CACHE[uri]

    # Log at DEBUG level with password redacted to aid debugging connection issues
    import logging

    debug_uri = uri
    try:
        from urllib.parse import urlparse, urlunparse

        _p = urlparse(uri)
        if _p.password:
            redacted = _p._replace(netloc=f"{_p.username}:***@{_p.hostname}:{_p.port}")
            debug_uri = urlunparse(redacted)
    except Exception:  # noqa: BLE001 – best-effort
        pass

    logging.getLogger(__name__).debug("Creating PGVectorStore for app %s with URI %s", app_id, debug_uri)

    store = PGVectorStore(uri=uri)
    _VSTORE_CACHE[uri] = store
    return store


async def get_multi_vector_store_for_app(app_id: str | None):
    """Return a MultiVectorStore bound to the connection URI of *app_id*.

    When *app_id* is ``None`` we return a store that points at the control-plane
    database so that shared/legacy apps keep working.  Instances are cached per
    URI to avoid duplicate connection pools.
    """

    if MultiVectorStore is None:  # Dependency missing – feature disabled
        return None

    settings = get_settings()

    # ------------------------------------------------------------------
    # 1) No per-app routing required –> control-plane store
    # ------------------------------------------------------------------
    if not app_id:
        uri = settings.POSTGRES_URI
    else:
        uri = await _resolve_connection_uri(app_id)

        # Fallback (should not normally happen)
        if uri is None:
            uri = settings.POSTGRES_URI

    # Convert asyncpg URI to plain psycopg (sync) variant expected by
    # MultiVectorStore and make sure *sslmode=require* is present so Neon
    # accepts the connection.
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    parsed = urlparse(uri)

    # Ensure the driver prefix is *postgresql://* (psycopg)
    if parsed.scheme.startswith("postgresql+asyncpg"):
        parsed = parsed._replace(scheme="postgresql")

    # Ensure sslmode=require in the query string (Neon tends to need it)
    q = parse_qs(parsed.query)
    if "sslmode" not in q:
        q["sslmode"] = ["require"]
    parsed = parsed._replace(query=urlencode(q, doseq=True))

    final_uri = urlunparse(parsed)

    if final_uri in _MVSTORE_CACHE:
        return _MVSTORE_CACHE[final_uri]

    store = MultiVectorStore(uri=final_uri)
    _MVSTORE_CACHE[final_uri] = store
    return store
