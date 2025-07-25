from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
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

try:
    from core.vector_store.fast_multivector_store import FastMultiVectorStore
except ImportError:  # pragma: no cover – module optional in some deployments
    FastMultiVectorStore = None  # type: ignore

__all__ = ["get_database_for_app"]

# ---------------------------------------------------------------------------
# Internal caches (process-wide)
# ---------------------------------------------------------------------------

_CONTROL_PLANE_DB: Optional[PostgresDatabase] = None
_DB_CACHE: Dict[str, PostgresDatabase] = {}
# PGVectorStore cache keyed by connection URI to avoid duplicate pools
_VSTORE_CACHE: Dict[str, "PGVectorStore"] = {}
_MVSTORE_CACHE: Dict[str, "MultiVectorStore"] = {}
_FAST_MVSTORE_CACHE: Dict[str, "FastMultiVectorStore"] = {}
# Dedicated cache for control-plane vector store (covers non-app requests)
_CONTROL_PLANE_VSTORE: Optional["PGVectorStore"] = None
# Dedicated cache for control-plane multi-vector store
_CONTROL_PLANE_MVSTORE: Optional["MultiVectorStore"] = None
_CONTROL_PLANE_FAST_MVSTORE: Optional["FastMultiVectorStore"] = None
_CONN_URI_CACHE: Dict[str, Optional[str]] = {}
# Reuse a single engine + sessionmaker for catalogue look-ups so we don't create
# a new connection pool on every request.  The pool is tiny because we only
# use it for simple SELECT queries.
_RESOLVE_ENGINE: Optional["AsyncEngine"] = None
_RESOLVE_SESSION: Optional[sessionmaker] = None


async def _resolve_connection_uri(app_id: str) -> Optional[str]:
    """Return ``connection_uri`` for *app_id* from **app_metadata** (async lookup).

    The first lookup for a given *app_id* performs an actual database query
    against the **app_metadata** catalogue.  The result is then cached so
    subsequent calls are O(1) dictionary look-ups which avoids the overhead
    of repeatedly creating a new SQLAlchemy engine *and* running the query.
    """

    # Fast path –> in-memory cache hit
    if app_id in _CONN_URI_CACHE:
        return _CONN_URI_CACHE[app_id]

    global _RESOLVE_ENGINE, _RESOLVE_SESSION  # noqa: PLW0603 – module-level cache

    # Lazily initialise the tiny connection pool that backs catalogue look-ups
    if _RESOLVE_ENGINE is None:
        settings = get_settings()
        _RESOLVE_ENGINE = create_async_engine(
            settings.POSTGRES_URI,
            pool_size=2,
            max_overflow=4,
            pool_pre_ping=True,
        )
        _RESOLVE_SESSION = sessionmaker(
            _RESOLVE_ENGINE,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    # At this point the sessionmaker is guaranteed to be initialised
    assert _RESOLVE_SESSION is not None  # mypy / static checkers

    async with _RESOLVE_SESSION() as sess:
        result = await sess.execute(select(AppMetadataModel.connection_uri).where(AppMetadataModel.id == app_id))
        connection_uri: Optional[str] = result.scalar_one_or_none()

    # Store (even None) so that we don't hit the DB again for invalid IDs
    _CONN_URI_CACHE[app_id] = connection_uri
    return connection_uri


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
            # Initialize the control-plane database
            await _CONTROL_PLANE_DB.initialize()
            logging.getLogger(__name__).info("Initialized control-plane PostgresDatabase")
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

    # If the resolved URI is identical to the control-plane DB we can simply
    # reuse the existing pool instead of creating another one.  This covers
    # Free / Pro customers that share the same logical database.
    settings = get_settings()
    if connection_uri == settings.POSTGRES_URI:
        return await get_database_for_app(None)  # type: ignore[return-value]

    # Fallback to control-plane DB when catalogue entry is missing (shouldn't
    # happen in normal operation but avoids 500s due to mis-configuration).
    if not connection_uri:
        return await get_database_for_app(None)  # type: ignore[return-value]

    db = PostgresDatabase(uri=connection_uri)
    # Initialize the per-app database
    await db.initialize()
    _DB_CACHE[app_id] = db
    logging.getLogger(__name__).info(f"Initialized PostgresDatabase for app_id={app_id}")
    return db


async def get_vector_store_for_app(app_id: str | None):
    """Return a :class:`PGVectorStore` bound to the connection URI of *app_id*."""
    if PGVectorStore is None:
        return None

    global _CONTROL_PLANE_VSTORE  # noqa: PLW0603 – module-level cache

    if not app_id:
        if _CONTROL_PLANE_VSTORE is None:
            settings = get_settings()
            _CONTROL_PLANE_VSTORE = PGVectorStore(uri=settings.POSTGRES_URI)
            # Initialize the control-plane vector store
            await _CONTROL_PLANE_VSTORE.initialize()
            logging.getLogger(__name__).info("Initialized control-plane PGVectorStore")
        return _CONTROL_PLANE_VSTORE

    # Fetch the raw connection URI directly from the catalogue – this string
    # already contains the password.  We augment it to make sure it has
    # sslmode=require and uses the asyncpg driver.

    uri = await _resolve_connection_uri(app_id)

    # Free / Pro plans share the control-plane database –> reuse shared store
    settings = get_settings()
    if uri == settings.POSTGRES_URI:
        return await get_vector_store_for_app(None)

    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    if uri is None:
        # No dedicated DB for this app (Free/Pro) –> use shared store
        return await get_vector_store_for_app(None)

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
    # Initialize the per-app vector store
    await store.initialize()
    _VSTORE_CACHE[uri] = store
    logging.getLogger(__name__).info(f"Initialized PGVectorStore for app_id={app_id}")
    return store


async def get_multi_vector_store_for_app(app_id: str | None):
    """Return a MultiVectorStore or FastMultiVectorStore bound to the connection URI of *app_id*.

    When *app_id* is ``None`` we return a store that points at the control-plane
    database so that shared/legacy apps keep working.  Instances are cached per
    URI to avoid duplicate connection pools.

    Respects MULTIVECTOR_STORE_PROVIDER setting to determine whether to create
    FastMultiVectorStore or MultiVectorStore.
    """

    if MultiVectorStore is None:  # Dependency missing – feature disabled
        return None

    global _CONTROL_PLANE_MVSTORE, _CONTROL_PLANE_FAST_MVSTORE  # noqa: PLW0603 – module-level cache
    settings = get_settings()

    # ------------------------------------------------------------------
    # 1) No per-app routing required –> control-plane store
    # ------------------------------------------------------------------
    if not app_id:
        # Choose store type based on configuration
        if settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
            if FastMultiVectorStore is None:
                return None
            if _CONTROL_PLANE_FAST_MVSTORE is not None:
                return _CONTROL_PLANE_FAST_MVSTORE

            # Build URI for FastMultiVectorStore (requires async URI)
            uri = settings.POSTGRES_URI
            if not settings.TURBOPUFFER_API_KEY:
                raise ValueError("TURBOPUFFER_API_KEY is required when using morphik multivector store provider")

            store = FastMultiVectorStore(uri=uri, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public")
            # Initialize the control-plane fast multi-vector store
            import asyncio

            await asyncio.to_thread(store.initialize)
            _CONTROL_PLANE_FAST_MVSTORE = store
            logging.getLogger(__name__).info("Initialized control-plane FastMultiVectorStore")
            return store
        else:
            # Use MultiVectorStore (legacy behavior)
            if _CONTROL_PLANE_MVSTORE is not None:
                return _CONTROL_PLANE_MVSTORE

            # Build URI once (include sslmode=require)
            uri = settings.POSTGRES_URI
            from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

            parsed = urlparse(uri)
            if parsed.scheme.startswith("postgresql+asyncpg"):
                parsed = parsed._replace(scheme="postgresql")
            q = parse_qs(parsed.query)
            if "sslmode" not in q:
                q["sslmode"] = ["require"]
            parsed = parsed._replace(query=urlencode(q, doseq=True))
            final_uri = urlunparse(parsed)

            # Reuse cache if created via generic cache (unlikely due first call)
            if final_uri in _MVSTORE_CACHE:
                _CONTROL_PLANE_MVSTORE = _MVSTORE_CACHE[final_uri]
                return _CONTROL_PLANE_MVSTORE

            store = MultiVectorStore(uri=final_uri)
            # Initialize the control-plane multi-vector store
            import asyncio

            await asyncio.to_thread(store.initialize)
            _MVSTORE_CACHE[final_uri] = store
            _CONTROL_PLANE_MVSTORE = store
            logging.getLogger(__name__).info("Initialized control-plane MultiVectorStore")
            return store

    # ------------------------------------------------------------------
    # 2) Per-app routing
    # ------------------------------------------------------------------
    uri = await _resolve_connection_uri(app_id)

    # Free / Pro plans share the control-plane DB – reuse shared store
    if uri == settings.POSTGRES_URI:
        return await get_multi_vector_store_for_app(None)

    # Fallback (should not normally happen)
    if uri is None:
        uri = settings.POSTGRES_URI

    # Choose store type based on configuration
    if settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
        if FastMultiVectorStore is None:
            return None

        # Use original URI for FastMultiVectorStore (keeps asyncpg)
        if uri in _FAST_MVSTORE_CACHE:
            return _FAST_MVSTORE_CACHE[uri]

        if not settings.TURBOPUFFER_API_KEY:
            raise ValueError("TURBOPUFFER_API_KEY is required when using morphik multivector store provider")

        store = FastMultiVectorStore(uri=uri, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public")
        # Initialize the per-app fast multi-vector store
        import asyncio

        await asyncio.to_thread(store.initialize)
        _FAST_MVSTORE_CACHE[uri] = store
        logging.getLogger(__name__).info(f"Initialized FastMultiVectorStore for app_id={app_id}")
        return store
    else:
        # Use MultiVectorStore (legacy behavior)
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
        # Initialize the per-app multi-vector store
        import asyncio

        await asyncio.to_thread(store.initialize)
        _MVSTORE_CACHE[final_uri] = store
        logging.getLogger(__name__).info(f"Initialized MultiVectorStore for app_id={app_id}")
        return store
