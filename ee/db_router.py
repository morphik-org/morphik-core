from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase

try:
    from core.vector_store.pgvector_store import PGVectorStore
except ImportError:  # pragma: no cover
    PGVectorStore = None  # type: ignore

try:
    from core.vector_store.multi_vector_store import MultiVectorStore
except ImportError:  # pragma: no cover
    MultiVectorStore = None  # type: ignore

try:
    from core.vector_store.fast_multivector_store import FastMultiVectorStore
except ImportError:  # pragma: no cover
    FastMultiVectorStore = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["get_database_for_app", "get_vector_store_for_app", "get_multi_vector_store_for_app"]


_CONTROL_PLANE_DB: Optional[PostgresDatabase] = None
_CONTROL_PLANE_VSTORE: Optional["PGVectorStore"] = None
_CONTROL_PLANE_MVSTORE: Optional["MultiVectorStore"] = None
_CONTROL_PLANE_FAST_MVSTORE: Optional["FastMultiVectorStore"] = None


async def get_database_for_app(app_id: str | None) -> PostgresDatabase:
    """Return the single PostgresDatabase instance configured via POSTGRES_URI.

    Enterprise per-app routing has been removed, so *app_id* is ignored.  The
    function signature stays intact so existing imports keep working.
    """

    global _CONTROL_PLANE_DB  # noqa: PLW0603

    if _CONTROL_PLANE_DB is None:
        settings = get_settings()
        _CONTROL_PLANE_DB = PostgresDatabase(uri=settings.POSTGRES_URI)
        await _CONTROL_PLANE_DB.initialize()
        logger.info("Initialized shared PostgresDatabase")

    return _CONTROL_PLANE_DB


async def get_vector_store_for_app(app_id: str | None):
    """Return the shared PGVectorStore instance (if the optional dependency exists)."""
    if PGVectorStore is None:
        return None

    global _CONTROL_PLANE_VSTORE  # noqa: PLW0603

    if _CONTROL_PLANE_VSTORE is None:
        settings = get_settings()
        _CONTROL_PLANE_VSTORE = PGVectorStore(uri=settings.POSTGRES_URI)
        await _CONTROL_PLANE_VSTORE.initialize()
        logger.info("Initialized shared PGVectorStore")

    return _CONTROL_PLANE_VSTORE


async def get_multi_vector_store_for_app(app_id: str | None):
    """Return the configured multivector store (FastMultiVectorStore or MultiVectorStore)."""
    settings = get_settings()

    if settings.MULTIVECTOR_STORE_PROVIDER == "morphik":
        if FastMultiVectorStore is None:
            return None

        global _CONTROL_PLANE_FAST_MVSTORE  # noqa: PLW0603

        if _CONTROL_PLANE_FAST_MVSTORE is None:
            if not settings.TURBOPUFFER_API_KEY:
                raise ValueError("TURBOPUFFER_API_KEY is required when using the morphik multivector store provider")

            store = FastMultiVectorStore(
                uri=settings.POSTGRES_URI, tpuf_api_key=settings.TURBOPUFFER_API_KEY, namespace="public"
            )
            await asyncio.to_thread(store.initialize)
            _CONTROL_PLANE_FAST_MVSTORE = store
            logger.info("Initialized shared FastMultiVectorStore")

        return _CONTROL_PLANE_FAST_MVSTORE

    if MultiVectorStore is None:
        return None

    global _CONTROL_PLANE_MVSTORE  # noqa: PLW0603

    if _CONTROL_PLANE_MVSTORE is None:
        store = MultiVectorStore(uri=settings.POSTGRES_URI)
        await asyncio.to_thread(store.initialize)
        _CONTROL_PLANE_MVSTORE = store
        logger.info("Initialized shared MultiVectorStore")

    return _CONTROL_PLANE_MVSTORE
