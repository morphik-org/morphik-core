from core.config import Settings
from .base_vector_store import BaseVectorStore
from .pgvector_store import PGVectorStore
from .qdrant_store import QdrantVectorStore


def vector_store_factory(settings: Settings) -> BaseVectorStore:
    prov = settings.VECTOR_STORE_PROVIDER
    if prov == "pgvector":
        if not settings.POSTGRES_URI:
            raise ValueError("PostgreSQL URI is required for pgvector store")
        return PGVectorStore(uri=settings.POSTGRES_URI)
    elif prov == "qdrant":
        if not settings.QDRANT_HOST:
            raise ValueError("Qdrant host is required for qdrant store")
        return QdrantVectorStore(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, https=settings.QDRANT_HTTPS)
    else:
        raise ValueError(f"Unknown vector store provider selected: '{prov}'")
