import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Dict, List, Optional

from sqlalchemy import Column, Index, Integer, String, Text, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime

from core.database.metadata_filters import InvalidMetadataFilterError, MetadataFilterBuilder
from core.models.auth import AuthContext
from core.vector_store.pgvector_store import Vector
from core.vector_store.utils import build_store_metrics

logger = logging.getLogger(__name__)
Base = declarative_base()
PGVECTOR_MAX_DIMENSIONS = 2000


class ChunkV2Model(Base):
    """SQLAlchemy model for v2 chunks."""

    __tablename__ = "chunk_v2"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    document_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector, nullable=False)
    page_number = Column(Integer)
    chunk_number = Column(Integer)
    app_id = Column(String)
    end_user_id = Column(String)
    folder_path = Column(String)
    doc_metadata = Column("doc_metadata", JSONB)
    metadata_types = Column(JSONB)
    filename = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "chunk_v2_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
        ),
        Index("chunk_v2_document_id_idx", "document_id"),
    )


class ChunkV2Store:
    """PostgreSQL + pgvector store for v2 chunks."""

    def __init__(
        self,
        uri: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        from core.config import get_settings

        settings = get_settings()

        pool_size = getattr(settings, "DB_POOL_SIZE", 20)
        max_overflow = getattr(settings, "DB_MAX_OVERFLOW", 30)
        pool_recycle = getattr(settings, "DB_POOL_RECYCLE", 3600)
        pool_timeout = getattr(settings, "DB_POOL_TIMEOUT", 10)
        pool_pre_ping = getattr(settings, "DB_POOL_PRE_PING", True)

        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        parsed = urlparse(uri)
        query_params = parse_qs(parsed.query)
        incompatible_params = ["sslmode", "channel_binding"]
        removed_params = []
        for param in incompatible_params:
            if param in query_params:
                query_params.pop(param, None)
                removed_params.append(param)

        if removed_params:
            logger.debug("Removing parameters from PostgreSQL URI (not compatible with asyncpg): %s", removed_params)
            parsed = parsed._replace(query=urlencode(query_params, doseq=True))
            uri = urlunparse(parsed)

        logger.info(
            "Initializing v2 chunk store database engine with pool size=%s, max_overflow=%s",
            pool_size,
            max_overflow,
        )

        self.engine = create_async_engine(
            uri,
            pool_pre_ping=pool_pre_ping,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_timeout=pool_timeout,
            echo=False,
        )
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_store_metrics: Dict[str, Any] = {}
        self._metadata_filter_builder = MetadataFilterBuilder(
            metadata_column="doc_metadata",
            metadata_types_column="metadata_types",
        )
        self.ivfflat_probes = max(1, int(getattr(settings, "VECTOR_IVFFLAT_PROBES", 100) or 100))

    @asynccontextmanager
    async def get_session_with_retry(self) -> AsyncContextManager[AsyncSession]:
        attempt = 0
        last_error = None
        while attempt < self.max_retries:
            try:
                async with self.async_session() as session:
                    await session.execute(text("SELECT 1"))
                    yield session
                    return
            except OperationalError as exc:
                last_error = exc
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(
                        "Chunk v2 DB connection attempt %s failed: %s. Retrying in %ss...",
                        attempt,
                        exc,
                        self.retry_delay,
                    )
                    await asyncio.sleep(self.retry_delay)

        logger.error("All chunk v2 DB connection attempts failed: %s", last_error)
        raise last_error

    async def initialize(self) -> bool:
        from core.config import get_settings

        settings = get_settings()
        dimensions = min(settings.VECTOR_DIMENSIONS, PGVECTOR_MAX_DIMENSIONS)

        try:
            attempt = 0
            last_error = None
            while attempt < self.max_retries:
                try:
                    async with self.engine.begin() as conn:
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    break
                except OperationalError as exc:
                    last_error = exc
                    attempt += 1
                    if attempt < self.max_retries:
                        logger.warning(
                            "Chunk v2 DB init attempt %s failed: %s. Retrying in %ss...",
                            attempt,
                            exc,
                            self.retry_delay,
                        )
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error("Chunk v2 DB init failed after retries: %s", last_error)
                        raise

            async with self.engine.begin() as conn:
                check_table_sql = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'chunk_v2'
                );
                """
                result = await conn.execute(text(check_table_sql))
                table_exists = result.scalar()

                if table_exists:
                    column_check_sql = """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'chunk_v2';
                    """
                    result = await conn.execute(text(column_check_sql))
                    columns = {row[0] for row in result.fetchall()}
                    if "doc_metadata" not in columns:
                        if "metadata" in columns:
                            await conn.execute(text("ALTER TABLE chunk_v2 RENAME COLUMN metadata TO doc_metadata;"))
                        else:
                            await conn.execute(text("ALTER TABLE chunk_v2 ADD COLUMN doc_metadata JSONB;"))

                    check_dim_sql = """
                    SELECT atttypmod - 4 AS dimensions
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_type t ON a.atttypid = t.oid
                    WHERE c.relname = 'chunk_v2'
                    AND a.attname = 'embedding'
                    AND t.typname = 'vector';
                    """
                    result = await conn.execute(text(check_dim_sql))
                    current_dim = result.scalar()

                    if current_dim is not None and (current_dim + 4) != dimensions:
                        logger.warning(
                            "chunk_v2 vector dimensions changed from %s to %s. Table recreation required.",
                            current_dim,
                            dimensions,
                        )
                        user_input = input(
                            f"WARNING: chunk_v2 embedding dimensions changed from {current_dim} to {dimensions}. "
                            "This will DELETE ALL existing v2 chunk data. Type 'yes' to continue: "
                        )
                        if user_input.lower() != "yes":
                            raise ValueError(
                                "Operation aborted by user. chunk_v2 dimension change requires recreation."
                            )

                        await conn.execute(text("DROP INDEX IF EXISTS chunk_v2_embedding_idx;"))
                        await conn.execute(text("DROP INDEX IF EXISTS chunk_v2_document_id_idx;"))
                        await conn.execute(text("DROP TABLE IF EXISTS chunk_v2;"))

                        create_table_sql = f"""
                        CREATE TABLE chunk_v2 (
                            id UUID PRIMARY KEY,
                            document_id VARCHAR NOT NULL REFERENCES documents(external_id),
                            content TEXT NOT NULL,
                            embedding vector({dimensions}) NOT NULL,
                            page_number INT,
                            chunk_number INT,
                            app_id VARCHAR,
                            end_user_id VARCHAR,
                            folder_path VARCHAR,
                            doc_metadata JSONB,
                            metadata_types JSONB,
                            filename VARCHAR,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                        await conn.execute(text(create_table_sql))
                        await conn.execute(
                            text(
                                """
                                CREATE INDEX chunk_v2_embedding_idx
                                ON chunk_v2
                                USING ivfflat (embedding vector_cosine_ops)
                                WITH (lists = 100);
                                """
                            )
                        )
                        await conn.execute(text("CREATE INDEX chunk_v2_document_id_idx ON chunk_v2 (document_id);"))
                    else:
                        logger.info("chunk_v2 table exists with matching vector dimensions (%s)", dimensions)
                else:
                    create_table_sql = f"""
                    CREATE TABLE chunk_v2 (
                        id UUID PRIMARY KEY,
                        document_id VARCHAR NOT NULL REFERENCES documents(external_id),
                        content TEXT NOT NULL,
                        embedding vector({dimensions}) NOT NULL,
                        page_number INT,
                        chunk_number INT,
                        app_id VARCHAR,
                        end_user_id VARCHAR,
                        folder_path VARCHAR,
                        doc_metadata JSONB,
                        metadata_types JSONB,
                        filename VARCHAR,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                    await conn.execute(text(create_table_sql))
                    await conn.execute(
                        text(
                            """
                            CREATE INDEX chunk_v2_embedding_idx
                            ON chunk_v2
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                            """
                        )
                    )
                    await conn.execute(text("CREATE INDEX chunk_v2_document_id_idx ON chunk_v2 (document_id);"))

            logger.info("chunk_v2 store initialized")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Error initializing chunk_v2 store: %s", exc)
            return False

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> tuple[bool, List[str], Dict[str, Any]]:
        if not chunks:
            self._last_store_metrics = build_store_metrics(
                chunk_payload_backend="none",
                multivector_backend="none",
                vector_store_backend="pgvector_v2",
            )
            return True, [], self._last_store_metrics

        rows = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            rows.append(
                {
                    "id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "content": chunk["content"],
                    "embedding": embedding,
                    "page_number": chunk.get("page_number"),
                    "chunk_number": chunk.get("chunk_number"),
                    "app_id": chunk.get("app_id"),
                    "end_user_id": chunk.get("end_user_id"),
                    "folder_path": chunk.get("folder_path"),
                    "doc_metadata": chunk.get("doc_metadata"),
                    "metadata_types": chunk.get("metadata_types"),
                    "filename": chunk.get("filename"),
                }
            )

        if not rows:
            self._last_store_metrics = build_store_metrics(
                chunk_payload_backend="none",
                multivector_backend="none",
                vector_store_backend="pgvector_v2",
            )
            return True, [], self._last_store_metrics

        chunk_payload_bytes = sum(len(row["content"].encode("utf-8")) for row in rows if row.get("content"))

        write_start = time.perf_counter()
        async with self.get_session_with_retry() as session:
            await session.execute(ChunkV2Model.__table__.insert().values(rows))
            await session.commit()
        write_duration = time.perf_counter() - write_start

        self._last_store_metrics = build_store_metrics(
            chunk_payload_backend="none",
            multivector_backend="none",
            vector_store_backend="pgvector_v2",
            chunk_payload_bytes=chunk_payload_bytes,
            vector_store_write_s=write_duration,
            vector_store_rows=len(rows),
        )

        stored_ids = [str(row["id"]) for row in rows]
        return True, stored_ids, self._last_store_metrics

    def latest_store_metrics(self) -> Dict[str, Any]:
        return dict(self._last_store_metrics) if self._last_store_metrics else {}

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        auth: AuthContext,
        document_ids: Optional[List[str]] = None,
        folder_paths: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        end_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            conditions = []
            if not auth.app_id:
                raise PermissionError("app_id is required for v2 chunk queries")
            conditions.append(ChunkV2Model.app_id == auth.app_id)

            if end_user_id is not None:
                conditions.append(ChunkV2Model.end_user_id == end_user_id)

            if document_ids:
                conditions.append(ChunkV2Model.document_id.in_(document_ids))

            if folder_paths:
                conditions.append(ChunkV2Model.folder_path.in_(folder_paths))

            if metadata_filters:
                clause = self._metadata_filter_builder.build(metadata_filters)
                if clause:
                    conditions.append(text(clause))

            distance = ChunkV2Model.embedding.op("<=>")(query_embedding)
            query = (
                select(
                    ChunkV2Model.id,
                    ChunkV2Model.document_id,
                    ChunkV2Model.content,
                    ChunkV2Model.page_number,
                    ChunkV2Model.chunk_number,
                    ChunkV2Model.filename,
                    distance.label("distance"),
                )
                .where(*conditions)
                .order_by(distance)
                .limit(k)
            )

            async with self.get_session_with_retry() as session:
                # Use set_config() for parameterized config - SET doesn't support parameters
                await session.execute(
                    text("SELECT set_config('ivfflat.probes', :probes, true)"), {"probes": str(self.ivfflat_probes)}
                )
                result = await session.execute(query)
                rows = result.all()

            chunks: List[Dict[str, Any]] = []
            for row in rows:
                distance_val = row.distance
                if isinstance(distance_val, (list, tuple)):
                    distance_val = distance_val[0] if distance_val else 0.0
                score = 1.0 - float(distance_val) / 2.0
                chunks.append(
                    {
                        "id": str(row.id),
                        "document_id": row.document_id,
                        "content": row.content,
                        "page_number": row.page_number,
                        "chunk_number": row.chunk_number,
                        "filename": row.filename,
                        "score": score,
                    }
                )

            return chunks
        except InvalidMetadataFilterError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Error querying chunk_v2 store: %s", exc)
            return []

    async def delete_chunks_by_document_id(self, document_id: str, auth: AuthContext) -> bool:
        """Delete chunks for a document, scoped by app_id to prevent cross-tenant deletion."""
        try:
            async with self.get_session_with_retry() as session:
                if not auth.app_id:
                    raise PermissionError("app_id is required for v2 chunk deletion")
                await session.execute(
                    text("DELETE FROM chunk_v2 WHERE document_id = :doc_id AND app_id = :app_id"),
                    {"doc_id": document_id, "app_id": auth.app_id},
                )
                await session.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Error deleting chunk_v2 rows for %s: %s", document_id, exc)
            return False
