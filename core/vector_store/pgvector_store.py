import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncContextManager, List, Optional, Tuple

from sqlalchemy import Column, Index, Integer, String, select, text, tuple_
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import UserDefinedType

from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)
Base = declarative_base()
PGVECTOR_MAX_DIMENSIONS = 2000  # Maximum dimensions for pgvector


class Vector(UserDefinedType):
    """Custom type for pgvector vectors."""

    # cache_ok = True  # Safe to use in cache key since vectors are immutable

    def get_col_spec(self, **kw):
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, list):
                return f"[{','.join(str(x) for x in value)}]"
            return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            # Handle different formats returned by the DB driver.
            # If the driver already gives us an iterable of floats, just cast and
            # return. Otherwise, fall back to parsing the string representation.
            if isinstance(value, (list, tuple)):
                # Some drivers may wrap the vector in an extra list layer
                # (e.g. [[0.1, 0.2, ...]]). Flatten one level so each element
                # fed to float() is a scalar, not another list.
                flattened: list = []
                for elem in value:
                    if isinstance(elem, (list, tuple)):
                        flattened.extend(elem)
                    else:
                        flattened.append(elem)

                return [float(x) for x in flattened]
            if isinstance(value, str):
                # Remove enclosing brackets (e.g. "[1,2,3]") and split.
                value = value.strip()[1:-1].split(",")
                return [float(x) for x in value if x]
            # In rare cases the driver may send a single float (e.g. when the
            # server happened to cast the vector to a numeric). Just return it
            # in a list so the rest of the code can keep the same expectations.
            if isinstance(value, (int, float)):
                return [float(value)]
            # Fallback – return as-is to avoid hard failure.
            return value

        return process


class VectorEmbedding(Base):
    """SQLAlchemy model for vector embeddings."""

    __tablename__ = "vector_embeddings"

    id = Column(Integer, primary_key=True)
    document_id = Column(String, nullable=False)
    chunk_number = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    chunk_metadata = Column(String, nullable=True)
    embedding = Column(Vector, nullable=False)

    # Create indexes
    __table_args__ = (
        Index("idx_document_id", "document_id"),
        Index(
            "idx_vector_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
        ),
    )


class PGVectorStore(BaseVectorStore):
    """PostgreSQL with pgvector implementation for vector storage."""

    def __init__(
        self,
        uri: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize PostgreSQL connection for vector storage.

        Args:
            uri: PostgreSQL connection URI
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
        """
        # Load settings from config
        from core.config import get_settings

        settings = get_settings()

        # Get database pool settings from config with defaults
        pool_size = getattr(settings, "DB_POOL_SIZE", 20)
        max_overflow = getattr(settings, "DB_MAX_OVERFLOW", 30)
        pool_recycle = getattr(settings, "DB_POOL_RECYCLE", 3600)
        pool_timeout = getattr(settings, "DB_POOL_TIMEOUT", 10)
        pool_pre_ping = getattr(settings, "DB_POOL_PRE_PING", True)

        # Strip parameters that asyncpg doesn't accept as keyword arguments
        # These will raise "unexpected keyword argument" errors
        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        parsed = urlparse(uri)
        query_params = parse_qs(parsed.query)

        # List of parameters that asyncpg doesn't accept
        incompatible_params = ["sslmode", "channel_binding"]
        removed_params = []

        for param in incompatible_params:
            if param in query_params:
                query_params.pop(param, None)
                removed_params.append(param)

        if removed_params:
            logger.debug(f"Removing parameters from PostgreSQL URI (not compatible with asyncpg): {removed_params}")
            parsed = parsed._replace(query=urlencode(query_params, doseq=True))
            uri = urlunparse(parsed)

        logger.info(
            f"Initializing vector store database engine with pool size={pool_size}, "
            f"max_overflow={max_overflow}, pool_recycle={pool_recycle}s"
        )

        # Create the engine with improved connection pool settings
        self.engine = create_async_engine(
            uri,
            # Prevent connection timeouts by keeping connections alive
            pool_pre_ping=pool_pre_ping,
            # Increase pool size to handle concurrent operations
            pool_size=pool_size,
            # Maximum overflow connections allowed beyond pool_size
            max_overflow=max_overflow,
            # Keep connections in the pool for up to 60 minutes
            pool_recycle=pool_recycle,
            # Time to wait for a connection from the pool (10 seconds)
            pool_timeout=pool_timeout,
            # Echo SQL for debugging (set to False in production)
            echo=False,
        )

        # Log success
        logger.info("Created vector store database engine successfully")
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @asynccontextmanager
    async def get_session_with_retry(self) -> AsyncContextManager[AsyncSession]:
        """Get a SQLAlchemy async session with retry logic.

        Yields:
            AsyncSession: A SQLAlchemy async session

        Raises:
            OperationalError: If all connection attempts fail
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                async with self.async_session() as session:
                    # Test if the connection is valid with a simple query
                    await session.execute(text("SELECT 1"))
                    yield session
                    return
            except OperationalError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(
                        f"Database connection attempt {attempt} failed: {str(e)}."
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    await asyncio.sleep(self.retry_delay)

        # If we get here, all retries failed
        logger.error(f"All database connection attempts failed after {self.max_retries} retries: {str(last_error)}")
        raise last_error

    async def initialize(self):
        """Initialize database tables and vector extension."""
        try:
            # Import config to get vector dimensions
            from core.config import get_settings

            settings = get_settings()
            dimensions = min(settings.VECTOR_DIMENSIONS, PGVECTOR_MAX_DIMENSIONS)

            logger.info(f"Initializing PGVector store with {dimensions} dimensions")

            # Use retry logic for initialization
            attempt = 0
            last_error = None

            while attempt < self.max_retries:
                try:
                    async with self.engine.begin() as conn:
                        # Enable pgvector extension
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        logger.info("Enabled pgvector extension")

                        # Rest of initialization code follows
                        break  # Success, exit the retry loop
                except OperationalError as e:
                    last_error = e
                    attempt += 1
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Database initialization attempt {attempt} failed: {str(e)}."
                            f"Retrying in {self.retry_delay} seconds..."
                        )
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(
                            f"All database initialization attempts failed after"
                            f"{self.max_retries} retries: {str(last_error)}"
                        )
                        raise last_error

            # Continue with the rest of the initialization
            async with self.engine.begin() as conn:
                # Check if vector_embeddings table exists
                check_table_sql = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'vector_embeddings'
                );
                """
                result = await conn.execute(text(check_table_sql))
                table_exists = result.scalar()

                if table_exists:
                    # Check current vector dimensions
                    check_dim_sql = """
                    SELECT atttypmod - 4 AS dimensions
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_type t ON a.atttypid = t.oid
                    WHERE c.relname = 'vector_embeddings'
                    AND a.attname = 'embedding'
                    AND t.typname = 'vector';
                    """
                    result = await conn.execute(text(check_dim_sql))
                    current_dim = result.scalar()

                    if (current_dim + 4) != dimensions:
                        logger.warning(
                            f"Vector dimensions changed from {current_dim} to {dimensions}."
                            "This requires recreating tables and will delete all existing vector data."
                        )

                        # Ask for explicit user confirmation
                        user_input = input(
                            f"WARNING: Embedding dimensions changed from {current_dim} to {dimensions}."
                            "This will DELETE ALL existing vector data. Type 'yes' to continue: "
                        )

                        if user_input.lower() != "yes":
                            logger.info("User aborted table recreation due to dimension change")
                            raise ValueError(
                                "Operation aborted by user. Vector dimension change requires recreating tables."
                            )

                        logger.info("User confirmed table recreation")

                        # Drop existing vector index if it exists
                        await conn.execute(text("DROP INDEX IF EXISTS vector_idx;"))

                        # Drop existing vector embeddings table
                        await conn.execute(text("DROP TABLE IF EXISTS vector_embeddings;"))

                        # Create vector embeddings table with proper vector column
                        create_table_sql = f"""
                        CREATE TABLE vector_embeddings (
                            id SERIAL PRIMARY KEY,
                            document_id VARCHAR(255) NOT NULL,
                            chunk_number INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            chunk_metadata TEXT,
                            embedding vector({dimensions}) NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                        await conn.execute(text(create_table_sql))
                        logger.info(f"Created vector_embeddings table with vector({dimensions})")

                        # Create indexes
                        await conn.execute(text("CREATE INDEX idx_document_id ON vector_embeddings(document_id);"))

                        # Create vector index
                        await conn.execute(
                            text(
                                """
                                CREATE INDEX vector_idx
                                ON vector_embeddings
                                USING ivfflat (embedding vector_cosine_ops)
                                WITH (lists = 100);
                                """
                            )
                        )
                        logger.info("Created IVFFlat index on vector_embeddings")

                        # Whether the table pre-existed or we just created it, make
                        # sure the application role can use the serial sequence.
                        try:
                            await conn.execute(
                                text("GRANT USAGE, SELECT ON SEQUENCE vector_embeddings_id_seq TO PUBLIC;")
                            )
                        except Exception as priv_exc:  # noqa: BLE001
                            # Log once at DEBUG level – most likely the current role *does*
                            # own the sequence already so the grant is unnecessary.
                            logger.debug("Privilege grant on sequence skipped: %s", priv_exc)
                    else:
                        logger.info(f"Vector dimensions unchanged ({dimensions}), using existing table")
                else:
                    # Create tables and indexes if they don't exist
                    create_table_sql = f"""
                    CREATE TABLE vector_embeddings (
                        id SERIAL PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        chunk_number INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        chunk_metadata TEXT,
                        embedding vector({dimensions}) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                    await conn.execute(text(create_table_sql))
                    logger.info(f"Created vector_embeddings table with vector({dimensions})")

                    # Create indexes
                    await conn.execute(text("CREATE INDEX idx_document_id ON vector_embeddings(document_id);"))

                    # Create vector index
                    await conn.execute(
                        text(
                            """
                            CREATE INDEX vector_idx
                            ON vector_embeddings
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                            """
                        )
                    )
                    logger.info("Created IVFFlat index on vector_embeddings")

            logger.info("PGVector store initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing PGVector store: {str(e)}")
            return False

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Bulk-insert embeddings in one go instead of row-by-row ORM adds.

        This avoids per-row overhead and cuts the SQL round-trips from *n*
        to 1, yielding a ~5-10× throughput boost for large documents.
        """

        if not chunks:
            return True, []

        # Flatten to plain dicts so SQLAlchemy can send one executemany call.
        rows = [
            {
                "document_id": c.document_id,
                "chunk_number": c.chunk_number,
                "content": c.content,
                "chunk_metadata": json.dumps(c.metadata or {}),
                "embedding": c.embedding,
            }
            for c in chunks
            if c.embedding  # Skip empty vectors early
        ]

        if not rows:
            logger.warning("No embeddings to store – all chunks had empty vectors")
            return True, []

        async with self.get_session_with_retry() as session:
            await session.execute(VectorEmbedding.__table__.insert().values(rows))
            await session.commit()

        stored_ids = [f"{r['document_id']}-{r['chunk_number']}" for r in rows]
        return True, stored_ids

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
        app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using cosine similarity."""
        try:
            async with self.get_session_with_retry() as session:
                # Build query with cosine distance calculation, which is normalized to [0, 2].
                # A distance of 0 is perfect similarity.
                distance = VectorEmbedding.embedding.op("<=>")(query_embedding)
                query = select(VectorEmbedding, distance).order_by(distance)

                if doc_ids:
                    query = query.filter(VectorEmbedding.document_id.in_(doc_ids))

                query = query.limit(k)
                result = await session.execute(query)
                embeddings = result.all()

                # Convert to DocumentChunks with similarity scores
                chunks = []
                for emb, distance in embeddings:
                    try:
                        metadata = json.loads(emb.chunk_metadata) if emb.chunk_metadata else {}
                    except Exception:
                        metadata = {}

                    # Drivers may apply the Vector result processor to the scalar
                    # distance value, wrapping it in a single-element list. Unwrap
                    # here so the float() cast below never fails.
                    if isinstance(distance, (list, tuple)):
                        # Use the first element – pgvector <=> always returns a
                        # single scalar distance.
                        distance = distance[0] if distance else 0.0

                    # Chunk scores are normalized to [0, 1] where 1 is a perfect match
                    chunk = DocumentChunk(
                        document_id=emb.document_id,
                        chunk_number=emb.chunk_number,
                        content=emb.content,
                        embedding=[],  # Don't send embeddings back
                        metadata=metadata,
                        score=1.0 - float(distance) / 2.0,
                    )
                    chunks.append(chunk)

                return chunks

        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
        app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve specific chunks by document ID and chunk number in a single database query.

        Args:
            chunk_identifiers: List of (document_id, chunk_number) tuples

        Returns:
            List of DocumentChunk objects
        """
        try:
            if not chunk_identifiers:
                return []

            unique_identifiers = list(dict.fromkeys(chunk_identifiers))
            async with self.get_session_with_retry() as session:
                # Build query to find all matching chunks in a single query using tuple comparison
                comparison_tuple = tuple_(VectorEmbedding.document_id, VectorEmbedding.chunk_number)
                query = select(VectorEmbedding).where(comparison_tuple.in_(unique_identifiers))

                logger.debug(f"Batch retrieving {len(unique_identifiers)} chunks with a single query")

                # Execute query
                result = await session.execute(query)
                chunk_models = result.scalars().all()

                # Convert to DocumentChunk objects
                chunks = []
                for chunk_model in chunk_models:
                    # Convert stored metadata string back to dict
                    try:
                        metadata = json.loads(chunk_model.chunk_metadata) if chunk_model.chunk_metadata else {}
                    except Exception:
                        metadata = {}

                    chunk = DocumentChunk(
                        document_id=chunk_model.document_id,
                        chunk_number=chunk_model.chunk_number,
                        content=chunk_model.content,
                        embedding=[],  # Don't send embeddings back
                        metadata=metadata,
                        score=0.0,  # No relevance score for direct retrieval
                    )
                    chunks.append(chunk)

                logger.debug(f"Found {len(chunks)} chunks in batch retrieval")
                return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by ID: {str(e)}")
            return []

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: ID of the document whose chunks should be deleted

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            async with self.get_session_with_retry() as session:
                # Delete all chunks for the specified document
                query = text("DELETE FROM vector_embeddings WHERE document_id = :doc_id")
                await session.execute(query, {"doc_id": document_id})
                await session.commit()

                logger.info(f"Deleted all chunks for document {document_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {str(e)}")
            return False
