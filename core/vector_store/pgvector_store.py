from typing import List, Optional, Tuple
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, Index, select, text
from sqlalchemy.sql.expression import func
from sqlalchemy.types import UserDefinedType

from .base_vector_store import BaseVectorStore
from core.models.chunk import DocumentChunk

logger = logging.getLogger(__name__)
Base = declarative_base()


class Vector(UserDefinedType):
    """Custom type for pgvector vectors."""

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
            # Remove brackets and split by comma
            value = value[1:-1].split(",")
            return [float(x) for x in value]

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
    ):
        """Initialize PostgreSQL connection for vector storage."""
        self.engine = create_async_engine(uri)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self):
        """Initialize database tables and vector extension."""
        try:
            async with self.engine.begin() as conn:
                # Enable pgvector extension
                await conn.execute(
                    func.create_extension("vector", schema="public", if_not_exists=True)
                )

                # Create tables and indexes
                await conn.run_sync(Base.metadata.create_all)

                # Create vector index
                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS vector_idx
                    ON vector_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
                    )
                )

            logger.info("PGVector store initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing PGVector store: {str(e)}")
            return False

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their embeddings."""
        try:
            if not chunks:
                return True, []

            async with self.async_session() as session:
                stored_ids = []
                for chunk in chunks:
                    if not chunk.embedding:
                        logger.error(
                            f"Missing embedding for chunk {chunk.document_id}-{chunk.chunk_number}"
                        )
                        continue

                    vector_embedding = VectorEmbedding(
                        document_id=chunk.document_id,
                        chunk_number=chunk.chunk_number,
                        content=chunk.content,
                        chunk_metadata=str(chunk.metadata),
                        embedding=chunk.embedding,
                    )
                    session.add(vector_embedding)
                    stored_ids.append(f"{chunk.document_id}-{chunk.chunk_number}")

                await session.commit()
                return len(stored_ids) > 0, stored_ids

        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False, []

    async def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using cosine similarity."""
        try:
            async with self.async_session() as session:
                # Build query
                query = select(VectorEmbedding).order_by(
                    VectorEmbedding.embedding.op("<->")(query_embedding)
                )

                if doc_ids:
                    query = query.filter(VectorEmbedding.document_id.in_(doc_ids))

                query = query.limit(k)
                result = await session.execute(query)
                embeddings = result.scalars().all()

                # Convert to DocumentChunks
                chunks = []
                for emb in embeddings:
                    try:
                        metadata = eval(emb.chunk_metadata) if emb.chunk_metadata else {}
                    except (ValueError, SyntaxError):
                        metadata = {}

                    chunk = DocumentChunk(
                        document_id=emb.document_id,
                        chunk_number=emb.chunk_number,
                        content=emb.content,
                        embedding=[],  # Don't send embeddings back
                        metadata=metadata,
                    )
                    chunks.append(chunk)

                return chunks

        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []
