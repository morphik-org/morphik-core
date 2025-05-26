import asyncio
import json
import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np
import psycopg
import torch
from pgvector.psycopg import Bit, register_vector
from psycopg_pool import ConnectionPool

from core.models.chunk import DocumentChunk

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class MultiVectorStore(BaseVectorStore):
    """PostgreSQL implementation for storing and querying multi-vector embeddings using psycopg."""

    def __init__(
        self,
        uri: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize PostgreSQL connection for multi-vector storage.

        Args:
            uri: PostgreSQL connection URI
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
        """
        # Convert SQLAlchemy URI to psycopg format if needed
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://")
        self.uri = uri
        # Shared connection pool – re-uses sockets across jobs, avoids TLS
        # handshakes and auth for every INSERT call.  A small pool is enough
        # because inserts are short-lived.
        self.pool: ConnectionPool = ConnectionPool(conninfo=self.uri, min_size=1, max_size=10, timeout=60)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Don't initialize here - initialization will be handled separately

    @contextmanager
    def get_connection(self):
        """Get a PostgreSQL connection with retry logic.

        Yields:
            A PostgreSQL connection object

        Raises:
            psycopg.OperationalError: If all connection attempts fail
        """
        attempt = 0
        last_error = None

        # Try to establish a new connection with retries
        while attempt < self.max_retries:
            try:
                # Borrow a pooled connection (blocking wait). Autocommit stays
                # disabled so we can batch-commit.
                conn = self.pool.getconn()

                try:
                    yield conn
                    return
                finally:
                    # Release connection back to the pool
                    try:
                        self.pool.putconn(conn)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
            except psycopg.OperationalError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(
                        f"Connection attempt {attempt} failed: {str(e)}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)

        # If we get here, all retries failed
        logger.error(f"All connection attempts failed after {self.max_retries} retries: {str(last_error)}")
        raise last_error

    def initialize(self):
        """Initialize database tables and max_sim function."""
        try:
            # Use the connection with retry logic
            with self.get_connection() as conn:
                # Register vector extension
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                register_vector(conn)

            # First check if the table exists and if it has the required columns
            with self.get_connection() as conn:
                check_table = conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'multi_vector_embeddings'
                    );
                """
                ).fetchone()[0]

                if check_table:
                    # Check if document_id column exists
                    has_document_id = conn.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns
                            WHERE table_name = 'multi_vector_embeddings' AND column_name = 'document_id'
                        );
                    """
                    ).fetchone()[0]

                    # If the table exists but doesn't have document_id, we need to add the required columns
                    if not has_document_id:
                        logger.info("Updating multi_vector_embeddings table with required columns")
                        conn.execute(
                            """
                            ALTER TABLE multi_vector_embeddings
                            ADD COLUMN document_id TEXT,
                            ADD COLUMN chunk_number INTEGER,
                            ADD COLUMN content TEXT,
                            ADD COLUMN chunk_metadata TEXT
                        """
                        )
                        conn.execute(
                            """
                            ALTER TABLE multi_vector_embeddings
                            ALTER COLUMN document_id SET NOT NULL
                        """
                        )

                        # Add a commit to ensure changes are applied
                        conn.commit()
                else:
                    # Create table if it doesn't exist with all required columns
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS multi_vector_embeddings (
                            id BIGSERIAL PRIMARY KEY,
                            document_id TEXT NOT NULL,
                            chunk_number INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            chunk_metadata TEXT,
                            embeddings BIT(128)[]
                        )
                    """
                    )

                # Add a commit to ensure table creation is complete
                conn.commit()

            try:
                # Create index on document_id
                with self.get_connection() as conn:
                    conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_multi_vector_document_id
                        ON multi_vector_embeddings (document_id)
                    """
                    )
            except Exception as e:
                # Log index creation failure but continue
                logger.warning(f"Failed to create index: {str(e)}")

            try:
                # First, try to drop the existing function if it exists
                with self.get_connection() as conn:
                    conn.execute(
                        """
                        DROP FUNCTION IF EXISTS max_sim(bit[], bit[])
                    """
                    )
                    logger.info("Dropped existing max_sim function")

                    # Create max_sim function
                    conn.execute(
                        """
                        CREATE OR REPLACE FUNCTION max_sim(document bit[], query bit[]) RETURNS double precision AS $$
                            WITH queries AS (
                                SELECT row_number() OVER () AS query_number, *
                                FROM (SELECT unnest(query) AS query) AS foo
                            ),
                            documents AS (
                                SELECT unnest(document) AS document
                            ),
                            similarities AS (
                                SELECT
                                    query_number,
                                    1.0 - (bit_count(document # query)::float /
                                        greatest(bit_length(query), 1)::float) AS similarity
                                FROM queries CROSS JOIN documents
                            ),
                            max_similarities AS (
                                SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
                            )
                            SELECT SUM(max_similarity) FROM max_similarities
                        $$ LANGUAGE SQL
                    """
                    )
                    logger.info("Created max_sim function successfully")
            except Exception as e:
                logger.error(f"Error creating max_sim function: {str(e)}")
                # Continue even if function creation fails - it might already exist and be usable

            logger.info("MultiVectorStore initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MultiVectorStore: {str(e)}")
            return False

    def _binary_quantize(self, embeddings: Union[np.ndarray, torch.Tensor, List]) -> List[Bit]:
        """Convert embeddings to binary format for PostgreSQL BIT[] arrays."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(embeddings, list) and not isinstance(embeddings[0], np.ndarray):
            embeddings = np.array(embeddings)

        return [Bit(embedding > 0) for embedding in embeddings]

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their multi-vector embeddings."""
        # Prepare a list of row tuples for executemany
        rows = []
        for chunk in chunks:
            if not hasattr(chunk, "embedding") or chunk.embedding is None:
                logger.error(f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}")
                continue

            binary_embeddings = self._binary_quantize(chunk.embedding)

            rows.append(
                (
                    chunk.document_id,
                    chunk.chunk_number,
                    chunk.content,
                    str(chunk.metadata),
                    binary_embeddings,
                )
            )

        if not rows:
            return True, []

        # Off-load blocking DB I/O to a thread so we don't block the event loop
        await asyncio.to_thread(self._bulk_insert_rows, rows)

        stored_ids = [f"{r[0]}-{r[1]}" for r in rows]
        logger.debug(f"{len(stored_ids)} multi-vector embeddings added in bulk")
        return True, stored_ids

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using the max_sim function for multi-vectors."""
        # try:
        # Convert query embeddings to binary format
        binary_query_embeddings = self._binary_quantize(query_embedding)

        def _bit_raw(b: Bit) -> str:
            """Return raw bit string without 'Bit(...)' wrapper"""
            s = str(b)
            # Expected formats: "Bit('1010')" or "Bit(1010)"
            if s.startswith("Bit("):
                s = s[4:-1]  # strip wrapper
                s = s.strip("'")
            return s

        bit_strings = [_bit_raw(b) for b in binary_query_embeddings]
        array_literal = "ARRAY[" + ",".join(f"B'{s}'" for s in bit_strings) + "]::bit(128)[]"

        # Start query with inlined array literal (internal usage only)
        query = (
            "SELECT id, document_id, chunk_number, content, chunk_metadata, "
            f"max_sim(embeddings, {array_literal}) AS similarity "
            "FROM multi_vector_embeddings"
        )

        params: List = []

        if doc_ids:
            placeholders = ", ".join(["%s"] * len(doc_ids))
            query += f" WHERE document_id IN ({placeholders})"
            params.extend(doc_ids)

        query += " ORDER BY similarity DESC LIMIT %s"
        params.append(k)

        with self.get_connection() as conn:
            result = conn.execute(query, tuple(params)).fetchall()

        # Convert to DocumentChunks
        chunks = []
        for row in result:
            try:
                metadata = json.loads(row[4]) if row[4] else {}
            except Exception:
                metadata = {}

            chunk = DocumentChunk(
                document_id=row[1],
                chunk_number=row[2],
                content=row[3],
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=float(row[5]),  # Use the similarity score from max_sim
            )
            chunks.append(chunk)

        return chunks

        # except Exception as e:
        #     logger.error(f"Error querying similar chunks: {str(e)}")
        #     raise e
        #     return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
    ) -> List[DocumentChunk]:
        """
        Retrieve specific chunks by document ID and chunk number in a single database query.

        Args:
            chunk_identifiers: List of (document_id, chunk_number) tuples

        Returns:
            List of DocumentChunk objects
        """
        # try:
        if not chunk_identifiers:
            return []

        # Construct the WHERE clause with OR conditions
        conditions = []
        for doc_id, chunk_num in chunk_identifiers:
            conditions.append(f"(document_id = '{doc_id}' AND chunk_number = {chunk_num})")

        where_clause = " OR ".join(conditions)

        # Build and execute query
        query = f"""
            SELECT document_id, chunk_number, content, chunk_metadata
            FROM multi_vector_embeddings
            WHERE {where_clause}
        """

        logger.debug(f"Batch retrieving {len(chunk_identifiers)} chunks from multi-vector store")

        with self.get_connection() as conn:
            result = conn.execute(query).fetchall()

        # Convert to DocumentChunks
        chunks = []
        for row in result:
            try:
                metadata = json.loads(row[3]) if row[3] else {}
            except Exception:
                metadata = {}

            chunk = DocumentChunk(
                document_id=row[0],
                chunk_number=row[1],
                content=row[2],
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=0.0,  # No relevance score for direct retrieval
            )
            chunks.append(chunk)

        logger.debug(f"Found {len(chunks)} chunks in batch retrieval from multi-vector store")
        return chunks

    async def delete_chunks_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: ID of the document whose chunks should be deleted

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Delete all chunks for the specified document with retry logic
            query = f"DELETE FROM multi_vector_embeddings WHERE document_id = '{document_id}'"
            with self.get_connection() as conn:
                conn.execute(query)

            logger.info(f"Deleted all chunks for document {document_id} from multi-vector store")
            return True

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id} from multi-vector store: {str(e)}")
            return False

    def close(self):
        """Close the database connection."""
        # Close pool gracefully – this will close all underlying connections
        try:
            self.pool.close()
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")

    # ----------------- internal helpers -----------------

    def _bulk_insert_rows(self, rows: List[Tuple]):
        """Sync helper executed in a worker thread to avoid blocking."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO multi_vector_embeddings
                    (document_id, chunk_number, content, chunk_metadata, embeddings)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )
                # Single commit for all rows – very fast
                conn.commit()
