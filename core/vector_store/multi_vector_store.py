import asyncio
import base64
import json
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg
import torch
from pgvector.psycopg import Bit, register_vector
from psycopg_pool import ConnectionPool

from core.config import get_settings
from core.models.chunk import DocumentChunk
from core.storage.base_storage import BaseStorage
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.storage.utils_file_extensions import detect_file_type

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Constants for external storage
MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"
DEFAULT_APP_ID = "default"  # Fallback for local usage when app_id is None


class MultiVectorStore(BaseVectorStore):
    """PostgreSQL implementation for storing and querying multi-vector embeddings using psycopg."""

    def __init__(
        self,
        uri: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_initialize: bool = True,
        enable_external_storage: bool = True,
    ):
        """Initialize PostgreSQL connection for multi-vector storage.

        Args:
            uri: PostgreSQL connection URI
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
            auto_initialize: Whether to automatically initialize the store
            enable_external_storage: Whether to use external storage for chunks
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

        # Initialize external storage if enabled
        self.enable_external_storage = enable_external_storage
        self.storage: Optional[BaseStorage] = None
        self._document_app_id_cache: Dict[str, str] = {}  # Cache for document app_ids

        if enable_external_storage:
            self.storage = self._init_storage()

        # Optionally initialize database objects (tables, functions, etc.)
        # This ensures that required items like the max_sim function exist and
        # avoids runtime errors when the store is first used.
        if auto_initialize:
            try:
                self.initialize()
            except Exception as exc:
                # Log the failure but do not crash the application – callers
                # can still attempt explicit initialization or handle errors.
                logger.error("Auto-initialization of MultiVectorStore failed: %s", exc)

    def _init_storage(self) -> BaseStorage:
        """Initialize appropriate storage backend based on settings."""
        try:
            settings = get_settings()
            if settings.STORAGE_PROVIDER == "aws-s3":
                logger.info("Initializing S3 storage for multi-vector chunks")
                return S3Storage(
                    aws_access_key=settings.AWS_ACCESS_KEY,
                    aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    default_bucket=MULTIVECTOR_CHUNKS_BUCKET,
                )
            else:
                logger.info("Initializing local storage for multi-vector chunks")
                storage_path = getattr(settings, "LOCAL_STORAGE_PATH", "./storage")
                return LocalStorage(storage_path=storage_path)
        except Exception as e:
            logger.error(f"Failed to initialize external storage: {e}")
            return None

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

            # Create max_sim function for multi-vector similarity search
            # This function is specific to multi-vector operations and belongs here
            try:
                with self.get_connection() as conn:
                    exists_check = conn.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_proc
                            WHERE proname = 'max_sim'
                            AND pg_get_function_arguments(oid) = 'document bit[], query bit[]'
                        )
                    """
                    ).fetchone()[0]

                    if not exists_check:
                        logger.info("Creating max_sim function for multi-vector similarity search")
                        conn.execute(
                            """
                            CREATE OR REPLACE FUNCTION public.max_sim(document bit[], query bit[])
                            RETURNS double precision
                            LANGUAGE SQL
                            IMMUTABLE
                            PARALLEL SAFE
                            AS $$
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
                                SELECT COALESCE(SUM(max_similarity), 0.0) FROM max_similarities
                            $$
                        """
                        )
                        conn.commit()
                        logger.info("Created max_sim function successfully")
                    else:
                        logger.debug("max_sim function already exists")

            except Exception as e:
                logger.error(f"Error creating or checking max_sim function: {str(e)}")
                # Continue - we'll get a runtime error if the function is actually missing

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

    async def _get_document_app_id(self, document_id: str) -> str:
        """Get app_id for a document, with caching."""
        if document_id in self._document_app_id_cache:
            return self._document_app_id_cache[document_id]

        try:
            query = "SELECT system_metadata->>'app_id' FROM documents WHERE external_id = %s"
            with self.get_connection() as conn:
                result = conn.execute(query, (document_id,)).fetchone()

            app_id = result[0] if result and result[0] else DEFAULT_APP_ID
            self._document_app_id_cache[document_id] = app_id
            return app_id
        except Exception as e:
            logger.warning(f"Failed to get app_id for document {document_id}: {e}")
            return DEFAULT_APP_ID

    def _determine_file_extension(self, content: str, chunk_metadata: Optional[str]) -> str:
        """Determine appropriate file extension based on content and metadata."""
        try:
            # Parse chunk metadata to check if it's an image
            if chunk_metadata:
                metadata = json.loads(chunk_metadata)
                is_image = metadata.get("is_image", False)

                if is_image:
                    # For images, auto-detect from base64 content
                    return detect_file_type(content)
                else:
                    # For text content, use .txt
                    return ".txt"
            else:
                # No metadata, try to auto-detect
                return detect_file_type(content)

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error parsing chunk metadata: {e}")
            # Fallback to auto-detection
            return detect_file_type(content)

    def _generate_storage_key(self, app_id: str, document_id: str, chunk_number: int, extension: str) -> str:
        """Generate storage key path."""
        return f"{app_id}/{document_id}/{chunk_number}{extension}"

    async def _store_content_externally(
        self,
        content: str,
        document_id: str,
        chunk_number: int,
        chunk_metadata: Optional[str],
        app_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store chunk content in external storage and return storage key."""
        if not self.storage:
            return None

        try:
            # Use provided app_id or fall back to document lookup
            if app_id is None:
                logger.warning(f"No app_id provided for document {document_id}, falling back to database lookup")
                app_id = await self._get_document_app_id(document_id)
            else:
                logger.debug(f"Using provided app_id: {app_id} for document {document_id}")

            # Determine file extension
            extension = self._determine_file_extension(content, chunk_metadata)

            # Generate storage key
            storage_key = self._generate_storage_key(app_id, document_id, chunk_number, extension)

            # Store content in external storage
            if extension == ".txt":
                # For text content, store as-is without base64 encoding
                # Convert content to base64 for storage interface compatibility
                content_bytes = content.encode("utf-8")
                content_b64 = base64.b64encode(content_bytes).decode("utf-8")
                await self.storage.upload_from_base64(
                    content=content_b64, key=storage_key, content_type="text/plain", bucket=MULTIVECTOR_CHUNKS_BUCKET
                )
            else:
                # For images, content should already be base64
                await self.storage.upload_from_base64(
                    content=content, key=storage_key, bucket=MULTIVECTOR_CHUNKS_BUCKET
                )

            logger.debug(f"Stored chunk content externally with key: {storage_key}")
            return storage_key

        except Exception as e:
            logger.error(f"Failed to store content externally for {document_id}-{chunk_number}: {e}")
            return None

    def _is_storage_key(self, content: str) -> bool:
        """Check if content field contains a storage key rather than actual content."""
        # Storage keys are short paths with slashes, not base64/long content
        return (
            len(content) < 500 and "/" in content and not content.startswith("data:") and not content.startswith("http")
        )

    async def _retrieve_content_from_storage(self, storage_key: str, chunk_metadata: Optional[str]) -> str:
        """Retrieve content from external storage and convert to expected format."""
        logger.debug(f"Attempting to retrieve content from storage key: {storage_key}")

        if not self.storage:
            logger.warning(f"External storage not available for retrieving key: {storage_key}")
            return storage_key  # Return storage key as fallback

        try:
            # Download content from storage
            logger.debug(f"Downloading from bucket: {MULTIVECTOR_CHUNKS_BUCKET}, key: {storage_key}")
            if isinstance(self.storage, S3Storage):
                attempted_key = f"{MULTIVECTOR_CHUNKS_BUCKET}/{storage_key}"
            try:
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=attempted_key)
            except Exception:
                attempted_key = f"{MULTIVECTOR_CHUNKS_BUCKET}/{storage_key}.txt"
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=attempted_key)

            if not content_bytes:
                logger.error(f"No content downloaded for storage key: {storage_key}")
                return storage_key

            logger.debug(f"Downloaded {len(content_bytes)} bytes for key: {storage_key}")

            # Determine if this should be returned as base64 or text
            try:
                if chunk_metadata:
                    metadata = json.loads(chunk_metadata)
                    is_image = metadata.get("is_image", False)
                    logger.debug(f"Chunk metadata indicates is_image: {is_image}")

                    if is_image:
                        # For images, return as base64 string
                        result = base64.b64encode(content_bytes).decode("utf-8")
                        logger.debug(f"Returning image as base64, length: {len(result)}")
                        return result
                    else:
                        # For text, return decoded string
                        result = content_bytes.decode("utf-8")
                        logger.debug(f"Returning text content, length: {len(result)}")
                        return result
                else:
                    # No metadata, try to determine based on content
                    logger.debug("No metadata, auto-detecting content type")
                    # If it's valid UTF-8, treat as text
                    try:
                        result = content_bytes.decode("utf-8")
                        logger.debug(f"Auto-detected as text, length: {len(result)}")
                        return result
                    except UnicodeDecodeError:
                        # If not valid UTF-8, treat as binary (image) and return base64
                        result = base64.b64encode(content_bytes).decode("utf-8")
                        logger.debug(f"Auto-detected as binary, returning base64, length: {len(result)}")
                        return result

            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error determining content type for {storage_key}: {e}")
                # Fallback: try text first, then base64
                try:
                    return content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    return base64.b64encode(content_bytes).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to retrieve content from storage key {storage_key}: {e}", exc_info=True)
            return storage_key  # Return storage key as fallback

    async def store_embeddings(
        self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Store document chunks with their multi-vector embeddings."""
        # Prepare a list of row tuples for executemany
        rows = []
        for chunk in chunks:
            if not hasattr(chunk, "embedding") or chunk.embedding is None:
                logger.error(f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}")
                continue

            binary_embeddings = self._binary_quantize(chunk.embedding)

            # Handle content storage (external vs database)
            content_to_store = chunk.content

            if self.enable_external_storage and self.storage:
                # Try to store content externally
                storage_key = await self._store_content_externally(
                    chunk.content, chunk.document_id, chunk.chunk_number, str(chunk.metadata), app_id
                )

                if storage_key:
                    content_to_store = storage_key
                    logger.debug(f"Stored chunk {chunk.document_id}-{chunk.chunk_number} externally")
                else:
                    logger.warning(
                        f"Failed to store chunk {chunk.document_id}-{chunk.chunk_number} externally, using database"
                    )

            rows.append(
                (
                    chunk.document_id,
                    chunk.chunk_number,
                    content_to_store,
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
        app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using the max_sim function for multi-vectors."""
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

        # Convert to DocumentChunks with external storage support
        chunks = []
        for row in result:
            try:
                metadata = json.loads(row[4]) if row[4] else {}
            except Exception:
                metadata = {}

            content = row[3]

            # Handle external storage retrieval
            logger.debug(
                f"Checking content for chunk {row[1]}-{row[2]}: is_storage_key={self._is_storage_key(content)}, enable_external_storage={self.enable_external_storage}"
            )
            if self.enable_external_storage and self._is_storage_key(content):
                logger.info(f"Retrieving external content for chunk {row[1]}-{row[2]} from storage key: {content}")
                try:
                    original_content = content
                    content = await self._retrieve_content_from_storage(content, row[4])
                    if content == original_content:
                        logger.warning(f"Content retrieval failed, still showing storage key: {content}")
                    else:
                        logger.info(
                            f"Successfully retrieved content for chunk {row[1]}-{row[2]}, length: {len(content)}"
                        )
                except Exception as e:
                    logger.error(f"Failed to retrieve content from storage for chunk {row[1]}-{row[2]}: {e}")
                    # Keep storage key as content if retrieval fails

            chunk = DocumentChunk(
                document_id=row[1],
                chunk_number=row[2],
                content=content,
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
        app_id: Optional[str] = None,
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

        # Convert to DocumentChunks with external storage support
        chunks = []
        for row in result:
            try:
                metadata = json.loads(row[3]) if row[3] else {}
            except Exception:
                metadata = {}

            content = row[2]

            # Handle external storage retrieval
            logger.debug(
                f"Checking content for chunk {row[0]}-{row[1]}: is_storage_key={self._is_storage_key(content)}, enable_external_storage={self.enable_external_storage}"
            )
            if self.enable_external_storage and self._is_storage_key(content):
                logger.info(f"Retrieving external content for chunk {row[0]}-{row[1]} from storage key: {content}")
                try:
                    original_content = content
                    content = await self._retrieve_content_from_storage(content, row[3])
                    if content == original_content:
                        logger.warning(f"Content retrieval failed, still showing storage key: {content}")
                    else:
                        logger.info(
                            f"Successfully retrieved content for chunk {row[0]}-{row[1]}, length: {len(content)}"
                        )
                except Exception as e:
                    logger.error(f"Failed to retrieve content from storage for chunk {row[0]}-{row[1]}: {e}")
                    # Keep storage key as content if retrieval fails

            chunk = DocumentChunk(
                document_id=row[0],
                chunk_number=row[1],
                content=content,
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=0.0,  # No relevance score for direct retrieval
            )
            chunks.append(chunk)

        logger.debug(f"Found {len(chunks)} chunks in batch retrieval from multi-vector store")
        return chunks

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
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
            # Register vector extension for this connection
            register_vector(conn)

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
