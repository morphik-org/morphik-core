import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Index, String, desc, func, select, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config import get_settings
from core.models.workflows import Workflow, WorkflowRun

from ..models.auth import AuthContext
from ..models.documents import Document, StorageFileInfo
from ..models.folders import Folder
from ..models.graph import Graph
from ..models.model_config import ModelConfig
from .base_database import BaseDatabase

logger = logging.getLogger(__name__)
Base = declarative_base()

SYSTEM_METADATA_SCOPE_KEYS = {"folder_name", "end_user_id", "app_id"}


class DocumentModel(Base):
    """SQLAlchemy model for document metadata."""

    __tablename__ = "documents"

    external_id = Column(String, primary_key=True)
    content_type = Column(String)
    filename = Column(String, nullable=True)
    doc_metadata = Column(JSONB, default=dict)
    storage_info = Column(JSONB, default=dict)
    system_metadata = Column(JSONB, default=dict)
    additional_metadata = Column(JSONB, default=dict)
    chunk_ids = Column(JSONB, default=list)
    storage_files = Column(JSONB, default=list)

    # Flattened auth columns for performance
    owner_id = Column(String)
    owner_type = Column(String)
    readers = Column(ARRAY(String), default=list)
    writers = Column(ARRAY(String), default=list)
    admins = Column(ARRAY(String), default=list)
    app_id = Column(String)
    folder_name = Column(String)
    end_user_id = Column(String)

    # Create indexes
    __table_args__ = (
        Index("idx_system_metadata", "system_metadata", postgresql_using="gin"),
        Index("idx_doc_metadata_gin", "doc_metadata", postgresql_using="gin"),
        # Flattened column indexes are created directly on the columns
        Index("idx_doc_app_id", "app_id"),
        Index("idx_doc_folder_name", "folder_name"),
        Index("idx_doc_end_user_id", "end_user_id"),
        Index("idx_doc_owner_id", "owner_id"),
    )


class GraphModel(Base):
    """SQLAlchemy model for graph data."""

    __tablename__ = "graphs"

    id = Column(String, primary_key=True)
    name = Column(String)  # Not unique globally anymore
    entities = Column(JSONB, default=list)
    relationships = Column(JSONB, default=list)
    graph_metadata = Column(JSONB, default=dict)  # Renamed from 'metadata' to avoid conflict
    system_metadata = Column(JSONB, default=dict)  # For folder_name and end_user_id
    document_ids = Column(JSONB, default=list)
    filters = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=func.now())

    # Flattened auth columns for performance
    owner_id = Column(String)
    owner_type = Column(String)
    readers = Column(ARRAY(String), default=list)
    writers = Column(ARRAY(String), default=list)
    admins = Column(ARRAY(String), default=list)
    app_id = Column(String)
    folder_name = Column(String)
    end_user_id = Column(String)

    # Create indexes
    __table_args__ = (
        Index("idx_graph_name", "name"),
        Index("idx_graph_system_metadata", "system_metadata", postgresql_using="gin"),
        # Create a unique constraint on name scoped by owner_id (flattened column)
        Index("idx_graph_owner_name", "name", "owner_id", unique=True),
        # Indexes on flattened columns
        Index("idx_graph_app_id", "app_id"),
        Index("idx_graph_folder_name", "folder_name"),
        Index("idx_graph_end_user_id", "end_user_id"),
    )


class FolderModel(Base):
    """SQLAlchemy model for folder data."""

    __tablename__ = "folders"

    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String, nullable=True)
    document_ids = Column(JSONB, default=list)
    system_metadata = Column(JSONB, default=dict)
    rules = Column(JSONB, default=list)
    workflow_ids = Column(JSONB, default=list)

    # Flattened auth columns for performance
    owner_id = Column(String)
    owner_type = Column(String)
    readers = Column(ARRAY(String), default=list)
    writers = Column(ARRAY(String), default=list)
    admins = Column(ARRAY(String), default=list)
    app_id = Column(String)
    end_user_id = Column(String)

    # Create indexes
    __table_args__ = (
        Index("idx_folder_name", "name"),
        # Indexes on flattened columns
        Index("idx_folder_app_id", "app_id"),
        Index("idx_folder_owner_id", "owner_id"),
        Index("idx_folder_end_user_id", "end_user_id"),
    )


class ChatConversationModel(Base):
    """SQLAlchemy model for persisted chat history."""

    __tablename__ = "chat_conversations"

    conversation_id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=True)
    app_id = Column(String, index=True, nullable=True)
    title = Column(String, nullable=True)
    history = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=func.now())

    # Avoid duplicate indexes – SQLAlchemy already creates BTREE indexes for
    # columns declared with `index=True` and the primary-key column has an
    # implicit index.  Removing the explicit duplicates prevents bloat and
    # guarantees they won't be re-added after we dropped them in production.
    __table_args__ = ()


class ModelConfigModel(Base):
    """SQLAlchemy model for user model configurations."""

    __tablename__ = "model_configs"

    id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    app_id = Column(String, index=True, nullable=False)
    provider = Column(String, nullable=False)
    config_data = Column(JSONB, default=dict)
    created_at = Column(String)
    updated_at = Column(String)

    __table_args__ = (
        Index("idx_model_config_user_app", "user_id", "app_id"),
        Index("idx_model_config_provider", "provider"),
    )


# ---------------------------------------------------------------------------
# Workflow models (JSONB payload) – persistent storage for workflow definitions
#   and execution runs. Placed here so later PostgresDatabase code can import
#   them without forward-reference issues or linter complaints.
# ---------------------------------------------------------------------------


class WorkflowModel(Base):
    """SQLAlchemy model for workflow definitions stored as JSONB."""

    __tablename__ = "workflows"

    id = Column(String, primary_key=True)
    data = Column(JSONB, nullable=False)
    owner_id = Column(String, index=True)
    app_id = Column(String, index=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=func.now())

    __table_args__ = (
        Index("idx_workflows_owner_app", "owner_id", "app_id"),
        Index("idx_workflows_owner_user", "owner_id", "user_id"),
    )


class WorkflowRunModel(Base):
    """SQLAlchemy model for individual workflow execution runs."""

    __tablename__ = "workflow_runs"

    id = Column(String, primary_key=True)
    workflow_id = Column(String, index=True, nullable=False)
    data = Column(JSONB, nullable=False)
    owner_id = Column(String, index=True)
    app_id = Column(String, index=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=func.now())

    __table_args__ = (
        Index("idx_workflow_runs_wf", "workflow_id"),
        Index("idx_workflow_runs_owner_app", "owner_id", "app_id"),
        Index("idx_workflow_runs_owner_user", "owner_id", "user_id"),
    )


def _serialize_datetime(obj: Any) -> Any:
    """Helper function to serialize datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


def _parse_datetime_field(value: Any) -> Any:
    """Helper function to parse datetime fields from PostgreSQL."""
    if isinstance(value, str):
        try:
            # Handle PostgreSQL datetime strings that might be missing timezone colon
            # e.g., '2025-06-25 21:35:49.22022+00' -> '2025-06-25 21:35:49.22022+00:00'
            if value.endswith("+00") and not value.endswith("+00:00"):
                value = value[:-3] + "+00:00"
            elif value.endswith("-00") and not value.endswith("-00:00"):
                value = value[:-3] + "-00:00"
            return datetime.fromisoformat(value)
        except ValueError:
            # If parsing fails, return the original value
            return value
    return value


class PostgresDatabase(BaseDatabase):
    """PostgreSQL implementation for document metadata storage."""

    async def delete_folder(self, folder_id: str, auth: AuthContext) -> bool:
        """Delete a folder row if user has admin access."""
        try:
            # Fetch the folder to check permissions
            async with self.async_session() as session:
                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found")
                    return False
                if not self._check_folder_model_access(folder_model, auth):
                    logger.error(f"User does not have admin access to folder {folder_id}")
                    return False
                await session.delete(folder_model)
                await session.commit()
                logger.info(f"Deleted folder {folder_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting folder: {e}")
            return False

    def __init__(
        self,
        uri: str,
    ):
        """Initialize PostgreSQL connection for document storage."""
        # Load settings from config
        settings = get_settings()

        # Get database pool settings from config with defaults
        pool_size = getattr(settings, "DB_POOL_SIZE", 20)
        max_overflow = getattr(settings, "DB_MAX_OVERFLOW", 30)
        pool_recycle = getattr(settings, "DB_POOL_RECYCLE", 3600)
        pool_timeout = getattr(settings, "DB_POOL_TIMEOUT", 10)
        pool_pre_ping = getattr(settings, "DB_POOL_PRE_PING", True)

        logger.info(
            f"Initializing PostgreSQL connection pool with size={pool_size}, "
            f"max_overflow={max_overflow}, pool_recycle={pool_recycle}s"
        )

        # Create async engine with explicit pool settings
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
            connect_args={"server_settings": {"statement_timeout": "30000"}},  # 30 second timeout
        )
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self._initialized = False

    async def initialize(self):
        """Initialize database tables and indexes."""
        if self._initialized:
            return True

        try:
            logger.info("Initializing PostgreSQL database tables and indexes...")
            # Create ORM models
            async with self.engine.begin() as conn:
                # Explicitly create all tables with checkfirst=True to avoid errors if tables already exist
                await conn.run_sync(lambda conn: Base.metadata.create_all(conn, checkfirst=True))

                # No need to manually create graphs table again since SQLAlchemy does it
                logger.info("Created database tables successfully")

                # Create caches table if it doesn't exist (kept as direct SQL for backward compatibility)
                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS caches (
                        name TEXT PRIMARY KEY,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )

                # Check if storage_files column exists
                result = await conn.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'storage_files'
                    """
                    )
                )
                if not result.first():
                    # Add storage_files column to documents table
                    await conn.execute(
                        text(
                            """
                        ALTER TABLE documents
                        ADD COLUMN IF NOT EXISTS storage_files JSONB DEFAULT '[]'::jsonb
                        """
                        )
                    )
                    logger.info("Added storage_files column to documents table")

                # Create folders table if it doesn't exist
                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS folders (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        document_ids JSONB DEFAULT '[]',
                        system_metadata JSONB DEFAULT '{}'
                    );
                    """
                    )
                )

                # Add rules column to folders table if it doesn't exist
                result = await conn.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'folders' AND column_name = 'rules'
                    """
                    )
                )
                if not result.first():
                    # Add rules column to folders table
                    await conn.execute(
                        text(
                            """
                        ALTER TABLE folders
                        ADD COLUMN IF NOT EXISTS rules JSONB DEFAULT '[]'::jsonb
                        """
                        )
                    )
                    logger.info("Added rules column to folders table")

                # Add workflow_ids column to folders table if it doesn't exist
                result = await conn.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'folders' AND column_name = 'workflow_ids'
                    """
                    )
                )
                if not result.first():
                    # Add workflow_ids column to folders table
                    await conn.execute(
                        text(
                            """
                        ALTER TABLE folders
                        ADD COLUMN IF NOT EXISTS workflow_ids JSONB DEFAULT '[]'::jsonb
                        """
                        )
                    )
                    logger.info("Added workflow_ids column to folders table")

                # Create indexes for folders table
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_folder_name ON folders (name);"))

                # Check if system_metadata column exists in graphs table
                result = await conn.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'graphs' AND column_name = 'system_metadata'
                    """
                    )
                )
                if not result.first():
                    # Add system_metadata column to graphs table
                    await conn.execute(
                        text(
                            """
                        ALTER TABLE graphs
                        ADD COLUMN IF NOT EXISTS system_metadata JSONB DEFAULT '{}'::jsonb
                        """
                        )
                    )
                    logger.info("Added system_metadata column to graphs table")

                # Note: Indexes for folder_name, end_user_id, and app_id are created as direct column indexes
                # in the flattened columns section below, so we don't need JSONB path indexes

                # ORM models already include workflows tables – ensure they are created
                await conn.run_sync(lambda c: WorkflowModel.__table__.create(c, checkfirst=True))
                await conn.run_sync(lambda c: WorkflowRunModel.__table__.create(c, checkfirst=True))

                # Add scoping columns to workflows table if they don't exist
                for column_name in ["owner_id", "app_id", "user_id"]:
                    result = await conn.execute(
                        text(
                            f"""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = 'workflows' AND column_name = '{column_name}'
                            """
                        )
                    )
                    if not result.first():
                        await conn.execute(
                            text(
                                f"""
                                ALTER TABLE workflows
                                ADD COLUMN IF NOT EXISTS {column_name} VARCHAR
                                """
                            )
                        )
                        logger.info(f"Added {column_name} column to workflows table")

                # Add scoping columns to workflow_runs table if they don't exist
                for column_name in ["owner_id", "app_id", "user_id"]:
                    result = await conn.execute(
                        text(
                            f"""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = 'workflow_runs' AND column_name = '{column_name}'
                            """
                        )
                    )
                    if not result.first():
                        await conn.execute(
                            text(
                                f"""
                                ALTER TABLE workflow_runs
                                ADD COLUMN IF NOT EXISTS {column_name} VARCHAR
                                """
                            )
                        )
                        logger.info(f"Added {column_name} column to workflow_runs table")

                # Create indexes for workflows table
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_workflows_owner_id ON workflows (owner_id);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_workflows_app_id ON workflows (app_id);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON workflows (user_id);"))
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_workflows_owner_app ON workflows (owner_id, app_id);")
                )
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_workflows_owner_user ON workflows (owner_id, user_id);")
                )

                # Create indexes for workflow_runs table
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_workflow_runs_owner_id ON workflow_runs (owner_id);")
                )
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_workflow_runs_app_id ON workflow_runs (app_id);")
                )
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_workflow_runs_user_id ON workflow_runs (user_id);")
                )
                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_workflow_runs_owner_app ON workflow_runs (owner_id, app_id);")
                )
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_workflow_runs_owner_user ON workflow_runs (owner_id, user_id);"
                    )
                )

                # Check if title column exists in chat_conversations table
                result = await conn.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'chat_conversations' AND column_name = 'title'
                    """
                    )
                )
                if not result.first():
                    # Add title column to chat_conversations table
                    await conn.execute(
                        text(
                            """
                        ALTER TABLE chat_conversations
                        ADD COLUMN IF NOT EXISTS title VARCHAR
                        """
                        )
                    )
                    logger.info("Added title column to chat_conversations table")

                # ---------------------------------------------
                # Legacy ALTER-TABLE loop removed: SQLAlchemy now
                # creates flattened auth columns (owner_id,…, ACL
                # arrays) during metadata.create_all().  We keep
                # *tables_to_update* for the index creation block
                # that follows, but skip redundant DDL.
                # ---------------------------------------------

                tables_to_update = ["documents", "graphs", "folders"]

                logger.info("Creating optimised indexes for flattened columns …")

                for table in tables_to_update:
                    # Create indexes for scalar columns
                    await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_owner_id ON {table}(owner_id);"))
                    await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_app_id ON {table}(app_id);"))

                    # Create GIN indexes for array columns
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_readers ON {table} USING gin(readers);")
                    )
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_writers ON {table} USING gin(writers);")
                    )
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_admins ON {table} USING gin(admins);")
                    )

                    # Create composite indexes for common query patterns
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_owner_app ON {table}(owner_id, app_id);")
                    )
                    if table != "folders":  # folders don't have folder_name column
                        await conn.execute(
                            text(f"CREATE INDEX IF NOT EXISTS idx_{table}_app_folder ON {table}(app_id, folder_name);")
                        )
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_app_end_user ON {table}(app_id, end_user_id);")
                    )

                logger.info("Flattened auth columns and indexes created successfully")

            logger.info("PostgreSQL tables and indexes created successfully")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error creating PostgreSQL tables and indexes: {str(e)}")
            return False

    async def store_document(self, document: Document, auth: AuthContext) -> bool:
        """Store document metadata."""
        try:
            doc_dict = document.model_dump()

            # Rename metadata to doc_metadata
            if "metadata" in doc_dict:
                doc_dict["doc_metadata"] = doc_dict.pop("metadata")
            doc_dict["doc_metadata"]["external_id"] = doc_dict["external_id"]

            # Ensure system metadata
            if "system_metadata" not in doc_dict:
                doc_dict["system_metadata"] = {}
            doc_dict["system_metadata"]["created_at"] = datetime.now(UTC)
            doc_dict["system_metadata"]["updated_at"] = datetime.now(UTC)

            # Handle storage_files
            if "storage_files" in doc_dict and doc_dict["storage_files"]:
                # Convert storage_files to the expected format for storage
                doc_dict["storage_files"] = [file.model_dump() for file in doc_dict["storage_files"]]

            # Serialize datetime objects to ISO format strings
            doc_dict = _serialize_datetime(doc_dict)

            # Simplified access control - only what's actually needed
            doc_dict["owner_id"] = auth.entity_id or "system"
            doc_dict["owner_type"] = "developer"  # Always developer, no need to check
            doc_dict["app_id"] = auth.app_id  # Primary access control in cloud mode

            # ACL arrays are required by the schema but never used
            doc_dict["readers"] = []
            doc_dict["writers"] = []
            doc_dict["admins"] = []

            # The flattened fields are already in doc_dict from the Document model

            async with self.async_session() as session:
                doc_model = DocumentModel(**doc_dict)
                session.add(doc_model)
                await session.commit()
            return True

        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            return False

    async def get_document(self, document_id: str, auth: AuthContext) -> Optional[Document]:
        """Retrieve document metadata by ID if user has access."""
        try:
            async with self.async_session() as session:
                # Build access filter and params
                access_filter = self._build_access_filter_optimized(auth)
                filter_params = self._build_filter_params(auth)

                # Query document with parameterized query
                query = (
                    select(DocumentModel)
                    .where(DocumentModel.external_id == document_id)
                    .where(text(f"({access_filter})").bindparams(**filter_params))
                )

                result = await session.execute(query)
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    return Document(**self._document_model_to_dict(doc_model))
                return None

        except Exception as e:
            logger.error(f"Error retrieving document metadata: {str(e)}")
            return None

    async def get_document_by_filename(
        self, filename: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """Retrieve document metadata by filename if user has access.
        If multiple documents have the same filename, returns the most recently updated one.

        Args:
            filename: The filename to search for
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)
        """
        try:
            async with self.async_session() as session:
                # Build access filter and params
                access_filter = self._build_access_filter_optimized(auth)
                system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)
                filter_params = self._build_filter_params(auth, system_filters)
                filter_params["filename"] = filename  # Add filename as a parameter

                # Construct where clauses
                where_clauses = [
                    f"({access_filter})",
                    "filename = :filename",  # Use parameterized query
                ]

                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")

                final_where_clause = " AND ".join(where_clauses)

                # Query document with system filters using parameterized query
                query = (
                    select(DocumentModel).where(text(final_where_clause).bindparams(**filter_params))
                    # Order by updated_at in system_metadata to get the most recent document
                    .order_by(text("system_metadata->>'updated_at' DESC"))
                )

                logger.debug(f"Querying document by filename with system filters: {system_filters}")

                result = await session.execute(query)
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    return Document(**self._document_model_to_dict(doc_model))
                return None

        except Exception as e:
            logger.error(f"Error retrieving document metadata by filename: {str(e)}")
            return None

    async def get_documents_by_id(
        self,
        document_ids: List[str],
        auth: AuthContext,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve multiple documents by their IDs in a single batch operation.
        Only returns documents the user has access to.
        Can filter by system metadata fields like folder_name and end_user_id.

        Args:
            document_ids: List of document IDs to retrieve
            auth: Authentication context
            system_filters: Optional filters for system metadata fields

        Returns:
            List of Document objects that were found and user has access to
        """
        try:
            if not document_ids:
                return []

            async with self.async_session() as session:
                # Build access filter and params
                access_filter = self._build_access_filter_optimized(auth)
                system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)
                filter_params = self._build_filter_params(auth, system_filters)

                # Add document IDs as array parameter
                filter_params["document_ids"] = document_ids

                # Construct where clauses
                where_clauses = [f"({access_filter})", "external_id = ANY(:document_ids)"]

                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")

                final_where_clause = " AND ".join(where_clauses)

                # Query documents with document IDs, access check, and system filters in a single query
                query = select(DocumentModel).where(text(final_where_clause).bindparams(**filter_params))

                logger.info(f"Batch retrieving {len(document_ids)} documents with a single query")

                # Execute batch query
                result = await session.execute(query)
                doc_models = result.scalars().all()

                documents = []
                for doc_model in doc_models:
                    documents.append(Document(**self._document_model_to_dict(doc_model)))

                logger.info(f"Found {len(documents)} documents in batch retrieval")
                return documents

        except Exception as e:
            logger.error(f"Error batch retrieving documents: {str(e)}")
            return []

    async def get_documents(
        self,
        auth: AuthContext,
        skip: int = 0,
        limit: int = 10000,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """List documents the user has access to."""
        try:
            async with self.async_session() as session:
                # Build query
                access_filter = self._build_access_filter_optimized(auth)
                metadata_filter = self._build_metadata_filter(filters)
                system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)
                filter_params = self._build_filter_params(auth, system_filters)

                where_clauses = [f"({access_filter})"]

                if metadata_filter:
                    where_clauses.append(f"({metadata_filter})")

                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")

                final_where_clause = " AND ".join(where_clauses)
                query = select(DocumentModel).where(text(final_where_clause).bindparams(**filter_params))

                query = query.offset(skip).limit(limit)

                result = await session.execute(query)
                doc_models = result.scalars().all()

                return [Document(**self._document_model_to_dict(doc)) for doc in doc_models]

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    async def update_document(self, document_id: str, updates: Dict[str, Any], auth: AuthContext) -> bool:
        """Update document metadata if user has write access."""
        try:
            if not await self.check_access(document_id, auth, "write"):
                return False

            # Get existing document to preserve system_metadata
            existing_doc = await self.get_document(document_id, auth)
            if not existing_doc:
                return False

            # Update system metadata
            updates.setdefault("system_metadata", {})

            # Merge with existing system_metadata instead of just preserving specific fields
            if existing_doc.system_metadata:
                # Start with existing system_metadata
                merged_system_metadata = dict(existing_doc.system_metadata)
                # Update with new values
                merged_system_metadata.update(updates["system_metadata"])
                # Replace with merged result
                updates["system_metadata"] = merged_system_metadata
                logger.debug("Merged system_metadata during document update, preserving existing fields")

            # Remove scope fields that are now stored as dedicated columns
            if isinstance(updates.get("system_metadata"), dict):
                updates["system_metadata"] = {
                    key: value
                    for key, value in updates["system_metadata"].items()
                    if key not in SYSTEM_METADATA_SCOPE_KEYS
                }

            # Always update the updated_at timestamp
            updates["system_metadata"]["updated_at"] = datetime.now(UTC)

            # Serialize datetime objects to ISO format strings
            updates = _serialize_datetime(updates)

            async with self.async_session() as session:
                result = await session.execute(select(DocumentModel).where(DocumentModel.external_id == document_id))
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    # Log what we're updating
                    logger.info(f"Document update: updating fields {list(updates.keys())}")

                    # Special handling for metadata/doc_metadata conversion
                    if "metadata" in updates and "doc_metadata" not in updates:
                        logger.info("Converting 'metadata' to 'doc_metadata' for database update")
                        updates["doc_metadata"] = updates.pop("metadata")

                    # The flattened fields (owner_id, owner_type, readers, writers, admins)
                    # should be in updates directly if they need to be updated

                    # Set all attributes
                    for key, value in updates.items():
                        if key == "storage_files" and isinstance(value, list):
                            serialized_value = [
                                _serialize_datetime(
                                    item.model_dump()
                                    if hasattr(item, "model_dump")
                                    else (item.dict() if hasattr(item, "dict") else item)
                                )
                                for item in value
                            ]
                            logger.debug("Serializing storage_files before setting attribute")
                            setattr(doc_model, key, serialized_value)
                        else:
                            logger.debug(f"Setting document attribute {key} = {value}")
                            setattr(doc_model, key, value)

                    await session.commit()
                    logger.info(f"Document {document_id} updated successfully")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error updating document metadata: {str(e)}")
            return False

    async def delete_document(self, document_id: str, auth: AuthContext) -> bool:
        """Delete document if user has write access."""
        try:
            if not await self.check_access(document_id, auth, "write"):
                return False

            async with self.async_session() as session:
                result = await session.execute(select(DocumentModel).where(DocumentModel.external_id == document_id))
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    await session.delete(doc_model)
                    await session.commit()

                    # --------------------------------------------------------------------------------
                    # Maintain referential integrity: remove the deleted document ID from any folders
                    # that still list it in their document_ids JSONB array.  This prevents the UI from
                    # requesting stale IDs after a delete.
                    # --------------------------------------------------------------------------------
                    try:
                        await session.execute(
                            text(
                                """
                                UPDATE folders
                                SET document_ids = document_ids - :doc_id
                                WHERE document_ids ? :doc_id
                                """
                            ),
                            {"doc_id": document_id},
                        )
                        await session.commit()
                    except Exception as upd_err:  # noqa: BLE001
                        # Non-fatal – log but keep the document deleted so user doesn't see it any more.
                        logger.error("Failed to remove deleted document %s from folders: %s", document_id, upd_err)

                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    async def find_authorized_and_filtered_documents(
        self,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Find document IDs matching filters and access permissions."""
        try:
            async with self.async_session() as session:
                # Build query
                access_filter = self._build_access_filter_optimized(auth)
                metadata_filter = self._build_metadata_filter(filters)
                system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)
                filter_params = self._build_filter_params(auth, system_filters)

                logger.debug(f"Access filter: {access_filter}")
                logger.debug(f"Metadata filter: {metadata_filter}")
                logger.debug(f"System metadata filter: {system_metadata_filter}")
                logger.debug(f"Original filters: {filters}")
                logger.debug(f"System filters: {system_filters}")

                where_clauses = [f"({access_filter})"]

                if metadata_filter:
                    where_clauses.append(f"({metadata_filter})")

                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")

                final_where_clause = " AND ".join(where_clauses)
                query = select(DocumentModel.external_id).where(text(final_where_clause).bindparams(**filter_params))

                logger.debug(f"Final query: {query}")

                result = await session.execute(query)
                doc_ids = [row[0] for row in result.all()]
                logger.debug(f"Found document IDs: {doc_ids}")
                return doc_ids

        except Exception as e:
            logger.error(f"Error finding authorized documents: {str(e)}")
            return []

    async def check_access(self, document_id: str, auth: AuthContext, required_permission: str = "read") -> bool:
        """Check if user has required permission for document."""
        try:
            async with self.async_session() as session:
                result = await session.execute(select(DocumentModel).where(DocumentModel.external_id == document_id))
                doc_model = result.scalar_one_or_none()

                if not doc_model:
                    return False

                # Simplified access check:
                # If app_id is present, check app_id match
                if auth.app_id:
                    return doc_model.app_id == auth.app_id

                # Otherwise check owner_id match
                return doc_model.owner_id == auth.entity_id

        except Exception as e:
            logger.error(f"Error checking document access: {str(e)}")
            return False

    def _build_access_filter_optimized(self, auth: AuthContext) -> str:
        """Build PostgreSQL filter for access control using flattened columns.

        Simplified strategy:
        - If app_id exists (cloud mode): Filter by app_id only
        - If no app_id (dev/self-hosted): Filter by owner_id

        Note: This returns a SQL string with named parameters.
        The caller must provide these parameters when executing the query.
        """
        # Primary access control: app_id based (for cloud mode with proper tokens)
        if auth.app_id:
            # When app_id is present, that's the primary access control
            # This is the case for all cloud mode operations with proper tokens
            return "app_id = :app_id"

        # Fallback for dev mode or self-hosted without app_id
        # Filter by owner_id to maintain backwards compatibility
        return "owner_id = :entity_id"

    def _build_metadata_filter(self, filters: Dict[str, Any]) -> str:
        """Build PostgreSQL filter for metadata."""
        if not filters:
            return ""

        filter_conditions = []
        for key, value in filters.items():
            # Handle list of values (IN operator)
            if isinstance(value, list):
                if not value:  # Skip empty lists
                    continue

                # New approach for lists: OR together multiple @> conditions
                # This allows each item in the list to be checked for containment.
                or_clauses_for_list = []
                for item_in_list in value:
                    json_filter_object = {key: item_in_list}
                    json_string_for_sql = json.dumps(json_filter_object)
                    sql_escaped_json_string = json_string_for_sql.replace("'", "''")
                    or_clauses_for_list.append(f"doc_metadata @> '{sql_escaped_json_string}'::jsonb")
                if or_clauses_for_list:
                    filter_conditions.append(f"({' OR '.join(or_clauses_for_list)})")

            else:
                # Handle single value (equality)
                # New approach for single value: Use JSONB containment operator @>
                json_filter_object = {key: value}
                json_string_for_sql = json.dumps(json_filter_object)
                sql_escaped_json_string = json_string_for_sql.replace("'", "''")
                filter_conditions.append(f"doc_metadata @> '{sql_escaped_json_string}'::jsonb")

        return " AND ".join(filter_conditions)

    def _build_system_metadata_filter_optimized(self, system_filters: Optional[Dict[str, Any]]) -> str:
        """Build PostgreSQL filter for system metadata using flattened columns.

        This optimized version uses direct column access instead of JSONB operations
        for better performance.

        Note: This returns a SQL string with named parameters like :app_id_0, :folder_name_0, etc.
        The caller must provide these parameters when executing the query.
        """
        if not system_filters:
            return ""

        key_clauses: List[str] = []
        self._filter_param_counter = 0  # Reset counter for parameter naming

        # Map system metadata keys to flattened columns
        column_map = {"app_id": "app_id", "folder_name": "folder_name", "end_user_id": "end_user_id"}

        for key, value in system_filters.items():
            if key not in column_map:
                continue

            column = column_map[key]
            values = value if isinstance(value, list) else [value]
            if not values and value is not None:
                continue

            value_clauses = []
            for item in values:
                if item is None:
                    value_clauses.append(f"{column} IS NULL")
                else:
                    # Use named parameter instead of string interpolation
                    param_name = f"{key}_{self._filter_param_counter}"
                    value_clauses.append(f"{column} = :{param_name}")
                    self._filter_param_counter += 1

            # OR all alternative values for this key
            if value_clauses:
                key_clauses.append("(" + " OR ".join(value_clauses) + ")")

        return " AND ".join(key_clauses)

    def _build_filter_params(
        self, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build parameter dictionary for the optimized filter methods.

        Returns:
            Dictionary with parameter values for SQL query execution
        """
        params = {}

        # Add auth parameters based on what's actually needed
        if auth.app_id:
            params["app_id"] = auth.app_id
        elif auth.entity_id:
            params["entity_id"] = auth.entity_id

        # Add system metadata filter parameters
        if system_filters:
            self._filter_param_counter = 0  # Reset counter
            column_map = {"app_id": "app_id", "folder_name": "folder_name", "end_user_id": "end_user_id"}

            for key, value in system_filters.items():
                if key not in column_map:
                    continue

                values = value if isinstance(value, list) else [value]
                if not values and value is not None:
                    continue

                for item in values:
                    if item is not None:
                        param_name = f"{key}_{self._filter_param_counter}"
                        params[param_name] = str(item)
                        self._filter_param_counter += 1

        return params

    def _graph_model_to_dict(self, graph_model) -> Dict[str, Any]:
        """Convert GraphModel to dictionary.

        Args:
            graph_model: GraphModel instance

        Returns:
            Dictionary ready to be passed to Graph constructor
        """
        return {
            "id": graph_model.id,
            "name": graph_model.name,
            "entities": graph_model.entities,
            "relationships": graph_model.relationships,
            "metadata": graph_model.graph_metadata,
            "system_metadata": graph_model.system_metadata or {},
            "document_ids": graph_model.document_ids,
            "filters": graph_model.filters,
            # Include flattened fields
            "folder_name": graph_model.folder_name,
            "app_id": graph_model.app_id,
            "end_user_id": graph_model.end_user_id,
        }

    def _document_model_to_dict(self, doc_model) -> Dict[str, Any]:
        """Convert DocumentModel to dictionary.

        Args:
            doc_model: DocumentModel instance

        Returns:
            Dictionary ready to be passed to Document constructor
        """
        # Convert storage_files from dict to StorageFileInfo
        storage_files = []
        if doc_model.storage_files:
            for file_info in doc_model.storage_files:
                if isinstance(file_info, dict):
                    storage_files.append(StorageFileInfo(**file_info))
                else:
                    storage_files.append(file_info)

        return {
            "external_id": doc_model.external_id,
            "content_type": doc_model.content_type,
            "filename": doc_model.filename,
            "metadata": doc_model.doc_metadata,
            "storage_info": doc_model.storage_info,
            "system_metadata": doc_model.system_metadata,
            "additional_metadata": doc_model.additional_metadata,
            "chunk_ids": doc_model.chunk_ids,
            "storage_files": storage_files,
            # Include flattened fields
            "folder_name": doc_model.folder_name,
            "app_id": doc_model.app_id,
            "end_user_id": doc_model.end_user_id,
        }

    async def store_cache_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Store metadata for a cache in PostgreSQL.

        Args:
            name: Name of the cache
            metadata: Cache metadata including model info and storage location

        Returns:
            bool: Whether the operation was successful
        """
        try:
            async with self.async_session() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO caches (name, metadata, updated_at)
                        VALUES (:name, :metadata, CURRENT_TIMESTAMP)
                        ON CONFLICT (name)
                        DO UPDATE SET
                            metadata = :metadata,
                            updated_at = CURRENT_TIMESTAMP
                        """
                    ),
                    {"name": name, "metadata": json.dumps(metadata)},
                )
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store cache metadata: {e}")
            return False

    async def get_cache_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cache from PostgreSQL.

        Args:
            name: Name of the cache

        Returns:
            Optional[Dict[str, Any]]: Cache metadata if found, None otherwise
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(text("SELECT metadata FROM caches WHERE name = :name"), {"name": name})
                row = result.first()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get cache metadata: {e}")
            return None

    async def store_graph(self, graph: Graph, auth: AuthContext) -> bool:
        """Store a graph in PostgreSQL.

        This method stores the graph metadata, entities, and relationships
        in a PostgreSQL table.

        Args:
            graph: Graph to store
            auth: Authentication context to set owner information

        Returns:
            bool: Whether the operation was successful
        """
        try:
            # First serialize the graph model to dict
            graph_dict = graph.model_dump()

            # Change 'metadata' to 'graph_metadata' to match our model
            if "metadata" in graph_dict:
                graph_dict["graph_metadata"] = graph_dict.pop("metadata")

            # Serialize datetime objects to ISO format strings, but preserve actual datetime objects
            # for created_at and updated_at fields that SQLAlchemy expects as datetime instances
            created_at = graph_dict.get("created_at")
            updated_at = graph_dict.get("updated_at")

            graph_dict = _serialize_datetime(graph_dict)

            # Restore datetime objects for SQLAlchemy columns
            if created_at:
                graph_dict["created_at"] = created_at
            if updated_at:
                graph_dict["updated_at"] = updated_at

            # Simplified access control - only what's actually needed
            graph_dict["owner_id"] = auth.entity_id or "system"
            graph_dict["owner_type"] = "developer"  # Always developer, no need to check
            graph_dict["app_id"] = auth.app_id  # Primary access control in cloud mode

            # ACL arrays are required by the schema but never used
            graph_dict["readers"] = []
            graph_dict["writers"] = []
            graph_dict["admins"] = []

            # The flattened fields are already in graph_dict from the Graph model

            # Store the graph metadata in PostgreSQL
            async with self.async_session() as session:
                # Store graph metadata in our table
                graph_model = GraphModel(**graph_dict)
                session.add(graph_model)
                await session.commit()
                logger.info(
                    f"Stored graph '{graph.name}' with {len(graph.entities)} entities "
                    f"and {len(graph.relationships)} relationships"
                )

            return True

        except Exception as e:
            logger.error(f"Error storing graph: {str(e)}")
            return False

    async def get_graph(
        self, name: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Graph]:
        """Get a graph by name.

        Args:
            name: Name of the graph
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            Optional[Graph]: Graph if found and accessible, None otherwise
        """
        try:
            async with self.async_session() as session:
                # Build access filter and params
                access_filter = self._build_access_filter_optimized(auth)
                filter_params = self._build_filter_params(auth)
                filter_params["graph_name"] = name

                # We need to check if the documents in the graph match the system filters
                # First get the graph without system filters
                query = select(GraphModel).where(
                    text(f"name = :graph_name AND ({access_filter})").bindparams(**filter_params)
                )

                result = await session.execute(query)
                graph_model = result.scalar_one_or_none()

                if graph_model:
                    # If system filters are provided, we need to filter the document_ids
                    document_ids = graph_model.document_ids

                    if system_filters and document_ids:
                        # Apply system_filters to document_ids
                        system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)

                        if system_metadata_filter:
                            # Get document IDs with system filters
                            system_params = self._build_filter_params(auth, system_filters)
                            system_params["doc_ids"] = document_ids
                            filter_query = f"""
                                SELECT external_id FROM documents
                                WHERE external_id = ANY(:doc_ids)
                                AND ({system_metadata_filter})
                            """

                            filter_result = await session.execute(text(filter_query).bindparams(**system_params))
                            filtered_doc_ids = [row[0] for row in filter_result.all()]

                            # If no documents match system filters, return None
                            if not filtered_doc_ids:
                                return None

                            # Update document_ids with filtered results
                            document_ids = filtered_doc_ids

                    # Convert to Graph model
                    graph_dict = self._graph_model_to_dict(graph_model)
                    # Override document_ids with filtered results if applicable
                    graph_dict["document_ids"] = document_ids
                    # Add datetime fields
                    graph_dict["created_at"] = _parse_datetime_field(graph_model.created_at)
                    graph_dict["updated_at"] = _parse_datetime_field(graph_model.updated_at)
                    return Graph(**graph_dict)

                return None

        except Exception as e:
            logger.error(f"Error retrieving graph: {str(e)}")
            return None

    async def list_graphs(self, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> List[Graph]:
        """List all graphs the user has access to.

        Args:
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            List[Graph]: List of graphs
        """
        try:
            async with self.async_session() as session:
                # Build access filter and params
                access_filter = self._build_access_filter_optimized(auth)
                filter_params = self._build_filter_params(auth)

                # Query graphs
                query = select(GraphModel).where(text(f"({access_filter})").bindparams(**filter_params))

                result = await session.execute(query)
                graph_models = result.scalars().all()

                graphs = []

                # If system filters are provided, we need to filter each graph's document_ids
                if system_filters:
                    system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)

                    for graph_model in graph_models:
                        document_ids = graph_model.document_ids

                        if document_ids and system_metadata_filter:
                            # Get document IDs with system filters
                            system_params = self._build_filter_params(auth, system_filters)
                            system_params["doc_ids"] = document_ids
                            filter_query = f"""
                                SELECT external_id FROM documents
                                WHERE external_id = ANY(:doc_ids)
                                AND ({system_metadata_filter})
                            """

                            filter_result = await session.execute(text(filter_query).bindparams(**system_params))
                            filtered_doc_ids = [row[0] for row in filter_result.all()]

                            # Only include graphs that have documents matching the system filters
                            if filtered_doc_ids:
                                graph_dict = self._graph_model_to_dict(graph_model)
                                # Override document_ids with filtered results
                                graph_dict["document_ids"] = filtered_doc_ids
                                # Add datetime fields
                                graph_dict["created_at"] = _parse_datetime_field(graph_model.created_at)
                                graph_dict["updated_at"] = _parse_datetime_field(graph_model.updated_at)
                                graphs.append(Graph(**graph_dict))
                else:
                    # No system filters, include all graphs
                    graphs = []
                    for graph_model in graph_models:
                        graph_dict = self._graph_model_to_dict(graph_model)
                        # Add datetime fields
                        graph_dict["created_at"] = _parse_datetime_field(graph_model.created_at)
                        graph_dict["updated_at"] = _parse_datetime_field(graph_model.updated_at)
                        graphs.append(Graph(**graph_dict))

                return graphs

        except Exception as e:
            logger.error(f"Error listing graphs: {str(e)}")
            return []

    async def update_graph(self, graph: Graph) -> bool:
        """Update an existing graph in PostgreSQL.

        This method updates the graph metadata, entities, and relationships
        in the PostgreSQL table.

        Args:
            graph: Graph to update

        Returns:
            bool: Whether the operation was successful
        """
        try:
            # First serialize the graph model to dict
            graph_dict = graph.model_dump()

            # Change 'metadata' to 'graph_metadata' to match our model
            if "metadata" in graph_dict:
                graph_dict["graph_metadata"] = graph_dict.pop("metadata")

            # Serialize datetime objects to ISO format strings, but preserve actual datetime objects
            # for created_at and updated_at fields that SQLAlchemy expects as datetime instances
            created_at = graph_dict.get("created_at")
            updated_at = graph_dict.get("updated_at")

            graph_dict = _serialize_datetime(graph_dict)

            # Restore datetime objects for SQLAlchemy columns
            if created_at:
                graph_dict["created_at"] = created_at
            if updated_at:
                graph_dict["updated_at"] = updated_at

            # The flattened fields are already in graph_dict from the Graph model
            # Note: owner_id, owner_type, and ACL fields should not be updated here
            # They should remain as set during graph creation

            # Update the graph in PostgreSQL
            async with self.async_session() as session:
                # Check if the graph exists
                result = await session.execute(select(GraphModel).where(GraphModel.id == graph.id))
                graph_model = result.scalar_one_or_none()

                if not graph_model:
                    logger.error(f"Graph '{graph.name}' with ID {graph.id} not found for update")
                    return False

                # Update the graph model with new values
                for key, value in graph_dict.items():
                    setattr(graph_model, key, value)

                await session.commit()
                logger.info(
                    f"Updated graph '{graph.name}' with {len(graph.entities)} entities "
                    f"and {len(graph.relationships)} relationships"
                )

            return True

        except Exception as e:
            logger.error(f"Error updating graph: {str(e)}")
            return False

    async def delete_graph(self, name: str, auth: AuthContext) -> bool:
        """Delete a graph by name.

        This method checks if the user has write access to the graph before deleting it.

        Args:
            name: Name of the graph to delete
            auth: Authentication context

        Returns:
            bool: Whether the operation was successful
        """
        try:
            async with self.async_session() as session:
                # First find the graph
                access_filter = self._build_access_filter_optimized(auth)
                filter_params = self._build_filter_params(auth)

                # Query to find the graph
                query = (
                    select(GraphModel)
                    .where(GraphModel.name == name)
                    .where(text(f"({access_filter})").bindparams(**filter_params))
                )

                result = await session.execute(query)
                graph_model = result.scalar_one_or_none()

                if not graph_model:
                    logger.error(f"Graph '{name}' not found")
                    return False

                # Simplified access check for deletion
                # If app_id is present, check app_id match
                if auth.app_id:
                    if graph_model.app_id != auth.app_id:
                        logger.error(f"User lacks write access to delete graph '{name}'")
                        return False
                else:
                    # Otherwise check owner_id match
                    if graph_model.owner_id != auth.entity_id:
                        logger.error(f"User lacks write access to delete graph '{name}'")
                        return False

                # Delete the graph
                await session.delete(graph_model)
                await session.commit()
                logger.info(f"Successfully deleted graph '{name}'")

            return True

        except Exception as e:
            logger.error(f"Error deleting graph: {str(e)}")
            return False

    async def create_folder(self, folder: Folder, auth: AuthContext) -> bool:
        """Create a new folder."""
        try:
            async with self.async_session() as session:
                folder_dict = folder.model_dump()

                # Convert datetime objects to strings for JSON serialization
                folder_dict = _serialize_datetime(folder_dict)

                # Simplified owner info
                owner_id = auth.entity_id or "system"
                owner_type = "developer"  # Always developer
                app_id_val = auth.app_id or folder_dict.get("app_id")

                # Check for existing folder with same name
                params = {"name": folder.name, "owner_id": owner_id}
                sql = """
                    SELECT id FROM folders
                    WHERE name = :name
                    AND owner_id = :owner_id
                    """
                if app_id_val:
                    sql += " AND app_id = :app_id"
                    params["app_id"] = app_id_val
                stmt = text(sql).bindparams(**params)

                result = await session.execute(stmt)
                existing_folder = result.scalar_one_or_none()

                if existing_folder:
                    logger.info(
                        f"Folder '{folder.name}' already exists with ID {existing_folder}, not creating a duplicate"
                    )
                    # Update the provided folder's ID to match the existing one
                    # so the caller gets the correct ID
                    folder.id = existing_folder
                    return True

                # Create a new folder model
                folder_model = FolderModel(
                    id=folder.id,
                    name=folder.name,
                    description=folder.description,
                    owner_id=owner_id,
                    owner_type=owner_type,
                    document_ids=folder_dict.get("document_ids", []),
                    system_metadata=folder_dict.get("system_metadata", {}),
                    readers=[],  # Required by schema but never used
                    writers=[],  # Required by schema but never used
                    admins=[],  # Required by schema but never used
                    app_id=app_id_val,
                    end_user_id=folder_dict.get("end_user_id"),
                    rules=folder_dict.get("rules", []),
                    workflow_ids=folder_dict.get("workflow_ids", []),
                )

                session.add(folder_model)
                await session.commit()

                logger.info(f"Created new folder '{folder.name}' with ID {folder.id}")
                return True

        except Exception as e:
            logger.error(f"Error creating folder: {e}")
            return False

    async def get_folder(self, folder_id: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by ID."""
        try:
            async with self.async_session() as session:
                # Get the folder
                logger.info(f"Getting folder with ID: {folder_id}")
                result = await session.execute(select(FolderModel).where(FolderModel.id == folder_id))
                folder_model = result.scalar_one_or_none()

                if not folder_model:
                    logger.error(f"Folder with ID {folder_id} not found in database")
                    return None

                # Convert to Folder object
                folder_dict = {
                    "id": folder_model.id,
                    "name": folder_model.name,
                    "description": folder_model.description,
                    "document_ids": folder_model.document_ids,
                    "system_metadata": folder_model.system_metadata,
                    "rules": folder_model.rules,
                    "workflow_ids": getattr(folder_model, "workflow_ids", []),
                }

                # Check if the user has access to the folder using the model
                if not self._check_folder_model_access(folder_model, auth):
                    return None

                folder = Folder(**folder_dict)
                return folder

        except Exception as e:
            logger.error(f"Error getting folder: {e}")
            return None

    async def get_folder_by_name(self, name: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by name."""
        try:
            async with self.async_session() as session:
                # Build query based on auth context
                params = {"name": name}

                if auth.app_id:
                    # Filter by app_id in cloud mode
                    stmt = text(
                        """
                        SELECT * FROM folders
                        WHERE name = :name AND app_id = :app_id
                    """
                    )
                    params["app_id"] = auth.app_id
                elif auth.entity_id:
                    # Filter by owner_id in dev/self-hosted mode
                    stmt = text(
                        """
                        SELECT * FROM folders
                        WHERE name = :name AND owner_id = :owner_id
                    """
                    )
                    params["owner_id"] = auth.entity_id
                else:
                    # No access without auth
                    return None

                result = await session.execute(stmt.bindparams(**params))
                folder_row = result.fetchone()

                if folder_row:
                    # Convert to Folder object
                    folder_dict = {
                        "id": folder_row.id,
                        "name": folder_row.name,
                        "description": folder_row.description,
                        "document_ids": folder_row.document_ids,
                        "system_metadata": folder_row.system_metadata,
                        "rules": folder_row.rules,
                        "workflow_ids": getattr(folder_row, "workflow_ids", []),
                    }
                    return Folder(**folder_dict)

                return None

        except Exception as e:
            logger.error(f"Error getting folder by name: {e}")
            return None

    async def list_folders(self, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> List[Folder]:
        """List all folders the user has access to using flattened columns."""
        try:
            current_params = {}

            # Simplified access control - same as documents/graphs
            if auth.app_id:
                # Filter by app_id when present (cloud mode)
                where_clause = text("app_id = :app_id_val")
                current_params["app_id_val"] = auth.app_id
            elif auth.entity_id:
                # Filter by owner_id as fallback (dev/self-hosted mode)
                where_clause = text("owner_id = :owner_id_val")
                current_params["owner_id_val"] = auth.entity_id
            else:
                # No access if no auth context
                where_clause = text("1=0")

            # Build and execute query
            async with self.async_session() as session:
                query = select(FolderModel).where(where_clause)
                result = await session.execute(query, current_params)
                folder_models = result.scalars().all()

                folders = []
                for folder_model in folder_models:
                    folder_dict = {
                        "id": folder_model.id,
                        "name": folder_model.name,
                        "description": folder_model.description,
                        "document_ids": folder_model.document_ids,
                        "system_metadata": folder_model.system_metadata,
                        "rules": folder_model.rules,
                        "workflow_ids": getattr(folder_model, "workflow_ids", []),
                    }
                    folders.append(Folder(**folder_dict))
                return folders

        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []

    async def add_document_to_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Add a document to a folder."""
        import asyncio

        try:
            # First, get the folder model and check access
            async with self.async_session() as session:
                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found")
                    return False

                # Check if user has write access to the folder
                if not self._check_folder_model_access(folder_model, auth):
                    logger.error(f"User does not have write access to folder {folder_id}")
                    return False

                # Convert to Folder object for document_ids access
                folder = await self.get_folder(folder_id, auth)
                if not folder:
                    return False

            # Check if the document exists and user has access with retry logic
            # This handles race conditions during ingestion where document might not be
            # immediately visible due to transaction isolation
            max_retries = 3
            retry_delay = 0.5  # Start with 500ms delay

            for attempt in range(max_retries):
                document = await self.get_document(document_id, auth)
                if document:
                    break

                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    logger.info(
                        f"Document {document_id} not found on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logger.error(
                        f"Document {document_id} not found or user does not have access after {max_retries} attempts"
                    )
                    return False

            # Final verification after retries
            if not document:
                logger.error(f"Document {document_id} not found or user does not have access after retries")
                return False

            # Check if the document is already in the folder
            if document_id in folder.document_ids:
                logger.info(f"Document {document_id} is already in folder {folder_id}")
                return True

            # Add the document to the folder
            async with self.async_session() as session:
                # Add document_id to document_ids array
                new_document_ids = folder.document_ids + [document_id]

                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found in database")
                    return False

                folder_model.document_ids = new_document_ids

                # Also update the document's folder_name flattened column
                stmt = text(
                    """
                    UPDATE documents
                    SET folder_name = :folder_name
                    WHERE external_id = :document_id
                    """
                ).bindparams(folder_name=folder.name, document_id=document_id)

                await session.execute(stmt)
                await session.commit()

                logger.info(f"Added document {document_id} to folder {folder_id}")
                return True

        except Exception as e:
            logger.error(f"Error adding document to folder: {e}")
            return False

    async def remove_document_from_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Remove a document from a folder."""
        try:
            # First, get the folder model and check access
            async with self.async_session() as session:
                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found")
                    return False

                # Check if user has write access to the folder
                if not self._check_folder_model_access(folder_model, auth):
                    logger.error(f"User does not have write access to folder {folder_id}")
                    return False

                # Convert to Folder object for document_ids access
                folder = await self.get_folder(folder_id, auth)
                if not folder:
                    return False

            # Check if the document is in the folder
            if document_id not in folder.document_ids:
                logger.warning(f"Tried to delete document {document_id} not in folder {folder_id}")
                return True

            # Remove the document from the folder
            async with self.async_session() as session:
                # Remove document_id from document_ids array
                new_document_ids = [doc_id for doc_id in folder.document_ids if doc_id != document_id]

                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found in database")
                    return False

                folder_model.document_ids = new_document_ids

                # Also update the document's folder_name to NULL
                stmt = text(
                    """
                    UPDATE documents
                    SET folder_name = NULL
                    WHERE external_id = :document_id
                    """
                ).bindparams(document_id=document_id)

                await session.execute(stmt)
                await session.commit()

                logger.info(f"Removed document {document_id} from folder {folder_id}")
                return True

        except Exception as e:
            logger.error(f"Error removing document from folder: {e}")
            return False

    async def associate_workflow_to_folder(self, folder_id: str, workflow_id: str, auth: AuthContext) -> bool:
        """Associate a workflow with a folder."""
        try:
            # First, get the folder model and check write access
            async with self.async_session() as session:
                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found")
                    return False

                # Check if user has write access to the folder
                if not self._check_folder_model_access(folder_model, auth):
                    logger.error(f"User does not have write access to folder {folder_id}")
                    return False

                # Get current workflow_ids
                workflow_ids = folder_model.workflow_ids or []

                # Check if workflow is already associated
                if workflow_id in workflow_ids:
                    logger.info(f"Workflow {workflow_id} is already associated with folder {folder_id}")
                    return True

                # Add the workflow to the folder
                workflow_ids.append(workflow_id)
                folder_model.workflow_ids = workflow_ids

                await session.commit()
                logger.info(f"Associated workflow {workflow_id} with folder {folder_id}")
                return True

        except Exception as e:
            logger.error(f"Error associating workflow with folder: {e}")
            return False

    async def disassociate_workflow_from_folder(self, folder_id: str, workflow_id: str, auth: AuthContext) -> bool:
        """Remove a workflow from a folder."""
        try:
            # First, get the folder model and check write access
            async with self.async_session() as session:
                folder_model = await session.get(FolderModel, folder_id)
                if not folder_model:
                    logger.error(f"Folder {folder_id} not found")
                    return False

                # Check if user has write access to the folder
                if not self._check_folder_model_access(folder_model, auth):
                    logger.error(f"User does not have write access to folder {folder_id}")
                    return False

                # Get current workflow_ids
                workflow_ids = folder_model.workflow_ids or []

                # Check if workflow is associated
                if workflow_id not in workflow_ids:
                    logger.info(f"Workflow {workflow_id} is not associated with folder {folder_id}")
                    return True

                # Remove the workflow from the folder
                workflow_ids.remove(workflow_id)
                folder_model.workflow_ids = workflow_ids

                await session.commit()
                logger.info(f"Disassociated workflow {workflow_id} from folder {folder_id}")
                return True

        except Exception as e:
            logger.error(f"Error disassociating workflow from folder: {e}")
            return False

    async def get_chat_history(
        self, conversation_id: str, user_id: Optional[str], app_id: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Return stored chat history for *conversation_id*."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ChatConversationModel).where(ChatConversationModel.conversation_id == conversation_id)
                )
                convo = result.scalar_one_or_none()
                if not convo:
                    return None
                if user_id and convo.user_id and convo.user_id != user_id:
                    return None
                if app_id and convo.app_id and convo.app_id != app_id:
                    return None
                return convo.history
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return None

    async def upsert_chat_history(
        self,
        conversation_id: str,
        user_id: Optional[str],
        app_id: Optional[str],
        history: List[Dict[str, Any]],
        title: Optional[str] = None,
    ) -> bool:
        """Store or update chat history."""
        try:
            now = datetime.now(UTC).isoformat()

            # Auto-generate title from first user message if not provided
            if title is None and history:
                # Find first user message
                for msg in history:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        # Extract first 50 chars as title
                        title = content[:50].strip()
                        if len(content) > 50:
                            title += "..."
                        break

            async with self.async_session() as session:
                # Check if conversation exists to determine if we need to preserve existing title
                result = await session.execute(
                    text("SELECT title FROM chat_conversations WHERE conversation_id = :cid"), {"cid": conversation_id}
                )
                existing = result.fetchone()

                # If conversation exists and has a title, preserve it unless a new title is provided
                if existing and existing[0] and title is None:
                    title = existing[0]

                await session.execute(
                    text(
                        """
                        INSERT INTO chat_conversations (conversation_id, user_id, app_id, history, title, created_at, updated_at)
                        VALUES (:cid, :uid, :aid, :hist, :title, CAST(:now AS TEXT), CAST(:now AS TEXT))
                        ON CONFLICT (conversation_id)
                        DO UPDATE SET
                            user_id = EXCLUDED.user_id,
                            app_id = EXCLUDED.app_id,
                            history = EXCLUDED.history,
                            title = COALESCE(EXCLUDED.title, chat_conversations.title),
                            updated_at = CAST(:now AS TEXT)
                        """
                    ),
                    {
                        "cid": conversation_id,
                        "uid": user_id,
                        "aid": app_id,
                        "hist": json.dumps(history),
                        "title": title,
                        "now": now,
                    },
                )
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error upserting chat history: {e}")
            return False

    async def list_chat_conversations(
        self,
        user_id: Optional[str],
        app_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return chat conversations for a given user (and optional app) ordered by last update.

        Args:
            user_id: ID of the user that owns the conversation (required for cloud-mode privacy).
            app_id: Optional application scope for developer tokens.
            limit: Maximum number of conversations to return.

        Returns:
            A list of dictionaries containing conversation_id, updated_at and a preview of the
            last message (if available).
        """
        try:
            async with self.async_session() as session:
                # Build WHERE clause dynamically to avoid parameter type ambiguity
                where_clauses = []
                params = {"limit": limit}

                if user_id is not None:
                    where_clauses.append("user_id = :user_id")
                    params["user_id"] = user_id

                if app_id is not None:
                    where_clauses.append("app_id = :app_id")
                    params["app_id"] = app_id

                where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

                # Use a raw SQL query to efficiently extract just the last message
                query = text(
                    f"""
                    SELECT
                        conversation_id,
                        title,
                        updated_at,
                        created_at,
                        CASE
                            WHEN history IS NOT NULL AND jsonb_array_length(history) > 0
                            THEN history->-1
                            ELSE NULL
                        END as last_message
                    FROM chat_conversations
                    {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT :limit
                """
                )

                result = await session.execute(query, params)

                conversations: List[Dict[str, Any]] = []
                for row in result:
                    conversations.append(
                        {
                            "chat_id": row.conversation_id,
                            "title": row.title,
                            "updated_at": row.updated_at,
                            "created_at": row.created_at,
                            "last_message": row.last_message,
                        }
                    )
                return conversations
        except Exception as exc:  # noqa: BLE001
            logger.error("Error listing chat conversations: %s", exc)
            return []

    async def update_chat_title(
        self,
        conversation_id: str,
        title: str,
        user_id: Optional[str],
        app_id: Optional[str] = None,
    ) -> bool:
        """Update the title of a chat conversation."""
        try:
            async with self.async_session() as session:
                # Build the WHERE clause based on user/app context
                where_clauses = ["conversation_id = :cid"]
                params = {"cid": conversation_id, "title": title}

                if user_id is not None:
                    where_clauses.append("user_id = :uid")
                    params["uid"] = user_id
                if app_id is not None:
                    where_clauses.append("app_id = :aid")
                    params["aid"] = app_id

                where_clause = " AND ".join(where_clauses)

                result = await session.execute(
                    text(
                        f"""
                        UPDATE chat_conversations
                        SET title = :title, updated_at = CURRENT_TIMESTAMP
                        WHERE {where_clause}
                    """
                    ),
                    params,
                )
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating chat title: {e}")
            return False

    def _check_folder_model_access(self, folder_model: FolderModel, auth: AuthContext) -> bool:
        """Check if the user has access to the folder."""
        # Simplified access check - consistent with documents/graphs
        if auth.app_id:
            # Check app_id match when present (cloud mode)
            return folder_model.app_id == auth.app_id

        # Otherwise check owner_id match (dev/self-hosted mode)
        return folder_model.owner_id == auth.entity_id

    # ------------------------------------------------------------------
    # PERFORMANCE: lightweight folder summaries (id, name, description)
    # ------------------------------------------------------------------

    async def list_folders_summary(self, auth: AuthContext) -> List[Dict[str, Any]]:  # noqa: D401 – returns plain dicts
        """Return folder summaries without the heavy *document_ids* payload.

        The UI only needs *id* and *name* to render the folder grid / sidebar.
        Excluding the potentially thousands-element ``document_ids`` array keeps
        the JSON response tiny and dramatically improves load time.
        """

        try:
            # Re-use the complex access logic of *list_folders* but post-process
            # the results to strip the large field.  This avoids duplicating
            # query-builder logic while still improving network payload size.
            full_folders = await self.list_folders(auth)

            summaries: List[Dict[str, Any]] = []
            for f in full_folders:
                summaries.append(
                    {
                        "id": f.id,
                        "name": f.name,
                        "description": getattr(f, "description", None),
                        "updated_at": (f.system_metadata or {}).get("updated_at"),
                        "doc_count": len(f.document_ids or []),
                    }
                )

            return summaries

        except Exception as exc:  # noqa: BLE001
            logger.error("Error building folder summary list: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Model Configuration Methods
    # ------------------------------------------------------------------

    async def store_model_config(self, model_config: ModelConfig) -> bool:
        """Store a model configuration."""
        try:
            config_dict = model_config.model_dump()

            # Serialize datetime objects
            config_dict = _serialize_datetime(config_dict)

            async with self.async_session() as session:
                config_model = ModelConfigModel(**config_dict)
                session.add(config_model)
                await session.commit()

            logger.info(f"Stored model config {model_config.id} for user {model_config.user_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing model config: {str(e)}")
            return False

    async def get_model_config(self, config_id: str, user_id: str, app_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID if user has access."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ModelConfigModel)
                    .where(ModelConfigModel.id == config_id)
                    .where(ModelConfigModel.user_id == user_id)
                    .where(ModelConfigModel.app_id == app_id)
                )
                config_model = result.scalar_one_or_none()

                if config_model:
                    return ModelConfig(
                        id=config_model.id,
                        user_id=config_model.user_id,
                        app_id=config_model.app_id,
                        provider=config_model.provider,
                        config_data=config_model.config_data,
                        created_at=config_model.created_at,
                        updated_at=config_model.updated_at,
                    )
                return None

        except Exception as e:
            logger.error(f"Error getting model config: {str(e)}")
            return None

    async def get_model_configs(self, user_id: str, app_id: str) -> List[ModelConfig]:
        """Get all model configurations for a user and app."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ModelConfigModel)
                    .where(ModelConfigModel.user_id == user_id)
                    .where(ModelConfigModel.app_id == app_id)
                    .order_by(ModelConfigModel.updated_at.desc())
                )
                config_models = result.scalars().all()

                configs = []
                for config_model in config_models:
                    configs.append(
                        ModelConfig(
                            id=config_model.id,
                            user_id=config_model.user_id,
                            app_id=config_model.app_id,
                            provider=config_model.provider,
                            config_data=config_model.config_data,
                            created_at=config_model.created_at,
                            updated_at=config_model.updated_at,
                        )
                    )

                return configs

        except Exception as e:
            logger.error(f"Error listing model configs: {str(e)}")
            return []

    async def update_model_config(self, config_id: str, user_id: str, app_id: str, updates: Dict[str, Any]) -> bool:
        """Update a model configuration if user has access."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ModelConfigModel)
                    .where(ModelConfigModel.id == config_id)
                    .where(ModelConfigModel.user_id == user_id)
                    .where(ModelConfigModel.app_id == app_id)
                )
                config_model = result.scalar_one_or_none()

                if not config_model:
                    logger.error(f"Model config {config_id} not found or user does not have access")
                    return False

                # Update fields
                if "config_data" in updates:
                    config_model.config_data = updates["config_data"]

                config_model.updated_at = datetime.now(UTC).isoformat()

                await session.commit()
                logger.info(f"Updated model config {config_id}")
                return True

        except Exception as e:
            logger.error(f"Error updating model config: {str(e)}")
            return False

    async def delete_model_config(self, config_id: str, user_id: str, app_id: str) -> bool:
        """Delete a model configuration if user has access."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ModelConfigModel)
                    .where(ModelConfigModel.id == config_id)
                    .where(ModelConfigModel.user_id == user_id)
                    .where(ModelConfigModel.app_id == app_id)
                )
                config_model = result.scalar_one_or_none()

                if not config_model:
                    logger.error(f"Model config {config_id} not found or user does not have access")
                    return False

                await session.delete(config_model)
                await session.commit()

                logger.info(f"Deleted model config {config_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting model config: {str(e)}")
            return False

    async def store_workflow(self, workflow: Workflow, auth: AuthContext) -> bool:  # noqa: D401 – override
        try:
            wf_json = _serialize_datetime(workflow.model_dump())

            # Extract scoping fields from auth context
            owner_id = auth.entity_id if auth.entity_type else None
            app_id = auth.app_id
            user_id = auth.user_id

            async with self.async_session() as session:
                existing = await session.get(WorkflowModel, workflow.id)
                if existing:
                    existing.data = wf_json
                    existing.owner_id = owner_id
                    existing.app_id = app_id
                    existing.user_id = user_id
                    existing.updated_at = datetime.now(UTC)
                else:
                    session.add(
                        WorkflowModel(id=workflow.id, data=wf_json, owner_id=owner_id, app_id=app_id, user_id=user_id)
                    )
                await session.commit()
            return True
        except Exception as exc:
            logger.error("Error storing workflow: %s", exc)
            return False

    async def list_workflows(self, auth: AuthContext) -> List[Workflow]:
        try:
            async with self.async_session() as session:
                # Build query with auth filtering
                query = select(WorkflowModel)

                # Simplified access control - consistent with other resources
                if auth.app_id:
                    # Filter by app_id in cloud mode
                    query = query.where(WorkflowModel.app_id == auth.app_id)
                elif auth.entity_id:
                    # Filter by owner_id in dev/self-hosted mode
                    query = query.where(WorkflowModel.owner_id == auth.entity_id)

                # In cloud mode, also filter by user_id if present
                if auth.user_id and get_settings().MODE == "cloud":
                    query = query.where(WorkflowModel.user_id == auth.user_id)

                res = await session.execute(query)
                rows = res.scalars().all()
                return [Workflow(**row.data) for row in rows]
        except Exception as exc:
            logger.error("Error listing workflows: %s", exc)
            return []

    async def get_workflow(self, workflow_id: str, auth: AuthContext) -> Optional[Workflow]:
        try:
            async with self.async_session() as session:
                wf_model = await session.get(WorkflowModel, workflow_id)
                if not wf_model:
                    return None

                # Simplified access check - consistent with other resources
                if auth.app_id:
                    # Check app_id in cloud mode
                    if wf_model.app_id != auth.app_id:
                        return None
                elif auth.entity_id:
                    # Check owner_id in dev/self-hosted mode
                    if wf_model.owner_id != auth.entity_id:
                        return None
                else:
                    # No access without auth
                    return None

                # In cloud mode, check user_id matches if present
                if auth.user_id and get_settings().MODE == "cloud":
                    if wf_model.user_id != auth.user_id:
                        return None

                return Workflow(**wf_model.data)
        except Exception as exc:
            logger.error("Error getting workflow: %s", exc)
            return None

    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any], auth: AuthContext) -> Optional[Workflow]:
        wf = await self.get_workflow(workflow_id, auth)
        if not wf:
            return None
        wf_dict = wf.model_dump()
        wf_dict.update(updates)
        wf_updated = Workflow(**wf_dict)
        await self.store_workflow(wf_updated, auth)
        return wf_updated

    async def delete_workflow(self, workflow_id: str, auth: AuthContext) -> bool:
        try:
            async with self.async_session() as session:
                # Get the workflow to check permissions
                wf_model = await session.get(WorkflowModel, workflow_id)
                if not wf_model:
                    return False

                # Check permissions - same logic as get_workflow
                # Simplified access check
                if auth.app_id:
                    if wf_model.app_id != auth.app_id:
                        logger.warning(
                            f"App {auth.app_id} attempted to delete workflow {workflow_id} from different app"
                        )
                        return False
                elif auth.entity_id:
                    if wf_model.owner_id != auth.entity_id:
                        logger.warning(
                            f"User {auth.entity_id} attempted to delete workflow {workflow_id} owned by {wf_model.owner_id}"
                        )
                        return False
                else:
                    return False

                if auth.user_id and get_settings().MODE == "cloud":
                    if wf_model.user_id != auth.user_id:
                        return False

                # First, remove workflow from all folders
                folders_query = text(
                    """
                    UPDATE folders
                    SET workflow_ids = workflow_ids - :workflow_id
                    WHERE workflow_ids @> :workflow_id_json
                """
                )
                await session.execute(
                    folders_query, {"workflow_id": workflow_id, "workflow_id_json": json.dumps([workflow_id])}
                )

                # Delete the workflow
                await session.delete(wf_model)
                await session.commit()
                logger.info(f"Deleted workflow {workflow_id} and removed from all folders")
                return True
        except Exception as exc:
            logger.error("Error deleting workflow: %s", exc)
            return False

    async def store_workflow_run(self, run: WorkflowRun) -> bool:
        try:
            run_json = _serialize_datetime(run.model_dump())

            async with self.async_session() as session:
                # Get owner_id from the workflow
                workflow = await session.get(WorkflowModel, run.workflow_id)
                owner_id = workflow.owner_id if workflow else None

                existing = await session.get(WorkflowRunModel, run.id)
                if existing:
                    existing.data = run_json
                    existing.workflow_id = run.workflow_id
                    existing.owner_id = owner_id
                    existing.app_id = run.app_id
                    existing.user_id = run.user_id
                    existing.updated_at = datetime.now(UTC)
                else:
                    session.add(
                        WorkflowRunModel(
                            id=run.id,
                            workflow_id=run.workflow_id,
                            data=run_json,
                            owner_id=owner_id,
                            app_id=run.app_id,
                            user_id=run.user_id,
                        )
                    )
                await session.commit()
            return True
        except Exception as exc:
            logger.error("Error storing workflow run: %s", exc)
            return False

    async def get_workflow_run(self, run_id: str, auth: AuthContext) -> Optional[WorkflowRun]:
        try:
            async with self.async_session() as session:
                run_model = await session.get(WorkflowRunModel, run_id)
                if not run_model:
                    return None

                # Check permissions - same logic as get_workflow
                # Check if user owns the workflow run
                # Simplified access check
                if auth.app_id:
                    if run_model.app_id != auth.app_id:
                        return None
                elif auth.entity_id:
                    if run_model.owner_id != auth.entity_id:
                        return None
                else:
                    return None

                # In cloud mode, check user_id matches if present
                if auth.user_id and get_settings().MODE == "cloud":
                    if run_model.user_id != auth.user_id:
                        return None

                return WorkflowRun(**run_model.data)
        except Exception as exc:
            logger.error("Error getting workflow run: %s", exc)
            return None

    async def list_workflow_runs(self, workflow_id: str, auth: AuthContext) -> List[WorkflowRun]:
        try:
            async with self.async_session() as session:
                # First check if the user has access to the workflow itself
                workflow = await session.get(WorkflowModel, workflow_id)
                if not workflow:
                    return []

                # Check workflow permissions - simplified
                if auth.app_id:
                    if workflow.app_id != auth.app_id:
                        return []
                elif auth.entity_id:
                    if workflow.owner_id != auth.entity_id:
                        return []
                else:
                    return []

                if auth.user_id and get_settings().MODE == "cloud":
                    if workflow.user_id != auth.user_id:
                        return []

                # Now get workflow runs with auth filtering
                query = select(WorkflowRunModel).where(WorkflowRunModel.workflow_id == workflow_id)

                # Simplified access control
                if auth.app_id:
                    query = query.where(WorkflowRunModel.app_id == auth.app_id)
                elif auth.entity_id:
                    query = query.where(WorkflowRunModel.owner_id == auth.entity_id)

                # In cloud mode, also filter by user_id if present
                if auth.user_id and get_settings().MODE == "cloud":
                    query = query.where(WorkflowRunModel.user_id == auth.user_id)

                res = await session.execute(query)
                rows = res.scalars().all()
                return [WorkflowRun(**r.data) for r in rows]
        except Exception as exc:
            logger.error("Error listing workflow runs: %s", exc)
            return []

    async def delete_workflow_run(self, run_id: str, auth: AuthContext) -> bool:
        try:
            async with self.async_session() as session:
                run_model = await session.get(WorkflowRunModel, run_id)
                if not run_model:
                    return False

                # Check permissions - same logic as get_workflow_run
                # Simplified access check
                if auth.app_id:
                    if run_model.app_id != auth.app_id:
                        logger.warning(
                            f"App {auth.app_id} attempted to delete workflow run {run_id} from different app"
                        )
                        return False
                elif auth.entity_id:
                    if run_model.owner_id != auth.entity_id:
                        logger.warning(
                            f"User {auth.entity_id} attempted to delete workflow run {run_id} owned by {run_model.owner_id}"
                        )
                        return False
                else:
                    return False

                if auth.user_id and get_settings().MODE == "cloud":
                    if run_model.user_id != auth.user_id:
                        return False

                await session.delete(run_model)
                await session.commit()
                logger.info(f"Deleted workflow run {run_id}")
                return True
        except Exception as exc:
            logger.error("Error deleting workflow run: %s", exc)
            return False

    async def search_documents_by_name(
        self,
        query: str,
        auth: AuthContext,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search documents by filename using PostgreSQL full-text search."""
        try:
            async with self.async_session() as session:
                # Build base query using existing patterns
                access_filter = self._build_access_filter_optimized(auth)
                metadata_filter = self._build_metadata_filter(filters)
                system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)
                filter_params = self._build_filter_params(auth, system_filters)

                # Build WHERE clauses
                where_clauses = [f"({access_filter})"]

                if metadata_filter:
                    where_clauses.append(f"({metadata_filter})")

                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")

                # Add search condition - try multiple approaches based on the article
                clean_query = query.strip()
                if clean_query:
                    filter_params["search_query"] = clean_query
                    filter_params["ilike_query"] = f"%{clean_query}%"

                    # Try multiple search strategies for better results with individual tracking
                    search_conditions = [
                        # Simple ILIKE for exact substring matches
                        "filename ILIKE :ilike_query",
                        # FTS with filename normalization - replace separators with spaces and remove extensions
                        """to_tsvector('english',
                            regexp_replace(
                                regexp_replace(COALESCE(filename, ''), '\\.[^.]*$', '', 'g'),
                                '[_-]+', ' ', 'g'
                            )
                        ) @@ plainto_tsquery('english', :search_query)""",
                        # FTS simple with same normalization
                        """to_tsvector('simple',
                            regexp_replace(
                                regexp_replace(COALESCE(filename, ''), '\\.[^.]*$', '', 'g'),
                                '[_-]+', ' ', 'g'
                            )
                        ) @@ plainto_tsquery('simple', :search_query)""",
                    ]

                    # Combine with OR - if any method matches, include the result
                    where_clauses.append(f"({' OR '.join(search_conditions)})")

                final_where_clause = " AND ".join(where_clauses)

                # Build the query properly using SQLAlchemy ORM
                base_query = select(DocumentModel).where(text(final_where_clause))

                # Add ordering based on whether we have a search query
                if clean_query:
                    # Order by FTS rank score with filename normalization
                    rank_expr = text(
                        """ts_rank(
                        to_tsvector('english',
                            regexp_replace(
                                regexp_replace(COALESCE(filename, ''), '\\.[^.]*$', '', 'g'),
                                '[_-]+', ' ', 'g'
                            )
                        ),
                        plainto_tsquery('english', :search_query)
                    )"""
                    )
                    query = base_query.order_by(
                        desc(rank_expr), text("(system_metadata->>'updated_at')::timestamp DESC NULLS LAST")
                    )
                else:
                    # No search query - order by recency only
                    query = base_query.order_by(text("(system_metadata->>'updated_at')::timestamp DESC NULLS LAST"))

                # Apply limit and bind parameters
                query = query.limit(limit)

                # Execute with parameter binding
                result = await session.execute(query, filter_params)
                doc_models = result.scalars().all()

                # Convert to Document objects using existing method
                documents = [Document(**self._document_model_to_dict(doc)) for doc in doc_models]

                logger.debug(f"Document name search for '{clean_query}' returned {len(documents)} results")
                return documents

        except Exception as e:
            logger.error(f"Error searching documents by name: {str(e)}")
            return []
