import json
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Index, select, text
from sqlalchemy.dialects.postgresql import JSONB

from .base_database import BaseDatabase
from ..models.documents import Document
from ..models.auth import AuthContext
from ..models.graph import Graph
from ..models.folders import Folder

logger = logging.getLogger(__name__)
Base = declarative_base()


class DocumentModel(Base):
    """SQLAlchemy model for document metadata."""

    __tablename__ = "documents"

    external_id = Column(String, primary_key=True)
    owner = Column(JSONB)
    content_type = Column(String)
    filename = Column(String, nullable=True)
    doc_metadata = Column(JSONB, default=dict)
    storage_info = Column(JSONB, default=dict)
    system_metadata = Column(JSONB, default=dict)
    additional_metadata = Column(JSONB, default=dict)
    access_control = Column(JSONB, default=dict)
    chunk_ids = Column(JSONB, default=list)
    storage_files = Column(JSONB, default=list)

    # Create indexes
    __table_args__ = (
        Index("idx_owner_id", "owner", postgresql_using="gin"),
        Index("idx_access_control", "access_control", postgresql_using="gin"),
        Index("idx_system_metadata", "system_metadata", postgresql_using="gin"),
    )


class GraphModel(Base):
    """SQLAlchemy model for graph data."""

    __tablename__ = "graphs"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True, index=True)
    entities = Column(JSONB, default=list)
    relationships = Column(JSONB, default=list)
    graph_metadata = Column(JSONB, default=dict)  # Renamed from 'metadata' to avoid conflict
    system_metadata = Column(JSONB, default=dict)  # For folder_name and end_user_id
    document_ids = Column(JSONB, default=list)
    filters = Column(JSONB, nullable=True)
    created_at = Column(String)  # ISO format string
    updated_at = Column(String)  # ISO format string
    owner = Column(JSONB)
    access_control = Column(JSONB, default=dict)

    # Create indexes
    __table_args__ = (
        Index("idx_graph_name", "name"),
        Index("idx_graph_owner", "owner", postgresql_using="gin"),
        Index("idx_graph_access_control", "access_control", postgresql_using="gin"),
        Index("idx_graph_system_metadata", "system_metadata", postgresql_using="gin"),
    )


class FolderModel(Base):
    """SQLAlchemy model for folder data."""

    __tablename__ = "folders"

    id = Column(String, primary_key=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    owner = Column(JSONB)
    document_ids = Column(JSONB, default=list)
    system_metadata = Column(JSONB, default=dict)
    access_control = Column(JSONB, default=dict)

    # Create indexes
    __table_args__ = (
        Index("idx_folder_name", "name"),
        Index("idx_folder_owner", "owner", postgresql_using="gin"),
        Index("idx_folder_access_control", "access_control", postgresql_using="gin"),
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


class PostgresDatabase(BaseDatabase):
    """PostgreSQL implementation for document metadata storage."""

    def __init__(
        self,
        uri: str,
    ):
        """Initialize PostgreSQL connection for document storage."""
        # Load settings from config
        from core.config import get_settings
        settings = get_settings()
        
        # Get database pool settings from config with defaults
        pool_size = getattr(settings, "DB_POOL_SIZE", 20)
        max_overflow = getattr(settings, "DB_MAX_OVERFLOW", 30)
        pool_recycle = getattr(settings, "DB_POOL_RECYCLE", 3600)
        pool_timeout = getattr(settings, "DB_POOL_TIMEOUT", 10)
        pool_pre_ping = getattr(settings, "DB_POOL_PRE_PING", True)
        
        logger.info(f"Initializing PostgreSQL connection pool with size={pool_size}, "
                   f"max_overflow={max_overflow}, pool_recycle={pool_recycle}s")
        
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
                    
                # Create indexes for folder_name and end_user_id in system_metadata for documents
                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_system_metadata_folder_name
                    ON documents ((system_metadata->>'folder_name'));
                    """
                    )
                )
                
                # Create folders table if it doesn't exist
                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS folders (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        owner JSONB,
                        document_ids JSONB DEFAULT '[]',
                        system_metadata JSONB DEFAULT '{}',
                        access_control JSONB DEFAULT '{}'
                    );
                    """
                    )
                )
                
                # Create indexes for folders table
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_folder_name ON folders (name);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_folder_owner ON folders USING gin (owner);"))
                await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_folder_access_control ON folders USING gin (access_control);"))
                
                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_system_metadata_end_user_id
                    ON documents ((system_metadata->>'end_user_id'));
                    """
                    )
                )
                
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
                
                # Create indexes for folder_name and end_user_id in system_metadata for graphs
                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_graph_system_metadata_folder_name
                    ON graphs ((system_metadata->>'folder_name'));
                    """
                    )
                )
                
                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_graph_system_metadata_end_user_id
                    ON graphs ((system_metadata->>'end_user_id'));
                    """
                    )
                )
                
                logger.info("Created indexes for folder_name and end_user_id in system_metadata")

            logger.info("PostgreSQL tables and indexes created successfully")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error creating PostgreSQL tables and indexes: {str(e)}")
            return False

    async def store_document(self, document: Document) -> bool:
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
                # Build access filter
                access_filter = self._build_access_filter(auth)

                # Query document
                query = (
                    select(DocumentModel)
                    .where(DocumentModel.external_id == document_id)
                    .where(text(f"({access_filter})"))
                )

                result = await session.execute(query)
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    # Convert doc_metadata back to metadata
                    doc_dict = {
                        "external_id": doc_model.external_id,
                        "owner": doc_model.owner,
                        "content_type": doc_model.content_type,
                        "filename": doc_model.filename,
                        "metadata": doc_model.doc_metadata,
                        "storage_info": doc_model.storage_info,
                        "system_metadata": doc_model.system_metadata,
                        "additional_metadata": doc_model.additional_metadata,
                        "access_control": doc_model.access_control,
                        "chunk_ids": doc_model.chunk_ids,
                        "storage_files": doc_model.storage_files or [],
                    }
                    return Document(**doc_dict)
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document metadata: {str(e)}")
            return None
            
    async def get_document_by_filename(self, filename: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> Optional[Document]:
        """Retrieve document metadata by filename if user has access.
        If multiple documents have the same filename, returns the most recently updated one.
        
        Args:
            filename: The filename to search for
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)
        """
        try:
            async with self.async_session() as session:
                # Build access filter
                access_filter = self._build_access_filter(auth)
                system_metadata_filter = self._build_system_metadata_filter(system_filters)
                filename = filename.replace('\'', '\'\'')
                # Construct where clauses
                where_clauses = [
                    f"({access_filter})",
                    f"filename = '{filename}'"  # Escape single quotes
                ]
                
                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")
                    
                final_where_clause = " AND ".join(where_clauses)

                # Query document with system filters
                query = (
                    select(DocumentModel)
                    .where(text(final_where_clause))
                    # Order by updated_at in system_metadata to get the most recent document
                    .order_by(text("system_metadata->>'updated_at' DESC"))
                )

                logger.debug(f"Querying document by filename with system filters: {system_filters}")
                
                result = await session.execute(query)
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    # Convert doc_metadata back to metadata
                    doc_dict = {
                        "external_id": doc_model.external_id,
                        "owner": doc_model.owner,
                        "content_type": doc_model.content_type,
                        "filename": doc_model.filename,
                        "metadata": doc_model.doc_metadata,
                        "storage_info": doc_model.storage_info,
                        "system_metadata": doc_model.system_metadata,
                        "additional_metadata": doc_model.additional_metadata,
                        "access_control": doc_model.access_control,
                        "chunk_ids": doc_model.chunk_ids,
                        "storage_files": doc_model.storage_files or [],
                    }
                    return Document(**doc_dict)
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document metadata by filename: {str(e)}")
            return None
            
    async def get_documents_by_id(self, document_ids: List[str], auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
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
                # Build access filter
                access_filter = self._build_access_filter(auth)
                system_metadata_filter = self._build_system_metadata_filter(system_filters)
                
                # Construct where clauses
                document_ids_linked = ', '.join([('\'' + doc_id + '\'') for doc_id in document_ids])
                where_clauses = [
                    f"({access_filter})",
                    f"external_id IN ({document_ids_linked})"
                ]
                
                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")
                    
                final_where_clause = " AND ".join(where_clauses)
                
                # Query documents with document IDs, access check, and system filters in a single query
                query = select(DocumentModel).where(text(final_where_clause))
                
                logger.info(f"Batch retrieving {len(document_ids)} documents with a single query")
                
                # Execute batch query
                result = await session.execute(query)
                doc_models = result.scalars().all()
                
                documents = []
                for doc_model in doc_models:
                    # Convert doc_metadata back to metadata
                    doc_dict = {
                        "external_id": doc_model.external_id,
                        "owner": doc_model.owner,
                        "content_type": doc_model.content_type,
                        "filename": doc_model.filename,
                        "metadata": doc_model.doc_metadata,
                        "storage_info": doc_model.storage_info,
                        "system_metadata": doc_model.system_metadata,
                        "additional_metadata": doc_model.additional_metadata,
                        "access_control": doc_model.access_control,
                        "chunk_ids": doc_model.chunk_ids,
                        "storage_files": doc_model.storage_files or [],
                    }
                    documents.append(Document(**doc_dict))
                
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
                access_filter = self._build_access_filter(auth)
                metadata_filter = self._build_metadata_filter(filters)
                system_metadata_filter = self._build_system_metadata_filter(system_filters)

                where_clauses = [f"({access_filter})"]
                
                if metadata_filter:
                    where_clauses.append(f"({metadata_filter})")
                    
                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")
                    
                final_where_clause = " AND ".join(where_clauses)
                query = select(DocumentModel).where(text(final_where_clause))

                query = query.offset(skip).limit(limit)

                result = await session.execute(query)
                doc_models = result.scalars().all()

                return [
                    Document(
                        external_id=doc.external_id,
                        owner=doc.owner,
                        content_type=doc.content_type,
                        filename=doc.filename,
                        metadata=doc.doc_metadata,
                        storage_info=doc.storage_info,
                        system_metadata=doc.system_metadata,
                        additional_metadata=doc.additional_metadata,
                        access_control=doc.access_control,
                        chunk_ids=doc.chunk_ids,
                        storage_files=doc.storage_files or [],
                    )
                    for doc in doc_models
                ]

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    async def update_document(
        self, document_id: str, updates: Dict[str, Any], auth: AuthContext
    ) -> bool:
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
            
            # Preserve folder_name and end_user_id if not explicitly overridden
            if existing_doc.system_metadata:
                if "folder_name" in existing_doc.system_metadata and "folder_name" not in updates["system_metadata"]:
                    updates["system_metadata"]["folder_name"] = existing_doc.system_metadata["folder_name"]
                
                if "end_user_id" in existing_doc.system_metadata and "end_user_id" not in updates["system_metadata"]:
                    updates["system_metadata"]["end_user_id"] = existing_doc.system_metadata["end_user_id"]
            
            updates["system_metadata"]["updated_at"] = datetime.now(UTC)

            # Serialize datetime objects to ISO format strings
            updates = _serialize_datetime(updates)

            async with self.async_session() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.external_id == document_id)
                )
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    for key, value in updates.items():
                        setattr(doc_model, key, value)
                    await session.commit()
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
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.external_id == document_id)
                )
                doc_model = result.scalar_one_or_none()

                if doc_model:
                    await session.delete(doc_model)
                    await session.commit()
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    async def find_authorized_and_filtered_documents(
        self, auth: AuthContext, filters: Optional[Dict[str, Any]] = None, system_filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Find document IDs matching filters and access permissions."""
        try:
            async with self.async_session() as session:
                # Build query
                access_filter = self._build_access_filter(auth)
                metadata_filter = self._build_metadata_filter(filters)
                system_metadata_filter = self._build_system_metadata_filter(system_filters)

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
                query = select(DocumentModel.external_id).where(text(final_where_clause))

                logger.debug(f"Final query: {query}")

                result = await session.execute(query)
                doc_ids = [row[0] for row in result.all()]
                logger.debug(f"Found document IDs: {doc_ids}")
                return doc_ids

        except Exception as e:
            logger.error(f"Error finding authorized documents: {str(e)}")
            return []

    async def check_access(
        self, document_id: str, auth: AuthContext, required_permission: str = "read"
    ) -> bool:
        """Check if user has required permission for document."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.external_id == document_id)
                )
                doc_model = result.scalar_one_or_none()

                if not doc_model:
                    return False

                # Check owner access
                owner = doc_model.owner
                if owner.get("type") == auth.entity_type and owner.get("id") == auth.entity_id:
                    return True

                # Check permission-specific access
                access_control = doc_model.access_control
                permission_map = {"read": "readers", "write": "writers", "admin": "admins"}
                permission_set = permission_map.get(required_permission)

                if not permission_set:
                    return False

                return auth.entity_id in access_control.get(permission_set, [])

        except Exception as e:
            logger.error(f"Error checking document access: {str(e)}")
            return False

    def _build_access_filter(self, auth: AuthContext) -> str:
        """Build PostgreSQL filter for access control."""
        filters = [
            f"owner->>'id' = '{auth.entity_id}'",
            f"access_control->'readers' ? '{auth.entity_id}'",
            f"access_control->'writers' ? '{auth.entity_id}'",
            f"access_control->'admins' ? '{auth.entity_id}'",
        ]

        if auth.entity_type == "DEVELOPER" and auth.app_id:
            # Add app-specific access for developers
            filters.append(f"access_control->'app_access' ? '{auth.app_id}'")
            
        # Add user_id filter in cloud mode
        if auth.user_id:
            from core.config import get_settings
            settings = get_settings()
            
            if settings.MODE == "cloud":
                # Filter by user_id in access_control
                filters.append(f"access_control->>'user_id' = '{auth.user_id}'")

        return " OR ".join(filters)

    def _build_metadata_filter(self, filters: Dict[str, Any]) -> str:
        """Build PostgreSQL filter for metadata."""
        if not filters:
            return ""

        filter_conditions = []
        for key, value in filters.items():
            # Convert boolean values to string 'true' or 'false'
            if isinstance(value, bool):
                value = str(value).lower()
                
            # Use proper SQL escaping for string values
            if isinstance(value, str):
                # Replace single quotes with double single quotes to escape them
                value = value.replace("'", "''") 
                
            filter_conditions.append(f"doc_metadata->>'{key}' = '{value}'")

        return " AND ".join(filter_conditions)
        
    def _build_system_metadata_filter(self, system_filters: Optional[Dict[str, Any]]) -> str:
        """Build PostgreSQL filter for system metadata."""
        if not system_filters:
            return ""
            
        conditions = []
        for key, value in system_filters.items():
            if value is None:
                continue
                
            if isinstance(value, str):
                # Replace single quotes with double single quotes to escape them
                escaped_value = value.replace("'", "''")
                conditions.append(f"system_metadata->>'{key}' = '{escaped_value}'")
            else:
                conditions.append(f"system_metadata->>'{key}' = '{value}'")
                
        return " AND ".join(conditions)

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
                result = await session.execute(
                    text("SELECT metadata FROM caches WHERE name = :name"), {"name": name}
                )
                row = result.first()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get cache metadata: {e}")
            return None

    async def store_graph(self, graph: Graph) -> bool:
        """Store a graph in PostgreSQL.

        This method stores the graph metadata, entities, and relationships
        in a PostgreSQL table.

        Args:
            graph: Graph to store

        Returns:
            bool: Whether the operation was successful
        """
        # Ensure database is initialized
        if not self._initialized:
            await self.initialize()

        try:
            # First serialize the graph model to dict
            graph_dict = graph.model_dump()

            # Change 'metadata' to 'graph_metadata' to match our model
            if "metadata" in graph_dict:
                graph_dict["graph_metadata"] = graph_dict.pop("metadata")

            # Serialize datetime objects to ISO format strings
            graph_dict = _serialize_datetime(graph_dict)

            # Store the graph metadata in PostgreSQL
            async with self.async_session() as session:
                # Store graph metadata in our table
                graph_model = GraphModel(**graph_dict)
                session.add(graph_model)
                await session.commit()
                logger.info(f"Stored graph '{graph.name}' with {len(graph.entities)} entities and {len(graph.relationships)} relationships")

            return True

        except Exception as e:
            logger.error(f"Error storing graph: {str(e)}")
            return False

    async def get_graph(self, name: str, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> Optional[Graph]:
        """Get a graph by name.

        Args:
            name: Name of the graph
            auth: Authentication context
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)

        Returns:
            Optional[Graph]: Graph if found and accessible, None otherwise
        """
        # Ensure database is initialized
        if not self._initialized:
            await self.initialize()

        try:
            async with self.async_session() as session:
                # Build access filter
                access_filter = self._build_access_filter(auth)

                # We need to check if the documents in the graph match the system filters
                # First get the graph without system filters
                query = (
                    select(GraphModel)
                    .where(GraphModel.name == name)
                    .where(text(f"({access_filter})"))
                )

                result = await session.execute(query)
                graph_model = result.scalar_one_or_none()

                if graph_model:
                    # If system filters are provided, we need to filter the document_ids
                    document_ids = graph_model.document_ids
                    
                    if system_filters and document_ids:
                        # Apply system_filters to document_ids
                        system_metadata_filter = self._build_system_metadata_filter(system_filters)
                        
                        if system_metadata_filter:
                            # Get document IDs with system filters
                            doc_id_placeholders = ", ".join([f"'{doc_id}'" for doc_id in document_ids])
                            filter_query = f"""
                                SELECT external_id FROM documents 
                                WHERE external_id IN ({doc_id_placeholders})
                                AND ({system_metadata_filter})
                            """
                            
                            filter_result = await session.execute(text(filter_query))
                            filtered_doc_ids = [row[0] for row in filter_result.all()]
                            
                            # If no documents match system filters, return None
                            if not filtered_doc_ids:
                                return None
                            
                            # Update document_ids with filtered results
                            document_ids = filtered_doc_ids
                    
                    # Convert to Graph model
                    graph_dict = {
                        "id": graph_model.id,
                        "name": graph_model.name,
                        "entities": graph_model.entities,
                        "relationships": graph_model.relationships,
                        "metadata": graph_model.graph_metadata,  # Reference the renamed column
                        "system_metadata": graph_model.system_metadata or {},  # Include system_metadata
                        "document_ids": document_ids,  # Use possibly filtered document_ids
                        "filters": graph_model.filters,
                        "created_at": graph_model.created_at,
                        "updated_at": graph_model.updated_at,
                        "owner": graph_model.owner,
                        "access_control": graph_model.access_control,
                    }
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
        # Ensure database is initialized
        if not self._initialized:
            await self.initialize()

        try:
            async with self.async_session() as session:
                # Build access filter
                access_filter = self._build_access_filter(auth)

                # Query graphs
                query = select(GraphModel).where(text(f"({access_filter})"))

                result = await session.execute(query)
                graph_models = result.scalars().all()
                
                graphs = []
                
                # If system filters are provided, we need to filter each graph's document_ids
                if system_filters:
                    system_metadata_filter = self._build_system_metadata_filter(system_filters)
                    
                    for graph_model in graph_models:
                        document_ids = graph_model.document_ids
                        
                        if document_ids and system_metadata_filter:
                            # Get document IDs with system filters
                            doc_id_placeholders = ", ".join([f"'{doc_id}'" for doc_id in document_ids])
                            filter_query = f"""
                                SELECT external_id FROM documents 
                                WHERE external_id IN ({doc_id_placeholders})
                                AND ({system_metadata_filter})
                            """
                            
                            filter_result = await session.execute(text(filter_query))
                            filtered_doc_ids = [row[0] for row in filter_result.all()]
                            
                            # Only include graphs that have documents matching the system filters
                            if filtered_doc_ids:
                                graph = Graph(
                                    id=graph_model.id,
                                    name=graph_model.name,
                                    entities=graph_model.entities,
                                    relationships=graph_model.relationships,
                                    metadata=graph_model.graph_metadata,  # Reference the renamed column
                                    system_metadata=graph_model.system_metadata or {},  # Include system_metadata
                                    document_ids=filtered_doc_ids,  # Use filtered document_ids
                                    filters=graph_model.filters,
                                    created_at=graph_model.created_at,
                                    updated_at=graph_model.updated_at,
                                    owner=graph_model.owner,
                                    access_control=graph_model.access_control,
                                )
                                graphs.append(graph)
                else:
                    # No system filters, include all graphs
                    graphs = [
                        Graph(
                            id=graph.id,
                            name=graph.name,
                            entities=graph.entities,
                            relationships=graph.relationships,
                            metadata=graph.graph_metadata,  # Reference the renamed column
                            system_metadata=graph.system_metadata or {},  # Include system_metadata
                            document_ids=graph.document_ids,
                            filters=graph.filters,
                            created_at=graph.created_at,
                            updated_at=graph.updated_at,
                            owner=graph.owner,
                            access_control=graph.access_control,
                        )
                        for graph in graph_models
                    ]
                
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
        # Ensure database is initialized
        if not self._initialized:
            await self.initialize()

        try:
            # First serialize the graph model to dict
            graph_dict = graph.model_dump()

            # Change 'metadata' to 'graph_metadata' to match our model
            if "metadata" in graph_dict:
                graph_dict["graph_metadata"] = graph_dict.pop("metadata")

            # Serialize datetime objects to ISO format strings
            graph_dict = _serialize_datetime(graph_dict)

            # Update the graph in PostgreSQL
            async with self.async_session() as session:
                # Check if the graph exists
                result = await session.execute(
                    select(GraphModel).where(GraphModel.id == graph.id)
                )
                graph_model = result.scalar_one_or_none()

                if not graph_model:
                    logger.error(f"Graph '{graph.name}' with ID {graph.id} not found for update")
                    return False

                # Update the graph model with new values
                for key, value in graph_dict.items():
                    setattr(graph_model, key, value)

                await session.commit()
                logger.info(f"Updated graph '{graph.name}' with {len(graph.entities)} entities and {len(graph.relationships)} relationships")

            return True

        except Exception as e:
            logger.error(f"Error updating graph: {str(e)}")
            return False
            
    async def create_folder(self, folder: Folder) -> bool:
        """Create a new folder."""
        try:
            async with self.async_session() as session:
                folder_dict = folder.model_dump()
                
                # Convert datetime objects to strings for JSON serialization
                folder_dict = _serialize_datetime(folder_dict)
                
                # Check if a folder with this name already exists for this owner
                # Use only the type/id format
                stmt = text(
                    """
                    SELECT id FROM folders
                    WHERE name = :name
                    AND owner->>'id' = :entity_id 
                    AND owner->>'type' = :entity_type
                    """
                ).bindparams(
                    name=folder.name,
                    entity_id=folder.owner["id"],
                    entity_type=folder.owner["type"]
                )
                
                result = await session.execute(stmt)
                existing_folder = result.scalar_one_or_none()
                
                if existing_folder:
                    logger.info(f"Folder '{folder.name}' already exists with ID {existing_folder}, not creating a duplicate")
                    # Update the provided folder's ID to match the existing one 
                    # so the caller gets the correct ID
                    folder.id = existing_folder
                    return True
                
                # Create a new folder model
                access_control = folder_dict.get("access_control", {})
                
                # Log access control to debug any issues
                if "user_id" in access_control:
                    logger.info(f"Storing folder with user_id: {access_control['user_id']}")
                else:
                    logger.info("No user_id found in folder access_control")
                
                folder_model = FolderModel(
                    id=folder.id,
                    name=folder.name,
                    description=folder.description,
                    owner=folder_dict["owner"],
                    document_ids=folder_dict.get("document_ids", []),
                    system_metadata=folder_dict.get("system_metadata", {}),
                    access_control=access_control
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
                result = await session.execute(
                    select(FolderModel).where(FolderModel.id == folder_id)
                )
                folder_model = result.scalar_one_or_none()
                
                if not folder_model:
                    logger.error(f"Folder with ID {folder_id} not found in database")
                    return None
                
                # Convert to Folder object
                folder_dict = {
                    "id": folder_model.id,
                    "name": folder_model.name,
                    "description": folder_model.description,
                    "owner": folder_model.owner,
                    "document_ids": folder_model.document_ids,
                    "system_metadata": folder_model.system_metadata,
                    "access_control": folder_model.access_control
                }
                
                folder = Folder(**folder_dict)
                
                # Check if the user has access to the folder
                if not self._check_folder_access(folder, auth, "read"):
                    return None
                    
                return folder
                
        except Exception as e:
            logger.error(f"Error getting folder: {e}")
            return None
    
    async def get_folder_by_name(self, name: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by name."""
        try:
            async with self.async_session() as session:
                # First try to get a folder owned by this entity
                if auth.entity_type and auth.entity_id:
                    stmt = text(
                        """
                        SELECT * FROM folders
                        WHERE name = :name
                        AND (owner->>'entity_id' = :entity_id)
                        AND (owner->>'entity_type' = :entity_type)
                        """
                    ).bindparams(
                        name=name,
                        entity_id=auth.entity_id,
                        entity_type=auth.entity_type.value
                    )
                    
                    result = await session.execute(stmt)
                    folder_row = result.fetchone()
                    
                    if folder_row:
                        # Convert to Folder object
                        folder_dict = {
                            "id": folder_row.id,
                            "name": folder_row.name,
                            "description": folder_row.description,
                            "owner": folder_row.owner,
                            "document_ids": folder_row.document_ids,
                            "system_metadata": folder_row.system_metadata,
                            "access_control": folder_row.access_control
                        }
                        
                        return Folder(**folder_dict)
                
                # If not found, try to find any accessible folder with that name
                result = await session.execute(
                    select(FolderModel).where(FolderModel.name == name)
                )
                folder_models = result.scalars().all()
                
                for folder_model in folder_models:
                    # Convert to Folder object
                    folder_dict = {
                        "id": folder_model.id,
                        "name": folder_model.name,
                        "description": folder_model.description,
                        "owner": folder_model.owner,
                        "document_ids": folder_model.document_ids,
                        "system_metadata": folder_model.system_metadata,
                        "access_control": folder_model.access_control
                    }
                    
                    folder = Folder(**folder_dict)
                    
                    # Check if the user has access to the folder
                    if self._check_folder_access(folder, auth, "read"):
                        return folder
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting folder by name: {e}")
            return None
    
    async def list_folders(self, auth: AuthContext) -> List[Folder]:
        """List all folders the user has access to."""
        try:
            folders = []
            
            async with self.async_session() as session:
                # Get all folders
                result = await session.execute(select(FolderModel))
                folder_models = result.scalars().all()
                
                for folder_model in folder_models:
                    # Convert to Folder object
                    folder_dict = {
                        "id": folder_model.id,
                        "name": folder_model.name,
                        "description": folder_model.description,
                        "owner": folder_model.owner,
                        "document_ids": folder_model.document_ids,
                        "system_metadata": folder_model.system_metadata,
                        "access_control": folder_model.access_control
                    }
                    
                    folder = Folder(**folder_dict)
                    
                    # Check if the user has access to the folder
                    if self._check_folder_access(folder, auth, "read"):
                        folders.append(folder)
                
                return folders
                
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []
    
    async def add_document_to_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Add a document to a folder."""
        try:
            # First, check if the user has access to the folder
            folder = await self.get_folder(folder_id, auth)
            if not folder:
                logger.error(f"Folder {folder_id} not found or user does not have access")
                return False
                
            # Check if user has write access to the folder
            if not self._check_folder_access(folder, auth, "write"):
                logger.error(f"User does not have write access to folder {folder_id}")
                return False
                
            # Check if the document exists and user has access
            document = await self.get_document(document_id, auth)
            if not document:
                logger.error(f"Document {document_id} not found or user does not have access")
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
                
                # Also update the document's system_metadata to include the folder_name
                folder_name_json = json.dumps(folder.name)
                stmt = text(
                    f"""
                    UPDATE documents 
                    SET system_metadata = jsonb_set(system_metadata, '{{folder_name}}', '{folder_name_json}'::jsonb)
                    WHERE external_id = :document_id
                    """
                ).bindparams(
                    document_id=document_id
                )
                
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
            # First, check if the user has access to the folder
            folder = await self.get_folder(folder_id, auth)
            if not folder:
                logger.error(f"Folder {folder_id} not found or user does not have access")
                return False
                
            # Check if user has write access to the folder
            if not self._check_folder_access(folder, auth, "write"):
                logger.error(f"User does not have write access to folder {folder_id}")
                return False
            
            # Check if the document is in the folder
            if document_id not in folder.document_ids:
                logger.info(f"Document {document_id} is not in folder {folder_id}")
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
                
                # Also update the document's system_metadata to remove the folder_name
                stmt = text(
                    f"""
                    UPDATE documents 
                    SET system_metadata = jsonb_set(system_metadata, '{{folder_name}}', 'null'::jsonb)
                    WHERE external_id = :document_id
                    """
                ).bindparams(
                    document_id=document_id
                )
                
                await session.execute(stmt)
                await session.commit()
                
                logger.info(f"Removed document {document_id} from folder {folder_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error removing document from folder: {e}")
            return False
    
    def _check_folder_access(self, folder: Folder, auth: AuthContext, permission: str = "read") -> bool:
        """Check if the user has the required permission for the folder."""
        # Admin always has access
        if "admin" in auth.permissions:
            return True
            
        # Check if folder is owned by the user
        if (auth.entity_type and auth.entity_id and 
            folder.owner.get("type") == auth.entity_type.value and 
            folder.owner.get("id") == auth.entity_id):
            
            # In cloud mode, also verify user_id if present
            if auth.user_id:
                from core.config import get_settings
                settings = get_settings()
                
                if settings.MODE == "cloud":
                    folder_user_ids = folder.access_control.get("user_id", [])
                    if auth.user_id not in folder_user_ids:
                        return False
            return True
            
        # Check access control lists
        access_control = folder.access_control or {}
        
        if permission == "read":
            readers = access_control.get("readers", [])
            if f"{auth.entity_type.value}:{auth.entity_id}" in readers:
                return True
                
        if permission == "write":
            writers = access_control.get("writers", [])
            if f"{auth.entity_type.value}:{auth.entity_id}" in writers:
                return True
                
        # For admin permission, check admins list
        if permission == "admin":
            admins = access_control.get("admins", [])
            if f"{auth.entity_type.value}:{auth.entity_id}" in admins:
                return True
                
        return False
