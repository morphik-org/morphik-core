import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.config import get_settings
from core.utils.folder_utils import normalize_folder_path
from core.utils.typed_metadata import TypedMetadataError, normalize_metadata

from ..models.auth import AuthContext
from ..models.documents import Document
from ..models.folders import Folder
from ..models.graph import Graph
from ..models.model_config import ModelConfig
from .metadata_filters import InvalidMetadataFilterError, MetadataFilterBuilder
from .models import Base, ChatConversationModel, DocumentModel, FolderModel, GraphModel, ModelConfigModel
from .serializers import (
    _document_model_to_dict,
    _folder_row_to_dict,
    _graph_model_to_dict,
    _parse_datetime_field,
    _serialize_datetime,
)

logger = logging.getLogger(__name__)
SYSTEM_METADATA_SCOPE_KEYS = {"folder_name", "folder_id", "end_user_id", "app_id"}


class PostgresDatabase:
    """PostgreSQL implementation for document metadata storage."""

    _metadata_filter_builder = MetadataFilterBuilder()
    # Map system filter keys to flattened column names (used by filter builder methods)
    _SYSTEM_FILTER_COLUMNS = {
        "app_id": "app_id",
        "folder_name": "folder_name",
        "folder_path": "folder_path",
        "end_user_id": "end_user_id",
    }

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

            # Ensure all declarative models (including ones defined outside this module)
            # are registered with SQLAlchemy's metadata before create_all runs.
            # Import is local to avoid circular import overhead at module load.
            from core.models.apps import AppModel  # noqa: F401
            from core.vector_store.pgvector_store import VectorEmbedding  # noqa: F401

            # Create all tables and indexes via SQLAlchemy metadata
            async with self.engine.begin() as conn:
                # Enable pgvector extension (required for Vector column type)
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("Enabled pgvector extension")

                await conn.run_sync(lambda conn: Base.metadata.create_all(conn, checkfirst=True))
                logger.info("Created database tables and indexes successfully")

            logger.info("PostgreSQL initialization complete")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing PostgreSQL: {str(e)}")
            return False

    async def store_document(self, document: Document, auth: AuthContext) -> bool:
        """Store document metadata."""
        try:
            doc_dict = document.model_dump()

            metadata = doc_dict.pop("metadata", {}) or {}
            metadata.setdefault("external_id", doc_dict["external_id"])
            metadata_type_hints = doc_dict.pop("metadata_types", {}) or {}
            normalized_metadata, normalized_types = normalize_metadata(metadata, metadata_type_hints)
            doc_dict["doc_metadata"] = normalized_metadata
            doc_dict["metadata_types"] = normalized_types
            # Mirror folder path into doc_metadata for convenience in downstream filters (allow clearing)
            path_for_metadata = doc_dict.get("folder_path") or doc_dict.get("folder_name")
            doc_dict["doc_metadata"]["folder_name"] = path_for_metadata
            doc_dict["folder_id"] = doc_dict.get("folder_id")
            if doc_dict.get("folder_id"):
                doc_dict["doc_metadata"]["folder_id"] = doc_dict["folder_id"]

            # Keep folder_path in sync with folder_name for backward compatibility
            folder_name_value = doc_dict.get("folder_name")
            if doc_dict.get("folder_path") is None and folder_name_value:
                try:
                    doc_dict["folder_path"] = normalize_folder_path(folder_name_value)
                except ValueError:
                    doc_dict["folder_path"] = folder_name_value

            # Ensure system metadata
            if "system_metadata" not in doc_dict:
                doc_dict["system_metadata"] = {}
            doc_dict["system_metadata"]["created_at"] = datetime.now(UTC)
            doc_dict["system_metadata"]["updated_at"] = datetime.now(UTC)

            # Remove storage_files - we only use storage_info now
            doc_dict.pop("storage_files", None)

            # Serialize datetime objects to ISO format strings
            doc_dict = _serialize_datetime(doc_dict)

            # Simplified access control - only what's actually needed
            doc_dict["owner_id"] = auth.user_id or "system"
            doc_dict["app_id"] = auth.app_id  # Primary access control in cloud mode

            # The flattened fields are already in doc_dict from the Document model

            async with self.async_session() as session:
                doc_model = DocumentModel(**doc_dict)
                session.add(doc_model)
                await session.commit()
            return True

        except TypedMetadataError as exc:
            logger.error("Invalid typed metadata for document %s: %s", document.external_id, exc)
            raise
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
                    return Document(**_document_model_to_dict(doc_model))
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
                    return Document(**_document_model_to_dict(doc_model))
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
                    documents.append(Document(**_document_model_to_dict(doc_model)))

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

                return [Document(**_document_model_to_dict(doc)) for doc in doc_models]

        except InvalidMetadataFilterError as exc:
            logger.warning("Invalid metadata filter while listing documents: %s", exc)
            raise
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    async def list_documents_flexible(
        self,
        auth: AuthContext,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
        status_filter: Optional[List[str]] = None,
        include_total_count: bool = False,
        include_status_counts: bool = False,
        include_folder_counts: bool = False,
        return_documents: bool = True,
        sort_by: Optional[str] = None,
        sort_direction: str = "desc",
    ) -> Dict[str, Any]:
        """List documents with optional aggregate metadata. Field projection is handled at application layer."""
        limit = max(limit, 0) if limit is not None else None
        skip = max(skip, 0)

        try:
            async with self.async_session() as session:
                access_filter = self._build_access_filter_optimized(auth)
                metadata_filter = self._build_metadata_filter(filters)
                system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)
                filter_params = self._build_filter_params(auth, system_filters)

                where_clauses = [f"({access_filter})"]
                if metadata_filter:
                    where_clauses.append(f"({metadata_filter})")
                if system_metadata_filter:
                    where_clauses.append(f"({system_metadata_filter})")
                if status_filter:
                    status_clauses: List[str] = []
                    include_null_status = any(item is None for item in status_filter)
                    normalized_statuses = [item for item in status_filter if item is not None]

                    for idx, status_value in enumerate(normalized_statuses):
                        param_name = f"status_filter_{idx}"
                        filter_params[param_name] = str(status_value)
                        status_clauses.append(f"(system_metadata->>'status') = :{param_name}")

                    if include_null_status:
                        status_clauses.append("(system_metadata->>'status') IS NULL")

                    if status_clauses:
                        where_clauses.append("(" + " OR ".join(status_clauses) + ")")

                final_where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

                documents: List[Document] = []
                returned_count = 0
                has_more = False

                fetch_documents = return_documents and (limit is None or limit > 0)

                if fetch_documents:
                    # Note: We always select all columns from the database
                    # Field projection is handled at the application layer for simplicity
                    base_query = select(DocumentModel).where(text(final_where_clause).bindparams(**filter_params))
                    order_clause = self._resolve_document_sort_clause(sort_by, sort_direction)
                    if order_clause is not None:
                        base_query = base_query.order_by(order_clause, DocumentModel.external_id.asc())
                    else:
                        base_query = base_query.order_by(DocumentModel.external_id.asc())

                    fetch_limit = limit + 1 if limit is not None else None
                    base_query = base_query.offset(skip)
                    if fetch_limit is not None:
                        base_query = base_query.limit(fetch_limit)

                    result = await session.execute(base_query)
                    doc_models = result.scalars().all()

                    if fetch_limit is not None and len(doc_models) > limit:
                        has_more = True
                        doc_models = doc_models[:limit]

                    documents = [Document(**_document_model_to_dict(doc_model)) for doc_model in doc_models]
                    returned_count = len(documents)

                total_count: Optional[int] = None
                if include_total_count:
                    count_query = text(f"SELECT COUNT(*) FROM documents WHERE {final_where_clause}")
                    count_result = await session.execute(count_query, filter_params)
                    total_count = count_result.scalar_one() if count_result is not None else 0
                    has_more = skip + returned_count < total_count if fetch_documents else skip < total_count

                status_counts: Optional[Dict[str, int]] = None
                if include_status_counts:
                    status_query = text(
                        f"""
                        SELECT COALESCE(NULLIF(system_metadata->>'status', ''), 'unknown') AS status,
                               COUNT(*) AS count
                        FROM documents
                        WHERE {final_where_clause}
                        GROUP BY status
                        """
                    )
                    status_result = await session.execute(status_query, filter_params)
                    status_counts = {}
                    for row in status_result.mappings():
                        status_value = row.get("status") or "unknown"
                        status_counts[status_value] = row.get("count", 0)

                folder_counts: Optional[List[Dict[str, Any]]] = None
                if include_folder_counts:
                    folder_query = text(
                        f"""
                        SELECT COALESCE(folder_path, folder_name) AS folder_name, COUNT(*) AS count
                        FROM documents
                        WHERE {final_where_clause}
                        GROUP BY COALESCE(folder_path, folder_name)
                        ORDER BY folder_name NULLS FIRST
                        """
                    )
                    folder_result = await session.execute(folder_query, filter_params)
                    folder_counts = [
                        {"folder": row.get("folder_name"), "count": row.get("count", 0)}
                        for row in folder_result.mappings()
                    ]

                if include_total_count and total_count is not None:
                    next_skip = (
                        skip + returned_count if fetch_documents and (skip + returned_count) < total_count else None
                    )
                elif has_more and fetch_documents:
                    next_skip = skip + returned_count
                else:
                    next_skip = None

                return {
                    "documents": documents if fetch_documents else [],
                    "returned_count": returned_count if fetch_documents else 0,
                    "total_count": total_count,
                    "status_counts": status_counts,
                    "folder_counts": folder_counts,
                    "has_more": has_more,
                    "next_skip": next_skip,
                }

        except InvalidMetadataFilterError as exc:
            logger.warning("Invalid metadata filter while listing documents with aggregates: %s", exc)
            raise
        except Exception as e:
            logger.error(f"Error listing documents with aggregates: {str(e)}")
            return {
                "documents": [],
                "returned_count": 0,
                "total_count": None,
                "status_counts": None,
                "folder_counts": None,
                "has_more": False,
                "next_skip": None,
            }

    def _resolve_document_sort_clause(self, sort_by: Optional[str], sort_direction: str):
        """Resolve ORDER BY clause for flexible document listings."""
        direction = "ASC" if (sort_direction or "").lower() == "asc" else "DESC"
        normalized_sort = (sort_by or "updated_at").lower()

        if normalized_sort == "filename":
            return text(f"filename {direction} NULLS LAST")
        if normalized_sort == "external_id":
            return text(f"external_id {direction}")
        if normalized_sort == "created_at":
            return text(
                "COALESCE((system_metadata->>'created_at')::timestamptz, "
                "(system_metadata->>'updated_at')::timestamptz) "
                f"{direction} NULLS LAST"
            )

        return text(
            "COALESCE((system_metadata->>'updated_at')::timestamptz, "
            "(system_metadata->>'created_at')::timestamptz) "
            f"{direction} NULLS LAST"
        )

    async def update_document(self, document_id: str, updates: Dict[str, Any], auth: AuthContext) -> bool:
        """Update document metadata if user has write access."""
        try:
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

            # Keep folder writes consistent: if folder_id is set, ensure folder_name/path are populated
            if "folder_id" in updates:
                folder_id_value = updates.get("folder_id")
                if folder_id_value:
                    needs_folder_name = updates.get("folder_name") in (None, "") if "folder_name" in updates else True
                    needs_folder_path = updates.get("folder_path") in (None, "") if "folder_path" in updates else True
                    if needs_folder_name or needs_folder_path:
                        folder_for_update = await self.get_folder(folder_id_value, auth)
                        if not folder_for_update:
                            logger.error(
                                "Folder %s not found or inaccessible while updating document %s",
                                folder_id_value,
                                document_id,
                            )
                            return False
                        if needs_folder_name:
                            updates["folder_name"] = folder_for_update.name
                        if needs_folder_path:
                            try:
                                updates["folder_path"] = folder_for_update.full_path or normalize_folder_path(
                                    folder_for_update.name
                                )
                            except ValueError:
                                updates["folder_path"] = folder_for_update.name
                else:
                    updates.setdefault("folder_name", None)
                    updates.setdefault("folder_path", None)

            # Keep folder_path aligned with folder_name when provided and no path supplied
            folder_name_for_alignment = updates["folder_name"] if "folder_name" in updates else existing_doc.folder_name
            if "folder_name" in updates and "folder_path" not in updates:
                if folder_name_for_alignment:
                    try:
                        updates["folder_path"] = normalize_folder_path(folder_name_for_alignment)
                    except ValueError:
                        updates["folder_path"] = folder_name_for_alignment
                else:
                    updates["folder_path"] = None

            # Decide which folder value to mirror into doc_metadata (prefer path)
            if "folder_path" in updates:
                folder_value_for_metadata = updates.get("folder_path")
            elif "folder_name" in updates:
                folder_value_for_metadata = updates.get("folder_name")
            else:
                folder_value_for_metadata = existing_doc.folder_path or existing_doc.folder_name

            # Serialize datetime objects to ISO format strings
            updates = _serialize_datetime(updates)

            if "metadata" in updates:
                logger.info("Converting 'metadata' to 'doc_metadata' for database update")
                metadata_payload = updates.pop("metadata") or {}
                metadata_payload.setdefault("external_id", document_id)
                metadata_type_hints = updates.pop("metadata_types", {}) or {}
                normalized_metadata, normalized_types = normalize_metadata(metadata_payload, metadata_type_hints)
                updates["doc_metadata"] = normalized_metadata
                updates["metadata_types"] = normalized_types

            async with self.async_session() as session:
                async with session.begin():
                    doc_model = await self._fetch_document_locked(session, document_id)

                    if not doc_model:
                        return False

                    if not self._has_document_access(doc_model, auth):
                        logger.error("User does not have write access to document %s", document_id)
                        return False
                    # Log what we're updating
                    logger.info(f"Document update: updating fields {list(updates.keys())}")

                    # The flattened fields (owner_id, app_id)
                    # should be in updates directly if they need to be updated

                    # Keep doc_metadata folder fields in sync with flattened columns (support clearing)
                    doc_metadata_update = updates.get("doc_metadata") if "doc_metadata" in updates else None
                    has_folder_change = any(key in updates for key in ("folder_name", "folder_path", "folder_id"))

                    if doc_metadata_update is not None:
                        folder_value = updates.get("folder_path", folder_value_for_metadata)
                        if folder_value is None:
                            folder_value = doc_model.folder_path or doc_model.folder_name
                        try:
                            if isinstance(doc_metadata_update, dict):
                                doc_metadata_update = dict(doc_metadata_update)
                                doc_metadata_update["folder_name"] = folder_value
                                if "folder_id" in updates:
                                    doc_metadata_update["folder_id"] = updates["folder_id"]
                                updates["doc_metadata"] = doc_metadata_update
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Unable to set folder fields in doc_metadata for %s: %s", document_id, exc)
                    elif has_folder_change:
                        new_doc_metadata = dict(doc_model.doc_metadata or {})
                        folder_value = updates.get("folder_path", folder_value_for_metadata)
                        if folder_value is None:
                            folder_value = doc_model.folder_path or doc_model.folder_name
                        new_doc_metadata["folder_name"] = folder_value
                        if "folder_id" in updates:
                            new_doc_metadata["folder_id"] = updates["folder_id"]
                        updates["doc_metadata"] = new_doc_metadata

                    # Set all attributes
                    for key, value in updates.items():
                        setattr(doc_model, key, value)

                    await session.commit()
                    logger.info(f"Document {document_id} updated successfully")
                    return True

        except TypedMetadataError as exc:
            logger.error("Invalid typed metadata for document %s: %s", document_id, exc)
            raise
        except Exception as e:
            logger.error(f"Error updating document metadata: {str(e)}")
            return False

    async def delete_document(self, document_id: str, auth: AuthContext) -> bool:
        """Delete document if user has write access."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    doc_model = await self._fetch_document_locked(session, document_id)

                    if not doc_model:
                        return False

                    if not self._has_document_access(doc_model, auth):
                        logger.error("User does not have write access to document %s", document_id)
                        return False

                    await session.delete(doc_model)

                    # Maintain referential integrity inside the same transaction
                    await session.execute(
                        text(
                            """
                            UPDATE folders
                            SET document_ids = COALESCE(document_ids, '[]'::jsonb) - :doc_id
                            WHERE document_ids ? :doc_id
                            """
                        ),
                        {"doc_id": document_id},
                    )

                    logger.info("Deleted document %s and removed folder references", document_id)
                    return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    async def find_authorized_and_filtered_documents(
        self,
        auth: AuthContext,
        filters: Optional[Dict[str, Any]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
        status_filter: Optional[List[str]] = None,
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

                if status_filter:
                    status_clauses = []
                    status_params: Dict[str, Any] = {}
                    for idx, status in enumerate(status_filter):
                        if status is None:
                            status_clauses.append("(system_metadata->>'status') IS NULL")
                        else:
                            param_name = f"status_filter_{idx}"
                            status_clauses.append(f"(system_metadata->>'status') = :{param_name}")
                            status_params[param_name] = str(status)

                    if status_clauses:
                        where_clauses.append("(" + " OR ".join(status_clauses) + ")")
                        filter_params.update(status_params)

                final_where_clause = " AND ".join(where_clauses)
                query = select(DocumentModel.external_id).where(text(final_where_clause).bindparams(**filter_params))

                logger.debug(f"Final query: {query}")

                result = await session.execute(query)
                doc_ids = [row[0] for row in result.all()]
                logger.debug(f"Found document IDs: {doc_ids}")
                return doc_ids

        except InvalidMetadataFilterError as exc:
            logger.warning("Invalid metadata filter while finding documents: %s", exc)
            raise
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
                return doc_model.owner_id == auth.user_id

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
        return "owner_id = :user_id"

    def _build_metadata_filter(self, filters: Optional[Dict[str, Any]]) -> str:
        """Delegate metadata filtering to the shared builder (supports arrays, regex, substring operators)."""
        return self._metadata_filter_builder.build(filters)

    def _build_system_metadata_filter_optimized(self, system_filters: Optional[Dict[str, Any]]) -> str:
        """Build PostgreSQL filter for system metadata using flattened columns.

        - Uses direct column access (e.g. folder_name, end_user_id) for performance
        - Backward-compatibility: treat empty string as NULL for folder_name/end_user_id
          since some legacy rows may have "" instead of NULL in flattened columns.

        Returns a SQL string with named parameters like :app_id_0, :folder_name_0, etc.
        The caller must also supply parameter values via ``_build_filter_params``.
        """
        if not system_filters:
            return ""

        key_clauses: List[str] = []
        param_counter = 0  # Local counter for thread safety

        for key, value in system_filters.items():
            if key == "folder_path_prefix":
                values = value if isinstance(value, list) else [value]
                if not values and value is not None:
                    continue

                prefix_clauses: List[str] = []
                for item in values:
                    if item is None:
                        prefix_clauses.append("(folder_path IS NULL OR folder_path = '')")
                        continue

                    param_eq = f"{key}_{param_counter}"
                    param_like = f"{param_eq}_like"
                    param_counter += 1
                    prefix_clauses.append(f"(folder_path = :{param_eq} OR folder_path LIKE :{param_like})")
                if prefix_clauses:
                    key_clauses.append("(" + " OR ".join(prefix_clauses) + ")")
                continue

            if key == "folder_path_prefix_depth":
                entries = value if isinstance(value, list) else [value]
                if not entries and value is not None:
                    continue

                scoped_clauses: List[str] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    prefix_val = entry.get("prefix")
                    max_depth = entry.get("max_depth")
                    if prefix_val is None:
                        continue
                    param_prefix = f"{key}_{param_counter}"
                    param_like = f"{param_prefix}_like"
                    clause = f"(folder_path = :{param_prefix} OR folder_path LIKE :{param_like})"
                    if max_depth is not None:
                        depth_param = f"{param_prefix}_depth"
                        clause = f"({clause} AND array_length(string_to_array(trim(BOTH '/' from folder_path), '/'), 1) <= :{depth_param})"
                    scoped_clauses.append(clause)
                    param_counter += 1

                if scoped_clauses:
                    key_clauses.append("(" + " OR ".join(scoped_clauses) + ")")
                continue

            if key not in self._SYSTEM_FILTER_COLUMNS:
                continue

            column = self._SYSTEM_FILTER_COLUMNS[key]
            values = value if isinstance(value, list) else [value]
            if not values and value is not None:
                continue

            value_clauses = []
            for item in values:
                if item is None:
                    # Backward-compat: for folder_name/folder_path/end_user_id, also match empty string values which
                    # historically represented "no folder/user" in some datasets.
                    if column in ("folder_name", "folder_path", "end_user_id"):
                        value_clauses.append(f"({column} IS NULL OR {column} = '')")
                    else:
                        value_clauses.append(f"{column} IS NULL")
                else:
                    # Use named parameter instead of string interpolation
                    param_name = f"{key}_{param_counter}"
                    value_clauses.append(f"{column} = :{param_name}")
                    param_counter += 1

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
        elif auth.user_id:
            params["user_id"] = auth.user_id

        # Add system metadata filter parameters
        if system_filters:
            param_counter = 0  # Local counter for thread safety
            for key, value in system_filters.items():
                if key == "folder_path_prefix":
                    values = value if isinstance(value, list) else [value]
                    if not values and value is not None:
                        continue
                    for item in values:
                        if item is None:
                            continue
                        param_name = f"{key}_{param_counter}"
                        params[param_name] = str(item)
                        params[f"{param_name}_like"] = f"{str(item).rstrip('/')}/%"
                        param_counter += 1
                    continue

                if key == "folder_path_prefix_depth":
                    entries = value if isinstance(value, list) else [value]
                    if not entries and value is not None:
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        prefix_val = entry.get("prefix")
                        max_depth = entry.get("max_depth")
                        if prefix_val is None:
                            continue
                        param_name = f"{key}_{param_counter}"
                        params[param_name] = str(prefix_val)
                        params[f"{param_name}_like"] = f"{str(prefix_val).rstrip('/')}/%"
                        if max_depth is not None:
                            params[f"{param_name}_depth"] = int(max_depth)
                        param_counter += 1
                    continue

                if key not in self._SYSTEM_FILTER_COLUMNS:
                    continue

                values = value if isinstance(value, list) else [value]
                if not values and value is not None:
                    continue

                for item in values:
                    if item is not None:
                        param_name = f"{key}_{param_counter}"
                        params[param_name] = str(item)
                        param_counter += 1

        return params

    async def _fetch_document_locked(self, session: AsyncSession, document_id: str) -> Optional[DocumentModel]:
        """Fetch a document row with a FOR UPDATE lock."""
        result = await session.execute(
            select(DocumentModel).where(DocumentModel.external_id == document_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def _fetch_folder_locked(self, session: AsyncSession, folder_id: str) -> Optional[FolderModel]:
        """Fetch a folder row with a FOR UPDATE lock."""
        result = await session.execute(select(FolderModel).where(FolderModel.id == folder_id).with_for_update())
        return result.scalar_one_or_none()

    def _has_document_access(self, doc_model: DocumentModel, auth: AuthContext) -> bool:
        """Check access against a fetched DocumentModel."""
        return doc_model.app_id == auth.app_id if auth.app_id else doc_model.owner_id == auth.user_id

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

            # Keep folder_path aligned with folder_name for backward compatibility
            if graph_dict.get("folder_path") is None and graph_dict.get("folder_name"):
                try:
                    graph_dict["folder_path"] = normalize_folder_path(graph_dict["folder_name"])
                except ValueError:
                    graph_dict["folder_path"] = graph_dict["folder_name"]

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
            graph_dict["owner_id"] = auth.user_id or "system"
            graph_dict["app_id"] = auth.app_id  # Primary access control in cloud mode

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
                    # Sanitize: filter out None, empty strings, and non-string values
                    raw_doc_ids = graph_model.document_ids or []
                    document_ids = [doc_id for doc_id in raw_doc_ids if doc_id and isinstance(doc_id, str)]
                    invalid_count = len(raw_doc_ids) - len(document_ids)
                    if invalid_count > 0:
                        logger.warning(
                            "Graph %s has %d invalid document_id values (None/empty/non-string)",
                            graph_model.id,
                            invalid_count,
                        )

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
                    graph_dict = _graph_model_to_dict(graph_model)
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
                access_filter = self._build_access_filter_optimized(auth)
                filter_params = self._build_filter_params(auth)

                query = select(GraphModel).where(text(f"({access_filter})").bindparams(**filter_params))
                result = await session.execute(query)
                graph_models = result.scalars().all()

                graphs: List[Graph] = []

                if system_filters:
                    system_metadata_filter = self._build_system_metadata_filter_optimized(system_filters)

                    # If no recognized system filters were provided, preserve the previous behavior (empty result)
                    if not system_metadata_filter:
                        return []

                    # Batch-fetch all document IDs that satisfy the system filter to avoid N+1 queries
                    system_params = self._build_filter_params(auth, system_filters)
                    # Filter out None, empty strings, and non-string values to prevent driver type errors
                    total_raw_count = sum(len(gm.document_ids or []) for gm in graph_models)
                    all_doc_ids = {
                        doc_id
                        for graph_model in graph_models
                        for doc_id in (graph_model.document_ids or [])
                        if doc_id and isinstance(doc_id, str)
                    }
                    invalid_count = total_raw_count - len(all_doc_ids)
                    if invalid_count > 0:
                        logger.warning(
                            "Found %d invalid document_id values across %d graphs (None/empty/non-string)",
                            invalid_count,
                            len(graph_models),
                        )

                    allowed_ids: set[str] = set()
                    if all_doc_ids:
                        system_params["doc_ids"] = list(all_doc_ids)
                        filter_query = text(
                            f"""
                            WITH filtered_docs AS (
                                SELECT external_id
                                FROM documents
                                WHERE external_id = ANY(:doc_ids)
                                AND ({system_metadata_filter})
                            )
                            SELECT external_id FROM filtered_docs
                            """
                        )
                        filter_result = await session.execute(filter_query.bindparams(**system_params))
                        allowed_ids = {row[0] for row in filter_result.all()}

                    for graph_model in graph_models:
                        filtered_doc_ids = [
                            doc_id for doc_id in (graph_model.document_ids or []) if doc_id in allowed_ids
                        ]
                        if not filtered_doc_ids:
                            continue

                        graph_dict = _graph_model_to_dict(graph_model)
                        graph_dict["document_ids"] = filtered_doc_ids
                        graph_dict["created_at"] = _parse_datetime_field(graph_model.created_at)
                        graph_dict["updated_at"] = _parse_datetime_field(graph_model.updated_at)
                        graphs.append(Graph(**graph_dict))
                else:
                    for graph_model in graph_models:
                        graph_dict = _graph_model_to_dict(graph_model)
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

            if graph_dict.get("folder_path") is None and graph_dict.get("folder_name"):
                try:
                    graph_dict["folder_path"] = normalize_folder_path(graph_dict["folder_name"])
                except ValueError:
                    graph_dict["folder_path"] = graph_dict["folder_name"]

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
            # Note: owner_id and app_id should not be updated here
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
                    if graph_model.owner_id != auth.user_id:
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

                # Derive canonical full_path and depth (single-level folders default to depth=1)
                try:
                    canonical_path = normalize_folder_path(folder_dict.get("full_path") or folder_dict.get("name"))
                except ValueError as exc:
                    logger.error("Invalid folder path '%s': %s", folder_dict.get("full_path"), exc)
                    return False

                folder_dict["full_path"] = canonical_path
                folder.full_path = canonical_path

                if folder_dict.get("depth") is None:
                    segments = canonical_path.strip("/").split("/") if canonical_path and canonical_path != "/" else []
                    folder_depth = len(segments) if canonical_path != "/" else 0
                    if canonical_path != "/" and folder_depth == 0:
                        folder_depth = 1
                    folder_dict["depth"] = folder_depth
                    folder.depth = folder_depth

                # Convert datetime objects to strings for JSON serialization
                folder_dict = _serialize_datetime(folder_dict)

                # Simplified owner info
                owner_id = auth.user_id or "system"
                app_id_val = auth.app_id or folder_dict.get("app_id")

                # Check for existing folder with same full_path (scoped by app or owner, matching uniqueness rules)
                if app_id_val:
                    params = {"full_path": canonical_path, "app_id": app_id_val}
                    stmt = text(
                        """
                        SELECT id FROM folders
                        WHERE full_path = :full_path
                        AND app_id = :app_id
                        """
                    )
                else:
                    params = {"full_path": canonical_path, "owner_id": owner_id}
                    stmt = text(
                        """
                        SELECT id FROM folders
                        WHERE full_path = :full_path
                        AND app_id IS NULL
                        AND owner_id = :owner_id
                        """
                    )

                result = await session.execute(stmt.bindparams(**params))
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
                    full_path=folder_dict.get("full_path"),
                    parent_id=folder_dict.get("parent_id"),
                    depth=folder_dict.get("depth"),
                    description=folder.description,
                    owner_id=owner_id,
                    document_ids=folder_dict.get("document_ids", []),
                    system_metadata=folder_dict.get("system_metadata", {}),
                    app_id=app_id_val,
                    end_user_id=folder_dict.get("end_user_id"),
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

                # Check if the user has access to the folder using the model
                if not self._check_folder_model_access(folder_model, auth):
                    return None

                folder = Folder(**_folder_row_to_dict(folder_model))
                return folder

        except Exception as e:
            logger.error(f"Error getting folder: {e}")
            return None

    async def get_folder_by_name(self, name: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by name."""
        try:
            async with self.async_session() as session:
                normalized_full_path = None
                try:
                    normalized_full_path = normalize_folder_path(name)
                except Exception:
                    normalized_full_path = None

                # Build query based on auth context
                params = {"name": name, "full_path": normalized_full_path}

                if auth.app_id:
                    # Filter by app_id in cloud mode
                    if normalized_full_path:
                        stmt = text(
                            """
                            SELECT * FROM folders
                            WHERE full_path = :full_path
                            AND app_id = :app_id
                        """
                        )
                        params["app_id"] = auth.app_id
                    else:
                        stmt = text(
                            """
                            SELECT * FROM folders
                            WHERE name = :name
                            AND app_id = :app_id
                        """
                        )
                        params["app_id"] = auth.app_id
                elif auth.user_id:
                    # Filter by owner_id in dev/self-hosted mode
                    if normalized_full_path:
                        stmt = text(
                            """
                            SELECT * FROM folders
                            WHERE full_path = :full_path
                            AND owner_id = :owner_id
                        """
                        )
                        params["owner_id"] = auth.user_id
                    else:
                        stmt = text(
                            """
                            SELECT * FROM folders
                            WHERE name = :name
                            AND owner_id = :owner_id
                        """
                        )
                        params["owner_id"] = auth.user_id
                else:
                    # No access without auth
                    return None

                result = await session.execute(stmt.bindparams(**params))
                folder_row = result.fetchone()

                if folder_row:
                    return Folder(**_folder_row_to_dict(folder_row))

                return None

        except Exception as e:
            logger.error(f"Error getting folder by name: {e}")
            return None

    async def get_folder_by_full_path(self, full_path: str, auth: AuthContext) -> Optional[Folder]:
        """Get a folder by canonical full_path."""
        try:
            normalized_full_path = normalize_folder_path(full_path)
            async with self.async_session() as session:
                params: Dict[str, Any] = {"full_path": normalized_full_path}
                if auth.app_id:
                    stmt = text(
                        """
                        SELECT * FROM folders
                        WHERE full_path = :full_path
                        AND app_id = :app_id
                    """
                    )
                    params["app_id"] = auth.app_id
                elif auth.user_id:
                    stmt = text(
                        """
                        SELECT * FROM folders
                        WHERE full_path = :full_path
                        AND owner_id = :owner_id
                    """
                    )
                    params["owner_id"] = auth.user_id
                else:
                    return None

                result = await session.execute(stmt.bindparams(**params))
                folder_row = result.fetchone()
                if not folder_row:
                    return None

                return Folder(**_folder_row_to_dict(folder_row))
        except Exception as e:
            logger.error(f"Error getting folder by full_path: {e}")
            return None

    async def list_folders(self, auth: AuthContext, system_filters: Optional[Dict[str, Any]] = None) -> List[Folder]:
        """List all folders the user has access to using flattened columns."""
        try:
            current_params: Dict[str, Any] = {}

            # Simplified access control - same as documents/graphs
            if auth.app_id:
                # Filter by app_id when present (cloud mode)
                access_condition = "app_id = :app_id_val"
                current_params["app_id_val"] = auth.app_id
            elif auth.user_id:
                # Filter by owner_id as fallback (dev/self-hosted mode)
                access_condition = "owner_id = :owner_id_val"
                current_params["owner_id_val"] = auth.user_id
            else:
                # No access if no auth context
                access_condition = "1=0"

            # Build and execute query
            async with self.async_session() as session:
                # Prefetch child counts to populate Folder.child_count
                child_counts_result = await session.execute(
                    text(
                        f"""
                        SELECT parent_id, COUNT(*) AS cnt
                        FROM folders
                        WHERE parent_id IS NOT NULL AND ({access_condition})
                        GROUP BY parent_id
                        """
                    ),
                    current_params,
                )
                child_counts = {row.parent_id: row.cnt for row in child_counts_result.mappings()}

                query = select(FolderModel).where(text(access_condition))
                result = await session.execute(query, current_params)
                folder_models = result.scalars().all()

                folders = []
                for folder_model in folder_models:
                    folder_dict = {
                        "id": folder_model.id,
                        "name": folder_model.name,
                        "full_path": folder_model.full_path,
                        "parent_id": folder_model.parent_id,
                        "depth": folder_model.depth,
                        "description": folder_model.description,
                        "document_ids": folder_model.document_ids,
                        "system_metadata": folder_model.system_metadata,
                        "app_id": folder_model.app_id,
                        "end_user_id": folder_model.end_user_id,
                        "child_count": child_counts.get(folder_model.id, 0),
                    }
                    folders.append(Folder(**folder_dict))
                return folders

        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []

    async def add_document_to_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Add a document to a folder."""
        import asyncio

        class _RetryableMissingDocument(Exception):
            """Internal sentinel to retry when a document hasn't been committed yet."""

            pass

        max_retries = 3
        retry_delay = 0.5  # Start with 500ms delay for transient visibility issues

        for attempt in range(max_retries):
            try:
                async with self.async_session() as session:
                    # Enforce a single transactional flow with row locks to avoid lost updates
                    async with session.begin():
                        folder_model = await self._fetch_folder_locked(session, folder_id)
                        if not folder_model:
                            logger.error(f"Folder {folder_id} not found")
                            return False

                        if not self._check_folder_model_access(folder_model, auth):
                            logger.error(f"User does not have write access to folder {folder_id}")
                            return False

                        doc_model = await self._fetch_document_locked(session, document_id)
                        if not doc_model:
                            raise _RetryableMissingDocument

                        if not self._has_document_access(doc_model, auth):
                            logger.error(f"User does not have access to document {document_id}")
                            return False

                        # Compute folder_path_value early so we can check alignment
                        try:
                            folder_path_value = folder_model.full_path or (
                                normalize_folder_path(folder_model.name) if folder_model.name else None
                            )
                        except ValueError:
                            folder_path_value = folder_model.name

                        # Early success if everything is already aligned (including folder_path)
                        current_folder_ids = folder_model.document_ids or []
                        if (
                            doc_model.folder_id == folder_id
                            and document_id in current_folder_ids
                            and doc_model.folder_path == folder_path_value
                        ):
                            logger.info(f"Document {document_id} is already in folder {folder_id}")
                            return True

                        # Remove the document from its previous folder (if any) using an atomic JSONB update
                        if doc_model.folder_id and doc_model.folder_id != folder_id:
                            await session.execute(
                                text(
                                    """
                                    UPDATE folders
                                    SET document_ids = COALESCE(document_ids, '[]'::jsonb) - :doc_id
                                    WHERE id = :previous_folder_id
                                    """
                                ),
                                {"doc_id": document_id, "previous_folder_id": doc_model.folder_id},
                            )

                        # Append to the target folder with the lock held to avoid lost updates
                        existing_ids = folder_model.document_ids or []
                        folder_model.document_ids = list(dict.fromkeys(existing_ids + [document_id]))

                        # Update the document's folder references and keep metadata aligned
                        # (folder_path_value was computed earlier for the alignment check)
                        doc_model.folder_id = folder_id
                        doc_model.folder_name = folder_model.name
                        doc_model.folder_path = folder_path_value

                        updated_metadata = dict(doc_model.doc_metadata or {})
                        updated_metadata["folder_name"] = folder_path_value
                        updated_metadata["folder_id"] = folder_id
                        doc_model.doc_metadata = updated_metadata

                        updated_system_metadata = dict(doc_model.system_metadata or {})
                        updated_system_metadata["updated_at"] = datetime.now(UTC)
                        doc_model.system_metadata = _serialize_datetime(updated_system_metadata)

                        logger.info(f"Added document {document_id} to folder {folder_id}")
                        return True

            except _RetryableMissingDocument:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Document {document_id} not found on attempt {attempt + 1}/{max_retries}, "
                        f"retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                logger.error(
                    f"Document {document_id} not found or user does not have access after {max_retries} attempts"
                )
                return False
            except Exception as e:
                logger.error(f"Error adding document to folder: {e}")
                return False

    async def remove_document_from_folder(self, folder_id: str, document_id: str, auth: AuthContext) -> bool:
        """Remove a document from a folder."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    # Lock the folder row to prevent concurrent modifications
                    folder_model = await self._fetch_folder_locked(session, folder_id)
                    if not folder_model:
                        logger.error(f"Folder {folder_id} not found")
                        return False

                    # Check if user has write access to the folder
                    if not self._check_folder_model_access(folder_model, auth):
                        logger.error(f"User does not have write access to folder {folder_id}")
                        return False

                    # Get document to check if we need to clear folder references
                    doc_model = await self._fetch_document_locked(session, document_id)
                    if not doc_model:
                        logger.error(f"Document {document_id} not found while removing from folder {folder_id}")
                        return False

                    # Check access to document
                    if not self._has_document_access(doc_model, auth):
                        logger.error(f"User does not have access to document {document_id}")
                        return False

                    should_clear_folder = doc_model.folder_id == folder_id
                    current_doc_ids = folder_model.document_ids or []

                    # Check if the document is in the folder (or recorded as such)
                    if document_id not in current_doc_ids and doc_model.folder_id != folder_id:
                        logger.warning(f"Tried to delete document {document_id} not in folder {folder_id}")
                        return True

                    # Remove document_id from document_ids array (filter invalid values too)
                    new_document_ids = [
                        doc_id
                        for doc_id in current_doc_ids
                        if doc_id and isinstance(doc_id, str) and doc_id != document_id
                    ]
                    # Check for invalid values being cleaned up (excluding the one being removed)
                    expected_count = len(current_doc_ids) - (1 if document_id in current_doc_ids else 0)
                    invalid_count = expected_count - len(new_document_ids)
                    if invalid_count > 0:
                        logger.warning(
                            "Folder %s had %d invalid document_id values (None/empty/non-string), cleaned up",
                            folder_id,
                            invalid_count,
                        )
                    folder_model.document_ids = new_document_ids

                    # Clear folder references on the document if it was attached to this folder
                    if should_clear_folder:
                        doc_model.folder_name = None
                        doc_model.folder_path = None
                        doc_model.folder_id = None
                        updated_metadata = dict(doc_model.doc_metadata or {})
                        updated_metadata["folder_name"] = None
                        updated_metadata["folder_id"] = None
                        doc_model.doc_metadata = updated_metadata

                    await session.commit()
                    logger.info(f"Removed document {document_id} from folder {folder_id}")
                    return True

        except Exception as e:
            logger.error(f"Error removing document from folder: {e}")
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
        return folder_model.owner_id == auth.user_id

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
            params: Dict[str, Any] = {}
            if auth.app_id:
                doc_access_condition = "d.app_id = :app_id_val"
                params["app_id_val"] = auth.app_id
            elif auth.user_id:
                doc_access_condition = "d.owner_id = :owner_id_val"
                params["owner_id_val"] = auth.user_id
            else:
                doc_access_condition = "1=0"

            # Build folder access condition (same logic as doc access)
            if auth.app_id:
                folder_access_condition = "f.app_id = :app_id_val"
            elif auth.user_id:
                folder_access_condition = "f.owner_id = :owner_id_val"
            else:
                folder_access_condition = "1=0"

            async with self.async_session() as session:
                # Single query: fetch folder summaries with doc counts via LEFT JOIN
                # This avoids loading the heavy document_ids JSONB array entirely
                result = await session.execute(
                    text(
                        f"""
                        SELECT
                            f.id,
                            f.name,
                            f.full_path,
                            f.depth,
                            f.description,
                            f.system_metadata->>'updated_at' AS updated_at,
                            COALESCE(dc.cnt, 0) AS doc_count
                        FROM folders f
                        LEFT JOIN (
                            SELECT COALESCE(d.folder_id, f2.id) AS fid, COUNT(*) AS cnt
                            FROM documents d
                            LEFT JOIN folders f2
                                ON d.folder_path IS NOT NULL
                                AND d.folder_path <> ''
                                AND f2.full_path = d.folder_path
                                AND (f2.app_id IS NOT DISTINCT FROM d.app_id)
                            WHERE (d.folder_id IS NOT NULL OR (d.folder_path IS NOT NULL AND d.folder_path <> ''))
                            AND ({doc_access_condition})
                            GROUP BY COALESCE(d.folder_id, f2.id)
                        ) dc ON dc.fid = f.id
                        WHERE {folder_access_condition}
                        ORDER BY f.name
                        """
                    ),
                    params,
                )

                summaries: List[Dict[str, Any]] = []
                for row in result.mappings():
                    summaries.append(
                        {
                            "id": row.id,
                            "name": row.name,
                            "full_path": row.full_path,
                            "depth": row.depth,
                            "description": row.description,
                            "updated_at": row.updated_at,
                            "doc_count": row.doc_count,
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

                # Convert to Document objects using serializer function
                documents = [Document(**_document_model_to_dict(doc)) for doc in doc_models]

                logger.debug(f"Document name search for '{clean_query}' returned {len(documents)} results")
                return documents

        except InvalidMetadataFilterError as exc:
            logger.warning("Invalid metadata filter while searching documents: %s", exc)
            raise
        except Exception as e:
            logger.error(f"Error searching documents by name: {str(e)}")
            return []
