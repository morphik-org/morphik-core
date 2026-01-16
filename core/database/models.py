import logging

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentModel(Base):
    """SQLAlchemy model for document metadata."""

    __tablename__ = "documents"

    external_id = Column(String, primary_key=True)
    content_type = Column(String)
    filename = Column(String, nullable=True)
    doc_metadata = Column(JSONB, default=dict)
    metadata_types = Column(JSONB, default=dict)
    storage_info = Column(JSONB, default=dict)
    system_metadata = Column(JSONB, default=dict)
    additional_metadata = Column(JSONB, default=dict)
    chunk_ids = Column(JSONB, default=list)

    # Flattened auth columns for performance
    owner_id = Column(String)
    app_id = Column(String)
    folder_name = Column(String)
    folder_path = Column(String)
    folder_id = Column(String)
    end_user_id = Column(String)

    __table_args__ = (
        Index("idx_system_metadata", "system_metadata", postgresql_using="gin"),
        Index("idx_doc_metadata_gin", "doc_metadata", postgresql_using="gin"),
        # Primary access control indexes (used in every query)
        Index("idx_doc_app_id", "app_id"),
        Index("idx_doc_owner_id", "owner_id"),
        # Composite indexes for scoped queries (app_id/owner_id first, then filter field)
        Index("idx_documents_owner_app", "owner_id", "app_id"),
        Index("idx_documents_app_folder", "app_id", "folder_name"),
        Index("idx_documents_app_folder_path", "app_id", "folder_path"),
        Index("idx_documents_app_folder_id", "app_id", "folder_id"),
        Index("idx_documents_app_end_user", "app_id", "end_user_id"),
    )


class DocumentStorageUsageModel(Base):
    """Per-document storage accounting for app-level aggregation."""

    __tablename__ = "document_storage_usage"

    document_id = Column(String, primary_key=True)
    app_id = Column(String, nullable=False)
    raw_bytes = Column(BigInteger, default=0)
    chunk_bytes = Column(BigInteger, default=0)
    multivector_bytes = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (Index("idx_doc_storage_app_id", "app_id"),)


class AppStorageUsageModel(Base):
    """Aggregated storage accounting by app."""

    __tablename__ = "app_storage_usage"

    app_id = Column(String, primary_key=True)
    raw_bytes = Column(BigInteger, default=0)
    chunk_bytes = Column(BigInteger, default=0)
    multivector_bytes = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
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
    updated_at = Column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )

    # Flattened auth columns for performance
    owner_id = Column(String)
    app_id = Column(String)
    folder_name = Column(String)
    folder_path = Column(String)
    end_user_id = Column(String)

    __table_args__ = (
        Index("idx_graph_system_metadata", "system_metadata", postgresql_using="gin"),
        # Unique constraint on name scoped by owner_id (also serves as index for name lookups)
        Index("idx_graph_owner_name", "name", "owner_id", unique=True),
        # Primary access control indexes
        Index("idx_graph_app_id", "app_id"),
        Index("idx_graph_owner_id", "owner_id"),
        # Composite indexes for scoped queries
        Index("idx_graphs_owner_app", "owner_id", "app_id"),
        Index("idx_graphs_app_folder", "app_id", "folder_name"),
        Index("idx_graphs_app_folder_path", "app_id", "folder_path"),
        Index("idx_graphs_app_end_user", "app_id", "end_user_id"),
    )


class FolderModel(Base):
    """SQLAlchemy model for folder data."""

    __tablename__ = "folders"

    id = Column(String, primary_key=True)
    name = Column(String)
    full_path = Column(String)
    parent_id = Column(String)
    depth = Column(Integer)
    description = Column(String, nullable=True)
    document_ids = Column(JSONB, default=list)
    system_metadata = Column(JSONB, default=dict)

    # Flattened auth columns for performance
    owner_id = Column(String)
    app_id = Column(String)
    end_user_id = Column(String)

    __table_args__ = (
        # Tree navigation index (finding children of a parent)
        Index("idx_folder_parent_id", "parent_id"),
        # Primary access control indexes
        Index("idx_folder_app_id", "app_id"),
        Index("idx_folder_owner_id", "owner_id"),
        # Composite indexes for scoped queries
        Index("idx_folders_owner_app", "owner_id", "app_id"),
        Index("idx_folders_app_end_user", "app_id", "end_user_id"),
        # Scoped uniqueness for full_path (also serves as index for path lookups)
        Index(
            "uq_folders_app_full_path",
            "app_id",
            "full_path",
            unique=True,
            postgresql_where=text("app_id IS NOT NULL"),
        ),
        Index(
            "uq_folders_owner_full_path",
            "owner_id",
            "full_path",
            unique=True,
            postgresql_where=text("app_id IS NULL"),
        ),
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
    updated_at = Column(
        DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )

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
