from datetime import datetime

from sqlalchemy import Column, DateTime, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID

from core.database.postgres_database import Base


class AppModel(Base):
    """Represents a lightweight record of a **provisioned** application.

    • Rows live in the control-plane Postgres database configured via ``settings.POSTGRES_URI``.
    • Purpose: keep enough metadata for dashboards and admin tooling to list apps quickly.

    `AppModel` is organization-scoped, storing the minimal public attributes that a
    front-end needs: ``app_id``, ``org_id``, human-friendly ``name`` and the
    generated Morphik ``uri`` plus light provenance (creator metadata).
    The ``user_id`` is retained for backward compatibility but ``org_id`` is now
    the primary scoping mechanism.
    """

    __tablename__ = "apps"
    __table_args__ = (
        Index(
            "apps_org_name_unique",
            "org_id",
            "name",
            unique=True,
            postgresql_where=text("org_id IS NOT NULL"),
        ),
        Index(
            "apps_user_name_unique",
            "user_id",
            "name",
            unique=True,
            postgresql_where=text("org_id IS NULL AND user_id IS NOT NULL"),
        ),
    )

    app_id = Column(String, primary_key=True)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)  # Legacy field, kept for compatibility
    org_id = Column(String, index=True, nullable=True)  # Primary scoping field (optional for legacy rows)
    created_by_user_id = Column(String, nullable=True)  # User who created the app
    name = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    token_version = Column(Integer, nullable=False, default=0, server_default=text("0"))
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
