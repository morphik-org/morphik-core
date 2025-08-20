from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from core.database.postgres_database import Base


class AppModel(Base):
    """Represents a lightweight record of a **provisioned** application.

    • The row lives in the *control-plane* Postgres database configured via
      ``settings.POSTGRES_URI`` (i.e. the **main instance database**, *not* the
      dynamically created per-app Neon database).
    • Purpose: allow dashboards / multi-tenant admin UIs to list apps quickly
      without having to join against the heavier *app_metadata* table.

    `AppModel` is now organization-scoped, storing the minimal public attributes that a
    front-end needs: ``app_id``, ``org_id``, human-friendly ``name`` and the
    generated Morphik ``uri``. The ``user_id`` is retained for backward compatibility
    but ``org_id`` is now the primary scoping mechanism. All operational details remain in
    :class:`core.models.app_metadata.AppMetadataModel`.
    """

    __tablename__ = "apps"

    app_id = Column(String, primary_key=True)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)  # Legacy field, kept for compatibility
    org_id = Column(String, index=True, nullable=False)  # Primary scoping field
    created_by_user_id = Column(String, nullable=True)  # User who created the app
    name = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    tier = Column(JSON, nullable=True)  # App tier/limits configuration
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
