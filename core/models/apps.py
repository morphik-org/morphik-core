from datetime import datetime

from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from core.database.postgres_database import Base


class AppModel(Base):
    """Represents a lightweight record of a **provisioned** application.

    • The row lives in the *control-plane* Postgres database configured via
      ``settings.POSTGRES_URI`` (i.e. the **main instance database**, *not* the
      dynamically created per-app Neon database).
    • Purpose: allow dashboards / multi-tenant admin UIs to list apps quickly
      without having to join against the heavier *app_metadata* table.

    `AppModel` intentionally stores only the minimal public attributes that a
    front-end needs: ``app_id``, ``user_id``, human-friendly ``name`` and the
    generated Morphik ``uri``.  All operational details remain in
    :class:`core.models.app_metadata.AppMetadataModel`.
    """

    __tablename__ = "apps"

    app_id = Column(String, primary_key=True)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    name = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
