import uuid
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB

# Import the shared declarative base that is already used throughout the codebase
from core.database.postgres_database import Base


class AppMetadataModel(Base):
    """SQLAlchemy model that stores metadata for provisioned Neon-backed apps."""

    __tablename__ = "app_metadata"

    # A surrogate primary key that uniquely identifies the record inside our control-plane DB
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # References the *logical* owner inside Morphik (typically the authenticated user / client)
    owner_id = Column(String, nullable=False, index=True)

    # Human-friendly name supplied by the caller (e.g. "my-new-campaign-data")
    app_name = Column(String, nullable=False, index=True)

    # The Neon project that was created for the app.  We store it so that we can tear the
    # database down at a later point in time if necessary.
    neon_project_id = Column(String, nullable=False, unique=True)

    # The fully-qualified connection string returned by Neon.  Storing it lets us rebuild the
    # Morphik-specific URI at any point without another API round-trip.
    connection_uri = Column(String, nullable=False)

    # A free-form JSON payload where additional implementation-specific details can be captured.
    extra = Column(JSONB, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Pydantic schema that we expose via the public API (FastAPI response model)
# ---------------------------------------------------------------------------


class AppMetadata(BaseModel):
    """Pydantic model that mirrors :class:`AppMetadataModel` for API responses."""

    id: str = Field(..., description="Internal surrogate ID")
    owner_id: str = Field(..., description="ID of the Morphik account that owns the app")
    app_name: str = Field(..., description="Human-friendly app name as provided by the caller")
    neon_project_id: str = Field(..., description="ID of the Neon project backing the app")
    connection_uri: str = Field(..., description="PostgreSQL connection URI for the provisioned Neon DB")
    extra: dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
