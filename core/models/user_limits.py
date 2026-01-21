from datetime import UTC, datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .tiers import AccountTier


class UserUsage(BaseModel):
    """Tracks user's actual usage of the system."""

    # Storage usage
    storage_file_count: int = 0
    storage_size_bytes: int = 0

    # Query usage - hourly
    hourly_query_count: int = 0
    hourly_query_reset: Optional[datetime] = None

    # Query usage - monthly
    monthly_query_count: int = 0
    monthly_query_reset: Optional[datetime] = None

    # Ingest usage - lifetime (no reset)
    ingest_count: int = 0


class UserLimits(BaseModel):
    """Stores user tier and usage information."""

    user_id: str
    tier: AccountTier = AccountTier.FREE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    usage: UserUsage = Field(default_factory=UserUsage)
    custom_limits: Optional[Dict[str, Any]] = None
