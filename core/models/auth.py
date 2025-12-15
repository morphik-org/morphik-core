from typing import Optional

from pydantic import BaseModel


class AuthContext(BaseModel):
    """JWT decoded context for authenticated requests."""

    user_id: str  # The authenticated user's ID (owner for ACL)
    app_id: Optional[str] = None  # The app scope for multi-tenancy
