"""Enterprise endpoint that generates Morphik Cloud URIs without explicitly
specifying *user_id* in the request body.  The user ID is harvested from the
JWT bearer token so that front-ends can call this endpoint with the same token
that they already possess.
"""

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.services.user_service import UserService

router = APIRouter(prefix="/ee", tags=["Enterprise"])


class GenerateUriEERequest(BaseModel):
    """Request body for the EE *generate_uri* endpoint (no ``user_id`` field)."""

    app_id: str = Field(..., description="ID of the application")
    name: str = Field(..., description="Name of the application")
    expiry_days: int = Field(default=30, description="Token validity in days")


@router.post("/create_app", include_in_schema=True)
async def create_app(
    request: GenerateUriEERequest,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, str]:
    """Generate a cloud URI for *request.app_id* owned by the calling user.

    The *user_id* is derived from the bearer token.  The caller can therefore
    not create applications for *other* users unless their token carries the
    ``admin`` permission (mirroring the community behaviour).
    """

    # "auth" is guaranteed by verify_token.  Reuse its user_id.
    user_id = auth.user_id or auth.entity_id

    # --- 2. Generate the cloud URI via the UserService ----------------------
    user_service = UserService()
    await user_service.initialize()

    name_clean = request.name.replace(" ", "_").lower()

    uri = await user_service.generate_cloud_uri(
        user_id=user_id,
        app_id=request.app_id,
        name=name_clean,
        expiry_days=request.expiry_days,
    )

    if not uri:
        # The UserService returns *None* when the user exceeded their app quota
        raise HTTPException(status_code=403, detail="Application limit reached for this account tier")

    return {"uri": uri, "app_id": request.app_id}
