"""Enterprise router that allows *clients* of a **dedicated Morphik instance** to
provision isolated Neon-backed databases ("apps").  The endpoint hides all Neon
complexities and returns a Morphik-specific connection URI that other
Databridge APIs can consume.

This route is only available in *Enterprise* deployments and is therefore
located inside the `ee` package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import toml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.models.tiers import AccountTier, get_tier_limits

# The provisioning logic lives in *core* because it is useful in background
# jobs and potentially community deployments as well.  The router is EE-only.
from core.services.app_provisioning_service import AppProvisioningService, ProvisionResult
from core.services.user_service import UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ee", tags=["Enterprise"])


# ---------------------------------------------------------------------------
# Request / Response models (kept intentionally minimal)
# ---------------------------------------------------------------------------


class CreateAppRequest(BaseModel):  # noqa: D101 – simple schema
    app_name: str = Field(..., description="Human-friendly name of the application to create")
    region: str | None = Field(
        default=None,
        description="Optional Neon region identifier (defaults to `aws-us-east-1`)",
    )


class CreateAppResponse(BaseModel):  # noqa: D101 – simple schema
    app_id: str
    app_name: str
    morphik_uri: str
    status: str


# ---------------------------------------------------------------------------
# Endpoint implementation
# ---------------------------------------------------------------------------


@router.post("/apps", response_model=CreateAppResponse, status_code=status.HTTP_201_CREATED, include_in_schema=True)
async def create_app_route(
    request: CreateAppRequest,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, str]:
    """Provision a **brand-new** Neon database for *request.app_name*.

    The caller must be authenticated to the dedicated Morphik instance.  The
    authenticated user (represented by the JWT's *user_id*) becomes the owner
    of the provisioned app.
    """

    # Ensure authentication has user_id (in dev mode it is generated)
    if not auth.user_id:
        raise HTTPException(status_code=403, detail="Missing user_id in token – cannot provision app")

    # Perform the heavy lifting via the service layer
    service = AppProvisioningService()
    await service.initialize()

    # Load morphik-host from ee.toml
    morphik_host: str | None = None
    try:
        # Assume ee.toml is in the 'ee' directory, so two parents up from 'ee/routers/apps.py' then into 'ee'
        # Adjust path if ee.toml is located elsewhere relative to this file.
        # For a typical structure where 'ee' is a top-level package and this file is ee/routers/apps.py
        # ee_config_path = Path(__file__).resolve().parent.parent / "ee.toml"
        # Correcting path assumption if 'ee' is the root of the 'ee' package content
        ee_package_dir = Path(__file__).resolve().parent.parent  # This should point to the 'ee' directory
        ee_config_path = ee_package_dir / "ee.toml"

        if ee_config_path.exists():
            ee_config = toml.load(ee_config_path)
            morphik_host = ee_config.get("morphik-host")
        else:
            logger.error(f"Configuration file not found: {ee_config_path}")
            raise HTTPException(status_code=500, detail="Server configuration error: ee.toml not found.")

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error loading ee.toml: {e}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Could not load ee.toml. {e}")

    if not morphik_host:
        logger.error("morphik-host not configured in ee.toml.")
        raise HTTPException(status_code=500, detail="Server configuration error: morphik-host not set in ee.toml.")

    try:
        # Pass morphik_host to the service
        result: ProvisionResult = await service.provision_new_app(
            auth.user_id,
            request.app_name,
            request.region,
            morphik_host=morphik_host,
        )

        # ------------------------------------------------------------------
        # The code snippet `user_service = UserService()` creates an instance of the `UserService`
        # class, which is responsible for interacting with the user service in the application.
        # Register the newly created app in the user limits table
        # ------------------------------------------------------------------
        user_service = UserService()
        await user_service.initialize()

        # ------------------------------------------------------------------
        # Check plan eligibility and enforce Teams app limit
        # ------------------------------------------------------------------

        user_limits = await user_service.get_user_limits(auth.user_id)

        if not user_limits:
            # Create default limits record if none exists
            await user_service.create_user(auth.user_id)
            user_limits = await user_service.get_user_limits(auth.user_id)

        tier = user_limits.get("tier", AccountTier.FREE)

        # Only Teams or Enterprise (self-hosted) customers may provision isolated apps
        if tier not in (AccountTier.TEAMS, AccountTier.SELF_HOSTED):
            raise HTTPException(
                status_code=403,
                detail="Only Teams or Enterprise plan customers can create isolated applications. "
                "Upgrade your plan to access this feature.",
            )

        # For Teams tier, make sure the user is still within the app limit
        if tier == AccountTier.TEAMS:
            tier_limits = get_tier_limits(tier, user_limits.get("custom_limits"))
            app_limit = tier_limits.get("app_limit", 50)
            current_apps = user_limits.get("app_ids", []) or []

            if len(current_apps) >= app_limit:
                raise HTTPException(
                    status_code=403,
                    detail=f"Application limit reached for Teams plan (maximum {app_limit}).",
                )

        # Record the new app in the user's limits profile
        await user_service.register_app(auth.user_id, result.app_id)

    except Exception as exc:  # noqa: BLE001 – capture NeonAPIError and others
        logger.exception("Failed to provision new app: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to provision app") from exc

    return result.as_dict()
