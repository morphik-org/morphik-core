import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
from fastapi import APIRouter, Depends, Form, Header, HTTPException

from core.auth_utils import verify_token
from core.config import get_settings
from core.models.auth import AuthContext, EntityType
from core.models.request import GenerateUriRequest
from core.services.telemetry import TelemetryService
from core.services_init import document_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Admin"])
settings = get_settings()
telemetry = TelemetryService()


@router.get("/usage/stats")
@telemetry.track(
    operation_type="get_usage_stats",
    metadata_resolver=telemetry.usage_stats_metadata,
)
async def get_usage_stats(
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, int]:
    """Get usage statistics for the authenticated user.

    Args:
        auth: Authentication context identifying the caller.

    Returns:
        A mapping of operation types to token usage counts.
    """
    try:
        stats = await document_service.db.get_usage_stats(
            auth.user_id, auth.app_id
        )
        return stats
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get usage stats")


@router.get("/usage/recent")
@telemetry.track(
    operation_type="get_recent_usage",
    metadata_resolver=telemetry.recent_usage_metadata,
)
async def get_recent_usage(
    auth: AuthContext = Depends(verify_token),
    operation_type: Optional[str] = None,
    since: Optional[datetime] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """Retrieve recent telemetry records for the user or application.

    Args:
        auth: Authentication context; admin users receive global records.
        operation_type: Optional operation type to filter by.
        since: Only return records newer than this timestamp.
        status: Optional status filter (e.g. ``success`` or ``error``).

    Returns:
        A list of usage entries sorted by timestamp, each represented as a
        dictionary.
    """
    try:
        # Default to last 24 hours if no since provided
        if not since:
            since = datetime.now(UTC) - timedelta(days=1)

        events = await document_service.db.get_recent_usage(
            auth.user_id,
            auth.app_id,
            operation_type=operation_type,
            since=since,
            status=status,
        )
        return events
    except Exception as e:
        logger.error(f"Error getting recent usage: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get recent usage"
        )


@router.post("/local/generate_uri", include_in_schema=True)
async def generate_local_uri(
    name: str = Form("admin"),
    expiry_days: int = Form(30),
) -> Dict[str, str]:
    """Generate a development URI for running Morphik locally.

    Args:
        name: Developer name to embed in the token payload.
        expiry_days: Number of days the generated token should remain valid.

    Returns:
        A dictionary containing the ``uri`` that can be used to connect to the
        local instance.
    """
    if settings.MODE == "cloud":
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available in self-hosted mode"
        )

    # Create token payload
    payload = {
        "entity_type": EntityType.DEVELOPER.value,
        "entity_id": name,
        "app_id": None,
        "permissions": ["read", "write", "admin"],
        "user_id": name,
        "exp": datetime.now(UTC) + timedelta(days=expiry_days),
        "iat": datetime.now(UTC),
    }

    # Generate JWT token
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    # Create URI
    base_url = settings.BASE_URL or "http://localhost:8000"
    uri = f"{base_url}/auth?token={token}"

    return {
        "uri": uri,
        "token": token,
        "expires_in_days": expiry_days,
        "name": name,
    }


@router.post("/cloud/generate_uri", include_in_schema=True)
async def generate_cloud_uri(
    request: GenerateUriRequest,
    authorization: str = Header(None),
) -> Dict[str, str]:
    """Generate an authenticated URI for a cloud-hosted Morphik application.

    Args:
        request: Parameters for URI generation including ``app_id`` and
            ``name``.
        authorization: Bearer token of the user requesting the URI.

    Returns:
        A dictionary with the generated ``uri`` and associated ``app_id``.
    """
    if settings.MODE != "cloud":
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available in cloud mode"
        )

    # Verify the control plane token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )

    control_token = authorization.replace("Bearer ", "")
    if control_token != settings.CONTROL_PLANE_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Invalid control plane token"
        )

    # Create app-specific token
    payload = {
        "entity_type": EntityType.DEVELOPER.value,
        "entity_id": request.app_id,
        "app_id": request.app_id,
        "permissions": ["read", "write"],
        "user_id": request.user_id,
        "exp": datetime.now(UTC) + timedelta(days=request.expiry_days),
        "iat": datetime.now(UTC),
    }

    # Generate JWT token
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    # Create URI
    base_url = settings.BASE_URL or "https://api.morphik.ai"
    uri = f"{base_url}/auth?token={token}"

    return {
        "uri": uri,
        "token": token,
        "app_id": request.app_id,
        "user_id": request.user_id,
        "expires_in_days": request.expiry_days,
    }


@router.delete("/cloud/apps")
async def delete_cloud_app(
    app_name: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Delete all data associated with a cloud application.

    This endpoint removes all documents, folders, graphs, and other
    data associated with the specified application.

    Args:
        app_name: Name of the application to delete
        auth: Authentication context (must have admin permissions)

    Returns:
        Dict with deletion summary
    """
    if settings.MODE != "cloud":
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available in cloud mode"
        )

    # Check admin permissions
    if "admin" not in auth.permissions:
        raise HTTPException(
            status_code=403,
            detail="Admin permissions required"
        )

    try:
        # Delete all app data
        deletion_summary = await document_service.db.delete_app_data(
            app_name, auth
        )

        return {
            "status": "deleted",
            "app_name": app_name,
            "summary": deletion_summary,
        }
    except Exception as e:
        logger.error(f"Error deleting app data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete app data: {str(e)}"
        )