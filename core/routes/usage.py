import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.config import get_settings
from core.models.auth import AuthContext
from core.services_init import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/usage", tags=["Usage"])


class AppStorageUsageResponse(BaseModel):
    app_id: str
    doc_raw_bytes_mb: float
    chunk_raw_bytes_mb: float
    multivector_mb: float
    total_mb: float
    document_count: int


def _bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2) if value else 0.0


async def _authorize_app_access(auth: AuthContext, target_app_id: str) -> None:
    settings = get_settings()
    if settings.bypass_auth_mode:
        return

    if not target_app_id:
        raise HTTPException(status_code=400, detail="app_id is required")

    if auth.app_id and auth.app_id == target_app_id:
        return

    target_app = await document_service.db.get_app_record(target_app_id)
    if not target_app:
        raise HTTPException(status_code=404, detail="App not found")

    target_user = target_app.get("created_by_user_id") or target_app.get("user_id")
    if target_user and target_user == auth.user_id:
        return

    if auth.app_id:
        auth_app = await document_service.db.get_app_record(auth.app_id)
    else:
        auth_app = None

    auth_org = auth_app.get("org_id") if auth_app else None
    target_org = target_app.get("org_id")
    if auth_org and target_org and auth_org == target_org:
        return

    if auth_app:
        auth_user = auth_app.get("created_by_user_id") or auth_app.get("user_id")
        if auth_user and auth_user == target_user:
            return

    raise HTTPException(status_code=403, detail="Not authorized for requested app")


@router.get("/app-storage", response_model=AppStorageUsageResponse)
async def get_app_storage_usage(
    auth: AuthContext = Depends(verify_token),
    app_id: Optional[str] = Query(None, description="Target app_id (defaults to token app_id)"),
) -> AppStorageUsageResponse:
    target_app_id = app_id or auth.app_id
    if not target_app_id:
        raise HTTPException(status_code=400, detail="app_id is required")

    await _authorize_app_access(auth, target_app_id)
    usage = await document_service.db.get_app_storage_usage(target_app_id)
    raw_bytes = int(usage.get("raw_bytes") or 0)
    chunk_bytes = int(usage.get("chunk_bytes") or 0)
    multivector_bytes = int(usage.get("multivector_bytes") or 0)
    total_bytes = raw_bytes + chunk_bytes + multivector_bytes
    return AppStorageUsageResponse(
        app_id=usage.get("app_id", target_app_id),
        doc_raw_bytes_mb=_bytes_to_mb(raw_bytes),
        chunk_raw_bytes_mb=_bytes_to_mb(chunk_bytes),
        multivector_mb=_bytes_to_mb(multivector_bytes),
        total_mb=_bytes_to_mb(total_bytes),
        document_count=int(usage.get("document_count") or 0),
    )
