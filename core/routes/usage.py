import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth_utils import verify_token
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


@router.get("/app-storage", response_model=AppStorageUsageResponse)
async def get_app_storage_usage(
    auth: AuthContext = Depends(verify_token),
) -> AppStorageUsageResponse:
    if not auth.app_id:
        raise HTTPException(status_code=400, detail="app_id is required")

    usage = await document_service.db.get_app_storage_usage(auth.app_id)
    raw_bytes = int(usage.get("raw_bytes") or 0)
    chunk_bytes = int(usage.get("chunk_bytes") or 0)
    multivector_bytes = int(usage.get("multivector_bytes") or 0)
    total_bytes = raw_bytes + chunk_bytes + multivector_bytes
    return AppStorageUsageResponse(
        app_id=usage.get("app_id", auth.app_id),
        doc_raw_bytes_mb=_bytes_to_mb(raw_bytes),
        chunk_raw_bytes_mb=_bytes_to_mb(chunk_bytes),
        multivector_mb=_bytes_to_mb(multivector_bytes),
        total_mb=_bytes_to_mb(total_bytes),
        document_count=int(usage.get("document_count") or 0),
    )
