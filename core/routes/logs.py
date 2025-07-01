import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.database.logs_db import LogsDB
from core.models.auth import AuthContext
from core.services.telemetry import TelemetryService, UsageRecord

router = APIRouter(prefix="/logs", tags=["Logs"])

telemetry = TelemetryService()


class LogResponse(BaseModel):
    """Public serialisable view of a UsageRecord."""

    timestamp: datetime
    user_id: str
    operation_type: str
    status: str
    tokens_used: int
    duration_ms: float
    app_id: str | None = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/", response_model=List[LogResponse])
@router.get("", response_model=List[LogResponse], include_in_schema=False)
async def get_logs(
    auth: AuthContext = Depends(verify_token),
    limit: int = Query(100, ge=1, le=500),
    since: Optional[datetime] = None,
    op_type: Optional[str] = Query(None, alias="op_type"),
    status_filter: Optional[str] = Query(None, alias="status"),
):
    """Return recent logs for the authenticated user (scoped by user_id)."""

    # Prefer persisted logs if table exists
    try:
        db = await LogsDB.get()
        rows = await db.query(
            auth.entity_id,
            auth.app_id or "unknown",
            limit,
            since.isoformat() if since else None,
            op_type,
            status_filter,
        )
        records = [
            UsageRecord(
                timestamp=row["timestamp"],
                operation_type=row["operation_type"],
                tokens_used=row["tokens_used"],
                user_id=row["user_id"],
                app_id=row["app_id"],
                duration_ms=float(row["duration_ms"] or 0),
                status=row["status"],
                metadata=(
                    row["metadata"] if not isinstance(row["metadata"], str) else json.loads(row["metadata"] or "{}")
                ),
                error=row["error"],
            )
            for row in rows
        ]
    except Exception:
        # Fallback to in-memory for backwards compatibility
        records = telemetry.get_recent_usage(
            user_id=auth.entity_id,
            app_id=auth.app_id,
            operation_type=op_type,
            since=since,
            status=status_filter,
        )

    # Return the *most recent* first, truncated by limit
    return [
        LogResponse(
            timestamp=r.timestamp,
            user_id=r.user_id,
            operation_type=r.operation_type,
            status=r.status,
            tokens_used=r.tokens_used,
            duration_ms=r.duration_ms,
            app_id=r.app_id,
            metadata=r.metadata,
            error=r.error,
        )
        for r in sorted(records, key=lambda x: x.timestamp, reverse=True)[:limit]
    ]
