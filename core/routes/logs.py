import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.services.telemetry_events import TelemetryEventReader

router = APIRouter(prefix="/logs", tags=["Logs"])
logger = logging.getLogger(__name__)

event_reader = TelemetryEventReader(log_dir=Path("logs/telemetry"))
LOGS_PROXY_URL = "https://logs.morphik.ai/api/events/query"
LOCAL_RETENTION_HOURS = 4


async def _query_proxy(
    app_id: str,
    since: datetime,
    limit: int,
    operation_type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Any]:
    """Query logs.morphik.ai for historical events filtered by app_id."""
    try:
        params = {
            "app_id": app_id,
            "since": since.isoformat(),
            "limit": str(limit),
        }
        if operation_type:
            params["operation_type"] = operation_type
        if status:
            params["status"] = status

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(LOGS_PROXY_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get("ok"):
                logger.warning("Proxy query failed: %s", data.get("error"))
                return []

            raw_events = data.get("events", [])
            from core.services.telemetry_events import TelemetryEvent

            return [
                TelemetryEvent(
                    timestamp=datetime.fromisoformat(e["timestamp"]),
                    installation_id=e.get("installation_id"),
                    operation_type=e["operation_type"],
                    status=e.get("status", "unknown"),
                    duration_ms=float(e.get("duration_ms", 0.0)),
                    user_id=e.get("user_id"),
                    app_id=e.get("app_id"),
                    tokens_used=int(e.get("tokens_used", 0)),
                    metadata=e.get("metadata"),
                    error=e.get("error"),
                    trace_id=e.get("trace_id"),
                    span_id=e.get("span_id"),
                    worker_pid=e.get("worker_pid"),
                )
                for e in raw_events
            ]
    except Exception as exc:
        logger.error("Failed to query logs proxy: %s", exc)
        return []


class LogResponse(BaseModel):
    """Public serialisable view of a telemetry event."""

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
    hours: float = Query(4.0, ge=0.1, le=168.0),
    op_type: Optional[str] = Query(None, alias="op_type"),
    status_filter: Optional[str] = Query(None, alias="status"),
):
    """Return recent logs for the authenticated user (scoped by app_id).

    Args:
        hours: Number of hours of history to retrieve (default 4)
               - <= 4 hours: reads from local files
               - > 4 hours: queries logs.morphik.ai proxy (requires proxy endpoint)
    """
    if not auth.app_id:
        return []

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    if hours <= LOCAL_RETENTION_HOURS:
        events = event_reader.recent_events(
            limit=limit,
            user_id=auth.user_id,
            app_id=auth.app_id,
            operation_type=op_type,
            status=status_filter,
            since=since,
        )
    else:
        events = await _query_proxy(
            app_id=auth.app_id,
            since=since,
            limit=limit,
            operation_type=op_type,
            status=status_filter,
        )

    return [
        LogResponse(
            timestamp=event.timestamp,
            user_id=event.user_id or auth.user_id,
            operation_type=event.operation_type,
            status=event.status,
            tokens_used=event.tokens_used,
            duration_ms=event.duration_ms,
            app_id=event.app_id,
            metadata=event.metadata,
            error=event.error,
        )
        for event in events
    ]
