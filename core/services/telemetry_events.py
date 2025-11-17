from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    timestamp: datetime
    installation_id: Optional[str]
    operation_type: str
    status: str
    duration_ms: float
    user_id: Optional[str]
    app_id: Optional[str]
    tokens_used: int
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]
    trace_id: Optional[str]
    span_id: Optional[str]
    worker_pid: Optional[int]


class TelemetryEventReader:
    """Read telemetry JSONL events emitted by the JSONLSpanExporter."""

    def __init__(self, log_dir: str | Path = Path("logs/telemetry")):
        self.log_dir = Path(log_dir)

    def recent_events(
        self,
        *,
        limit: int = 100,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[TelemetryEvent]:
        """Return the most recent events that satisfy the provided filters."""
        if limit <= 0:
            return []

        events = self._collect_events(
            limit=limit,
            user_id=user_id,
            app_id=app_id,
            operation_type=operation_type,
            status=status,
            since=since,
            until=None,
        )
        return events

    def events_between(
        self,
        *,
        since: datetime,
        until: Optional[datetime] = None,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TelemetryEvent]:
        """Return all events recorded between ``since`` (inclusive) and ``until`` (exclusive)."""
        since_normalized = self._normalize_since(since)
        until_normalized = self._normalize_since(until) if until else None
        return self._collect_events(
            limit=None,
            user_id=user_id,
            app_id=app_id,
            operation_type=operation_type,
            status=status,
            since=since_normalized,
            until=until_normalized,
        )

    # ------------------------------------------------------------------ #
    def _ordered_log_files(self) -> List[Path]:
        if not self.log_dir.exists():
            return []
        files = sorted(
            self.log_dir.glob("usage_events_worker_*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return files

    def _iter_file(self, path: Path):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return

        for line in reversed(lines):
            event = self._parse_event(line)
            if event:
                yield event

    def _parse_event(self, line: str) -> Optional[TelemetryEvent]:
        if not line:
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None

        timestamp_raw = payload.get("timestamp")
        if not timestamp_raw:
            return None

        try:
            timestamp = datetime.fromisoformat(timestamp_raw)
        except ValueError:
            return None

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        operation_type = payload.get("operation_type")
        if not operation_type:
            return None

        return TelemetryEvent(
            timestamp=timestamp,
            installation_id=payload.get("installation_id"),
            operation_type=operation_type,
            status=payload.get("status", "unknown"),
            duration_ms=float(payload.get("duration_ms", 0.0) or 0.0),
            user_id=payload.get("user_id"),
            app_id=payload.get("app_id"),
            tokens_used=int(payload.get("tokens_used", 0) or 0),
            metadata=payload.get("metadata"),
            error=payload.get("error"),
            trace_id=payload.get("trace_id"),
            span_id=payload.get("span_id"),
            worker_pid=payload.get("worker_pid"),
        )

    def _matches(
        self,
        event: TelemetryEvent,
        user_id: Optional[str],
        app_id: Optional[str],
        operation_type: Optional[str],
        status_filter: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> bool:
        if user_id and event.user_id != user_id:
            return False
        if app_id and event.app_id != app_id:
            return False
        if operation_type and event.operation_type != operation_type:
            return False
        if status_filter and event.status.lower() != status_filter:
            return False
        if since and event.timestamp < since:
            return False
        if until and event.timestamp >= until:
            return False
        return True

    @staticmethod
    def _normalize_since(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _collect_events(
        self,
        *,
        limit: Optional[int],
        user_id: Optional[str],
        app_id: Optional[str],
        operation_type: Optional[str],
        status: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> List[TelemetryEvent]:
        since_normalized = self._normalize_since(since)
        until_normalized = self._normalize_since(until) if until else None
        status_filter = status.lower() if status else None
        files = self._ordered_log_files()
        results: List[TelemetryEvent] = []

        for path in files:
            for event in self._iter_file(path):
                if not self._matches(
                    event,
                    user_id=user_id,
                    app_id=app_id,
                    operation_type=operation_type,
                    status_filter=status_filter,
                    since=since_normalized,
                    until=until_normalized,
                ):
                    continue
                results.append(event)
                if limit and len(results) >= limit:
                    break
            if limit and len(results) >= limit:
                break

        results.sort(key=lambda ev: ev.timestamp, reverse=True)
        if limit:
            return results[:limit]
        return results
