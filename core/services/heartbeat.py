from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

from core.utils.telemetry_signature import compute_telemetry_signature

LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 10


class Heartbeat:
    """Sends a lightweight ping to Morphik to report active installations."""

    def __init__(
        self,
        *,
        heartbeat_url: str,
        project_name: str,
        installation_id: str,
        version: str,
        interval_hours: float,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.heartbeat_url = (heartbeat_url or "").strip()
        self.project_name = project_name
        self.installation_id = installation_id
        self.version = version or "unknown"
        self.interval_seconds = max(interval_hours, 0) * 3600
        self.timeout = timeout
        self._signature = compute_telemetry_signature(installation_id)

        self._is_first_ping = True
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._session = requests.Session()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self.heartbeat_url:
            LOGGER.debug("Heartbeat disabled (no URL configured)")
            return
        if self.interval_seconds <= 0:
            LOGGER.debug("Heartbeat disabled (interval=%s)", self.interval_seconds)
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, name="heartbeat", daemon=True)
        self._thread.start()
        LOGGER.debug("Heartbeat thread started for project %s", self.project_name)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._session.close()

    # ------------------------------------------------------------------
    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            self._send_ping()
            wait_seconds = self.interval_seconds or 3600
            if self._stop_event.wait(wait_seconds):
                break

    def _send_ping(self) -> None:
        event_type = "first_start" if self._is_first_ping else "heartbeat"
        payload = {
            "project_name": self.project_name,
            "installation_id": self.installation_id,
            # RFC3339 timestamp with explicit Zulu suffix for zod validator.
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "version": self.version,
            "event_type": event_type,
            "signature": self._signature,
        }

        try:
            response = self._session.post(self.heartbeat_url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            LOGGER.warning("Heartbeat request failed: %s", exc)
            return

        if response.status_code >= 300:
            LOGGER.warning(
                "Heartbeat rejected (%s): %s",
                response.status_code,
                response.text[:200],
            )
            return

        if self._is_first_ping:
            LOGGER.debug("Sent first_start heartbeat for project: %s", self.project_name)
        else:
            LOGGER.debug("Sent heartbeat for project: %s", self.project_name)
        self._is_first_ping = False
