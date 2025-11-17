from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import requests

LOGGER = logging.getLogger(__name__)
TELEMETRY_PATTERN = "usage_events_worker_*.jsonl"
PROXY_SIGNATURE_SALT = "morphik-telemetry-upload-v1"
PROXY_UPLOAD_URL = "https://logs.morphik.ai/api/events/upload"
PROXY_TIMEOUT_SECONDS = 10.0


@dataclass
class TelemetryBundle:
    compressed: bytes
    files: int
    event_count: int
    invalid_lines: int
    earliest: Optional[datetime]
    latest: Optional[datetime]
    worker_pids: set[int]
    decompressed_bytes: int
    compressed_bytes: int


class LogUploader:
    """Background job that ships Morphik Core telemetry logs to the usage proxy."""

    def __init__(
        self,
        *,
        log_dir: Path,
        project_name: str,
        installation_id: str,
        interval_hours: float,
        max_local_bytes: int,
        service_name: str = "morphik-core",
        environment: str = "production",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir = self.log_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name.strip() or "oss"
        self.installation_id = installation_id
        self.interval_seconds = max(interval_hours, 0) * 3600
        self.max_local_bytes = max(int(max_local_bytes), 0)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._proxy_url = PROXY_UPLOAD_URL
        self._proxy_timeout = PROXY_TIMEOUT_SECONDS
        self._proxy_session = requests.Session()
        self._project_slug = self._sanitize_slug(self.project_name)
        self._service_name = service_name
        self._environment = environment
        self._uploader_version = os.getenv("DATABRIDGE_VERSION", "unknown")

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.interval_seconds <= 0:
            LOGGER.info("Log uploader disabled (interval=%s)", self.interval_seconds)
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, name="log-uploader", daemon=True)
        self._thread.start()
        LOGGER.info("Telemetry uploader started for project %s", self.project_name)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._proxy_session:
            self._proxy_session.close()

    # ------------------------------------------------------------------
    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            self._upload_cycle()
            wait_seconds = self.interval_seconds or 60
            if self._stop_event.wait(wait_seconds):
                break

    def _upload_cycle(self) -> None:
        with self._lock:
            telemetry_paths = self._gather_telemetry_files()
            if not telemetry_paths:
                LOGGER.debug("No telemetry events to upload")
                return

            bundle = self._build_telemetry_bundle(telemetry_paths)
            if not bundle:
                LOGGER.debug("Telemetry bundle contained no events")
                return

            if self._post_usage_payload(bundle):
                self._truncate_files(telemetry_paths)
                self._enforce_local_budget()
            else:
                LOGGER.warning("Telemetry proxy upload failed; retaining local files")

    def _gather_telemetry_files(self) -> List[Path]:
        files: List[Path] = []
        for path in sorted(self.telemetry_dir.glob(TELEMETRY_PATTERN)):
            try:
                if path.stat().st_size <= 0:
                    continue
            except FileNotFoundError:
                continue
            files.append(path)
        return files

    def _build_telemetry_bundle(self, paths: Sequence[Path]) -> Optional[TelemetryBundle]:
        buffer = io.BytesIO()
        event_count = 0
        invalid_lines = 0
        earliest: Optional[datetime] = None
        latest: Optional[datetime] = None
        worker_pids: set[int] = set()
        decompressed_bytes = 0
        files_processed = 0
        has_lines = False

        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            for path in paths:
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        files_processed += 1
                        for line in handle:
                            if line is None:
                                continue
                            encoded = line.encode("utf-8")
                            gz.write(encoded)
                            has_lines = True
                            stripped = line.strip()
                            if not stripped:
                                continue
                            decompressed_bytes += len(encoded)
                            try:
                                event = json.loads(stripped)
                            except json.JSONDecodeError:
                                invalid_lines += 1
                                continue

                            event_count += 1
                            timestamp_raw = event.get("timestamp")
                            if timestamp_raw:
                                try:
                                    parsed = datetime.fromisoformat(timestamp_raw)
                                except ValueError:
                                    parsed = None
                                if parsed:
                                    if parsed.tzinfo is None:
                                        parsed = parsed.replace(tzinfo=timezone.utc)
                                    if earliest is None or parsed < earliest:
                                        earliest = parsed
                                    if latest is None or parsed > latest:
                                        latest = parsed

                            pid = event.get("worker_pid")
                            if isinstance(pid, int):
                                worker_pids.add(pid)
                except FileNotFoundError:
                    continue

        if not has_lines:
            return None

        compressed = buffer.getvalue()
        return TelemetryBundle(
            compressed=compressed,
            files=files_processed,
            event_count=event_count,
            invalid_lines=invalid_lines,
            earliest=earliest,
            latest=latest,
            worker_pids=worker_pids,
            decompressed_bytes=decompressed_bytes,
            compressed_bytes=len(compressed),
        )

    def _post_usage_payload(self, bundle: TelemetryBundle) -> bool:
        payload = self._build_usage_payload(bundle)
        try:
            response = self._proxy_session.post(
                self._proxy_url,
                json=payload,
                timeout=self._proxy_timeout,
            )
        except requests.RequestException as exc:
            LOGGER.warning("Telemetry proxy upload failed: %s", exc)
            return False

        if response.status_code >= 300:
            preview = response.text[:200] if response.text else "<empty>"
            LOGGER.warning("Telemetry proxy rejected upload (%s): %s", response.status_code, preview)
            return False

        response_key = None
        try:
            body = response.json()
            response_key = body.get("key") if isinstance(body, dict) else None
        except ValueError:
            response_key = None

        LOGGER.info(
            "Uploaded telemetry via proxy (events=%s, invalid_lines=%s, key=%s)",
            bundle.event_count,
            bundle.invalid_lines,
            response_key,
        )
        return True

    def _build_usage_payload(self, bundle: TelemetryBundle) -> dict:
        started = self._format_timestamp(bundle.earliest)
        finished = self._format_timestamp(bundle.latest or bundle.earliest)
        metadata = {
            "projectName": self.project_name,
            "projectSlug": self._project_slug,
            "serviceName": self._service_name,
            "environment": self._environment,
            "files": bundle.files,
            "invalidLines": bundle.invalid_lines,
            "compressedBytes": bundle.compressed_bytes,
            "decompressedBytes": bundle.decompressed_bytes,
        }

        return {
            "schemaVersion": 1,
            "installationId": self.installation_id,
            "startedAt": started,
            "finishedAt": finished,
            "eventCount": bundle.event_count,
            "uploaderVersion": self._uploader_version,
            "workerPids": sorted(bundle.worker_pids),
            "metadata": metadata,
            "compression": "gzip",
            "payloadEncoding": "base64",
            "payload": base64.b64encode(bundle.compressed).decode("ascii"),
            "signature": self._compute_signature(),
        }

    def _compute_signature(self) -> str:
        value = f"{self.installation_id}{PROXY_SIGNATURE_SALT}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def _truncate_files(self, paths: Sequence[Path]) -> None:
        for path in paths:
            try:
                with path.open("r+b") as handle:
                    handle.truncate(0)
            except FileNotFoundError:
                continue
            except OSError as exc:
                LOGGER.warning("Unable to truncate %s: %s", path, exc)

    def _enforce_local_budget(self) -> None:
        if self.max_local_bytes <= 0:
            return
        files: List[tuple[Path, int, float]] = []
        total = 0
        for path in self.log_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            total += stat.st_size
            files.append((path, stat.st_size, stat.st_mtime))
        if total <= self.max_local_bytes:
            return

        for path, size, _mtime in sorted(files, key=lambda item: item[2]):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            LOGGER.warning("Removed %s to enforce %s-byte log budget", path, self.max_local_bytes)
            total -= size
            if total <= self.max_local_bytes:
                break

    @staticmethod
    def _format_timestamp(value: Optional[datetime]) -> str:
        current = value or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _sanitize_slug(value: str) -> str:
        cleaned = [ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip()]
        slug = "".join(cleaned).strip("-")
        return slug or "installation"
