import os
import threading
from pathlib import Path
from typing import Optional

import pytest

from core.services.log_uploader import LogUploader

pytestmark = pytest.mark.unit


def _make_uploader(tmp_path: Path, *, max_local_bytes: int) -> LogUploader:
    uploader = object.__new__(LogUploader)
    uploader.log_dir = tmp_path
    uploader.telemetry_dir = tmp_path / "telemetry"
    uploader.telemetry_dir.mkdir(parents=True, exist_ok=True)
    uploader.max_local_bytes = max_local_bytes
    uploader._lock = threading.Lock()
    return uploader


def _write_file(path: Path, content: str, *, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _event_line(operation: str = "query") -> str:
    return f'{{"timestamp":"2026-01-01T00:00:00+00:00","worker_pid":123,"operation":"{operation}"}}\n'


def test_upload_failure_enforces_budget_only_for_telemetry_files(tmp_path, monkeypatch):
    uploader = _make_uploader(tmp_path, max_local_bytes=len(_event_line("newer")))
    app_log = _write_file(tmp_path / "app.log", "x" * 80, mtime=1)
    old_telemetry = _write_file(
        tmp_path / "telemetry" / "usage_events_worker_1.jsonl",
        _event_line("older"),
        mtime=2,
    )
    new_telemetry = _write_file(
        tmp_path / "telemetry" / "usage_events_worker_2.jsonl",
        _event_line("newer"),
        mtime=3,
    )
    monkeypatch.setattr(LogUploader, "_post_usage_payload", lambda _self, _bundle: False)

    uploader._upload_cycle()

    assert app_log.exists()
    assert not old_telemetry.exists()
    assert new_telemetry.exists()
    assert new_telemetry.read_text(encoding="utf-8") == _event_line("newer")


def test_success_truncates_uploaded_files_before_global_budget_enforcement(tmp_path, monkeypatch):
    uploader = _make_uploader(tmp_path, max_local_bytes=5)
    stale_log = _write_file(tmp_path / "app.log", "x" * 80, mtime=1)
    telemetry_file = _write_file(tmp_path / "telemetry" / "usage_events_worker_1.jsonl", _event_line(), mtime=2)
    retained_log = _write_file(tmp_path / "recent.log", "fresh", mtime=3)
    monkeypatch.setattr(LogUploader, "_post_usage_payload", lambda _self, _bundle: True)

    uploader._upload_cycle()

    assert telemetry_file.exists()
    assert telemetry_file.read_text(encoding="utf-8") == ""
    assert not stale_log.exists()
    assert retained_log.exists()
    assert retained_log.read_text(encoding="utf-8") == "fresh"


def test_cycle_without_telemetry_files_does_not_run_global_budget(tmp_path):
    uploader = _make_uploader(tmp_path, max_local_bytes=5)
    stale_log = _write_file(tmp_path / "app.log", "x" * 40, mtime=1)

    uploader._upload_cycle()

    assert stale_log.exists()
    assert stale_log.read_text(encoding="utf-8") == "x" * 40


def test_empty_bundle_does_not_run_budget_cleanup(tmp_path, monkeypatch):
    uploader = _make_uploader(tmp_path, max_local_bytes=5)
    app_log = _write_file(tmp_path / "app.log", "x" * 80, mtime=1)
    old_telemetry = _write_file(tmp_path / "telemetry" / "usage_events_worker_1.jsonl", _event_line(), mtime=2)
    telemetry_file = tmp_path / "telemetry" / "usage_events_worker_2.jsonl"

    monkeypatch.setattr(LogUploader, "_gather_telemetry_files", lambda _self: [old_telemetry, telemetry_file])
    monkeypatch.setattr(LogUploader, "_build_telemetry_bundle", lambda _self, _paths: None)
    monkeypatch.setattr(
        LogUploader,
        "_post_usage_payload",
        lambda _self, _bundle: pytest.fail("empty bundles must not be uploaded"),
    )

    uploader._upload_cycle()

    assert app_log.exists()
    assert old_telemetry.exists()
    assert old_telemetry.read_text(encoding="utf-8") == _event_line()


def test_cleanup_failure_does_not_escape_upload_cycle(tmp_path, monkeypatch, caplog):
    uploader = _make_uploader(tmp_path, max_local_bytes=1)
    _write_file(tmp_path / "telemetry" / "usage_events_worker_1.jsonl", _event_line(), mtime=1)
    monkeypatch.setattr(LogUploader, "_post_usage_payload", lambda _self, _bundle: False)

    def fail_budget(_self: LogUploader, *, root: Optional[Path] = None) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(LogUploader, "_enforce_local_budget", fail_budget)

    uploader._upload_cycle()

    assert "Unable to enforce telemetry log budget" in caplog.text


def test_zero_budget_does_not_delete_retained_files_on_upload_failure(tmp_path, monkeypatch):
    uploader = _make_uploader(tmp_path, max_local_bytes=0)
    stale_log = _write_file(tmp_path / "app.log", "x" * 80, mtime=1)
    telemetry_file = _write_file(tmp_path / "telemetry" / "usage_events_worker_1.jsonl", _event_line(), mtime=2)
    monkeypatch.setattr(LogUploader, "_post_usage_payload", lambda _self, _bundle: False)

    uploader._upload_cycle()

    assert stale_log.exists()
    assert telemetry_file.exists()
    assert telemetry_file.read_text(encoding="utf-8") == _event_line()
