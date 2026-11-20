"""convert_with_libreoffice: private profile per call, nothing left running after a timeout, errors raise."""

import os
import time
from pathlib import Path

import pytest

from core.utils import libreoffice
from core.utils.libreoffice import LibreOfficeConversionError, convert_with_libreoffice

pytestmark = pytest.mark.unit

# Stand-in for the real launcher: like `soffice` -> oosplash -> soffice.bin, the work happens in a child
# process the launcher waits on. FAKE_SOFFICE_MODE picks the behaviour.
FAKE_SOFFICE = """#!/usr/bin/env bash
outdir=""; args="$*"
while [ $# -gt 0 ]; do
  case "$1" in --outdir) outdir="$2"; shift 2 ;; *) source="$1"; shift ;; esac
done
name=$(basename "${source%.*}")
case "$FAKE_SOFFICE_MODE" in
  ok) mkdir -p "$outdir"; printf '%s' "$args" > "$outdir/$name.pdf" ;;
  hang) sleep 300 & echo $! > "$FAKE_SOFFICE_CHILD_PID"; wait ;;
  fail) echo "conversion crashed" >&2; exit 1 ;;
  no-output) echo "Error: source file could not be loaded" >&2 ;;
esac
"""


@pytest.fixture
def fake_soffice(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "soffice"
    script.write_text(FAKE_SOFFICE)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


def _wait_until_gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_returns_output_and_uses_a_private_profile(fake_soffice, monkeypatch):
    monkeypatch.setenv("FAKE_SOFFICE_MODE", "ok")

    first = convert_with_libreoffice(b"doc bytes", ".doc", "pdf").decode()
    second = convert_with_libreoffice(b"doc bytes", ".doc", "pdf").decode()

    profile_args = [arg for arg in (first + " " + second).split() if arg.startswith("-env:UserInstallation=file://")]
    assert len(profile_args) == 2
    assert profile_args[0] != profile_args[1]


def test_timeout_kills_the_converter_the_launcher_started(fake_soffice, monkeypatch):
    child_pid_file = fake_soffice / "child.pid"
    monkeypatch.setenv("FAKE_SOFFICE_MODE", "hang")
    monkeypatch.setenv("FAKE_SOFFICE_CHILD_PID", str(child_pid_file))

    with pytest.raises(LibreOfficeConversionError, match="within 1s"):
        convert_with_libreoffice(b"doc bytes", ".doc", "pdf", timeout_s=1)

    assert _wait_until_gone(int(child_pid_file.read_text()))


@pytest.mark.parametrize("mode,message", [("fail", "exited with code 1"), ("no-output", "produced no pdf output")])
def test_failures_raise(fake_soffice, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_SOFFICE_MODE", mode)

    with pytest.raises(LibreOfficeConversionError, match=message):
        convert_with_libreoffice(b"doc bytes", ".doc", "pdf")


def test_missing_libreoffice_raises(monkeypatch):
    monkeypatch.setattr(libreoffice.shutil, "which", lambda name: None)

    with pytest.raises(LibreOfficeConversionError, match="not installed"):
        convert_with_libreoffice(b"doc bytes", ".doc", "pdf")


def test_memory_cap_wraps_the_command_when_prlimit_exists(fake_soffice, monkeypatch):
    prlimit = fake_soffice / "bin" / "prlimit"
    prlimit.write_text('#!/usr/bin/env bash\nshift 2\nexec "$@"\n')  # drop "--as=N --", run the rest
    prlimit.chmod(0o755)
    calls = []
    real_popen = libreoffice.subprocess.Popen

    def recording_popen(cmd, **kwargs):
        calls.append(cmd)
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(libreoffice.subprocess, "Popen", recording_popen)
    monkeypatch.setenv("FAKE_SOFFICE_MODE", "ok")

    convert_with_libreoffice(b"doc bytes", ".doc", "pdf")

    assert Path(calls[0][0]).name == "prlimit"
    assert calls[0][1:4] == [f"--as={libreoffice.MEMORY_LIMIT_BYTES}", "--", "soffice"]
