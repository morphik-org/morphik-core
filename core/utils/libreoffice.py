"""Run LibreOffice conversions without letting one bad file hurt the rest of the server."""

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_S = 300

# Address-space cap for one conversion. A malformed file can make LibreOffice allocate without
# bound, and the converter shares the machine with the API.
MEMORY_LIMIT_BYTES = 4 * 1024**3


class LibreOfficeConversionError(Exception):
    """LibreOffice could not convert a document."""


def convert_with_libreoffice(content: bytes, suffix: str, target: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> bytes:
    """Convert document bytes (for example a ``.doc``) to ``target`` (``pdf``, ``docx``, ...).

    Each call gets a private LibreOffice profile, its own process group and a memory cap. Without the
    private profile a second conversion hands its file to an already running instance and waits on it;
    without the process group a timeout kills only the ``soffice`` launcher script and leaves the real
    converter (``soffice.bin``) running.

    Raises:
        LibreOfficeConversionError: LibreOffice is missing, failed, timed out or produced no output.
    """
    if not shutil.which("soffice"):
        raise LibreOfficeConversionError("LibreOffice (soffice) is not installed")

    with tempfile.TemporaryDirectory(prefix="morphik-soffice-") as work_dir:
        work = Path(work_dir)
        source = work / f"source{suffix}"
        source.write_bytes(content)
        out_dir = work / "out"
        cmd = [
            "soffice",
            f"-env:UserInstallation={(work / 'profile').as_uri()}",
            "--headless",
            "--convert-to",
            target,
            "--outdir",
            str(out_dir),
            str(source),
        ]
        prlimit = shutil.which("prlimit")  # util-linux; not available on macOS, where we run uncapped
        if prlimit:
            cmd = [prlimit, f"--as={MEMORY_LIMIT_BYTES}", "--", *cmd]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except BaseException as exc:
            # Timeout or interruption: end the launcher and everything it started.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise LibreOfficeConversionError(
                    f"LibreOffice did not finish converting {suffix} to {target} within {timeout_s:.0f}s"
                ) from None
            raise

        if proc.returncode != 0:
            raise LibreOfficeConversionError(
                f"LibreOffice exited with code {proc.returncode} converting {suffix} to {target}: {stderr.strip()[:500]}"
            )
        output = out_dir / f"source.{target}"
        if not output.is_file() or output.stat().st_size == 0:
            raise LibreOfficeConversionError(
                f"LibreOffice produced no {target} output for {suffix}: {stderr.strip()[:500]}"
            )
        return output.read_bytes()
