"""Request-scoped profiling middleware using yappi.

Set ``enable_profiling = true`` in [service] section of morphik.toml to enable.
For each HTTP request we create a fresh yappi profile and store it as
``logs/profile_<timestamp>.prof`` which can later be opened with ``snakeviz``
or converted to callgrind with ``pyprof2calltree``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from core.config import get_settings

try:
    import yappi  # type: ignore
except ImportError:  # pragma: no cover
    yappi = None  # type: ignore  # fallback when profiling deps not installed

logger = logging.getLogger("profiler")

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

settings = get_settings()


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Starts yappi before each request and stops it afterwards."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        # Paths we do NOT want to profile (low-value, high-frequency endpoints)
        SKIP_PATH_SUFFIXES = ("/status", "/documents", "/folders")

        if (
            not settings.ENABLE_PROFILING
            or yappi is None
            or any(request.url.path.endswith(suf) for suf in SKIP_PATH_SUFFIXES)
        ):
            # Profiling disabled or yappi unavailable – continue normally.
            return await call_next(request)

        # Clean previous stats to avoid cross-request leakage
        yappi.clear_stats()
        yappi.set_clock_type("cpu")  # wall for real-time, cpu for deterministic
        yappi.start()

        start_ts = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration = time.perf_counter() - start_ts
            yappi.stop()

            ts = int(start_ts)
            fname = f"logs/profile_{ts}.prof"
            try:
                yappi.get_func_stats().save(fname, type="pstat")
                logger.info(
                    "Saved yappi stats for %s %s (%.3fs) to %s", request.method, request.url.path, duration, fname
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to save yappi stats: %s", exc)

        return response
