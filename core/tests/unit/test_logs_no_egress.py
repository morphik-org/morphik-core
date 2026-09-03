"""No-egress behavior for historical log requests."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")

from core.models.auth import AuthContext  # noqa: E402
from core.routes import logs  # noqa: E402


@pytest.mark.asyncio
async def test_historical_logs_do_not_call_proxy_when_telemetry_is_disabled(monkeypatch):
    async def unexpected_proxy_call(**kwargs):
        raise AssertionError("historical log proxy must not be called")

    monkeypatch.setattr(logs, "get_settings", lambda: SimpleNamespace(TELEMETRY_ENABLED=False))
    monkeypatch.setattr(logs, "_query_proxy", unexpected_proxy_call)

    result = await logs.get_logs(
        auth=AuthContext(user_id="iqor-user", app_id="iqor-app"),
        limit=100,
        hours=24,
        op_type=None,
        status_filter=None,
    )

    assert result == []
