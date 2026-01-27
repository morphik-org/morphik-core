import pytest
from morphik.async_ import AsyncMorphik
from morphik.models import DetailedHealthCheckResponse, LogResponse, RequeueIngestionJob
from morphik.sync import Morphik


def _make_sync_client():
    client = Morphik()
    calls = []

    def fake_request(method, endpoint, data=None, files=None, params=None):
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "data": data,
                "params": params,
            }
        )
        if endpoint == "logs/":
            return [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "user_id": "u1",
                    "operation_type": "query",
                    "status": "ok",
                    "tokens_used": 1,
                    "duration_ms": 2.0,
                }
            ]
        if endpoint == "health":
            return {
                "status": "healthy",
                "services": [{"name": "db", "status": "healthy"}],
                "timestamp": "2024-01-01T00:00:00Z",
            }
        if endpoint == "ingest/requeue":
            return {"results": []}
        return {"ok": True}

    client._request = fake_request  # type: ignore[attr-defined]
    return client, calls


async def _make_async_client():
    client = AsyncMorphik()
    calls = []

    async def fake_request(method, endpoint, data=None, files=None, params=None):
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "data": data,
                "params": params,
            }
        )
        if endpoint == "logs/":
            return [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "user_id": "u1",
                    "operation_type": "query",
                    "status": "ok",
                    "tokens_used": 1,
                    "duration_ms": 2.0,
                }
            ]
        if endpoint == "health":
            return {
                "status": "healthy",
                "services": [{"name": "db", "status": "healthy"}],
                "timestamp": "2024-01-01T00:00:00Z",
            }
        if endpoint == "ingest/requeue":
            return {"results": []}
        return {"ok": True}

    client._request = fake_request  # type: ignore[attr-defined]
    return client, calls


def test_sync_app_ops_payloads():
    client, calls = _make_sync_client()
    try:
        client.list_apps(org_id="org", app_id_filter={"$eq": "a"}, limit=501, offset=-1)
        call = calls.pop()
        assert call["endpoint"] == "apps"
        assert call["params"]["limit"] == 500
        assert call["params"]["offset"] == 0
        assert call["params"]["org_id"] == "org"
        assert call["params"]["app_id_filter"] == '{"$eq": "a"}'

        client.rename_app(new_name="new", app_id="app-1")
        call = calls.pop()
        assert call["endpoint"] == "apps/rename"
        assert call["params"] == {"new_name": "new", "app_id": "app-1"}

        client.rotate_app_token(app_name="demo", expiry_days=10)
        call = calls.pop()
        assert call["endpoint"] == "apps/rotate_token"
        assert call["params"] == {"app_name": "demo", "expiry_days": 10}

        client.create_app(name="demo")
        call = calls.pop()
        assert call["endpoint"] == "cloud/generate_uri"
        assert call["data"] == {"name": "demo"}

        client.requeue_ingestion_jobs(jobs=[RequeueIngestionJob(external_id="doc-1")])
        call = calls.pop()
        assert call["endpoint"] == "ingest/requeue"
        assert call["data"]["jobs"] == [{"external_id": "doc-1", "use_colpali": None}]

        logs = client.get_logs(limit=2, hours=5)
        call = calls.pop()
        assert call["endpoint"] == "logs/"
        assert isinstance(logs[0], LogResponse)

        health = client.get_health()
        call = calls.pop()
        assert call["endpoint"] == "health"
        assert isinstance(health, DetailedHealthCheckResponse)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_app_ops_payloads():
    client, calls = await _make_async_client()
    try:
        await client.list_apps(org_id="org", app_name_filter={"$eq": "demo"}, limit=1, offset=5)
        call = calls.pop()
        assert call["endpoint"] == "apps"
        assert call["params"]["limit"] == 1
        assert call["params"]["offset"] == 5
        assert call["params"]["app_name_filter"] == '{"$eq": "demo"}'

        await client.rename_app(new_name="new", app_name="old")
        call = calls.pop()
        assert call["endpoint"] == "apps/rename"
        assert call["params"] == {"new_name": "new", "app_name": "old"}

        await client.rotate_app_token(app_id="app-1")
        call = calls.pop()
        assert call["endpoint"] == "apps/rotate_token"
        assert call["params"] == {"app_id": "app-1"}

        await client.create_app(name="demo")
        call = calls.pop()
        assert call["endpoint"] == "cloud/generate_uri"
        assert call["data"] == {"name": "demo"}

        await client.requeue_ingestion_jobs(include_all=True, statuses=["failed"])
        call = calls.pop()
        assert call["endpoint"] == "ingest/requeue"
        assert call["data"]["include_all"] is True
        assert call["data"]["statuses"] == ["failed"]

        logs = await client.get_logs()
        call = calls.pop()
        assert call["endpoint"] == "logs/"
        assert isinstance(logs[0], LogResponse)

        health = await client.get_health()
        call = calls.pop()
        assert call["endpoint"] == "health"
        assert isinstance(health, DetailedHealthCheckResponse)
    finally:
        await client.close()
