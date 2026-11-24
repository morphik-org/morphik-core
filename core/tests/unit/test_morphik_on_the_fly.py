import json

import httpx
import pytest

from core.config import get_settings


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
    from core.services import morphik_on_the_fly_structured_output

    yield morphik_on_the_fly_structured_output
    get_settings.cache_clear()


@pytest.fixture
def requests(monkeypatch, service):
    seen = []

    def mock_request(request):
        body = json.loads(request.content)
        seen.append((request.url, body))
        if request.url.host == "generativelanguage.googleapis.com" and request.url.path.endswith(
            "gemini-2.5-flash:generateContent"
        ):
            return httpx.Response(404, json={"error": {"message": "Model unavailable to new projects"}})
        if request.url.path.endswith("gemini-3.5-flash-lite:generateContent"):
            config = body["generationConfig"]
            if "temperature" in config or "thinkingBudget" in config.get("thinkingConfig", {}):
                return httpx.Response(400, json={"error": {"message": "Deprecated generation parameter"}})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"invoice_number": "A-42"}'}]}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_request))
    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **_: client)
    return seen


@pytest.mark.asyncio
async def test_default_extracts_metadata_with_current_gemini(service, requests, monkeypatch):
    monkeypatch.delenv("GEMINI_API_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_METADATA_MODEL", raising=False)
    get_settings.cache_clear()
    config = get_settings()
    monkeypatch.setattr(service, "settings", config)

    result = await service.generate_morphik_on_the_fly_content(
        prompt="Extract metadata from this document",
        schema={"invoice_number": "string"},
        document_bytes=b"invoice fixture",
        api_key="fake-api-key",
    )

    assert result.structured_output == {"invoice_number": "A-42"}
    assert len(requests) == 1
    url, body = requests[0]
    assert url.host == "generativelanguage.googleapis.com"
    assert url.path == "/v1beta/models/gemini-3.5-flash-lite:generateContent"
    generation_config = body["generationConfig"]
    assert "temperature" not in generation_config
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseSchema"]["properties"]["invoice_number"]["type"] == "STRING"
    assert body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "application/pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_model", "explicit_model", "expected_model"),
    [
        ("gemini-custom", None, "gemini-custom"),
        ("gemini-custom", "models/gemini-override", "gemini-override"),
    ],
)
async def test_model_overrides_keep_legacy_parameters(
    service, requests, monkeypatch, configured_model, explicit_model, expected_model
):
    monkeypatch.setenv("GEMINI_METADATA_MODEL", configured_model)
    get_settings.cache_clear()
    config = get_settings()
    monkeypatch.setattr(service, "settings", config)

    await service.generate_morphik_on_the_fly_content(
        prompt="Extract text",
        model=explicit_model,
        api_key="fake-api-key",
        temperature=0.125,
        thinking_budget=64,
    )

    url, body = requests[0]
    assert url.path == f"/v1beta/models/{expected_model}:generateContent"
    assert body["generationConfig"] == {"temperature": 0.125, "thinkingConfig": {"thinkingBudget": 64}}


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_endpoint", [True, False])
async def test_custom_endpoint_keeps_implicit_model(service, requests, monkeypatch, configured_endpoint):
    gateway = "https://gemini-gateway.example"
    monkeypatch.delenv("GEMINI_METADATA_MODEL", raising=False)
    if configured_endpoint:
        monkeypatch.setenv("GEMINI_API_BASE_URL", gateway)
    else:
        monkeypatch.delenv("GEMINI_API_BASE_URL", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(service, "settings", get_settings())

    await service.generate_morphik_on_the_fly_content(
        prompt="Extract text",
        api_key="fake-api-key",
        api_base_url=None if configured_endpoint else gateway,
        thinking_budget=64,
    )

    url, body = requests[0]
    assert url.host == "gemini-gateway.example"
    assert url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert body["generationConfig"] == {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 64}}


@pytest.mark.asyncio
async def test_native_flash_lite_rejects_legacy_budget_before_request(service, requests, monkeypatch):
    monkeypatch.delenv("GEMINI_API_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_METADATA_MODEL", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(service, "settings", get_settings())

    with pytest.raises(service.MorphikOnTheFlyContentError, match="thinking_budget"):
        await service.generate_morphik_on_the_fly_content(
            prompt="Extract text",
            api_key="fake-api-key",
            thinking_budget=64,
        )
    assert requests == []
