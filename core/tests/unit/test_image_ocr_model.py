"""[parser] image_ocr_model sends raster images to a vision model instead of Docling's local OCR."""

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from PIL import Image

from core import config
from core.parser import morphik_parser
from core.parser.morphik_parser import OCR_MAX_SIDE, MorphikParser

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

REGISTERED = {
    "vision_ocr": {"model_name": "gpt-4.1-mini", "vision": True, "api_base": "https://llm.internal/v1"},
    # As documented in DOCKER.md. The conftest runs LiteLLM on its bundled model list, which predates gpt-6.1-sol.
    "reasoning_ocr": {"model_name": "openai/gpt-6.1-sol", "reasoning_effort": "low"},
    "local_ocr": {"model_name": "ollama_chat/qwen2.5vl:latest", "api_base": "http://localhost:11434", "vision": True},
}


def _image(fmt: str, size=(40, 20)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=fmt)
    return output.getvalue()


def _sent_image(call) -> Image.Image:
    url = call["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


@pytest.fixture
def make_parser(monkeypatch):
    monkeypatch.setattr(
        morphik_parser, "get_settings", lambda: SimpleNamespace(REGISTERED_MODELS=REGISTERED, PARSER_MODE="local")
    )

    def make(image_ocr_model="vision_ocr"):
        parser = MorphikParser(image_ocr_model=image_ocr_model)

        async def docling(file, filename):
            docling_calls.append(filename)
            return "docling text"

        parser._parse_document_local = docling
        parser._parse_document_local_deep = docling
        return parser

    docling_calls: list = []
    make.docling_calls = docling_calls
    return make


@pytest.fixture
def completions(monkeypatch):
    calls = []

    async def fake_acompletion(**params):
        calls.append(params)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Order #4521\nStatus: Open"))])

    monkeypatch.setattr(morphik_parser.litellm, "acompletion", fake_acompletion)
    return calls


@pytest.mark.asyncio
async def test_png_is_read_by_the_vision_model(make_parser, completions):
    parser = make_parser()

    metadata, text = await parser.parse_file_to_text(_image("PNG"), "image.png")

    assert (metadata, text) == ({}, "Order #4521\nStatus: Open")
    assert make_parser.docling_calls == []
    (call,) = completions
    assert call["model"] == "gpt-4.1-mini"
    assert call["api_base"] == "https://llm.internal/v1"
    assert "vision" not in call
    assert _sent_image(call).size == (40, 20)


@pytest.mark.asyncio
async def test_formats_the_api_rejects_are_sent_as_png(make_parser, completions):
    parser = make_parser()

    await parser.parse_file_to_text(_image("BMP"), "scan.bmp")
    await parser.parse_file_to_text(_image("TIFF"), "scan.tiff")

    assert [_sent_image(call).format for call in completions] == ["PNG", "PNG"]


@pytest.mark.asyncio
async def test_large_images_are_downscaled(make_parser, completions):
    parser = make_parser()

    await parser.parse_file_to_text(_image("JPEG", size=(4096, 1024)), "wide.jpg")

    assert _sent_image(completions[0]).size == (OCR_MAX_SIDE, OCR_MAX_SIDE // 4)


@pytest.mark.asyncio
async def test_without_the_setting_images_use_docling(make_parser, completions):
    parser = make_parser(image_ocr_model=None)

    _, text = await parser.parse_file_to_text(_image("PNG"), "image.png")

    assert text == "docling text"
    assert completions == []


@pytest.mark.asyncio
async def test_other_files_still_use_docling(make_parser, completions):
    parser = make_parser()

    _, text = await parser.parse_file_to_text(b"%PDF-1.4\n%fake", "spec.pdf")

    assert text == "docling text"
    assert completions == []


@pytest.mark.asyncio
async def test_deep_fallback_does_not_rerun_local_ocr_on_images(make_parser, completions):
    parser = make_parser()

    metadata, text = await parser.parse_file_to_text_deep(_image("PNG"), "image.png")

    assert (metadata, text) == ({"parse_fallback": "deep"}, "")
    assert make_parser.docling_calls == []


@pytest.mark.asyncio
async def test_rate_limits_are_retryable(make_parser, monkeypatch):
    async def rate_limited(**params):
        raise litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4.1-mini")

    monkeypatch.setattr(morphik_parser.litellm, "acompletion", rate_limited)
    parser = make_parser()

    # The ingestion worker retries ConnectionError instead of failing the document.
    with pytest.raises(ConnectionError):
        await parser.parse_file_to_text(_image("PNG"), "image.png")


@pytest.mark.asyncio
async def test_auth_errors_fail_without_retry(make_parser, monkeypatch):
    async def unauthorized(**params):
        raise litellm.AuthenticationError("bad key", llm_provider="openai", model="gpt-4.1-mini")

    monkeypatch.setattr(morphik_parser.litellm, "acompletion", unauthorized)
    parser = make_parser()

    with pytest.raises(litellm.AuthenticationError):
        await parser.parse_file_to_text(_image("PNG"), "image.png")


def test_unknown_model_key_is_rejected(make_parser):
    with pytest.raises(ValueError, match="registered_models"):
        make_parser(image_ocr_model="missing")


@pytest.mark.asyncio
async def test_openai_models_get_max_completion_tokens(make_parser, completions):
    # OpenAI's reasoning models reject max_tokens, and LiteLLM only renames it for models it already knows.
    await make_parser().parse_file_to_text(_image("PNG"), "image.png")
    await make_parser("reasoning_ocr").parse_file_to_text(_image("PNG"), "image.png")

    for call in completions:
        assert "max_tokens" not in call
        assert call["max_completion_tokens"] == morphik_parser.OCR_MAX_OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_reasoning_effort_is_passed_through(make_parser, completions):
    await make_parser("reasoning_ocr").parse_file_to_text(_image("PNG"), "image.png")

    (call,) = completions
    assert call["model"] == "openai/gpt-6.1-sol"
    assert call["reasoning_effort"] == "low"
    assert call["allowed_openai_params"] == ["reasoning_effort"]


@pytest.mark.asyncio
async def test_other_providers_keep_max_tokens(make_parser, completions):
    await make_parser("local_ocr").parse_file_to_text(_image("PNG"), "image.png")

    (call,) = completions
    assert call["max_tokens"] == morphik_parser.OCR_MAX_OUTPUT_TOKENS
    assert "max_completion_tokens" not in call


@pytest.mark.asyncio
async def test_running_out_of_tokens_before_any_text_fails_the_document(make_parser, monkeypatch):
    async def all_reasoning(**params):
        message = SimpleNamespace(content="")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="length")])

    monkeypatch.setattr(morphik_parser.litellm, "acompletion", all_reasoning)

    with pytest.raises(ValueError, match="reasoning_effort"):
        await make_parser("reasoning_ocr").parse_file_to_text(_image("PNG"), "image.png")


@pytest.fixture
def load_docker_config(tmp_path, monkeypatch):
    """Load morphik.docker.toml with one extra registered model set as image_ocr_model."""

    def load(model_name):
        toml = (REPO_ROOT / "morphik.docker.toml").read_text()
        toml = toml.replace(
            "[registered_models]\n", f'[registered_models]\nocr_model = {{ model_name = "{model_name}" }}\n', 1
        )
        toml = toml.replace('# image_ocr_model = "openai_gpt4-1-mini"', 'image_ocr_model = "ocr_model"', 1)
        (tmp_path / "morphik.toml").write_text(toml)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("POSTGRES_URI", "postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
        config.get_settings.cache_clear()
        try:
            return config.get_settings()
        finally:
            config.get_settings.cache_clear()

    return load


def test_documented_reasoning_model_loads(load_docker_config):
    settings = load_docker_config("openai/gpt-6.1-sol")

    assert settings.IMAGE_OCR_MODEL == "ocr_model"
    assert litellm.get_llm_provider(settings.REGISTERED_MODELS["ocr_model"]["model_name"])[1] == "openai"


def test_model_missing_from_litellm_list_fails_at_startup(load_docker_config):
    # LiteLLM 1.74.7's bundled list doesn't have gpt-6.1-sol. Without the check, every image fails at ingest.
    with pytest.raises(ValueError, match=r'ocr_model.*use model_name = "openai/gpt-6.1-sol"'):
        load_docker_config("gpt-6.1-sol")
