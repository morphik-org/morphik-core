"""Unit tests for the bulk-ingestion throughput/reliability fixes.

Covers:
- embed-client transient retry + EmbeddingUnavailableError semantics
- worker transient-error classification
- blank-page detection on raw pixmap samples (pre-encode)
- _image_bytes omission in ColPali API mode
- S3 bucket-check caching, direct byte upload, and HEAD elimination
- base64 size arithmetic replacing HEAD-after-PUT
"""

import asyncio
import base64
import io
import json

import fitz
import httpx
import numpy as np
import pytest

from core.embedding.colpali_api_embedding_model import ColpaliApiEmbeddingModel, EmbeddingUnavailableError
from core.services import ingestion_service as ingestion_module
from core.services.ingestion_service import IngestionService
from core.storage.s3_storage import S3Storage
from core.vector_store.fast_multivector_store import _decoded_base64_size
from core.workers.ingestion_worker import _is_transient_error

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _npz_response_bytes(n: int = 1) -> bytes:
    buf = io.BytesIO()
    arrays = {f"emb_{i}": np.ones((4, 128), dtype=np.float32) for i in range(n)}
    np.savez(buf, count=n, input_type="text", **arrays)
    return buf.getvalue()


def _make_model(transport: httpx.MockTransport) -> ColpaliApiEmbeddingModel:
    model = object.__new__(ColpaliApiEmbeddingModel)
    model.api_key = "test-key"
    model.endpoints = ["http://embed-a/embeddings", "http://embed-b/embeddings"]
    model.healthy_endpoints = set(model.endpoints)
    model._endpoint_unhealthy_since = {}
    model._unhealthy_recovery_seconds = 60.0
    model._endpoint_latencies = {}
    model.endpoint = model.endpoints[0]
    model._latest_ingest_metrics = {}
    model._http_client = httpx.AsyncClient(transport=transport)
    return model


def _render_pixmap(color=(255, 255, 255), dot=False) -> fitz.Pixmap:
    doc = fitz.open()
    page = doc.new_page(width=100, height=100)
    if color != (255, 255, 255):
        rgb = tuple(c / 255 for c in color)
        page.draw_rect(fitz.Rect(0, 0, 100, 100), color=rgb, fill=rgb)
    if dot:
        page.draw_rect(fitz.Rect(40, 40, 60, 60), color=(0, 0, 0), fill=(0, 0, 0))
    pix = page.get_pixmap()
    doc.close()
    return pix


# ---------------------------------------------------------------------------
# embed client: transient retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_api_endpoint_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep())
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, content=_npz_response_bytes(1))

    model = _make_model(httpx.MockTransport(handler))
    result = await model._call_api_endpoint(model.endpoint, ["hello"], "text")
    assert calls["n"] == 3
    assert len(result) == 1 and result[0].shape == (4, 128)


@pytest.mark.asyncio
async def test_call_api_endpoint_exhausts_retries_and_raises_transient(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep())
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    model = _make_model(httpx.MockTransport(handler))
    with pytest.raises(EmbeddingUnavailableError):
        await model._call_api_endpoint(model.endpoint, ["hello"], "text")
    assert calls["n"] == 4  # initial + 3 retries


@pytest.mark.asyncio
async def test_call_api_endpoint_does_not_retry_413_or_400(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep())
    for status in (413, 400):
        calls = {"n": 0}

        def handler(request, _status=status):
            calls["n"] += 1
            return httpx.Response(_status)

        model = _make_model(httpx.MockTransport(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await model._call_api_endpoint(model.endpoint, ["hello"], "text")
        assert calls["n"] == 1, f"status {status} must not be retried"


@pytest.mark.asyncio
async def test_call_api_endpoint_retries_connect_errors(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep())
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=_npz_response_bytes(1))

    model = _make_model(httpx.MockTransport(handler))
    result = await model._call_api_endpoint(model.endpoint, ["hello"], "text")
    assert calls["n"] == 2
    assert len(result) == 1


@pytest.mark.asyncio
async def test_all_endpoints_failed_raises_embedding_unavailable(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep())

    def handler(request):
        return httpx.Response(503)

    model = _make_model(httpx.MockTransport(handler))
    with pytest.raises(EmbeddingUnavailableError):
        await model._embed_inputs_distributed([(0, "a"), (1, "b")], "text")


def _instant_sleep():
    real_sleep = asyncio.sleep

    async def fake_sleep(_delay):
        await real_sleep(0)

    return fake_sleep


# ---------------------------------------------------------------------------
# worker: transient classification
# ---------------------------------------------------------------------------


def test_transient_classification():
    assert _is_transient_error(EmbeddingUnavailableError("down"))
    assert _is_transient_error(httpx.ConnectError("refused"))
    assert _is_transient_error(httpx.ReadTimeout("slow"))
    assert _is_transient_error(ConnectionError("reset"))
    assert not _is_transient_error(ValueError("bad document"))
    assert not _is_transient_error(RuntimeError("some other failure"))

    # wrapped cause counts
    try:
        try:
            raise httpx.ConnectError("inner")
        except httpx.ConnectError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as wrapped:
        assert _is_transient_error(wrapped)


# ---------------------------------------------------------------------------
# render path: blank check + encoding
# ---------------------------------------------------------------------------


def test_blank_pixmap_detection_matches_old_behavior():
    svc = object.__new__(IngestionService)
    blank = _render_pixmap()
    content = _render_pixmap(dot=True)
    assert svc._is_blank_pixmap(blank) is True
    assert svc._is_blank_pixmap(content) is False
    # agreement with the legacy bytes-based check on the encoded PNGs
    assert svc._is_blank_image_bytes(blank.tobytes("png")) is True
    assert svc._is_blank_image_bytes(content.tobytes("png")) is False


def test_image_bytes_omitted_in_api_mode(monkeypatch):
    svc = object.__new__(IngestionService)
    payload = b"\x89PNG-fake"

    monkeypatch.setattr(ingestion_module.settings, "COLPALI_MODE", "local")
    chunk_local = svc._image_bytes_to_chunk(payload, mime_type="image/png")
    assert chunk_local.metadata["_image_bytes"] == payload

    monkeypatch.setattr(ingestion_module.settings, "COLPALI_MODE", "api")
    chunk_api = svc._image_bytes_to_chunk(payload, mime_type="image/png")
    assert "_image_bytes" not in chunk_api.metadata
    assert chunk_api.metadata["is_image"] is True
    assert chunk_api.content.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# S3 storage: bucket cache, direct byte upload, no HEADs
# ---------------------------------------------------------------------------


class RecordingS3Client:
    def __init__(self):
        self.calls = []
        self.objects = {}

    def head_bucket(self, Bucket):
        self.calls.append("head_bucket")

    def put_object(self, Bucket, Key, Body, ContentType=None, **kw):
        self.calls.append("put_object")
        self.objects[Key] = (Body, ContentType)

    def upload_fileobj(self, fobj, bucket, key, ExtraArgs=None):
        self.calls.append("upload_fileobj")
        self.objects[key] = (fobj.read(), (ExtraArgs or {}).get("ContentType"))

    def head_object(self, Bucket, Key):
        self.calls.append("head_object")
        return {"ContentLength": len(self.objects[Key][0])}

    class meta:
        region_name = "us-east-1"


def _make_storage(client) -> S3Storage:
    storage = object.__new__(S3Storage)
    storage.default_bucket = "bucket"
    storage.s3_client = client
    storage._upload_semaphore = asyncio.Semaphore(16)
    storage._buckets_verified = set()
    return storage


@pytest.mark.asyncio
async def test_upload_from_base64_single_head_bucket_and_no_temp_files():
    client = RecordingS3Client()
    storage = _make_storage(client)
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 100
    uri = "data:image/png;base64," + base64.b64encode(png).decode()

    await storage.upload_from_base64(uri, "chunks/a", bucket="bucket")
    await storage.upload_from_base64(uri, "chunks/b", bucket="bucket")
    await storage.upload_from_base64(uri, "chunks/c", bucket="bucket")

    assert client.calls.count("head_bucket") == 1, "bucket check must be cached per process"
    assert client.calls.count("put_object") == 3
    assert client.calls.count("head_object") == 0
    stored, content_type = client.objects["chunks/a.png"]
    assert stored == png
    assert content_type == "image/png"


@pytest.mark.asyncio
async def test_upload_file_bytes_uses_put_object_with_content_type():
    client = RecordingS3Client()
    storage = _make_storage(client)
    bucket, key = await storage.upload_file(b"raw-bytes", "k.bin", content_type="application/octet-stream")
    assert (bucket, key) == ("bucket", "k.bin")
    assert client.objects["k.bin"] == (b"raw-bytes", "application/octet-stream")


def test_decoded_base64_size_arithmetic():
    for payload in (b"", b"a", b"ab", b"abc", b"abcd", b"\x00" * 1000, b"x" * 12345):
        b64 = base64.b64encode(payload).decode()
        assert _decoded_base64_size(b64) == len(payload)
        assert _decoded_base64_size("data:image/png;base64," + b64) == len(payload)


# ---------------------------------------------------------------------------
# render pipeline end-to-end on a real PDF (pair return, blank skip)
# ---------------------------------------------------------------------------


def _small_pdf_with_blank() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=100, height=100)
    page.draw_rect(fitz.Rect(20, 20, 80, 80), color=(0, 0, 0), fill=(0.2, 0.4, 0.6))
    doc.new_page(width=100, height=100)  # blank page
    return doc.tobytes()


def test_render_pdf_returns_png_pairs_and_skips_blank():
    svc = object.__new__(IngestionService)
    pages = svc._render_pdf_with_pymupdf(_small_pdf_with_blank(), dpi=72, include_bytes=True)
    assert len(pages) == 1  # blank page skipped
    b64, raw = pages[0]
    assert b64.startswith("data:image/png;base64,")
    assert raw.startswith(b"\x89PNG")
    assert base64.b64decode(b64.split(",", 1)[1]) == raw


# ---------------------------------------------------------------------------
# chunk storage size accounting stays consistent with what was uploaded
# ---------------------------------------------------------------------------


def test_chunk_metadata_roundtrip_serializable():
    svc = object.__new__(IngestionService)
    chunk = svc._image_bytes_to_chunk(b"\x89PNGdata", mime_type="image/png")
    # metadata (minus raw bytes) must stay JSON serializable for storage paths
    meta = {k: v for k, v in chunk.metadata.items() if k != "_image_bytes"}
    json.dumps(meta)


# ---------------------------------------------------------------------------
# review-driven fixes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_path_retries_once_only(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep())
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    model = _make_model(httpx.MockTransport(handler))
    with pytest.raises(EmbeddingUnavailableError):
        await model.embed_for_query("hello")
    assert calls["n"] == 2, "query path must retry once, not the full ingestion ladder"


@pytest.mark.asyncio
async def test_retry_after_is_capped(monkeypatch):
    sleeps = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", recording_sleep)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "86400"})
        return httpx.Response(200, content=_npz_response_bytes(1))

    model = _make_model(httpx.MockTransport(handler))
    await model._call_api_endpoint(model.endpoint, ["hello"], "text")
    assert max(sleeps) <= 30.0, f"Retry-After must be capped at 30s, slept {max(sleeps)}"


def test_render_page_error_skips_page_not_document(monkeypatch):
    svc = object.__new__(IngestionService)

    calls = {"n": 0}
    real_blank = IngestionService._is_blank_pixmap  # staticmethod -> plain function

    def flaky_blank(pix, tolerance=2):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("render step blew up on page 1")
        return real_blank(pix, tolerance)

    monkeypatch.setattr(svc, "_is_blank_pixmap", flaky_blank)

    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page(width=100, height=100)
        page.draw_rect(fitz.Rect(20, 20, 80, 80), color=(0, 0, 0), fill=(0.3, 0.3, 0.3))
    pdf = doc.tobytes()

    pages = svc._render_pdf_with_pymupdf(pdf, dpi=72, include_bytes=True)
    assert len(pages) == 2, "one bad page must not abort the document"
