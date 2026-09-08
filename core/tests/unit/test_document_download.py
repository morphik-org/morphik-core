"""Exercise the real documents router with synthetic bytes and isolated services."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import unquote_to_bytes

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core.models.auth import AuthContext
from core.models.documents import Document
from core.utils.content_disposition import build_content_disposition

pytestmark = pytest.mark.unit

REPORTED_FILENAMES = [
    "8 Major Food Allergens _ 🕮Knowledge _ Salesforce.html",
    "B Vitamins and How They Support The Body _ 🕮Knowledge _ Salesforce.html",
    "Age Restricted Ingredients 18+ _ 🕮Knowledge _ Salesforce.html",
]
FILE_CONTENT = b"<!doctype html><meta charset='utf-8'><p>Synthetic download fixture.</p>\n"


def _load_module(name, relative_path):
    """Load production code without leaving mocked imports cached for other tests."""
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def download_route(monkeypatch):
    doc = Document(
        external_id="document-1",
        filename="example.html",
        content_type="text/html",
        storage_info={"bucket": "test-bucket", "key": "original-file"},
    )
    auth = AuthContext(user_id="test-user", app_id="test-app")
    service = SimpleNamespace(
        db=SimpleNamespace(get_document=AsyncMock(return_value=doc)),
        storage=SimpleNamespace(
            download_file=AsyncMock(return_value=FILE_CONTENT),
            get_download_url=AsyncMock(return_value="https://storage.example.test/original-file"),
        ),
    )
    telemetry = Mock()
    telemetry.track.side_effect = lambda **kwargs: lambda handler: handler

    # Import-time singletons otherwise initialize database, models and telemetry.
    # Keep the production router, response classes and auth dependency intact.
    stubs = {
        "core.config": {
            "get_settings": lambda: SimpleNamespace(
                bypass_auth_mode=False,
                JWT_SECRET_KEY="synthetic-test-secret",
                JWT_ALGORITHM="HS256",
            )
        },
        "core.database.postgres_database": {
            "InvalidMetadataFilterError": type("InvalidMetadataFilterError", (ValueError,), {})
        },
        "core.services.telemetry": {"TelemetryService": lambda: telemetry},
        "core.services_init": {"document_service": service, "ingestion_service": Mock()},
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    auth_module = _load_module("core.auth_utils", "auth_utils.py")
    monkeypatch.setitem(sys.modules, "core.auth_utils", auth_module)
    documents = _load_module("core.routes.documents", "routes/documents.py")
    authenticate = AsyncMock(return_value=auth)

    async def authenticated():
        return await authenticate()

    app = FastAPI()
    app.include_router(documents.router)
    app.dependency_overrides[documents.verify_token] = authenticated
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, doc=doc, auth=auth, authenticate=authenticate, service=service)


def _assert_download(route, response, expected_filename, expected_fallback=None):
    assert response.status_code == 200
    assert response.content == route.service.storage.download_file.return_value
    assert response.headers["content-length"] == str(len(response.content))
    header = response.headers["content-disposition"]
    assert header.isascii()
    assert all(32 <= ord(char) < 127 for char in header)
    assert len(response.headers.get_list("content-disposition")) == 1
    # Validate both parameters, including quoted-string and RFC 8187 syntax.
    match = re.fullmatch(
        r"""inline; filename="([^"\\]*)"; filename\*=UTF-8''((?:[A-Za-z0-9!#$&+.^_`|~-]|%[0-9A-F]{2})+)""",
        header,
    )
    assert match is not None, header
    fallback, encoded = match.groups()
    assert fallback
    if expected_fallback is not None:
        assert fallback == expected_fallback
    assert unquote_to_bytes(encoded).decode("utf-8") == expected_filename
    route.authenticate.assert_awaited_once_with()
    route.service.db.get_document.assert_awaited_once_with("document-1", route.auth)
    route.service.storage.download_file.assert_awaited_once_with("test-bucket", "original-file")


@pytest.mark.parametrize("filename", REPORTED_FILENAMES)
def test_reported_unicode_filenames_download(download_route, filename):
    download_route.doc.filename = filename

    response = download_route.client.get("/documents/document-1/file")

    _assert_download(download_route, response, filename, filename.replace("🕮", "_"))
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert download_route.doc.filename == filename


@pytest.mark.parametrize("filename", ["café.pdf", "日本語.txt", "e\u0301 🧪.txt", "🕮", "résumé + 100% #1's*.txt"])
def test_unicode_filename_round_trip(download_route, filename):
    download_route.doc.filename = filename

    response = download_route.client.get("/documents/document-1/file")

    _assert_download(download_route, response, filename)
    assert download_route.doc.filename == filename


@pytest.mark.parametrize("filename", ["report.txt", "a report 18+.html", None, ""])
def test_ascii_and_missing_filenames(download_route, filename):
    download_route.doc.filename = filename

    response = download_route.client.get("/documents/document-1/file")

    _assert_download(download_route, response, filename or "document", filename or "document")
    assert download_route.doc.filename == filename


@pytest.mark.parametrize(
    ("filename", "expected_filename", "fallback"),
    [
        ('quoted"name.txt', 'quoted"name.txt', "quoted_name.txt"),
        ("back\\slash.txt", "back\\slash.txt", "back_slash.txt"),
        ('a"; filename="injected.html', 'a"; filename="injected.html', "a_; filename=_injected.html"),
        ("report\r\nX-Injected: yes.txt", "report__X-Injected: yes.txt", "report__X-Injected: yes.txt"),
        ("report\rX-Injected: yes.txt", "report_X-Injected: yes.txt", "report_X-Injected: yes.txt"),
        ("report\nX-Injected: yes.txt", "report_X-Injected: yes.txt", "report_X-Injected: yes.txt"),
        ("control\x00\t\x7f.txt", "control___.txt", "control___.txt"),
        ("literal%0D%0A.txt", "literal%0D%0A.txt", "literal_0D_0A.txt"),
    ],
)
def test_header_injection_is_safe(download_route, filename, expected_filename, fallback):
    download_route.doc.filename = filename

    response = download_route.client.get("/documents/document-1/file")

    _assert_download(download_route, response, expected_filename, fallback)
    assert "x-injected" not in response.headers
    assert download_route.doc.filename == filename


@pytest.mark.parametrize(
    ("content_type", "expected_type", "content"),
    [
        ("application/pdf", "application/pdf", b"%PDF-1.7\nSynthetic\x00\xff\r\n"),
        ("text/plain; charset=iso-8859-1", "text/plain; charset=iso-8859-1", b"caf\xe9\r\n"),
        (None, "application/octet-stream", bytes(range(256))),
        ("", "application/octet-stream", b""),
    ],
)
def test_download_preserves_bytes_and_content_type(download_route, content_type, expected_type, content):
    download_route.doc.content_type = content_type
    download_route.service.storage.download_file.return_value = content

    response = download_route.client.get("/documents/document-1/file")

    _assert_download(download_route, response, "example.html", "example.html")
    assert response.headers["content-type"] == expected_type


@pytest.mark.parametrize("endpoint", ["file", "download_url"])
@pytest.mark.parametrize("authorization", [None, "Basic invalid", "Bearer invalid"])
def test_download_rejects_unauthenticated_requests(download_route, endpoint, authorization):
    # Exercise the existing verify_token dependency for missing/malformed auth.
    download_route.app.dependency_overrides.clear()
    headers = {} if authorization is None else {"Authorization": authorization}

    response = download_route.client.get(f"/documents/document-1/{endpoint}", headers=headers)

    assert response.status_code == 401
    if authorization is None:
        assert response.json() == {"detail": "Missing authorization header"}
        assert response.headers["www-authenticate"] == "Bearer"
    download_route.service.db.get_document.assert_not_awaited()
    download_route.service.storage.download_file.assert_not_awaited()
    download_route.service.storage.get_download_url.assert_not_awaited()


@pytest.mark.parametrize("endpoint", ["file", "download_url"])
def test_download_preserves_auth_rejection(download_route, endpoint):
    download_route.authenticate.side_effect = HTTPException(status_code=403, detail="Access denied")

    response = download_route.client.get(f"/documents/document-1/{endpoint}")

    assert response.status_code == 403
    assert response.json() == {"detail": "Access denied"}
    download_route.service.db.get_document.assert_not_awaited()
    download_route.service.storage.download_file.assert_not_awaited()
    download_route.service.storage.get_download_url.assert_not_awaited()


@pytest.mark.parametrize(
    ("endpoint", "detail"), [("file", "Document not found: document-1"), ("download_url", "Document not found")]
)
def test_download_document_missing_or_inaccessible(download_route, endpoint, detail):
    # The database returns None for both missing documents and ACL misses.
    download_route.service.db.get_document.return_value = None

    response = download_route.client.get(f"/documents/document-1/{endpoint}")

    assert response.status_code == 404
    assert response.json() == {"detail": detail}
    download_route.service.db.get_document.assert_awaited_once_with("document-1", download_route.auth)
    download_route.service.storage.download_file.assert_not_awaited()
    download_route.service.storage.get_download_url.assert_not_awaited()


@pytest.mark.parametrize("endpoint", ["file", "download_url"])
@pytest.mark.parametrize("storage_info", [None, {}, {"bucket": "test-bucket"}, {"key": "original-file"}])
def test_download_missing_storage_info(download_route, endpoint, storage_info):
    download_route.doc.storage_info = storage_info

    response = download_route.client.get(f"/documents/document-1/{endpoint}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document file not found in storage"}
    download_route.service.storage.download_file.assert_not_awaited()
    download_route.service.storage.get_download_url.assert_not_awaited()


def test_download_missing_stored_file(download_route):
    download_route.service.storage.download_file.side_effect = FileNotFoundError("synthetic missing file")

    response = download_route.client.get("/documents/document-1/file")

    assert response.status_code == 404
    assert response.json() == {"detail": "File not found in storage: synthetic missing file"}


@pytest.mark.parametrize("filename", REPORTED_FILENAMES + ["report.txt", None])
def test_download_url_preserves_response_and_stored_filename(download_route, filename):
    download_route.doc.filename = filename

    response = download_route.client.get("/documents/document-1/download_url?expires_in=120")

    assert response.status_code == 200
    # The existing response model exposes only the URL and its expiry.
    assert response.json() == {
        "download_url": "https://storage.example.test/original-file",
        "expires_in": 120,
    }
    assert "content-disposition" not in response.headers
    assert download_route.doc.filename == filename
    download_route.service.db.get_document.assert_awaited_once_with("document-1", download_route.auth)
    download_route.service.storage.get_download_url.assert_awaited_once_with(
        "test-bucket", "original-file", expires_in=120
    )
    download_route.service.storage.download_file.assert_not_awaited()


def test_shared_builder_supports_attachment():
    assert build_content_disposition("café.pdf", disposition="attachment") == (
        "attachment; filename=\"caf_.pdf\"; filename*=UTF-8''caf%C3%A9.pdf"
    )


def test_shared_builder_rejects_invalid_disposition():
    with pytest.raises(ValueError, match="Unsupported content disposition"):
        build_content_disposition("report.txt", disposition="inline\r\nX-Injected: yes")
