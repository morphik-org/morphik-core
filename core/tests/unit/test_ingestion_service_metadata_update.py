"""Unit tests for ingestion metadata-only update validation."""

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException

from core.models.auth import AuthContext
from core.models.documents import Document

os.environ.setdefault("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")

# Avoid importing optional embedding backends through core.embedding.__init__.
_embedding_package = sys.modules.get("core.embedding")
if _embedding_package is None:
    _embedding_stub = ModuleType("core.embedding")
    _embedding_stub.__path__ = [str(Path(__file__).resolve().parents[2] / "embedding")]
    sys.modules["core.embedding"] = _embedding_stub

from core.services.ingestion_service import IngestionService  # noqa: E402

if _embedding_package is None:
    sys.modules.pop("core.embedding", None)


class FakeDatabase:
    def __init__(self, doc: Document):
        self.doc = doc
        self.update_calls = []

    async def get_document(self, document_id: str, auth: AuthContext):
        if document_id == self.doc.external_id:
            return self.doc
        return None

    async def check_access(self, document_id: str, auth: AuthContext, required_permission: str = "read") -> bool:
        return document_id == self.doc.external_id

    async def update_document(self, document_id: str, updates, auth: AuthContext, metadata_bundle=None):
        self.update_calls.append(
            {
                "document_id": document_id,
                "updates": updates,
                "auth": auth,
                "metadata_bundle": metadata_bundle,
            }
        )
        return True


class FakeStoreFailureDatabase(FakeDatabase):
    def __init__(self, doc: Document):
        super().__init__(doc)
        self.store_calls = []

    async def store_document(self, document: Document, auth: AuthContext, metadata_bundle=None):
        self.store_calls.append(
            {
                "document": document,
                "auth": auth,
                "metadata_bundle": metadata_bundle,
            }
        )
        return False


class FakeRedis:
    def __init__(self):
        self.calls = []

    async def enqueue_job(self, function_name, **payload):
        self.calls.append({"function_name": function_name, "payload": payload})
        return SimpleNamespace(job_id=payload["_job_id"])


def _auth() -> AuthContext:
    return AuthContext(user_id="user-1", app_id="app-1")


def _document() -> Document:
    return Document(
        external_id="doc-1",
        content_type="text/plain",
        filename="report.txt",
        metadata={
            "external_id": "doc-1",
            "folder_name": "/Team/Reports",
            "folder_id": "folder-1",
            "custom": "old",
        },
        metadata_types={
            "external_id": "string",
            "folder_name": "string",
            "folder_id": "string",
            "custom": "string",
        },
        folder_name="Reports",
        folder_path="/Team/Reports",
        folder_id="folder-1",
        app_id="app-1",
    )


def _service(doc: Document):
    db = FakeDatabase(doc)
    return IngestionService(db, None, None, None, None), db


@pytest.mark.asyncio
async def test_file_ingest_aborts_when_initial_document_store_fails():
    doc = _document()
    db = FakeStoreFailureDatabase(doc)
    service = IngestionService(db, None, None, None, None)

    async def noop_limit_check(auth, content_length, document_id):
        return None

    service._verify_ingest_and_storage_limits = noop_limit_check
    service._resolve_content_type = lambda content, filename, content_type: "text/plain"

    with pytest.raises(HTTPException) as exc_info:
        await service.ingest_file_content(
            file_content_bytes=b"hello",
            filename="report.txt",
            content_type="text/plain",
            metadata={"custom": "value"},
            auth=_auth(),
            redis=None,
            metadata_types={"custom": "string"},
            use_colpali=False,
            external_id="doc-1",
        )

    assert exc_info.value.status_code == 409
    assert "doc-1" in exc_info.value.detail
    assert len(db.store_calls) == 1
    assert db.update_calls == []


@pytest.mark.asyncio
async def test_metadata_only_update_allows_unchanged_managed_metadata_fields():
    doc = _document()
    service, db = _service(doc)

    updated = await service.update_document(
        document_id="doc-1",
        auth=_auth(),
        metadata={
            "external_id": "doc-1",
            "folder_name": "/Team/Reports",
            "folder_id": "folder-1",
            "custom": "new",
        },
        metadata_types={
            "external_id": "string",
            "folder_name": "string",
            "folder_id": "string",
            "custom": "string",
        },
    )

    assert updated is doc
    assert doc.metadata["custom"] == "new"
    assert len(db.update_calls) == 1
    assert db.update_calls[0]["updates"]["metadata"]["external_id"] == "doc-1"
    assert db.update_calls[0]["updates"]["metadata"]["folder_name"] == "/Team/Reports"


@pytest.mark.asyncio
async def test_metadata_only_update_rejects_folder_path_with_folder_endpoint_message():
    doc = _document()
    service, db = _service(doc)

    with pytest.raises(ValueError, match="folder_path.*update metadata endpoint.*folder"):
        await service.update_document(
            document_id="doc-1",
            auth=_auth(),
            metadata={
                "folder_path": "/Team/Reports",
                "custom": "new",
            },
        )

    assert "folder_path" not in doc.metadata
    assert doc.metadata["custom"] == "old"
    assert db.update_calls == []


@pytest.mark.asyncio
async def test_metadata_only_update_rejects_changed_managed_metadata_fields():
    doc = _document()
    service, db = _service(doc)

    with pytest.raises(ValueError, match="folder_name"):
        await service.update_document(
            document_id="doc-1",
            auth=_auth(),
            metadata={
                "folder_name": "/Team/Other",
                "custom": "new",
            },
        )

    assert doc.metadata["custom"] == "old"
    assert db.update_calls == []


@pytest.mark.asyncio
async def test_content_update_still_rejects_unchanged_managed_metadata_fields():
    doc = _document()
    service, db = _service(doc)

    with pytest.raises(ValueError, match="external_id"):
        await service.update_document(
            document_id="doc-1",
            auth=_auth(),
            content="replacement",
            metadata={"external_id": "doc-1"},
        )

    assert db.update_calls == []


@pytest.mark.asyncio
async def test_queued_metadata_only_update_allows_unchanged_managed_metadata_fields():
    doc = _document()
    service, db = _service(doc)

    updated = await service.queue_document_update(
        document_id="doc-1",
        auth=_auth(),
        redis=None,
        metadata={
            "external_id": "doc-1",
            "folder_name": "/Team/Reports",
            "custom": "queued",
        },
    )

    assert updated is doc
    assert doc.metadata["custom"] == "queued"
    assert len(db.update_calls) == 1


@pytest.mark.asyncio
async def test_queued_text_update_preserves_identity_metadata_and_queues_reindex():
    doc = _document()
    original_metadata = dict(doc.metadata)
    service, db = _service(doc)
    redis = FakeRedis()

    async def noop_limit_check(auth, content_length, document_id):
        return None

    async def fake_upload(*, content_bytes, filename, content_type):
        assert content_bytes == b"corrected QA backlog text"
        return "app-1", "ingest_uploads/replacement/report.txt", "report.txt"

    async def noop_record(*args, **kwargs):
        return None

    async def no_stored_size(*args, **kwargs):
        return None

    service._verify_ingest_and_storage_limits = noop_limit_check
    service._upload_content_bytes = fake_upload
    service._record_storage_usage = noop_record
    service._get_storage_object_size = no_stored_size

    updated = await service.queue_document_update(
        document_id="doc-1",
        auth=_auth(),
        redis=redis,
        content="corrected QA backlog text",
        use_colpali=False,
    )

    assert updated is doc
    assert updated.external_id == "doc-1"
    assert updated.metadata == original_metadata
    assert updated.system_metadata["status"] == "processing"
    assert updated.storage_info["key"] == "ingest_uploads/replacement/report.txt"

    persisted = db.update_calls[-1]
    assert persisted["document_id"] == "doc-1"
    assert persisted["updates"]["metadata"] == original_metadata
    assert persisted["updates"]["system_metadata"]["status"] == "processing"

    assert len(redis.calls) == 1
    queued = redis.calls[0]
    assert queued["function_name"] == "process_ingestion_job"
    assert queued["payload"]["document_id"] == "doc-1"
    assert queued["payload"]["file_key"] == "ingest_uploads/replacement/report.txt"
    assert queued["payload"]["_job_id"] == "ingest:doc-1"
