"""Unit tests for ingestion metadata-only update validation."""

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
