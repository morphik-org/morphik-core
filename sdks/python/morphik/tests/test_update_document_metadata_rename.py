import pytest
from morphik.async_ import AsyncMorphik
from morphik.models import Document
from morphik.sync import Morphik


def _make_doc():
    return Document(external_id="doc-1", content_type="text/plain")


def test_sync_update_metadata_rename_uses_file_roundtrip():
    client = Morphik()
    calls = {}

    client.get_document_by_filename = lambda filename: _make_doc()
    client.get_document_file = lambda document_id: b"content"

    def fake_update_document_with_file(document_id, file, filename, metadata, use_colpali=None):
        calls["update_file"] = {
            "document_id": document_id,
            "file": file,
            "filename": filename,
            "metadata": metadata,
        }
        return _make_doc()

    def fake_update_document_metadata(*args, **kwargs):
        raise AssertionError("update_document_metadata should not be called when renaming")

    client.update_document_with_file = fake_update_document_with_file  # type: ignore[assignment]
    client.update_document_metadata = fake_update_document_metadata  # type: ignore[assignment]

    try:
        client.update_document_by_filename_metadata("old.txt", {"k": "v"}, new_filename="new.txt")
        assert calls["update_file"]["document_id"] == "doc-1"
        assert calls["update_file"]["file"] == b"content"
        assert calls["update_file"]["filename"] == "new.txt"
        assert calls["update_file"]["metadata"] == {"k": "v"}
    finally:
        client.close()


def test_sync_update_metadata_no_rename_calls_metadata_update():
    client = Morphik()
    calls = {}

    client.get_document_by_filename = lambda filename: _make_doc()

    def fake_update_document_metadata(document_id, metadata):
        calls["metadata"] = {"document_id": document_id, "metadata": metadata}
        return _make_doc()

    def fake_update_document_with_file(*args, **kwargs):
        raise AssertionError("update_document_with_file should not be called without renaming")

    client.update_document_metadata = fake_update_document_metadata  # type: ignore[assignment]
    client.update_document_with_file = fake_update_document_with_file  # type: ignore[assignment]

    try:
        client.update_document_by_filename_metadata("old.txt", {"k": "v"})
        assert calls["metadata"] == {"document_id": "doc-1", "metadata": {"k": "v"}}
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_update_metadata_rename_uses_file_roundtrip():
    client = AsyncMorphik()
    calls = {}

    async def fake_get_document_by_filename(filename):
        return _make_doc()

    async def fake_get_document_file(document_id):
        return b"content"

    async def fake_update_document_with_file(document_id, file, filename, metadata, use_colpali=None):
        calls["update_file"] = {
            "document_id": document_id,
            "file": file,
            "filename": filename,
            "metadata": metadata,
        }
        return _make_doc()

    async def fake_update_document_metadata(*args, **kwargs):
        raise AssertionError("update_document_metadata should not be called when renaming")

    client.get_document_by_filename = fake_get_document_by_filename  # type: ignore[assignment]
    client.get_document_file = fake_get_document_file  # type: ignore[assignment]
    client.update_document_with_file = fake_update_document_with_file  # type: ignore[assignment]
    client.update_document_metadata = fake_update_document_metadata  # type: ignore[assignment]

    try:
        await client.update_document_by_filename_metadata("old.txt", {"k": "v"}, new_filename="new.txt")
        assert calls["update_file"]["document_id"] == "doc-1"
        assert calls["update_file"]["file"] == b"content"
        assert calls["update_file"]["filename"] == "new.txt"
        assert calls["update_file"]["metadata"] == {"k": "v"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_update_metadata_no_rename_calls_metadata_update():
    client = AsyncMorphik()
    calls = {}

    async def fake_get_document_by_filename(filename):
        return _make_doc()

    async def fake_update_document_metadata(document_id, metadata):
        calls["metadata"] = {"document_id": document_id, "metadata": metadata}
        return _make_doc()

    async def fake_update_document_with_file(*args, **kwargs):
        raise AssertionError("update_document_with_file should not be called without renaming")

    client.get_document_by_filename = fake_get_document_by_filename  # type: ignore[assignment]
    client.update_document_metadata = fake_update_document_metadata  # type: ignore[assignment]
    client.update_document_with_file = fake_update_document_with_file  # type: ignore[assignment]

    try:
        await client.update_document_by_filename_metadata("old.txt", {"k": "v"})
        assert calls["metadata"] == {"document_id": "doc-1", "metadata": {"k": "v"}}
    finally:
        await client.close()
