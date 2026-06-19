"""Document status is a local snapshot read — no per-access network calls."""

import pytest
from morphik.models import Document


def test_status_reads_local_snapshot_without_a_call():
    doc = Document(
        external_id="d1",
        content_type="text/plain",
        system_metadata={"status": "failed", "error": "boom", "updated_at": "2026-06-01T00:00:00"},
    )
    snap = doc.status
    assert snap["status"] == "failed"
    assert snap["error"] == "boom"
    assert snap["source"] == "local"
    assert snap["as_of"]  # stamped at construction
    assert doc.is_failed and not doc.is_processing and not doc.is_ingested
    assert doc.error == "boom"


def test_status_not_loaded_makes_no_call():
    # Status projected away (no system_metadata) and no client attached: must not call out.
    doc = Document(external_id="d2", content_type="text/plain", metadata={"a": 1})
    snap = doc.status
    assert snap["status"] == "unknown"
    assert snap["source"] == "not_loaded"
    assert doc.is_failed is False
    assert doc.is_processing is False
    assert doc.is_ingested is False


def test_projected_status_is_read_locally():
    # Shape the server returns for list_documents(fields=[..., "status"]).
    doc = Document(external_id="d3", content_type="text/plain", system_metadata={"status": "completed"})
    assert doc.is_ingested
    assert doc.status["source"] == "local"


def test_refresh_requires_client():
    doc = Document(external_id="d4", content_type="text/plain")
    with pytest.raises(ValueError):
        doc.refresh()


class _CountingClient:
    """Records any status/document fetch so tests can assert zero calls."""

    def __init__(self):
        self.calls = []

    def get_document_status(self, *args, **kwargs):
        self.calls.append("get_document_status")
        return {"status": "processing"}

    def get_document(self, *args, **kwargs):
        self.calls.append("get_document")
        return None


def test_is_star_make_zero_client_calls_when_status_local():
    # Regression guard for the N+1: reading status/is_* on a document that already carries
    # its status must NOT make any client call, even with a client attached.
    doc = Document(external_id="d5", content_type="text/plain", system_metadata={"status": "completed"})
    client = _CountingClient()
    doc._client = client
    _ = (doc.status, doc.is_failed, doc.is_processing, doc.is_ingested, doc.error)
    assert client.calls == [], f"is_*/status must make zero client calls, made: {client.calls}"


def test_is_star_make_zero_calls_when_not_loaded_even_with_client():
    doc = Document(external_id="d6", content_type="text/plain")  # status not loaded
    client = _CountingClient()
    doc._client = client
    _ = (doc.is_failed, doc.is_processing, doc.is_ingested)
    assert client.calls == [], f"not-loaded status must make zero calls, made: {client.calls}"
    assert doc.status["source"] == "not_loaded"
