"""Unit tests for document field projection (list_docs `fields`)."""

from sqlalchemy import select

from core.database.postgres_database import PostgresDatabase
from core.routes.utils import project_document_fields


class TestResolveProjectionFields:
    """PostgresDatabase._resolve_document_projection_fields."""

    def test_no_fields_returns_none(self):
        assert PostgresDatabase._resolve_document_projection_fields(None) is None
        assert PostgresDatabase._resolve_document_projection_fields([]) is None
        assert PostgresDatabase._resolve_document_projection_fields(["  "]) is None

    def test_always_includes_external_id(self):
        assert PostgresDatabase._resolve_document_projection_fields(["metadata"]) == {
            "external_id",
            "metadata",
        }

    def test_nested_field_resolves_to_root_column(self):
        # "metadata.client" only needs the doc_metadata column.
        assert PostgresDatabase._resolve_document_projection_fields(["metadata.client"]) == {
            "external_id",
            "metadata",
        }

    def test_summary_key_requires_system_metadata(self):
        assert PostgresDatabase._resolve_document_projection_fields(["summary_storage_key"]) == {
            "external_id",
            "system_metadata",
        }

    def test_page_count_requires_system_metadata_and_chunk_ids(self):
        assert PostgresDatabase._resolve_document_projection_fields(["page_count"]) == {
            "external_id",
            "system_metadata",
            "chunk_ids",
        }


class TestProjectionColumns:
    """The generated SQL only selects the projected columns."""

    def test_metadata_projection_does_not_read_content(self):
        fields = PostgresDatabase._resolve_document_projection_fields(["metadata"])
        sql = str(select(*PostgresDatabase._document_projection_columns(fields)))
        assert "documents.external_id" in sql
        assert "documents.doc_metadata" in sql
        # The heavy column (system_metadata holds the full document text) is not selected.
        assert "system_metadata" not in sql


class TestProjectionRowToDict:
    """PostgresDatabase._document_projection_row_to_dict."""

    def test_none_jsonb_normalized(self):
        row = {"external_id": "doc-1", "metadata": None, "chunk_ids": None}
        out = PostgresDatabase._document_projection_row_to_dict(row, {"external_id", "metadata", "chunk_ids"})
        assert out["metadata"] == {}
        assert out["chunk_ids"] == []

    def test_summary_keys_derived_when_system_metadata_present(self):
        row = {"external_id": "doc-1", "system_metadata": {"summary_storage_key": "s3://x"}}
        out = PostgresDatabase._document_projection_row_to_dict(row, {"external_id", "system_metadata"})
        assert out["summary_storage_key"] == "s3://x"


class TestProjectDocumentFields:
    """core.routes.utils.project_document_fields (application-layer shaping)."""

    def test_projects_requested_fields_only(self):
        doc = {"external_id": "d1", "content_type": "text/plain", "metadata": {"a": 1}, "system_metadata": {"big": "x"}}
        out = project_document_fields(doc, ["metadata"])
        assert set(out) == {"external_id", "metadata"}
        assert out["metadata"] == {"a": 1}

    def test_nested_projection(self):
        doc = {"external_id": "d1", "metadata": {"client": "ExampleCo", "doc_type": "invoice", "secret": "z"}}
        out = project_document_fields(doc, ["metadata.client", "metadata.doc_type"])
        assert out["metadata"] == {"client": "ExampleCo", "doc_type": "invoice"}

    def test_no_fields_returns_all(self):
        doc = {"external_id": "d1", "metadata": {"a": 1}}
        out = project_document_fields(doc, None)
        assert out["metadata"] == {"a": 1}
        assert out["external_id"] == "d1"
