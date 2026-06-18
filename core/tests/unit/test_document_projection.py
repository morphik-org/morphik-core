from core.database.postgres_database import PostgresDatabase


def test_document_projection_fields_resolve_metadata_columns():
    fields = PostgresDatabase._resolve_document_projection_fields(
        ["metadata.source", "system_metadata.status", "summary_version", "unknown"]
    )

    assert fields == {"external_id", "metadata", "system_metadata"}


def test_document_projection_fields_include_chunk_ids_for_page_count_fallback():
    fields = PostgresDatabase._resolve_document_projection_fields(["page_count"])

    assert fields == {"external_id", "system_metadata", "chunk_ids"}


def test_document_projection_row_to_dict_normalizes_json_and_summary_fields():
    row = {
        "external_id": "doc-1",
        "metadata": None,
        "metadata_types": None,
        "system_metadata": {
            "status": "completed",
            "summary_version": 3,
            "summary_storage_key": "summaries/doc-1.md",
        },
        "chunk_ids": None,
    }

    document = PostgresDatabase._document_projection_row_to_dict(
        row,
        {"external_id", "metadata", "metadata_types", "system_metadata", "chunk_ids"},
    )

    assert document["metadata"] == {}
    assert document["metadata_types"] == {}
    assert document["chunk_ids"] == []
    assert document["summary_version"] == 3
    assert document["summary_storage_key"] == "summaries/doc-1.md"
