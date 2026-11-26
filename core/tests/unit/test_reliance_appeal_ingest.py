from pathlib import Path

from scripts.ingest_reliance_appeal import (
    build_record,
    discover_documents,
    extract_dates,
    write_manifest,
)


def test_discover_documents_excludes_generated_export_files(tmp_path: Path):
    (tmp_path / "letter.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.docx").write_bytes(b"docx")
    (tmp_path / "scan.TIF").write_bytes(b"tif")
    (tmp_path / "export.xml").write_text("<xml />")
    (tmp_path / "archive.zip").write_bytes(b"zip")
    (tmp_path / "nested" / "generated").mkdir(parents=True)
    (tmp_path / "nested" / "generated" / "page.PNG").write_bytes(b"png")

    assert [path.relative_to(tmp_path).as_posix() for path in discover_documents(tmp_path)] == [
        "letter.pdf",
        "nested/generated/page.PNG",
        "notes.docx",
        "scan.TIF",
    ]


def test_extract_dates_returns_valid_sorted_iso_dates():
    assert extract_dates("Letter 09.08.2025", "Filed January 3, 2024 and 2025-02-14") == [
        "2024-01-03",
        "2025-02-14",
        "2025-09-08",
    ]


def test_build_record_contains_typed_retrieval_metadata(tmp_path: Path):
    path = tmp_path / "Appeal 45-Day Letter 09.08.2025.pdf"
    path.write_bytes(b"")

    record = build_record(path, tmp_path)

    assert record["source_path"] == path.name
    assert record["document_dates"] == ["2025-09-08"]
    assert record["metadata"]["source_sha256"]
    assert record["metadata_types"] == {"document_date": "date", "source_modified_at": "datetime"}


def test_write_manifest_serializes_records(tmp_path: Path):
    output = tmp_path / "out" / "manifest.json"
    write_manifest(output, [{"filename": "letter.pdf", "external_id": "doc-1"}])

    assert '"external_id": "doc-1"' in output.read_text(encoding="utf-8")
