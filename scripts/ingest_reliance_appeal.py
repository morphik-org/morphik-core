"""Inventory and ingest the Reliance Appeal document set into Morphik.

The source folder is intentionally supplied at runtime. This script never copies
source documents or generated manifests into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".tif", ".tiff", ".png", ".jpg", ".jpeg"})
DATE_PATTERNS = (
    re.compile(r"\b(?P<month>\d{1,2})[./-](?P<day>\d{1,2})[./-](?P<year>\d{4})\b"),
    re.compile(r"\b(?P<year>20\d{2})[./-](?P<month>\d{1,2})[./-](?P<day>\d{1,2})\b"),
    re.compile(
        r"\b(?P<month_name>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(?P<day>\d{1,2})(?:,|\s)\s*(?P<year>\d{4})\b",
        re.IGNORECASE,
    ),
)
MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
MONTHS.update({name[:3].lower(): number for name, number in list(MONTHS.items())})


def discover_documents(source: Path) -> list[Path]:
    """Return supported source documents in stable relative-path order."""
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    return sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.relative_to(source).as_posix().lower(),
    )


def extract_text(path: Path) -> str | None:
    """Extract readily available text for the manifest and date detection."""
    if path.suffix.lower() == ".pdf":
        import fitz

        with fitz.open(path) as document:
            return "\n".join(page.get_text() for page in document).strip() or None
    if path.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
        except (KeyError, OSError, ElementTree.ParseError, zipfile.BadZipFile) as exc:
            logger.warning("Could not extract local DOCX text from %s: %s", path, exc)
            return None
        return "\n".join(text for text in root.itertext() if text.strip()).strip() or None
    return None


def extract_dates(*values: str | None) -> list[str]:
    """Extract valid ISO dates from filenames and locally extracted text."""
    found: set[date] = set()
    for value in values:
        if not value:
            continue
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(value):
                parts = match.groupdict()
                month = parts.get("month")
                if month is None:
                    month = MONTHS[parts["month_name"].lower()]
                try:
                    found.add(date(int(parts["year"]), int(month), int(parts["day"])))
                except (TypeError, ValueError):
                    continue
    return [item.isoformat() for item in sorted(found)]


def build_record(path: Path, source: Path) -> dict[str, Any]:
    """Build the local inventory record and Morphik metadata for one source."""
    content = extract_text(path)
    raw_bytes = path.read_bytes()
    dates = extract_dates(path.name, content)
    modified_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    metadata: dict[str, Any] = {
        "source_path": path.relative_to(source).as_posix(),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_modified_at": modified_at,
        "content_chars": len(content) if content is not None else None,
        "document_dates": dates,
        "document_date": dates[0] if dates else None,
        "content_extracted_locally": content is not None,
    }
    return {
        "source_path": metadata["source_path"],
        "filename": path.name,
        "content": content,
        "document_dates": dates,
        "metadata": metadata,
        "metadata_types": {"document_date": "date", "source_modified_at": "datetime"},
    }


def write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write a UTF-8 JSON manifest, creating only the requested output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"documents": list(records)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ingest(
    source: Path,
    manifest: Path,
    uri: str | None,
    folder: str | None,
    batch_size: int,
    use_colpali: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    paths = discover_documents(source)
    records = [build_record(path, source) for path in paths]
    if dry_run:
        write_manifest(manifest, records)
        return records

    from morphik import Morphik

    client = Morphik(uri, timeout=10_000, is_local=not uri or uri.startswith("http://localhost"))
    scoped_client = client.get_folder(folder) if folder else client
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_records = records[start : start + batch_size]
        documents = scoped_client.ingest_files(
            files=batch_paths,
            metadata=[record["metadata"] for record in batch_records],
            use_colpali=use_colpali,
            parallel=True,
        )
        if len(documents) != len(batch_records):
            raise RuntimeError(f"Morphik returned {len(documents)} documents for {len(batch_records)} files")
        for record, document in zip(batch_records, documents):
            completed = document.wait_for_completion(timeout_seconds=10_000)
            record["external_id"] = completed.external_id
            record["morphik_status"] = completed.status
            record["content"] = record["content"] or None
    write_manifest(manifest, records)
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Reliance Appeal source directory")
    parser.add_argument("--manifest", type=Path, default=Path("reliance-appeal-manifest.json"))
    parser.add_argument("--uri", default=None, help="Morphik URI; defaults to MORPHIK_URI")
    parser.add_argument("--folder", default=None, help="Morphik folder path for the ingested documents")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--use-colpali", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Write the inventory without calling Morphik")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    import os

    uri = args.uri or os.getenv("MORPHIK_URI")
    records = ingest(args.source, args.manifest, uri, args.folder, args.batch_size, args.use_colpali, args.dry_run)
    print(f"{'Inventoried' if args.dry_run else 'Ingested'} {len(records)} documents; manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
