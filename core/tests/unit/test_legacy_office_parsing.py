"""Legacy Office files (.doc, .ppt, .xls, ...) are converted with LibreOffice before Docling sees them."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.parser import morphik_parser
from core.parser.morphik_parser import MorphikParser

pytestmark = pytest.mark.unit


def _soffice_works() -> bool:
    if not shutil.which("soffice"):
        return False
    try:
        return subprocess.run(["soffice", "--version"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(
        morphik_parser, "get_settings", lambda: SimpleNamespace(REGISTERED_MODELS={}, PARSER_MODE="local")
    )
    parser = MorphikParser()
    parser.docling_calls = []

    async def docling(file, filename):
        parser.docling_calls.append((file, filename))
        return "docling text"

    parser._parse_document_local = docling
    return parser


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,target",
    [("policy.doc", "docx"), ("notes.rtf", "docx"), ("deck.ppt", "pptx"), ("slides.odp", "pptx")],
)
async def test_legacy_files_reach_docling_converted(parser, monkeypatch, filename, target):
    conversions = []

    def convert(file, suffix, to):
        conversions.append((file, suffix, to))
        return b"converted"

    monkeypatch.setattr(MorphikParser, "_convert_with_libreoffice", staticmethod(convert))

    _, text = await parser.parse_file_to_text(b"legacy", filename)

    assert text == "docling text"
    assert conversions == [(b"legacy", Path(filename).suffix, target)]
    assert parser.docling_calls == [(b"converted", f"{Path(filename).stem}.{target}")]


@pytest.mark.asyncio
async def test_converted_xls_uses_fast_excel_path(parser, monkeypatch):
    monkeypatch.setattr(MorphikParser, "_convert_with_libreoffice", staticmethod(lambda file, suffix, to: b"xlsx"))
    monkeypatch.setattr(parser, "_parse_excel_to_markdown", lambda file: f"sheet from {file.decode()}")

    _, text = await parser.parse_file_to_text(b"legacy", "budget.xls")

    assert text == "sheet from xlsx"
    assert parser.docling_calls == []


@pytest.mark.asyncio
async def test_failed_conversion_falls_back_to_original_file(parser, monkeypatch):
    monkeypatch.setattr(MorphikParser, "_convert_with_libreoffice", staticmethod(lambda file, suffix, to: None))

    await parser.parse_file_to_text(b"legacy", "policy.doc")

    assert parser.docling_calls == [(b"legacy", "policy.doc")]


@pytest.mark.asyncio
async def test_modern_files_are_not_converted(parser, monkeypatch):
    def convert(*args):
        raise AssertionError("should not convert")

    monkeypatch.setattr(MorphikParser, "_convert_with_libreoffice", staticmethod(convert))

    await parser.parse_file_to_text(b"modern", "policy.docx")

    assert parser.docling_calls == [(b"modern", "policy.docx")]


@pytest.mark.skipif(not _soffice_works(), reason="LibreOffice is not installed")
def test_libreoffice_converts_doc_to_docx(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("Break time policy")
    subprocess.run(
        ["soffice", f"-env:UserInstallation={(tmp_path / 'profile').as_uri()}", "--headless", "--convert-to", "doc"]
        + ["--outdir", str(tmp_path), str(source)],
        check=True,
        capture_output=True,
        timeout=120,
    )

    docx = MorphikParser._convert_with_libreoffice((tmp_path / "policy.doc").read_bytes(), ".doc", "docx")

    assert docx is not None
    assert docx[:2] == b"PK"  # OOXML files are zip archives
