"""Text Docling's OCR finds inside a picture region is kept instead of a bare <!-- image --> placeholder."""

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from core.parser.morphik_parser import docling_markdown

pytestmark = pytest.mark.unit


def test_picture_text_replaces_the_placeholder():
    # Docling's layout model often tags a whole screenshot as a picture and files its OCR text under it.
    doc = DoclingDocument(name="screenshot")
    doc.add_text(label=DocItemLabel.TEXT, text="Steps to reproduce")
    form = doc.add_picture()
    doc.add_text(label=DocItemLabel.TEXT, text="Add Check", parent=form)
    doc.add_text(label=DocItemLabel.TEXT, text="Bank Account", parent=form)
    doc.add_text(label=DocItemLabel.TEXT, text="Expected: saved", parent=None)

    markdown = docling_markdown(doc)

    assert "<!-- image -->" not in markdown
    assert markdown.index("Steps to reproduce") < markdown.index("Add Check") < markdown.index("Expected: saved")
    assert "Add Check\nBank Account" in markdown


def test_each_picture_gets_its_own_text_and_empty_pictures_disappear():
    doc = DoclingDocument(name="two-panels")
    doc.add_picture()  # a logo with no text
    panel = doc.add_picture()
    doc.add_text(label=DocItemLabel.TEXT, text="Expenses $0.00", parent=panel)

    markdown = docling_markdown(doc)

    assert markdown.strip() == "Expenses $0.00"


def test_documents_without_pictures_are_unchanged():
    doc = DoclingDocument(name="plain")
    doc.add_heading(text="Error 0x80070057")
    doc.add_text(label=DocItemLabel.TEXT, text="Workaround: clear cache")

    assert docling_markdown(doc) == doc.export_to_markdown()
