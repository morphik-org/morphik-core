"""Unit tests for the RecursiveCharacterTextSplitter used by StandardChunker.

The splitter is defined inside ``core.parser.morphik_parser`` which imports heavy
optional dependencies (docling, openpyxl, ...). We stub those modules so the pure
chunking logic can be exercised in isolation.
"""

import sys
import types

import pytest


MODULE_UNDER_TEST = "core.parser.morphik_parser"


def _drop_module(name):
    sys.modules.pop(name, None)
    if "." not in name:
        return
    parent_name, attr_name = name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None and hasattr(parent, attr_name):
        delattr(parent, attr_name)


def _stub_module(monkeypatch, name):
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, attr_name, module, raising=False)
    return module


def _install_dependency_stubs(monkeypatch):
    _stub_module(monkeypatch, "docling")
    _stub_module(monkeypatch, "docling.datamodel")
    docling_base_models = _stub_module(monkeypatch, "docling.datamodel.base_models")
    docling_base_models.InputFormat = types.SimpleNamespace(PDF="pdf")

    docling_pipeline_options = _stub_module(monkeypatch, "docling.datamodel.pipeline_options")
    docling_pipeline_options.PdfPipelineOptions = type("PdfPipelineOptions", (), {})
    docling_pipeline_options.EasyOcrOptions = type("EasyOcrOptions", (), {})
    docling_pipeline_options.TableStructureOptions = type("TableStructureOptions", (), {})

    docling_document_converter = _stub_module(monkeypatch, "docling.document_converter")
    docling_document_converter.DocumentConverter = type(
        "DocumentConverter", (), {"__init__": lambda self, *a, **kw: None}
    )
    docling_document_converter.PdfFormatOption = type(
        "PdfFormatOption", (), {"__init__": lambda self, *a, **kw: None}
    )

    assemblyai = _stub_module(monkeypatch, "assemblyai")
    assemblyai.settings = types.SimpleNamespace(api_key=None)
    assemblyai.Transcript = type("Transcript", (), {})
    assemblyai.TranscriptionConfig = type("TranscriptionConfig", (), {"__init__": lambda self, *a, **kw: None})
    assemblyai.Transcriber = type("Transcriber", (), {"__init__": lambda self, *a, **kw: None})

    cv2 = _stub_module(monkeypatch, "cv2")
    cv2.CAP_PROP_FPS = 0
    cv2.CAP_PROP_FRAME_COUNT = 1
    cv2.VideoCapture = lambda path: None

    _stub_module(monkeypatch, "litellm")
    _stub_module(monkeypatch, "openpyxl")

    filetype = _stub_module(monkeypatch, "filetype")
    filetype.guess = lambda content: None


@pytest.fixture
def splitter_cls(monkeypatch):
    _drop_module(MODULE_UNDER_TEST)
    _install_dependency_stubs(monkeypatch)

    from core.parser.morphik_parser import RecursiveCharacterTextSplitter

    yield RecursiveCharacterTextSplitter

    _drop_module(MODULE_UNDER_TEST)


def test_duplicate_token_keeps_word_boundary(splitter_cls):
    """A token that also equals the final token must not lose its trailing separator.

    Regression: the old code compared ``part != splits[-1]`` by value, so the first
    "cat" had its separator dropped and got merged into the next word ("catdog").
    """
    splitter = splitter_cls(chunk_size=8, chunk_overlap=0, separators=[" ", ""])
    chunks = [chunk.content for chunk in splitter.split_text("cat dog bird cat")]

    rejoined = "".join(chunks)
    assert "catdog" not in rejoined
    assert "cat dog" in rejoined
    # The reconstructed text must contain every original word with boundaries intact.
    for word in ("cat", "dog", "bird"):
        assert word in rejoined.split()


def test_no_separator_loss_on_repeated_tokens(splitter_cls):
    """Repeated tokens elsewhere in the text keep their separators too."""
    splitter = splitter_cls(chunk_size=12, chunk_overlap=0, separators=[" ", ""])
    chunks = [chunk.content for chunk in splitter.split_text("the quick fox the")]

    rejoined = "".join(chunks)
    assert "thequick" not in rejoined
    assert rejoined.split() == ["the", "quick", "fox", "the"]
