import logging
import sys
import types

import pytest


MODULES_UNDER_TEST = [
    "core.models.video",
    "core.parser.video.parse_video",
    "core.parser.morphik_parser",
]


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
        "DocumentConverter",
        (),
        {"__init__": lambda self, *a, **kw: None},
    )
    docling_document_converter.PdfFormatOption = type(
        "PdfFormatOption",
        (),
        {"__init__": lambda self, *a, **kw: None},
    )

    assemblyai = _stub_module(monkeypatch, "assemblyai")
    assemblyai.settings = types.SimpleNamespace(api_key=None)
    assemblyai.Transcript = type("Transcript", (), {})
    assemblyai.TranscriptionConfig = type(
        "TranscriptionConfig",
        (),
        {"__init__": lambda self, *a, **kw: None},
    )
    assemblyai.Transcriber = type(
        "Transcriber",
        (),
        {"__init__": lambda self, *a, **kw: None},
    )

    cv2 = _stub_module(monkeypatch, "cv2")
    cv2.CAP_PROP_FPS = 0
    cv2.CAP_PROP_FRAME_COUNT = 1
    cv2.VideoCapture = lambda path: None

    _stub_module(monkeypatch, "litellm")
    _stub_module(monkeypatch, "openpyxl")

    filetype = _stub_module(monkeypatch, "filetype")
    filetype.guess = lambda content: None


@pytest.fixture
def video_modules(monkeypatch):
    for module_name in MODULES_UNDER_TEST:
        _drop_module(module_name)

    _install_dependency_stubs(monkeypatch)

    from core.models.video import ParseVideoResult, TimeSeriesData
    from core.parser import morphik_parser as morphik_parser_module
    from core.parser.video import parse_video as parse_video_module
    from core.parser.morphik_parser import MorphikParser
    from core.parser.video.parse_video import VideoParser

    yield types.SimpleNamespace(
        MorphikParser=MorphikParser,
        ParseVideoResult=ParseVideoResult,
        TimeSeriesData=TimeSeriesData,
        VideoParser=VideoParser,
        morphik_parser_module=morphik_parser_module,
        parse_video_module=parse_video_module,
    )

    for module_name in MODULES_UNDER_TEST:
        _drop_module(module_name)


def _fake_video_parser_class(ParseVideoResult, TimeSeriesData):
    class _FakeVideoParser:
        instances = []

        def __init__(self, video_path, assemblyai_api_key=None, frame_sample_rate=None):
            self.video_path = video_path
            self.assemblyai_api_key = assemblyai_api_key
            self.frame_sample_rate = frame_sample_rate
            self.instances.append(self)

        async def process_video(self):
            return ParseVideoResult(
                metadata={
                    "duration": 1.0,
                    "fps": 1.0,
                    "total_frames": 1,
                    "frame_sample_rate": self.frame_sample_rate,
                },
                frame_descriptions=TimeSeriesData(time_to_content={0.0: "visible frame"}),
                transcript=TimeSeriesData(time_to_content={0.5: "spoken words"}),
            )

    return _FakeVideoParser


@pytest.mark.asyncio
async def test_parse_video_skips_transcript_without_assemblyai_key(monkeypatch, video_modules):
    fake_video_parser = _fake_video_parser_class(video_modules.ParseVideoResult, video_modules.TimeSeriesData)
    monkeypatch.setattr(video_modules.morphik_parser_module, "VideoParser", fake_video_parser)
    monkeypatch.setattr(
        video_modules.morphik_parser_module,
        "load_config",
        lambda: {"parser": {"vision": {"frame_sample_rate": 5}}},
    )
    parser = object.__new__(video_modules.MorphikParser)
    parser._assemblyai_api_key = None
    parser.frame_sample_rate = 1

    metadata, text = await parser._parse_video(b"video bytes")

    assert fake_video_parser.instances[0].assemblyai_api_key is None
    assert text == "Frame Descriptions:\nvisible frame"
    assert metadata["frame_timestamps"] == [0.0]
    assert "transcript_timestamps" not in metadata
    assert "Transcript" not in text


@pytest.mark.asyncio
async def test_parse_video_includes_transcript_when_assemblyai_key_is_configured(monkeypatch, video_modules):
    fake_video_parser = _fake_video_parser_class(video_modules.ParseVideoResult, video_modules.TimeSeriesData)
    monkeypatch.setattr(video_modules.morphik_parser_module, "VideoParser", fake_video_parser)
    monkeypatch.setattr(
        video_modules.morphik_parser_module,
        "load_config",
        lambda: {"parser": {"vision": {"frame_sample_rate": 5}}},
    )
    parser = object.__new__(video_modules.MorphikParser)
    parser._assemblyai_api_key = "assembly-key"
    parser.frame_sample_rate = 1

    metadata, text = await parser._parse_video(b"video bytes")

    assert fake_video_parser.instances[0].assemblyai_api_key == "assembly-key"
    assert text == "Frame Descriptions:\nvisible frame\n\nTranscript:\nspoken words"
    assert metadata["frame_timestamps"] == [0.0]
    assert metadata["transcript_timestamps"] == [0.5]


class _FakeCapture:
    def __init__(self):
        self.calls = 0

    def read(self):
        self.calls += 1
        if self.calls == 1:
            return True, object()
        return False, None

    def release(self):
        pass

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == 0:
            return 1.0
        if property_id == 1:
            return 1
        return 0


class _FakeVisionClient:
    def __init__(self):
        self.contexts = []

    async def get_frame_description(self, image_base64, context):
        self.contexts.append(context)
        return "visible frame"


class _FakeVisionClientClass:
    def __init__(self, config):
        self.config = config


def test_video_parser_warns_when_assemblyai_key_is_missing(monkeypatch, caplog, video_modules):
    monkeypatch.setattr(video_modules.parse_video_module.cv2, "VideoCapture", lambda path: _FakeCapture())
    monkeypatch.setattr(video_modules.parse_video_module, "VisionModelClient", _FakeVisionClientClass)

    with caplog.at_level(logging.WARNING, logger="core.parser.video.parse_video"):
        parser = video_modules.VideoParser("/tmp/video.mp4", assemblyai_api_key=None, frame_sample_rate=1)

    parser.cap.release()
    assert "AssemblyAI API key is not available; skipping transcription" in caplog.text


@pytest.mark.asyncio
async def test_frame_descriptions_do_not_mention_transcripts_when_transcript_is_empty(video_modules):
    parser = object.__new__(video_modules.VideoParser)
    parser.cap = _FakeCapture()
    parser.fps = 1.0
    parser.frame_sample_rate = 1
    parser.transcript = video_modules.TimeSeriesData(time_to_content={})
    parser.vision_client = _FakeVisionClient()
    parser.frame_to_base64 = lambda frame: "image"

    result = await parser.get_frame_descriptions()

    assert result.time_to_content == {0.0: "visible frame"}
    assert len(parser.vision_client.contexts) == 1
    assert "transcript" not in parser.vision_client.contexts[0].lower()
