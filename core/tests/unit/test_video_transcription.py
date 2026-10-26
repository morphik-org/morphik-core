"""Video audio transcription through an OpenAI transcription model, and skipping unchanged frames."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest

from core import config
from core.models.video import TimeSeriesData
from core.parser.video import parse_video
from core.parser.video.parse_video import VideoParser, format_timestamp, has_audio_stream, split_audio

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg and ffprobe are not installed"
)


class _FakeTranscriptions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **params):
        self.calls.append({**params, "file": params["file"].name})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _parser(video_path, config, responses):
    parser = object.__new__(VideoParser)
    parser.video_path = str(video_path)
    parser.transcription_config = config
    transcriptions = _FakeTranscriptions(responses)
    parser._transcription_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
    return parser, transcriptions


def _audio_pieces(tmp_path, starts):
    pieces = []
    for index, start in enumerate(starts):
        path = tmp_path / f"audio_{index:05d}.mp3"
        path.write_bytes(b"mp3")
        pieces.append((start, str(path)))
    return pieces


def _make_video(path, with_audio, seconds=3):
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=10:duration={seconds}"]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(command, check=True)


@pytest.fixture
def tools_present(monkeypatch):
    monkeypatch.setattr(parse_video.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_timestamps_switch_to_hours_after_an_hour():
    assert [format_timestamp(t) for t in (0, 75.9, 3725)] == ["00:00", "01:15", "1:02:05"]


@pytest.mark.asyncio
async def test_silent_video_is_not_sent_for_transcription(monkeypatch, tools_present, tmp_path):
    monkeypatch.setattr(parse_video, "has_audio_stream", lambda path: False)
    monkeypatch.setattr(parse_video, "split_audio", lambda *a: pytest.fail("audio should not be extracted"))
    parser, transcriptions = _parser(tmp_path / "v.mp4", {"model_name": "openai/gpt-transcribe"}, [])

    transcript = await parser.get_model_transcript()

    assert transcript.time_to_content == {}
    assert transcriptions.calls == []


@pytest.mark.asyncio
async def test_pieces_are_timed_by_their_start_or_by_returned_segments(monkeypatch, tools_present, tmp_path):
    monkeypatch.setattr(parse_video, "has_audio_stream", lambda path: True)
    monkeypatch.setattr(parse_video, "split_audio", lambda path, out, seconds: _audio_pieces(tmp_path, [0.0, 120.0]))
    responses = [
        SimpleNamespace(text=" The login page hangs. "),
        SimpleNamespace(text="ignored", segments=[{"start": 3.5, "text": " Then it times out."}, {"start": 9}]),
    ]
    parser, _ = _parser(tmp_path / "v.mp4", {"model_name": "openai/gpt-transcribe"}, responses)

    transcript = await parser.get_model_transcript()

    assert transcript.time_to_content == {0.0: "The login page hangs.", 123.5: "Then it times out."}


@pytest.mark.asyncio
async def test_gpt_transcribe_asks_for_json_and_whisper_for_segments(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"mp3")
    config = {"model_name": "openai/gpt-transcribe", "language": "pl", "vision": False, "api_base": None}
    parser, transcriptions = _parser(tmp_path / "v.mp4", config, [SimpleNamespace(text="x")])
    await parser._transcribe_segment(str(tmp_path / "a.mp3"))

    whisper, whisper_calls = _parser(tmp_path / "v.mp4", {"model_name": "whisper-1"}, [SimpleNamespace(text="x")])
    await whisper._transcribe_segment(str(tmp_path / "a.mp3"))

    assert transcriptions.calls == [
        {"model": "gpt-transcribe", "file": str(tmp_path / "a.mp3"), "language": "pl", "response_format": "json"}
    ]
    assert whisper_calls.calls[0]["response_format"] == "verbose_json"
    assert whisper_calls.calls[0]["timestamp_granularities"] == ["segment"]


@pytest.mark.asyncio
async def test_rate_limits_become_retryable_connection_errors(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"mp3")
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    error = openai.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    parser, _ = _parser(tmp_path / "v.mp4", {"model_name": "openai/gpt-transcribe"}, [error])

    with pytest.raises(ConnectionError, match="gpt-transcribe"):
        await parser._transcribe_segment(str(tmp_path / "a.mp3"))


class _Capture:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        pass


class _Vision:
    def __init__(self):
        self.calls = 0

    async def get_frame_description(self, image_base64, context):
        self.calls += 1
        return f"description {self.calls}"


def _frame(value, noise=0):
    frame = np.full((90, 160, 3), value, dtype=np.uint8)
    frame[:2, :2] = min(255, value + noise)  # a cursor-sized change
    return frame


def _screen(*boxes):
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    for x, y, w, h, value in boxes:
        frame[y : y + h, x : x + w] = value
    return frame


def test_a_new_line_of_text_counts_as_a_change_and_a_moved_cursor_does_not():
    blank = parse_video.frame_signature(_screen())
    cursor_here = parse_video.frame_signature(_screen((100, 400, 12, 18, 0)))
    cursor_there = parse_video.frame_signature(_screen((400, 400, 12, 18, 0)))
    error_line = parse_video.frame_signature(_screen((100, 100, 60, 16, 50)))
    threshold = parse_video.DEFAULT_FRAME_CHANGE_THRESHOLD

    assert parse_video.frame_difference(cursor_here, cursor_there) < threshold
    assert parse_video.frame_difference(blank, error_line) > threshold


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold, expected_calls", [(0.025, 2), (0, 4)])
async def test_unchanged_frames_are_not_captioned_again(threshold, expected_calls):
    parser = object.__new__(VideoParser)
    parser.cap = _Capture([_frame(0), _frame(0, noise=200), _frame(255), _frame(250)])
    parser.fps = 1.0
    parser.frame_sample_rate = 1
    parser.frame_change_threshold = threshold
    parser.frames_unchanged = 0
    parser.transcript = TimeSeriesData(time_to_content={})
    parser.vision_client = _Vision()
    parser.frame_to_base64 = lambda frame: "image"

    descriptions = await parser.get_frame_descriptions()

    assert parser.vision_client.calls == expected_calls
    assert parser.frames_unchanged == 4 - expected_calls
    if threshold:
        assert descriptions.timestamps == [0.0, 2.0]


@needs_ffmpeg
def test_ffmpeg_finds_audio_and_cuts_it_into_timed_pieces(tmp_path):
    talking, silent = tmp_path / "talking.mp4", tmp_path / "silent.webm"
    _make_video(talking, with_audio=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2", str(silent)],
        check=True,
    )
    output = tmp_path / "pieces"
    output.mkdir()

    pieces = split_audio(str(talking), str(output), 1)

    assert has_audio_stream(str(talking)) is True
    assert has_audio_stream(str(silent)) is False
    # AAC encoder padding can leave a few milliseconds for a fourth piece.
    assert [round(start) for start, _ in pieces][:3] == [0, 1, 2]
    assert len(pieces) <= 4
    assert all((output / path).stat().st_size > 0 for _, path in pieces)


@needs_ffmpeg
@pytest.mark.asyncio
async def test_real_video_is_transcribed_piece_by_piece(tmp_path, monkeypatch):
    video = tmp_path / "talking.mp4"
    _make_video(video, with_audio=True, seconds=5)
    monkeypatch.setattr(parse_video, "TRANSCRIPTION_SEGMENT_SECONDS", 2)
    responses = [SimpleNamespace(text=f"piece {n}") for n in range(3)]
    parser, transcriptions = _parser(video, {"model_name": "openai/gpt-transcribe"}, responses)

    transcript = await parser.get_model_transcript()

    assert [round(t) for t in transcript.timestamps] == [0, 2, 4]
    assert transcript.contents == ["piece 0", "piece 1", "piece 2"]
    assert all(call["file"].endswith(".mp3") for call in transcriptions.calls)


@pytest.mark.parametrize("model_name", ["openai/gpt-transcribe", "whisper-1"])
def test_documented_transcription_model_loads(tmp_path, monkeypatch, model_name):
    toml = (REPO_ROOT / "morphik.docker.toml").read_text()
    toml = toml.replace(
        "[registered_models]\n", f'[registered_models]\nopenai_gpt_transcribe = {{ model_name = "{model_name}" }}\n', 1
    )
    toml = toml.replace(
        '# transcription_model = "openai_gpt_transcribe"', 'transcription_model = "openai_gpt_transcribe"'
    )
    (tmp_path / "morphik.toml").write_text(toml)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
    config.get_settings.cache_clear()
    try:
        settings = config.get_settings()
    finally:
        config.get_settings.cache_clear()

    assert settings.TRANSCRIPTION_MODEL == "openai_gpt_transcribe"


def test_transcription_model_must_be_openai(tmp_path, monkeypatch):
    toml = (REPO_ROOT / "morphik.docker.toml").read_text()
    toml = toml.replace(
        "[registered_models]\n", '[registered_models]\nlocal_audio = { model_name = "ollama/whisper" }\n', 1
    )
    toml = toml.replace('# transcription_model = "openai_gpt_transcribe"', 'transcription_model = "local_audio"')
    (tmp_path / "morphik.toml").write_text(toml)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_URI", "postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
    config.get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="transcription_model needs an OpenAI model"):
            config.get_settings()
    finally:
        config.get_settings.cache_clear()
