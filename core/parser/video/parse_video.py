import asyncio
import base64
import csv
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import assemblyai as aai
import cv2
import litellm
import openai
import tomli
from openai import AsyncOpenAI

from core.config import get_settings
from core.models.video import ParseVideoResult, TimeSeriesData

logger = logging.getLogger(__name__)

# Audio is cut into pieces this long before transcription. OpenAI accepts at most 25 MB per request; two minutes of
# the 32 kbps mono MP3 below is about 0.5 MB. The piece start is the timestamp when the model returns no segments.
TRANSCRIPTION_SEGMENT_SECONDS = 120
# A sampled frame is captioned only if some part of it changed since the last captioned frame. Frames are compared
# as 128x72 grayscale copies cut into an 8x8 grid; the change is the largest mean absolute difference (0-1) of any
# tile. A new line of text or a dialog moves one tile well past the threshold; a moving cursor or video compression
# noise doesn't. Screen recordings stay on one screen for most of their length.
DEFAULT_FRAME_CHANGE_THRESHOLD = 0.025
FRAME_COMPARE_SIZE = (128, 72)
FRAME_COMPARE_GRID = 8
# Keys of a registered transcription model passed on to the transcription request
TRANSCRIPTION_OPTIONS = ("language", "prompt", "temperature", "response_format")


def debug_object(title, obj):
    logger.debug("\n".join(["-" * 100, title, "-" * 100, f"{obj}", "-" * 100]))


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "../../../morphik.toml")
    with open(config_path, "rb") as f:
        return tomli.load(f)


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def has_audio_stream(video_path: str) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning("ffprobe could not read %s: %s", video_path, result.stderr.strip()[-500:])
        return False
    return bool(result.stdout.strip())


def split_audio(video_path: str, output_dir: str, segment_seconds: int) -> List[Tuple[float, str]]:
    """Extract the audio track as mono 16 kHz MP3 pieces. Returns (start seconds, path) for each piece."""
    segment_list = os.path.join(output_dir, "segments.csv")
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-segment_list",
            segment_list,
            "-segment_list_type",
            "csv",
            "-reset_timestamps",
            "1",
            os.path.join(output_dir, "audio_%05d.mp3"),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise ValueError(f"ffmpeg could not extract audio: {result.stderr.strip()[-500:]}")
    segments = []
    with open(segment_list, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                segments.append((float(row[1]), os.path.join(output_dir, row[0])))
    return segments


def frame_signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, FRAME_COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype("float32")


def frame_difference(first, second) -> float:
    width, height = FRAME_COMPARE_SIZE
    grid = FRAME_COMPARE_GRID
    tiles = abs(first - second).reshape(grid, height // grid, grid, width // grid).mean(axis=(1, 3))
    return float(tiles.max()) / 255.0


def _value(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


class VisionModelClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["parser"]["vision"]
        self.model_key = self.config.get("model")
        self.settings = get_settings()

        # Get the model configuration from registered_models
        if not hasattr(self.settings, "REGISTERED_MODELS") or self.model_key not in self.settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{self.model_key}' not found in registered_models configuration")

        self.model_config = self.settings.REGISTERED_MODELS[self.model_key]
        logger.info(f"Initialized VisionModelClient with model_key={self.model_key}, config={self.model_config}")

        # Check if the model has vision capability
        if not self.model_config.get("vision", False):
            logger.warning(f"Model '{self.model_key}' does not have vision capability marked in config")

    async def get_frame_description(self, image_base64: str, context: str) -> str:
        # Create a system message
        system_message = {
            "role": "system",
            "content": "You are a video frame description assistant. Describe the frame clearly and concisely.",
        }

        # Determine if the model is OpenAI compatible or Ollama compatible based on model name
        model_name = self.model_config.get("model_name", "")

        if "ollama" in model_name.lower():
            # Ollama format with images parameter
            messages = [system_message, {"role": "user", "content": context}]

            model_params = {"model": model_name, "messages": messages, "images": [image_base64]}
        else:
            # Standard format with image_url (OpenAI, Anthropic, etc.)
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]

            model_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 300,
            }

        # Add all additional parameters from the model config
        for key, value in self.model_config.items():
            if key not in ["model_name", "vision"]:
                model_params[key] = value

        # Use litellm for the completion
        response = await litellm.acompletion(**model_params)
        return response.choices[0].message.content


class VideoParser:
    def __init__(
        self,
        video_path: str,
        assemblyai_api_key: Optional[str] = None,
        frame_sample_rate: Optional[int] = None,
        transcription_config: Optional[Dict[str, Any]] = None,
        frame_change_threshold: Optional[float] = None,
    ):
        """
        Initialize the video parser

        Args:
            video_path: Path to the video file
            assemblyai_api_key: Optional API key for AssemblyAI, used when transcription_config is not set.
            frame_sample_rate: Sample every nth frame for description (optional, defaults to config value)
            transcription_config: Registered model entry for an OpenAI or OpenAI-compatible transcription model,
                e.g. gpt-transcribe. Takes precedence over AssemblyAI. With neither, audio is not transcribed.
            frame_change_threshold: Skip sampled frames closer than this to the last captioned frame; 0 captions all.
        """
        logger.info(f"Initializing VideoParser for {video_path}")
        self.config = load_config()
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate or self.config["parser"]["vision"].get("frame_sample_rate", 120)
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        self.transcript = TimeSeriesData(time_to_content={})
        self.transcription_config = transcription_config
        self._transcription_client: Optional[AsyncOpenAI] = None
        if frame_change_threshold is None:
            frame_change_threshold = self.config["parser"]["vision"].get(
                "frame_change_threshold", DEFAULT_FRAME_CHANGE_THRESHOLD
            )
        self.frame_change_threshold = frame_change_threshold
        self.frames_unchanged = 0

        # Initialize AssemblyAI only when audio transcription is configured.
        self.transcriber = None
        if transcription_config:
            logger.info("Transcribing audio with %s", transcription_config.get("model_name"))
        elif assemblyai_api_key:
            aai.settings.api_key = assemblyai_api_key
            aai_config = aai.TranscriptionConfig(speaker_labels=True)
            self.transcriber = aai.Transcriber(config=aai_config)
        else:
            logger.warning("AssemblyAI API key is not available; skipping transcription")

        # Initialize vision model client
        self.vision_client = VisionModelClient(self.config)

        logger.info(f"Video loaded: {self.duration:.2f}s duration, {self.fps:.2f} FPS")

    def frame_to_base64(self, frame) -> str:
        """Convert a frame to base64 string"""
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode frame to JPEG")
            raise ValueError("Failed to encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def get_transcript_object(self) -> aai.Transcript:
        """
        Get the transcript object from AssemblyAI
        """
        if self.transcriber is None:
            raise ValueError("AssemblyAI API key is required for video transcription")

        logger.info("Starting video transcription")
        transcript = self.transcriber.transcribe(self.video_path)
        if transcript.status == "error":
            logger.error(f"Transcription failed: {transcript.error}")
            raise ValueError(f"Transcription failed: {transcript.error}")
        if not transcript.words:
            logger.warning("No words found in transcript")
        logger.info("Transcription completed successfully!")

        return transcript

    def get_transcript(self) -> TimeSeriesData:
        """
        Get timestamped transcript of the video using AssemblyAI

        Returns:
            TimeSeriesData object containing transcript
        """
        if self.transcriber is None:
            self.transcript = TimeSeriesData(time_to_content={})
            return self.transcript

        logger.info("Starting video transcription")
        transcript = self.get_transcript_object()
        # divide by 1000 because assemblyai timestamps are in milliseconds
        time_to_text = {u.start / 1000: u.text for u in transcript.utterances} if transcript.utterances else {}
        debug_object("Time to text", time_to_text)
        self.transcript = TimeSeriesData(time_to_content=time_to_text)
        return self.transcript

    async def get_model_transcript(self) -> TimeSeriesData:
        """Timestamped transcript from the registered transcription model, two minutes of audio per request."""
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            logger.warning("ffmpeg or ffprobe not found in PATH; skipping transcription")
            return TimeSeriesData(time_to_content={})
        if not await asyncio.to_thread(has_audio_stream, self.video_path):
            logger.info("Video has no audio track; skipping transcription")
            return TimeSeriesData(time_to_content={})

        time_to_text: Dict[float, str] = {}
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                segments = await asyncio.to_thread(
                    split_audio, self.video_path, output_dir, TRANSCRIPTION_SEGMENT_SECONDS
                )
            except (ValueError, OSError, subprocess.TimeoutExpired) as e:
                # The frames can still be described, so a broken audio track doesn't fail the video.
                logger.warning("Skipping transcription of %s: %s", self.video_path, e)
                return TimeSeriesData(time_to_content={})
            for start, path in segments:
                response = await self._transcribe_segment(path)
                pieces = _value(response, "segments") or []
                timed = [
                    (start + float(_value(piece, "start") or 0), (_value(piece, "text") or "").strip())
                    for piece in pieces
                ]
                timed = [(time, text) for time, text in timed if text]
                if not timed:
                    text = (_value(response, "text") or "").strip()
                    timed = [(start, text)] if text else []
                time_to_text.update(timed)
        logger.info("Transcribed %d audio pieces into %d timestamped lines", len(segments), len(time_to_text))
        return TimeSeriesData(time_to_content=time_to_text)

    async def _transcribe_segment(self, path: str) -> Any:
        model = self.transcription_config["model_name"].removeprefix("openai/")
        request: Dict[str, Any] = {k: v for k, v in self.transcription_config.items() if k in TRANSCRIPTION_OPTIONS}
        # Only whisper-1 documents verbose_json, which carries segment timestamps. OpenAI's gpt-*-transcribe
        # models answer json, timed by the start of the audio piece. LiteLLM 1.74 forces verbose_json on every
        # OpenAI model without "gpt-4o" in its name, so this calls the OpenAI SDK directly.
        request.setdefault("response_format", "verbose_json" if model.startswith("whisper") else "json")
        if request["response_format"] == "verbose_json":
            request.setdefault("timestamp_granularities", ["segment"])
        if self._transcription_client is None:
            try:
                self._transcription_client = AsyncOpenAI(
                    api_key=self.transcription_config.get("api_key") or os.environ.get("OPENAI_API_KEY"),
                    base_url=self.transcription_config.get("api_base"),
                )
            except openai.OpenAIError as e:
                raise ValueError(f"Transcription model {model} has no API key: set OPENAI_API_KEY") from e
        with open(path, "rb") as audio:
            try:
                return await self._transcription_client.audio.transcriptions.create(model=model, file=audio, **request)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as e:
                # A later retry can fix these. ConnectionError makes the ingestion worker retry the job.
                raise ConnectionError(f"Transcription call to {model} failed: {e}") from e

    async def get_frame_descriptions(self) -> TimeSeriesData:
        """
        Get descriptions for sampled frames using configured vision model

        Returns:
            TimeSeriesData object containing frame descriptions
        """
        logger.info("Starting frame description generation")

        # Return empty TimeSeriesData if frame_sample_rate is -1 (captioning disabled)
        if self.frame_sample_rate == -1:
            logger.info("Frame captioning is disabled (frame_sample_rate = -1)")
            return TimeSeriesData(time_to_content={})

        frame_count = 0
        time_to_description = {}
        last_description = None
        last_signature = None
        logger.info("Starting main loop for frame description generation")
        while True:
            logger.debug(f"Frame count: {frame_count}")
            ret, frame = self.cap.read()
            if not ret:
                logger.info("Reached end of video")
                break

            if frame_count % self.frame_sample_rate == 0:
                signature = frame_signature(frame) if self.frame_change_threshold > 0 else None
                if (
                    signature is not None
                    and last_signature is not None
                    and frame_difference(signature, last_signature) < self.frame_change_threshold
                ):
                    self.frames_unchanged += 1
                    frame_count += 1
                    continue
                last_signature = signature

                logger.info(f"Processing frame at {frame_count / self.fps:.2f}s")
                timestamp = frame_count / self.fps
                logger.debug(f"Processing frame at {timestamp:.2f}s")

                img_base64 = self.frame_to_base64(frame)

                if last_description:
                    previous_frame_context = last_description
                else:
                    previous_frame_context = "No previous frame description available, this is the first frame"

                description_instruction = (
                    "Describe this frame from a video. Focus on the main elements, actions, and any notable details."
                )
                previous_frame_section = (
                    "Here is a description of the previous frame:\n"
                    "---\n"
                    f"{previous_frame_context}\n"
                    "---"
                )
                transcript_context = self.transcript.at_time(timestamp, padding=10)
                if transcript_context:
                    context = (
                        f"{description_instruction} Here is the transcript around the time of the frame:\n"
                        "---\n"
                        f"{transcript_context}\n"
                        "---\n\n"
                        f"{previous_frame_section}\n\n"
                        "In your response, only provide the description of the current frame, using the above "
                        "information as context."
                    )
                else:
                    context = (
                        f"{description_instruction}\n\n"
                        f"{previous_frame_section}\n\n"
                        "In your response, only provide the description of the current frame, using the above "
                        "information as context."
                    )

                last_description = await self.vision_client.get_frame_description(img_base64, context)
                time_to_description[timestamp] = last_description

            frame_count += 1

        logger.info(
            f"Generated descriptions for {len(time_to_description)} frames; "
            f"skipped {self.frames_unchanged} unchanged frames"
        )
        return TimeSeriesData(time_to_content=time_to_description)

    async def process_video(self) -> ParseVideoResult:
        """
        Process the video to get frame descriptions and transcript when configured.

        Returns:
            Dictionary containing transcript and frame descriptions as TimeSeriesData objects
        """
        logger.info("Starting full video processing")
        metadata = {
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frame_sample_rate": self.frame_sample_rate,
            "frame_change_threshold": self.frame_change_threshold,
        }
        if self.transcription_config:
            self.transcript = await self.get_model_transcript()
        elif self.transcriber is not None:
            self.get_transcript()
        # Frames are described after transcription so each caption sees the speech around it.
        frame_descriptions = await self.get_frame_descriptions()
        metadata["frames_unchanged"] = self.frames_unchanged
        result = ParseVideoResult(metadata=metadata, transcript=self.transcript, frame_descriptions=frame_descriptions)
        logger.info("Video processing completed successfully")
        return result

    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, "cap"):
            logger.debug("Releasing video capture resources")
            self.cap.release()
