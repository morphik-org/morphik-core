from numbers import Number
import cv2
from typing import Dict, Union
import base64
from openai import OpenAI
import assemblyai as aai
import logging
from core.models.video import TimeSeriesData, ParseVideoResult

logger = logging.getLogger(__name__)


def debug_object(title, obj):
    logger.debug("\n".join(["-" * 100, title, "-" * 100, f"{obj}", "-" * 100]))


class VideoParser:
    def __init__(
        self, video_path: str, assemblyai_api_key: str, frame_sample_rate: int = 120
    ):
        """
        Initialize the video parser

        Args:
            video_path: Path to the video file
            assemblyai_api_key: API key for AssemblyAI
            frame_sample_rate: Sample every nth frame for description
        """
        logger.info(f"Initializing VideoParser for {video_path}")
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        aai.settings.api_key = assemblyai_api_key
        aai_config = aai.TranscriptionConfig(
            speaker_labels=True
        )  # speech_model=aai.SpeechModel.nano
        self.transcriber = aai.Transcriber(config=aai_config)
        self.transcript = TimeSeriesData(
            time_to_content={}
        )  # empty transcript initially - TODO: have this be a lateinit somehow
        self.gpt = OpenAI()

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
        logger.info("Starting video transcription")
        transcript = self.get_transcript_object()
        # divide by 1000 because assemblyai timestamps are in milliseconds
        time_to_text = (
            {u.start / 1000: u.text for u in transcript.utterances}
            if transcript.utterances
            else {}
        )
        debug_object("Time to text", time_to_text)
        self.transcript = TimeSeriesData(time_to_text)
        return self.transcript

    def get_frame_descriptions(self) -> TimeSeriesData:
        """
        Get descriptions for sampled frames using GPT-4

        Returns:
            TimeSeriesData object containing frame descriptions
        """
        logger.info("Starting frame description generation")
        frame_count = 0
        time_to_description = {}
        last_description = None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % self.frame_sample_rate == 0:
                timestamp = frame_count / self.fps
                logger.debug(f"Processing frame at {timestamp:.2f}s")

                img_base64 = self.frame_to_base64(frame)

                response = self.gpt.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Describe this frame from a video. Focus on the main elements, actions, and any notable details. Here is the transcript around the time of the frame:
                                ---
                                {self.transcript.at_time(timestamp, padding=10)}
                                ---

                                Here is a description of the previous frame:
                                ---
                                {last_description if last_description else 'No previous frame description available, this is the first frame'}
                                ---

                                In your response, only provide the description of the current frame, using the above information as context.
                                """,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                last_description = response.choices[0].message.content
                time_to_description[timestamp] = last_description

            frame_count += 1

        logger.info(f"Generated descriptions for {len(time_to_description)} frames")
        return TimeSeriesData(time_to_description)

    def process_video(self) -> ParseVideoResult:
        """
        Process the video to get both transcript and frame descriptions

        Returns:
            Dictionary containing transcript and frame descriptions as TimeSeriesData objects
        """
        logger.info("Starting full video processing")
        metadata = {
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frame_sample_rate": self.frame_sample_rate,
        }
        result = ParseVideoResult(
            metadata=metadata,
            transcript=self.get_transcript(),
            frame_descriptions=self.get_frame_descriptions(),
        )
        logger.info("Video processing completed successfully")
        return result

    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, "cap"):
            logger.debug("Releasing video capture resources")
            self.cap.release()
