import cv2
from typing import Dict
import base64
from openai import OpenAI
import assemblyai as aai
import logging
from core.config import get_settings
from core.models.time_series import TimeSeriesData

logger = logging.getLogger(__name__)

def debug_object(title, obj):
    logger.debug("\n".join(["-" * 100, title, "-" * 100, f"{obj}", "-" * 100]))

class VideoParser():
    def __init__(self, video_path: str, frame_sample_rate: int = 120):
        """
        Initialize the video parser
        
        Args:
            video_path: Path to the video file
            frame_sample_rate: Sample every nth frame for description
        """
        settings = get_settings()
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
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        aai_config = aai.TranscriptionConfig(speaker_labels=True) # speech_model=aai.SpeechModel.nano
        self.transcriber = aai.Transcriber(config=aai_config)
        self.gpt = OpenAI()
        
        logger.info(f"Video loaded: {self.duration:.2f}s duration, {self.fps:.2f} FPS")
    
    def frame_to_base64(self, frame) -> str:
        """Convert a frame to base64 string"""
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            logger.error("Failed to encode frame to JPEG")
            raise ValueError("Failed to encode frame")
        return base64.b64encode(buffer).decode('utf-8')
    
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
        time_to_text = {u.start/1000: u.text for u in transcript.utterances}
        debug_object("Time to text", time_to_text)
        return TimeSeriesData(time_to_text)
    
    def get_frame_descriptions(self) -> TimeSeriesData:
        """
        Get descriptions for sampled frames using GPT-4
        
        Returns:
            TimeSeriesData object containing frame descriptions
        """
        logger.info("Starting frame description generation")
        frame_count = 0
        time_to_description = {}
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_sample_rate == 0:
                timestamp = frame_count / self.fps
                logger.debug(f"Processing frame at {timestamp:.2f}s")
                
                try:
                    img_base64 = self.frame_to_base64(frame)
                    
                    response = self.gpt.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this frame from the video in a detailed manner. Focus on the main elements, actions, and any notable details."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }],
                        max_tokens=300
                    )
                    
                    time_to_description[timestamp] = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Failed to process frame at {timestamp:.2f}s: {str(e)}")
                
            frame_count += 1
            
        logger.info(f"Generated descriptions for {len(time_to_description)} frames")
        return TimeSeriesData(time_to_description)
    
    async def process_video(self) -> Dict:
        """
        Process the video to get both transcript and frame descriptions
        
        Returns:
            Dictionary containing transcript and frame descriptions as TimeSeriesData objects
        """
        logger.info("Starting full video processing")
        result = {
            "metadata": {
                "duration": self.duration,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "frame_sample_rate": self.frame_sample_rate
            },
            "transcript": self.get_transcript(),
            "frame_descriptions": self.get_frame_descriptions()
        }
        logger.info("Video processing completed successfully")
        return result
    
    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, 'cap'):
            logger.debug("Releasing video capture resources")
            self.cap.release()
