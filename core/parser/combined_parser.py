from typing import List, Union
import logging
import os
from fastapi import UploadFile
import tempfile
import magic

from core.models.time_series import (
    TimeSeriesData,
)  # python-magic library for file type detection

from .base_parser import BaseParser
from .unstructured_parser import UnstructuredAPIParser
from .video.parse_video import VideoParser

logger = logging.getLogger(__name__)


class CombinedParser(BaseParser):
    def __init__(
        self,
        unstructured_api_key: str,
        assemblyai_api_key: str,
        chunk_size: int,
        chunk_overlap: int,
        frame_sample_rate: int,
    ):
        self.unstructured_parser = UnstructuredAPIParser(
            api_key=unstructured_api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.assemblyai_api_key = assemblyai_api_key
        self.frame_sample_rate = frame_sample_rate
        self.magic = magic.Magic(mime=True)

    def _is_video_file(
        self, file_path: str = None, file_bytes: bytes = None, filename: str = None
    ) -> bool:
        """
        Detect if a file is a video using multiple methods:
        1. Magic numbers/file signatures
        2. File extension (as fallback)
        """
        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".mpeg",
            ".mpg",
        }

        # Check magic numbers if we have file content
        if file_bytes:
            mime_type = self.magic.from_buffer(file_bytes)
            if mime_type.startswith("video/"):
                return True
        elif file_path:
            mime_type = self.magic.from_file(file_path)
            if mime_type.startswith("video/"):
                return True

        # Fallback to extension check if we have a filename
        if filename:
            ext = os.path.splitext(filename.lower())[1]
            if ext in video_extensions:
                return True

        return False

    async def split_text(self, text: str) -> List[str]:
        """Split plain text into chunks using unstructured parser"""
        return await self.unstructured_parser.split_text(text)

    async def parse_file(
        self, file: Union[UploadFile, bytes], content_type: str
    ) -> List[str]:
        """Parse file content into text chunks"""
        # For UploadFile, check both filename and content
        if isinstance(file, UploadFile):
            content = await file.read()
            is_video = self._is_video_file(file_bytes=content, filename=file.filename)
            # Reset file position for later use
            await file.seek(0)
        else:
            # For bytes, we can only check content
            is_video = self._is_video_file(file_bytes=file)

        if is_video:
            return await self._parse_video(file)
        else:
            return await self.unstructured_parser.parse_file(file, content_type)

    async def _parse_video(self, file: Union[UploadFile, bytes]) -> List[str]:
        """Parse video file and combine transcript and frame descriptions into chunks"""
        # Save video to temporary file if needed
        if isinstance(file, bytes):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(file)
            temp_file.close()
            video_path = temp_file.name
        else:
            # For UploadFile, save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            video_path = temp_file.name

        try:
            # Process video
            parser = VideoParser(
                video_path,
                assemblyai_api_key=self.assemblyai_api_key,
                frame_sample_rate=self.frame_sample_rate,
            )
            results = parser.process_video()
            # Get all frame descriptions
            frame_descriptions: TimeSeriesData = results["frame_descriptions"]
            # Get all transcript text
            transcript: TimeSeriesData = results["transcript"]
            return frame_descriptions.contents + transcript.contents

        finally:
            # Clean up temporary file
            os.unlink(video_path)
