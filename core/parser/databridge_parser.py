from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import tempfile
import io
import filetype
from abc import ABC, abstractmethod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

from core.models.chunk import Chunk
from core.parser.base_parser import BaseParser
from core.parser.video.parse_video import VideoParser, load_config

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Base class for text chunking strategies"""

    @abstractmethod
    def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks"""
        pass


class StandardChunker(BaseChunker):
    """Standard chunking using langchain's RecursiveCharacterTextSplitter"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_text(self, text: str) -> List[Chunk]:
        chunks = self.text_splitter.split_text(text)
        return [Chunk(content=chunk, metadata={}) for chunk in chunks]


class ContextualChunker(BaseChunker):
    """Contextual chunking using LLMs to add context to each chunk"""

    DOCUMENT_CONTEXT_PROMPT = """
    <document>
    {doc_content}
    </document>
    """

    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, anthropic_api_key: str):
        self.standard_chunker = StandardChunker(chunk_size, chunk_overlap)

        # Get the config for contextual chunking
        config = load_config()
        parser_config = config.get("parser", {})
        self.model_key = parser_config.get("contextual_chunking_model", "claude_sonnet")

        # Get the settings for registered models
        from core.config import get_settings

        self.settings = get_settings()

        # Make sure the model exists in registered_models
        if (
            not hasattr(self.settings, "REGISTERED_MODELS")
            or self.model_key not in self.settings.REGISTERED_MODELS
        ):
            raise ValueError(
                f"Model '{self.model_key}' not found in registered_models configuration"
            )

        self.model_config = self.settings.REGISTERED_MODELS[self.model_key]
        logger.info(f"Initialized ContextualChunker with model_key={self.model_key}")

    def _situate_context(self, doc: str, chunk: str) -> str:
        import litellm

        # Extract model name from config
        model_name = self.model_config.get("model_name")

        # Create system and user messages
        system_message = {
            "role": "system",
            "content": "You are an AI assistant that situates a chunk within a document for the purposes of improving search retrieval of the chunk.",
        }

        # Add document context and chunk to user message
        user_message = {
            "role": "user",
            "content": f"{self.DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)}\n\n{self.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)}",
        }

        # Prepare parameters for litellm
        model_params = {
            "model": model_name,
            "messages": [system_message, user_message],
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        # Add all model-specific parameters from the config
        for key, value in self.model_config.items():
            if key != "model_name":
                model_params[key] = value

        # Use litellm for completion
        response = litellm.completion(**model_params)
        return response.choices[0].message.content

    def split_text(self, text: str) -> List[Chunk]:
        base_chunks = self.standard_chunker.split_text(text)
        contextualized_chunks = []

        for chunk in base_chunks:
            context = self._situate_context(text, chunk.content)
            content = f"{context}; {chunk.content}"
            contextualized_chunks.append(Chunk(content=content, metadata=chunk.metadata))

        return contextualized_chunks


class DatabridgeParser(BaseParser):
    """Unified parser that handles different file types and chunking strategies"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_unstructured_api: bool = False,
        unstructured_api_key: Optional[str] = None,
        assemblyai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        frame_sample_rate: int = 1,
        use_contextual_chunking: bool = False,
    ):
        # Initialize basic configuration
        self.use_unstructured_api = use_unstructured_api
        self._unstructured_api_key = unstructured_api_key
        self._assemblyai_api_key = assemblyai_api_key
        self._anthropic_api_key = anthropic_api_key
        self.frame_sample_rate = frame_sample_rate

        # Initialize chunker based on configuration
        if use_contextual_chunking:
            self.chunker = ContextualChunker(chunk_size, chunk_overlap, anthropic_api_key)
        else:
            self.chunker = StandardChunker(chunk_size, chunk_overlap)

    def _is_video_file(self, file: bytes, filename: str) -> bool:
        """Check if the file is a video file."""
        try:
            kind = filetype.guess(file)
            return kind is not None and kind.mime.startswith("video/")
        except Exception as e:
            logging.error(f"Error detecting file type: {str(e)}")
            return False

    async def _parse_video(self, file: bytes) -> Tuple[Dict[str, Any], str]:
        """Parse video file to extract transcript and frame descriptions"""
        if not self._assemblyai_api_key:
            raise ValueError("AssemblyAI API key is required for video parsing")

        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(file)
            video_path = temp_file.name

        try:
            # Load the config to get the frame_sample_rate from databridge.toml
            config = load_config()
            parser_config = config.get("parser", {})
            vision_config = parser_config.get("vision", {})
            frame_sample_rate = vision_config.get("frame_sample_rate", self.frame_sample_rate)

            # Process video
            parser = VideoParser(
                video_path,
                assemblyai_api_key=self._assemblyai_api_key,
                frame_sample_rate=frame_sample_rate,
            )
            results = await parser.process_video()

            # Combine frame descriptions and transcript
            frame_text = "\n".join(results.frame_descriptions.time_to_content.values())
            transcript_text = "\n".join(results.transcript.time_to_content.values())
            combined_text = f"Frame Descriptions:\n{frame_text}\n\nTranscript:\n{transcript_text}"

            metadata = {
                "video_metadata": results.metadata,
                "frame_timestamps": list(results.frame_descriptions.time_to_content.keys()),
                "transcript_timestamps": list(results.transcript.time_to_content.keys()),
            }

            return metadata, combined_text
        finally:
            os.unlink(video_path)

    async def _parse_document(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
        """Parse document using unstructured"""
        elements = partition(
            file=io.BytesIO(file),
            content_type=None,
            metadata_filename=filename,
            strategy="hi_res",
            api_key=self._unstructured_api_key if self.use_unstructured_api else None,
        )

        text = "\n\n".join(str(element) for element in elements if str(element).strip())
        return {}, text

    async def parse_file_to_text(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
        """Parse file content into text based on file type"""
        if self._is_video_file(file, filename):
            return await self._parse_video(file)
        return await self._parse_document(file, filename)

    async def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks using configured chunking strategy"""
        return self.chunker.split_text(text)
