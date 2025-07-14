import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import filetype
from unstructured.partition.auto import partition

from core.models.chunk import Chunk
from core.parser.base_parser import BaseParser
from core.parser.video.parse_video import VideoParser, load_config
from core.parser.xml_chunker import XMLChunker

# Custom RecursiveCharacterTextSplitter replaces langchain's version


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
        return self.text_splitter.split_text(text)


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        chunks = self._split_recursive(text, self.separators)
        return [Chunk(content=chunk, metadata={}) for chunk in chunks]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if self.length_function(text) <= self.chunk_size:
            return [text] if text else []
        if not separators:
            # No separators left, split at chunk_size boundaries
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep = separators[0]
        if sep:
            splits = text.split(sep)
        else:
            # Last fallback: split every character
            splits = list(text)
        chunks = []
        current = ""
        for part in splits:
            add_part = part + (sep if sep and part != splits[-1] else "")
            if self.length_function(current + add_part) > self.chunk_size:
                if current:
                    chunks.append(current)
                current = add_part
            else:
                current += add_part
        if current:
            chunks.append(current)
        # If any chunk is too large, recurse further
        final_chunks = []
        for chunk in chunks:
            if self.length_function(chunk) > self.chunk_size and len(separators) > 1:
                final_chunks.extend(self._split_recursive(chunk, separators[1:]))
            else:
                final_chunks.append(chunk)
        # Handle overlap
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            overlapped = []
            for i in range(len(final_chunks)):
                chunk = final_chunks[i]
                if i > 0:
                    prev = final_chunks[i - 1]
                    overlap = prev[-self.chunk_overlap :]
                    chunk = overlap + chunk
                overlapped.append(chunk)
            return overlapped
        return final_chunks


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
        if not hasattr(self.settings, "REGISTERED_MODELS") or self.model_key not in self.settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{self.model_key}' not found in registered_models configuration")

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


class MorphikParser(BaseParser):
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
        settings: Optional[Any] = None,
    ):
        # Initialize basic configuration
        self.use_unstructured_api = use_unstructured_api
        self._unstructured_api_key = unstructured_api_key
        self._assemblyai_api_key = assemblyai_api_key
        self._anthropic_api_key = anthropic_api_key
        self.frame_sample_rate = frame_sample_rate
        self.settings = settings
        self.logger = logging.getLogger(__name__)

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

    def _is_xml_file(self, file: bytes, filename: str) -> bool:
        """Check if the file is an XML file."""
        # Check file extension
        if filename and filename.lower().endswith('.xml'):
            return True
        
        # Check content type by trying to detect XML content
        try:
            kind = filetype.guess(file)
            if kind and kind.mime in ['application/xml', 'text/xml']:
                return True
        except Exception:
            pass
        
        # Fallback: check if content starts with XML declaration or has XML-like structure
        try:
            content_start = file[:1000].decode('utf-8', errors='ignore').strip()
            if content_start.startswith('<?xml') or (content_start.startswith('<') and '>' in content_start):
                return True
        except Exception:
            pass
        
        return False

    async def _parse_xml(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
        """Parse XML file using XMLChunker and return metadata and combined text."""
        if not self.settings or not hasattr(self.settings, 'PARSER_XML'):
            # Fallback to default settings if not available
            xml_config = {
                "max_tokens": 350,
                "preferred_unit_tags": ["SECTION", "Section", "Article", "clause"],
                "ignore_tags": ["TOC", "INDEX"]
            }
        else:
            xml_config = self.settings.PARSER_XML.model_dump()
        
        self.logger.info(f"Processing '{filename}' with dedicated XML chunker.")
        
        try:
            # Create XMLChunker and get chunks
            xml_chunker = XMLChunker(content=file, config=xml_config)
            xml_chunks_data = xml_chunker.chunk()
            
            # Combine all chunk texts for the text processing pipeline
            combined_text = "\n\n".join([chunk_data['text'] for chunk_data in xml_chunks_data])
            
            # Create metadata that includes the XML chunks for later use
            metadata = {
                "xml_chunks": xml_chunks_data,
                "is_xml": True,
                "chunk_count": len(xml_chunks_data)
            }
            
            self.logger.info(f"XML chunking created {len(xml_chunks_data)} chunks")
            return metadata, combined_text
            
        except Exception as e:
            self.logger.error(f"Error parsing XML file '{filename}': {e}")
            # Fallback to regular text extraction
            try:
                text = file.decode('utf-8', errors='ignore')
                return {}, text
            except Exception:
                return {}, ""

    async def _parse_video(self, file: bytes) -> Tuple[Dict[str, Any], str]:
        """Parse video file to extract transcript and frame descriptions"""
        if not self._assemblyai_api_key:
            raise ValueError("AssemblyAI API key is required for video parsing")

        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(file)
            video_path = temp_file.name

        try:
            # Load the config to get the frame_sample_rate from morphik.toml
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
        # Choose a lighter parsing strategy for text-based files. Using
        # `hi_res` on plain PDFs/Word docs invokes OCR which can be 20-30×
        # slower.  A simple extension check covers the majority of cases.
        strategy = "hi_res"
        file_content_type: Optional[str] = None  # Default to None for auto-detection
        if filename.lower().endswith((".pdf", ".doc", ".docx")):
            strategy = "fast"
        elif filename.lower().endswith(".txt"):
            strategy = "fast"
            file_content_type = "text/plain"  # Explicitly set for .txt files
        elif filename.lower().endswith(".json"):
            strategy = "fast"  # or can be omitted if it doesn't apply to json
            file_content_type = "application/json"  # Explicitly set for .json files

        elements = partition(
            file=io.BytesIO(file),
            content_type=file_content_type,  # Use the determined content_type
            metadata_filename=filename,
            strategy=strategy,
            api_key=self._unstructured_api_key if self.use_unstructured_api else None,
        )

        text = "\n\n".join(str(element) for element in elements if str(element).strip())
        return {}, text

    async def parse_file_to_text(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
        """Parse file content into text based on file type"""
        if self._is_xml_file(file, filename):
            return await self._parse_xml(file, filename)
        elif self._is_video_file(file, filename):
            return await self._parse_video(file)
        return await self._parse_document(file, filename)

    async def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks using configured chunking strategy"""
        # Check if this is pre-chunked XML content
        if metadata and metadata.get("is_xml") and "xml_chunks" in metadata:
            xml_chunks_data = metadata["xml_chunks"]
            chunks = []
            
            for i, chunk_data in enumerate(xml_chunks_data):
                # Create metadata for the chunk
                chunk_metadata = {
                    "unit": chunk_data.get("unit"),
                    "xml_id": chunk_data.get("xml_id"),
                    "breadcrumbs": chunk_data.get("breadcrumbs"),
                    "source_path": chunk_data.get("source_path"),
                    "prev_chunk_xml_id": chunk_data.get("prev"),
                    "next_chunk_xml_id": chunk_data.get("next"),
                    "token_count": chunk_data.get("token_count"),
                }
                
                chunks.append(Chunk(
                    content=chunk_data['text'],
                    chunk_number=i,
                    metadata=chunk_metadata
                ))
            
            self.logger.info(f"Converted {len(chunks)} XML chunks to Chunk objects")
            return chunks
        
        # Use regular chunking strategy for non-XML files
        return self.chunker.split_text(text)
