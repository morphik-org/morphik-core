import logging
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field, field_validator

from core.models.video import TimeSeriesData

logger = logging.getLogger(__name__)


class QueryReturnType(str, Enum):
    CHUNKS = "chunks"
    DOCUMENTS = "documents"


class StorageFileInfo(BaseModel):
    """Information about a file stored in storage"""

    bucket: str
    key: str
    version: int = 1
    filename: Optional[str] = None
    content_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(BaseModel):
    """Represents a document stored in the database documents collection"""

    external_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """user-defined metadata"""
    storage_info: Dict[str, Any] = Field(default_factory=dict)
    """Legacy field for backwards compatibility - for single file storage"""
    storage_files: List[StorageFileInfo] = Field(default_factory=list)
    """List of files associated with this document"""
    system_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "version": 1,
            "status": "processing",  # Status can be: processing, completed, failed
        }
    )
    """metadata such as creation date etc."""
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)
    """metadata to help with querying eg. frame descriptions and time-stamped transcript for videos"""
    chunk_ids: List[str] = Field(default_factory=list)

    # Flattened fields from system_metadata for performance
    folder_name: Optional[str] = None
    end_user_id: Optional[str] = None
    app_id: Optional[str] = None

    # Ensure storage_info values are strings to maintain backward compatibility
    @field_validator("storage_info", mode="before")
    def _coerce_storage_info_values(cls, v):
        if isinstance(v, dict):
            return {k: str(val) if val is not None else "" for k, val in v.items()}
        return v

    def __hash__(self):
        return hash(self.external_id)

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.external_id == other.external_id


class DocumentContent(BaseModel):
    """Represents either a URL or content string"""

    type: Literal["url", "string"]
    value: str
    filename: Optional[str] = Field(None, description="Filename when type is url")

    @field_validator("filename")
    def filename_only_for_url(cls, v, values):
        logger.debug(f"Value looks like: {values}")
        if values.data.get("type") == "string" and v is not None:
            raise ValueError("filename can only be set when type is url")
        if values.data.get("type") == "url" and v is None:
            raise ValueError("filename is required when type is url")
        return v


class DocumentResult(BaseModel):
    """Query result at document level"""

    score: float  # Highest chunk score
    document_id: str  # external_id
    metadata: Dict[str, Any]
    content: DocumentContent
    additional_metadata: Dict[str, Any]


class ChunkResult(BaseModel):
    """Query result at chunk level"""

    content: str
    score: float
    document_id: str  # external_id
    chunk_number: int
    metadata: Dict[str, Any]
    content_type: str
    filename: Optional[str] = None
    download_url: Optional[str] = None
    is_padding: bool = Field(default=False, description="Whether this chunk was added as padding")

    def augmented_content(self, doc: DocumentResult) -> str | Image.Image:
        match self.metadata:
            case m if "timestamp" in m:
                # if timestamp present, then must be a video. In that case,
                # obtain the original document and augment the content with
                # frame/transcript information as well.
                frame_description = doc.additional_metadata.get("frame_description")
                transcript = doc.additional_metadata.get("transcript")
                if not isinstance(frame_description, dict) or not isinstance(transcript, dict):
                    logger.warning("Invalid frame description or transcript - not a dictionary")
                    return self.content
                ts_frame = TimeSeriesData(time_to_content=frame_description)
                ts_transcript = TimeSeriesData(time_to_content=transcript)
                timestamps = ts_frame.content_to_times[self.content] + ts_transcript.content_to_times[self.content]
                augmented_contents = [
                    f"Frame description: {ts_frame.at_time(t)} \n \n Transcript: {ts_transcript.at_time(t)}"
                    for t in timestamps
                ]
                return "\n\n".join(augmented_contents)
            # case m if m.get("is_image", False):
            #     try:
            #         # Handle data URI format "data:image/png;base64,..."
            #         content = self.content
            #         if content.startswith('data:'):
            #             # Extract the base64 part after the comma
            #             content = content.split(',', 1)[1]

            #         # Now decode the base64 string
            #         image_bytes = base64.b64decode(content)
            #         content = Image.open(io.BytesIO(image_bytes))
            #         return content
            #     except Exception as e:
            #         print(f"Error processing image: {str(e)}")
            #         # Fall back to using the content as text
            #         return self.content
            case _:
                return self.content


class ChunkGroup(BaseModel):
    """Represents a group of chunks: one main match + its padding chunks"""

    main_chunk: ChunkResult
    padding_chunks: List[ChunkResult] = Field(default_factory=list)
    total_chunks: int = Field(description="Total number of chunks in this group")

    @property
    def all_chunks(self) -> List[ChunkResult]:
        """Get all chunks in display order (padding before + main + padding after)"""
        # Sort padding chunks by chunk_number
        sorted_padding = sorted(self.padding_chunks, key=lambda x: x.chunk_number)

        # Split into before and after the main chunk
        before_main = [c for c in sorted_padding if c.chunk_number < self.main_chunk.chunk_number]
        after_main = [c for c in sorted_padding if c.chunk_number > self.main_chunk.chunk_number]

        return before_main + [self.main_chunk] + after_main


class GroupedChunkResponse(BaseModel):
    """Response that includes both flat results and grouped results for UI"""

    chunks: List[ChunkResult] = Field(description="Flat list of all chunks (for backward compatibility)")
    groups: List[ChunkGroup] = Field(description="Grouped chunks for UI display")
    total_results: int = Field(description="Total number of unique chunks")
    has_padding: bool = Field(description="Whether padding was applied to any results")
