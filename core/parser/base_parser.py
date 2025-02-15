from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from core.models.chunk import Chunk


class BaseParser(ABC):
    """Base class for document parsing"""

    @abstractmethod
    async def parse_file_to_text(
        self, file: bytes, content_type: str, filename: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Parse file content into text.

        Args:
            file: Raw file bytes
            content_type: MIME type of the file
            filename: Name of the file

        Returns:
            Tuple[Dict[str, Any], str]: (metadata, extracted_text)
            - metadata: Additional metadata extracted during parsing
            - extracted_text: Raw text extracted from the file
        """
        pass

    @abstractmethod
    async def split_text(self, text: str) -> List[Chunk]:
        """
        Split plain text into chunks.

        Args:
            text: Text to split into chunks

        Returns:
            List[Chunk]: List of text chunks with metadata
        """
        pass
