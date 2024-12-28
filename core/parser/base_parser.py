from abc import ABC, abstractmethod
from typing import List, Union
from fastapi import UploadFile
from core.models.documents import Chunk


class BaseParser(ABC):
    """Base class for document parsing"""

    @abstractmethod
    async def split_text(self, text: str) -> List[Chunk]:
        """Split plain text into chunks"""
        pass

    @abstractmethod
    async def parse_file(self, file: bytes, content_type: str) -> List[Chunk]:
        """Parse file content into text chunks"""
        pass
