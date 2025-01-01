from typing import Any, Dict, List, Tuple
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from core.models.chunk import Chunk
import logging

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class UnstructuredParser(BaseParser):
    def __init__(
        self,
        use_api: bool,
        api_key: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        if use_api and not api_key:
            logger.error("API key is required if use_api is set to True")
            raise ValueError("[UnstructuredParser] API key is required if use_api is True")
        self.use_api = use_api
        self.api_key = api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def split_text(self, text: str) -> List[Chunk]:
        """Split plain text into chunks"""
        return [Chunk(content=chunk, metadata={}) for chunk in self.text_splitter.split_text(text)]

    async def parse_file(
        self, file: bytes, content_type: str, filename: str
    ) -> Tuple[Dict[str, Any], List[Chunk]]:
        """Parse file content using unstructured"""
        # Parse with unstructured
        loader = UnstructuredLoader(
            file=io.BytesIO(file),
            partition_via_api=self.use_api,
            api_key=self.api_key if self.use_api else None,
            metadata_filename=None if self.use_api else filename,
            chunking_strategy="by_title",
        )
        elements = loader.load()
        return {}, [Chunk(content=element.page_content, metadata={}) for element in elements]
