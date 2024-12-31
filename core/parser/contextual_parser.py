import logging
import time
from typing import Any, Dict, List, Tuple
import anthropic

from core.models.chunk import Chunk
from core.parser.base_parser import BaseParser
from core.parser.combined_parser import CombinedParser

logger = logging.getLogger(__name__)

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


class ContextualParser(BaseParser):
    def __init__(
        self,
        unstructured_api_key: str,
        assemblyai_api_key: str,
        chunk_size: int,
        chunk_overlap: int,
        frame_sample_rate: int,
        anthropic_api_key: str,
    ):
        self.combined_parser = CombinedParser(
            unstructured_api_key=unstructured_api_key,
            assemblyai_api_key=assemblyai_api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            frame_sample_rate=frame_sample_rate,
        )
        self.llm = anthropic.Anthropic(api_key=anthropic_api_key)

    def situate_context(self, doc: str, chunk: str) -> str:
        response = self.llm.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            system=[
                {
                    "type": "text",
                    "text": "You are an AI assistant that situates a chunk within a document for the purposes of improving search retrieval of the chunk.",
                },
                {
                    "type": "text",
                    "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        }
                    ],
                }
            ],
        )

        context = response.content[0]
        if context.type == "text":
            return context.text
        else:
            message = f"Anthropic client returned non-text response when situating context for chunk: {chunk} \n Response: {response}"
            logger.error(message)
            raise ValueError(message)

    def situate_all_chunks(self, text: str, chunks: List[Chunk]) -> List[Chunk]:
        new_chunks = []
        chunks_situated = 0
        for chunk in chunks:
            context = self.situate_context(text, chunk.content)
            content = f"{context}; {chunk.content}"
            new_chunk = Chunk(content=content, metadata=chunk.metadata)
            new_chunks.append(new_chunk)
            logger.info(f"Situating the {chunks_situated}th chunk:\n {new_chunk.content[:100]}")
            logger.info("Sleeping to avoid rate limiting...")
            time.sleep(1.25)
            chunks_situated += 1
        return new_chunks

    async def parse_file(
        self, file: bytes, content_type: str
    ) -> Tuple[Dict[str, Any], List[Chunk]]:
        document_metadata, chunks = await self.combined_parser.parse_file(file, content_type)
        document_text = "\n".join([chunk.content for chunk in chunks])
        new_chunks = self.situate_all_chunks(document_text, chunks)
        return document_metadata, new_chunks

    async def split_text(self, text: str) -> List[Chunk]:
        chunks = await self.combined_parser.split_text(text)
        new_chunks = self.situate_all_chunks(text, chunks)
        return new_chunks
