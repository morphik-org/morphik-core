import json
import logging
import re
from typing import Any, Dict, List, Optional

from litellm import acompletion

from core.config import get_settings
from core.models.auth import AuthContext
from core.models.completion import ChunkSource
from core.models.documents import ChunkResult
from core.services.document_service import DocumentService

logger = logging.getLogger(__name__)

# Regex to parse the sourceChunkId
CHUNK_ID_REGEX = re.compile(r"doc:(?P<document_id>[^:]+)::chunk:(?P<chunk_number>\d+)::type:(?P<type>image|text)")


class GroundingService:
    """
    Service to perform a second-pass analysis on agent responses to
    ground citations in source material (e.g., add bounding boxes for images,
    refine text snippets).
    """

    def __init__(self, document_service: DocumentService):
        self.settings = get_settings()
        self.document_service = document_service
        # Ensure GEMINI_API_KEY is in the environment for LiteLLM to use Gemini.
        # LiteLLM typically reads this from os.environ['GEMINI_API_KEY']

    async def _get_image_url_from_chunk(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> Optional[str]:
        """Helper to convert chunk content (potentially base64) to a data URL."""
        if chunk_content.startswith("data:image"):
            logger.debug(f"Image content is already a data URI: {chunk_content[:100]}...")
            return chunk_content
        elif chunk_metadata.get("is_image", False):
            mime_type = chunk_metadata.get("mime_type", "image/png")
            # Ensure base64 content is not empty
            if not chunk_content:
                logger.warning("Attempted to create data URI from empty image chunk content.")
                return None
            data_uri = f"data:{mime_type};base64,{chunk_content}"
            logger.debug(f"Generated data URI for image: {data_uri[:100]}...")
            return data_uri
        logger.warning("Chunk content not a data URI and not marked as image, or content is empty.")
        return None

    async def ground_citations(
        self,
        citations: List[Dict[str, Any]],
        original_user_query: str,
        agent_answer_body: str,
        auth: AuthContext,  # unused for now, but could be for fetching secured assets
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of citations, attempting to ground them.
        For image citations, it uses sourceChunkId to fetch image data, then gets bounding boxes.
        For text citations, it uses sourceChunkId to fetch full chunk content for the snippet.
        """
        if not citations:
            return []

        updated_citations = []
        for citation_input in citations:
            citation = citation_input.copy()
            citation["grounded"] = False  # Default to not grounded
            citation_id_log = citation.get("id", "unknown_citation")  # For logging

            source_chunk_id = citation.get("sourceChunkId")

            if not source_chunk_id:
                logger.warning(
                    f"Citation {citation_id_log} (type: {citation.get('type')}) "
                    "missing sourceChunkId. Skipping grounding."
                )
                updated_citations.append(citation)
                continue

            parsed_id_match = CHUNK_ID_REGEX.match(source_chunk_id)
            if not parsed_id_match:
                logger.warning(
                    f"Could not parse sourceChunkId '{source_chunk_id}' for citation {citation_id_log}. Skipping."
                )
                updated_citations.append(citation)
                continue

            parsed_info = parsed_id_match.groupdict()
            doc_id = parsed_info["document_id"]
            try:
                chunk_num = int(parsed_info["chunk_number"])
            except ValueError:
                logger.warning(
                    f"Invalid chunk_number in sourceChunkId '{source_chunk_id}' for citation {citation_id_log}. "
                    "Skipping."
                )
                updated_citations.append(citation)
                continue

            # Attempt to retrieve the chunk regardless of type first, as both need it.
            retrieved_chunk_result: Optional[ChunkResult] = None
            try:
                chunk_sources = [ChunkSource(document_id=doc_id, chunk_number=chunk_num)]
                retrieved_chunks_list = await self.document_service.batch_retrieve_chunks(
                    chunk_ids=chunk_sources, auth=auth, use_colpali=True  # use_colpali might be relevant for images
                )
                if retrieved_chunks_list:
                    retrieved_chunk_result = retrieved_chunks_list[0]
                else:
                    logger.warning(
                        f"Could not retrieve chunk for sourceChunkId '{source_chunk_id}' (citation {citation_id_log})."
                    )
            except Exception as e_fetch:
                logger.error(
                    f"Error fetching chunk for sourceChunkId '{source_chunk_id}' (citation {citation_id_log}): "
                    f"{e_fetch}",
                    exc_info=True,
                )

            if not retrieved_chunk_result:
                updated_citations.append(citation)  # Append as is, marked not grounded
                continue

            # Now handle based on type
            if citation.get("type") == "image":
                image_to_ground_url: Optional[str] = None
                try:
                    image_to_ground_url = await self._get_image_url_from_chunk(
                        retrieved_chunk_result.content, retrieved_chunk_result.metadata
                    )
                    if not image_to_ground_url:
                        logger.warning(
                            f"Failed to create a usable image URL from chunk"
                            f"{source_chunk_id} for citation {citation_id_log}. "
                            "Skipping grounding."
                        )
                        updated_citations.append(citation)
                        continue

                    citation["imageUrl"] = image_to_ground_url
                    # BBox detection logic (as before)
                    image_description_clue = (
                        citation.get("snippet") or citation.get("reasoning") or "the relevant part of this image"
                    )
                    prompt_parts = [
                        f"The user asked: '{original_user_query}'.",
                        f"The assistant's answer (which might reference this image) is: '{agent_answer_body}'.",
                        "Regarding the following image, focus on the area related to this clue:",
                        f"'{image_description_clue}'.",
                        "Please identify the main subject or region in the image that corresponds to that clue.",
                        "Return a JSON object with a single key 'bbox' containing a list of four",
                        "numbers [x1, y1, x2, y2] representing the bounding box coordinates.",
                        "These coordinates should be normalized (ranging from 0.0 to 1.0) relative to the image",
                        "dimensions, where (0,0) is the top-left corner.",
                        "If you cannot determine a specific bounding box for the clue, return null for the",
                        "'bbox' value (e.g., {'bbox': null}).",
                    ]
                    prompt_text = " ".join(prompt_parts)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": image_to_ground_url}},
                            ],
                        }
                    ]
                    logger.info(
                        f"Sending image grounding request for citation {citation_id_log} "
                        f"(source: {source_chunk_id}) with image snippet: {image_to_ground_url[:100]}..."
                    )
                    # Use gemini-1.5-flash-latest as a more current default, allow override via settings
                    grounding_model_name = (
                        getattr(self.settings, "GROUNDING_MODEL", None) or "gemini/gemini-1.5-flash-latest"
                    )
                    response = await acompletion(
                        model=grounding_model_name, messages=messages, response_format={"type": "json_object"}
                    )
                    response_content = response.choices[0].message.content
                    logger.info(f"Grounding response for {citation_id_log}: {response_content}")

                    if response_content:
                        try:
                            parsed_response = json.loads(response_content)
                            bbox = parsed_response.get("bbox")
                            if bbox is None:
                                logger.info(f"Grounding model returned null bbox for {citation_id_log}.")
                                citation["grounded"] = False
                            elif (
                                isinstance(bbox, list)
                                and len(bbox) == 4
                                and all(isinstance(coord, (int, float)) for coord in bbox)
                            ):
                                if all(0.0 <= coord <= 1.0 for coord in bbox):
                                    citation["bbox"] = bbox
                                    citation["grounded"] = True
                                    logger.info(f"Successfully extracted normalized bbox for {citation_id_log}: {bbox}")
                                else:
                                    logger.warning(
                                        f"Extracted bbox for {citation_id_log} is not normalized: {bbox}. Discarding."
                                    )
                                    citation["grounded"] = False
                            else:
                                logger.warning(
                                    f"Invalid bbox format for {citation_id_log}. Response: {parsed_response}"
                                )
                                citation["grounded"] = False
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse JSON from grounding response for {citation_id_log}: "
                                f"{response_content}"
                            )
                            citation["grounded"] = False
                    else:
                        logger.warning(f"No content in grounding response for {citation_id_log}")
                        citation["grounded"] = False
                except Exception as e:
                    logger.error(
                        f"Error grounding image citation {citation_id_log} (source: {source_chunk_id}): {e}",
                        exc_info=True,
                    )
                    citation["grounded"] = False  # Ensure it's marked not grounded on any error within image block

            elif citation.get("type") == "text":
                # If snippet is already provided by LLM and is good, we could use it.
                # Otherwise, or to ensure full context, use the retrieved chunk's content.
                if retrieved_chunk_result.content:
                    citation["snippet"] = retrieved_chunk_result.content  # Use full content from retrieved chunk
                    citation["grounded"] = True
                    logger.info(
                        f"Text citation {citation_id_log} (source: {source_chunk_id}) grounded with full chunk content."
                    )
                else:
                    logger.warning(
                        f"Text citation {citation_id_log} (source: {source_chunk_id}) - retrieved chunk has no content."
                    )
                    # Keep existing snippet if LLM provided one, otherwise it remains un-grounded
                    if not citation.get("snippet"):  # If LLM didn't provide a snippet either
                        citation["grounded"] = False
                    # If LLM provided snippet, it's already in 'citation' object.
                    # Do not overwrite it if chunk fetch failed.

            updated_citations.append(citation)

        return updated_citations
