import json
import logging
from typing import Any, Dict, List

from litellm import acompletion

from core.config import get_settings
from core.models.auth import AuthContext

logger = logging.getLogger(__name__)


class GroundingService:
    """
    Service to perform a second-pass analysis on agent responses to
    ground citations in source material (e.g., add bounding boxes for images,
    refine text snippets).
    """

    def __init__(self):
        self.settings = get_settings()
        # Ensure GEMINI_API_KEY is in the environment for LiteLLM to use Gemini.
        # LiteLLM typically reads this from os.environ['GEMINI_API_KEY']

    async def ground_citations(
        self,
        citations: List[Dict[str, Any]],
        original_user_query: str,
        agent_answer_body: str,
        auth: AuthContext,  # unused for now, but could be for fetching secured assets
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of citations, attempting to ground them.
        For image citations, it tries to get bounding boxes.
        For text citations, it (eventually) will refine snippets.
        """
        if not citations:
            return []

        updated_citations = []
        for citation_input in citations:
            citation = citation_input.copy()
            citation["grounded"] = False  # Default to not grounded

            if citation.get("type") == "image":
                try:
                    image_url = citation.get("imageUrl")
                    citation_id = citation.get("id", "unknown_image_citation")

                    if not image_url:
                        logger.warning(f"Image citation {citation_id} missing imageUrl. Skipping grounding.")
                        updated_citations.append(citation)
                        continue

                    image_description_clue = citation.get("snippet", "the relevant part of this image")

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
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ]

                    logger.info(
                        f"Sending image grounding request for citation {citation_id}"
                        f"with image URL snippet: {image_url[:100]}..."
                    )

                    # Determine the grounding model from settings or use a default
                    grounding_model_name = (
                        getattr(self.settings, "GROUNDING_MODEL", None) or "gemini/gemini-1.5-flash-latest"
                    )

                    response = await acompletion(
                        model=grounding_model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        # Ensure API keys are set in the environment for LiteLLM
                    )

                    response_content = response.choices[0].message.content
                    logger.info(f"Grounding response for {citation_id}: {response_content}")

                    if response_content:
                        try:
                            parsed_response = json.loads(response_content)
                            bbox = parsed_response.get("bbox")  # Expected to be [x1, y1, x2, y2] or null

                            if bbox is None:
                                logger.info(
                                    f"Grounding model returned null bbox for"
                                    f"{citation_id}. Marking as not specifically grounded."
                                )
                                citation["grounded"] = False  # Explicitly not grounded if bbox is null
                            elif (
                                isinstance(bbox, list)
                                and len(bbox) == 4
                                and all(isinstance(coord, (int, float)) for coord in bbox)
                            ):
                                # Validate normalized coordinates (0.0 to 1.0)
                                if all(0.0 <= coord <= 1.0 for coord in bbox):
                                    citation["bbox"] = bbox
                                    citation["grounded"] = True
                                    logger.info(f"Successfully extracted normalized bbox for {citation_id}: {bbox}")
                                else:
                                    logger.warning(
                                        f"Extracted bbox for {citation_id} is not normalized: {bbox}. Discarding."
                                    )
                                    citation["grounded"] = False
                            else:
                                logger.warning(
                                    f"Invalid or non-list/non-4-item bbox in response"
                                    f"for {citation_id}. Response: {parsed_response}"
                                )
                                citation["grounded"] = False
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse JSON from grounding response for {citation_id}: {response_content}"
                            )
                            citation["grounded"] = False
                    else:
                        logger.warning(f"No content in grounding response for {citation_id}")
                        citation["grounded"] = False

                except Exception as e:
                    logger.error(
                        f"Error grounding image citation {citation.get('id', 'unknown_image_citation')}: {e}",
                        exc_info=True,
                    )
                    citation["grounded"] = False  # Ensure it's marked not grounded on any error

            elif citation.get("type") == "text":
                # Placeholder for actual text grounding/refinement
                # For now, if a snippet exists, we consider it provisionally grounded.
                if citation.get("snippet"):
                    citation["grounded"] = True
                logger.info(
                    f"Text citation {citation.get('id', 'unknown_text_citation')} - basic snippet considered grounded."
                )

            updated_citations.append(citation)

        return updated_citations
