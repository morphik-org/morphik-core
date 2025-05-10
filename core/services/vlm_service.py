from typing import List, Dict, Any, Optional
import os
import json
import logging
from core.config import get_settings

logger = logging.getLogger(__name__)

class VLMService:
    """
    Service for processing image elements with a Visual Language Model (VLM) to generate bounding boxes.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings for VLM processing")
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4-vision-preview"  # Use a vision-capable model

    async def process_image_elements(
        self,
        image_elements: List[Dict[str, Any]],
        image_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process image elements and chunks with a VLM to generate bounding boxes for each description.

        Args:
            image_elements (List[Dict[str, Any]]): List of image elements with descriptions and IDs.
            image_chunks (List[Dict[str, Any]]): List of image chunks retrieved from tool calls with metadata.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of description IDs to bounding box data and image numbers.
        """
        try:
            import httpx
            from litellm import acompletion

            # Prepare the list of image descriptions for the prompt
            descriptions = []
            for elem in image_elements:
                desc_id = elem.get('element', {}).get('id', '')
                desc_text = elem.get('element', {}).get('description', '')
                if desc_id and desc_text:
                    descriptions.append(f"ID: {desc_id}, Description: {desc_text}")

            # Prepare image content (assuming image_chunks have base64 encoded data or URLs)
            image_contents = []
            for idx, chunk in enumerate(image_chunks, 1):
                image_data = chunk.get('data', '')  # Adjust based on actual data structure
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": image_data if image_data.startswith('http') else f"data:image/jpeg;base64,{image_data}"}
                })

            # Construct the prompt for VLM
            prompt = (
                "Given the following list of image descriptions, analyze the provided images and output a mapping "
                "of each description ID to the corresponding image number and bounding box coordinates. "
                "Format your response as a JSON object where each key is a description ID and the value is an object "
                "with 'image_number' (integer) and 'bbox' (array of four numbers [x1, y1, x2, y2]).\n\n"
                f"List of image descriptions:\n{'; '.join(descriptions)}\n\n"
                "Images are provided below."
            )

            # Prepare messages for the VLM API call
            messages = [
                {"role": "system", "content": "You are an expert in visual analysis and bounding box detection."},
                {"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}
            ]

            # Make the API call using litellm
            response = await acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.endpoint,
                temperature=0.0
            )

            # Parse the response
            response_content = response.choices[0].message.content
            logger.info(f"VLM response content: {response_content}")

            # Assuming the VLM returns JSON-formatted data
            try:
                bounding_boxes = json.loads(response_content)
                return bounding_boxes
            except json.JSONDecodeError:
                logger.error("Failed to parse VLM response as JSON")
                return {}

        except Exception as e:
            logger.error(f"Error processing image elements with VLM: {str(e)}")
            return {}
