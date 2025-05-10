from typing import Dict, List, Any
from pydantic import BaseModel

class TextElement(BaseModel):
    content: str

class ImageElement(BaseModel):
    description: str
    id: str = ""

class DisplayElement(BaseModel):
    type: str  # 'text' or 'image'
    element: Any  # TextElement or ImageElement

class DesignAgent:
    """
    DesignAgent processes display instructions from MorphikAgent to create structured display elements.
    """

    def process_display_instructions(self, response: Dict[str, Any]) -> List[DisplayElement]:
        """
        Parse display instructions and convert them into a list of structured display elements.

        Args:
            response (Dict[str, Any]): The response dictionary from MorphikAgent containing display_instructions.

        Returns:
            List[DisplayElement]: A list of structured display elements (TextElement or ImageElement).
        """
        display_elements = []
        instructions = response.get('display_instructions', '')
        body = response.get('body', '')

        # Default logic to handle basic instructions
        if 'markdown formatting for text' in instructions.lower():
            display_elements.append(DisplayElement(
                type='text',
                element=TextElement(content=body)
            ))

        if 'place any referenced images or graphs below' in instructions.lower():
            # Extract potential image references from body (e.g., [ref:fig1])
            import re
            image_refs = re.findall(r'\[ref:([^\]]+)\]', body)
            for ref_id in image_refs:
                display_elements.append(DisplayElement(
                    type='image',
                    element=ImageElement(
                        description=f"Image or graph referenced as {ref_id}",
                        id=ref_id
                    )
                ))

        return display_elements
