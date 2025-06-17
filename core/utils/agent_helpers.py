import base64
import io
import json
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def extract_display_object(item: dict, source_map: dict):
    valid_object = isinstance(item, dict) and "type" in item and "content" in item
    if not valid_object:
        return {"invalid": True}
    match item["type"]:
        case "text":
            return {"type": item["type"], "source": item.get("source", "agent-response"), "content": item["content"]}
        case "image":
            source_id = item.get("source", "agent-response")
            # Get image content from source_map if available
            if source_id in source_map and "content" in source_map[source_id]:
                image_content = source_map[source_id]["content"]
            else:
                # Fallback to empty string if source not found
                image_content = ""

            return {
                "type": item["type"],
                "source": source_id,
                "content": image_content,
                "caption": item["content"],
            }
        case _:
            return {"invalid": True}


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def scale_and_clamp(val1, val2, current_scale, desired_scale, padding_percent):
    padding_multiplier1, padding_multiplier2 = 1 - padding_percent / 200, 1 + padding_percent / 200
    true_val1 = int((val1 / current_scale) * desired_scale * padding_multiplier1)
    true_val2 = int((val2 / current_scale) * desired_scale * padding_multiplier2)
    return max(true_val1, 0), min(true_val2, desired_scale)


def process_single_image(client: genai.Client, base64_image: str, description: str) -> str:
    if base64_image.startswith("data:image/"):
        base64_data = base64_image.split(",")[1]
        mime_type = base64_image.split(":")[1].split(";")[0]
    else:
        base64_data = base64_image
        mime_type = "image/png"

    try:
        image_bytes = base64.b64decode(base64_data)
    except ValueError:
        logger.error(f"Error decoding base64 image: {base64_image}. Potentially bad output from the agent.")
        return ""
    image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    prompt = "Find a SINGLE bounding box in the following image that best represents the following description:"
    prompt += f"{description}. Return the bounding box as a JSON object with the key 'box_2d'."
    prompt += "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000. Never return masks or code fencing."

    logger.info(f"Gemini bounding box prompt: {prompt}")

    response = client.models.generate_content(model="gemini-2.5-flash-preview-05-20", contents=[prompt, image])

    logger.info(f"Gemini bounding box response: {response.text}")
    json_output = parse_json(response.text)
    box = json.loads(json_output)
    ymin, xmin, ymax, xmax = box.get("box_2d", [0, 0, 1000, 1000])
    pil_image = Image.open(io.BytesIO(image_bytes))
    width, height = pil_image.size
    abs_y1, abs_y2 = scale_and_clamp(ymin, ymax, 1000, height, 20)
    abs_x1, abs_x2 = scale_and_clamp(xmin, xmax, 1000, width, 20)
    cropped_image = pil_image.crop((abs_x1, abs_y1, abs_x2, abs_y2))

    buffered = io.BytesIO()
    cropped_image.save(buffered, format="PNG")
    cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return cropped_image_base64


def crop_images_in_display_objects(display_objects: list):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for display_object in display_objects:
        if display_object["type"] == "image":
            display_object["content"] = process_single_image(
                client, display_object["content"], display_object["caption"]
            )
    return display_objects
