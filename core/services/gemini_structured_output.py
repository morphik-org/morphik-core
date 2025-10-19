import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

# Cache settings (pydantic BaseSettings is safe to reuse)
settings = get_settings()

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise metadata extraction assistant. "
    "Only return JSON with the requested fields and use null for missing values."
)
TYPE_MAP = {
    "string": "STRING",
    "str": "STRING",
    "text": "STRING",
    "number": "NUMBER",
    "float": "NUMBER",
    "double": "NUMBER",
    "integer": "NUMBER",
    "int": "NUMBER",
    "boolean": "BOOLEAN",
    "bool": "BOOLEAN",
    "array": "ARRAY",
    "list": "ARRAY",
    "object": "OBJECT",
}


class GeminiContentError(RuntimeError):
    """Raised when Gemini content generation fails."""


# Backwards compatibility for previous name
GeminiStructuredOutputError = GeminiContentError


@dataclass(slots=True)
class GeminiGenerationResult:
    """Container for Gemini generation output."""

    text_output: str
    structured_output: Optional[Dict[str, Any]] = None


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return default


def _normalize_type(value: Optional[str]) -> str:
    if not value:
        return "STRING"
    lowered = value.lower()
    return TYPE_MAP.get(lowered, value.upper())


def _build_field_from_str(value: str) -> Dict[str, Any]:
    lowered = value.lower()
    if lowered == "date":
        return {"type": "STRING", "format": "date-time", "nullable": True}
    return {"type": _normalize_type(value), "nullable": True}


def _build_field_definition(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        field: Dict[str, Any] = {}
        raw_type = raw.get("type")
        field_type = _normalize_type(raw_type) if isinstance(raw_type, str) else "STRING"

        field["type"] = field_type
        field["nullable"] = _coerce_bool(raw.get("nullable"), True)

        if "description" in raw and raw["description"] is not None:
            field["description"] = str(raw["description"])

        if field_type == "STRING":
            fmt = raw.get("format")
            if isinstance(fmt, str) and fmt:
                field["format"] = fmt
            elif raw_type and str(raw_type).lower() == "date":
                field["format"] = "date-time"

        if "enum" in raw and raw["enum"] is not None:
            field["enum"] = raw["enum"]

        if field_type == "ARRAY" and raw.get("items") is not None:
            field["items"] = raw["items"]

        if field_type == "OBJECT":
            props = raw.get("properties")
            if isinstance(props, dict):
                field["properties"] = props
            required = raw.get("required")
            if isinstance(required, list):
                field["required"] = required

        return field

    if isinstance(raw, str):
        return _build_field_from_str(raw)

    return {"type": "STRING", "nullable": True}


def build_gemini_schema(schema_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise user-provided schema definitions into Gemini's responseSchema format.

    Supports two formats:
      • A full Gemini-compatible schema dict (with "type"/"properties" keys) – returned unchanged.
      • A simple mapping of field name -> {type, description, nullable, ...} or string shorthand.
    """
    if not isinstance(schema_input, dict):
        raise ValueError("Schema must be a JSON object")

    schema_type_value = schema_input.get("type")
    schema_type = schema_type_value.upper() if isinstance(schema_type_value, str) else None
    if schema_type:
        if schema_type == "OBJECT":
            if "properties" not in schema_input:
                raise ValueError("Object schemas must include a 'properties' section.")
            return schema_input
        if schema_type == "ARRAY":
            if "items" not in schema_input:
                raise ValueError("Array schemas must include an 'items' definition.")
            return schema_input
        return schema_input

    properties: Dict[str, Dict[str, Any]] = {}
    required: list[str] = []
    ordering: list[str] = []

    for key, raw in schema_input.items():
        properties[key] = _build_field_definition(raw)
        required.append(key)
        ordering.append(key)

    return {
        "type": "OBJECT",
        "properties": properties,
        "required": required,
        "propertyOrdering": ordering,
    }


def normalize_model_name(model: Optional[str]) -> str:
    selected = model or getattr(settings, "GEMINI_METADATA_MODEL", None) or "gemini-2.5-flash"
    return selected if selected.startswith("models/") else f"models/{selected}"


def _extract_text_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    first = candidates[0]
    content = first.get("content") if isinstance(first, dict) else None
    if not isinstance(content, dict):
        return None

    parts = content.get("parts")
    if not isinstance(parts, list):
        return None

    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    joined = "".join(texts).strip()
    return joined or None


async def generate_gemini_content(
    *,
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    document_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    thinking_budget: Optional[int] = None,
    timeout_seconds: float = 60,
) -> GeminiGenerationResult:
    """
    Invoke Gemini to generate content, optionally enforcing structured output via schema.
    """
    if not prompt or not prompt.strip():
        raise GeminiContentError("Prompt is required for Gemini generation")

    api_key = api_key or getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        raise GeminiContentError("Gemini API key is not configured")

    api_base_url = (api_base_url or getattr(settings, "GEMINI_API_BASE_URL", None) or "").rstrip("/")
    if not api_base_url:
        raise GeminiContentError("Gemini API base URL is not configured")

    response_schema: Optional[Dict[str, Any]] = None
    if schema:
        try:
            response_schema = build_gemini_schema(schema)
        except ValueError as exc:
            raise GeminiContentError(str(exc)) from exc
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

    parts: list[Dict[str, Any]] = [{"text": prompt}]
    if document_bytes:
        encoded = base64.b64encode(document_bytes).decode("utf-8")
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type or "application/pdf",
                    "data": encoded,
                }
            }
        )

    generation_config: Dict[str, Any] = {"temperature": temperature}
    if response_schema:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
    if thinking_budget is not None:
        generation_config.setdefault("thinkingConfig", {})["thinkingBudget"] = thinking_budget

    body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": generation_config,
    }

    if system_prompt:
        body["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system_prompt}],
        }

    model_name = normalize_model_name(model)
    url = f"{api_base_url}/v1beta/{model_name}:generateContent"

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, params={"key": api_key}, json=body)

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except json.JSONDecodeError:
            error_payload = response.text
        logger.error("Gemini request failed: %s", error_payload)
        raise GeminiContentError(f"Gemini request failed with status {response.status_code}: {response.text}")

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise GeminiContentError(f"Failed to parse Gemini response: {exc}") from exc

    text = _extract_text_from_payload(payload)
    if text is None:
        logger.error("Gemini response missing textual content: %s", payload)
        raise GeminiContentError("Gemini response did not include textual output")

    structured_output: Optional[Dict[str, Any]] = None
    if response_schema:
        try:
            structured_output = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Gemini response was not valid JSON: %s", text)
            raise GeminiContentError(f"Failed to parse Gemini JSON output: {exc}") from exc

    return GeminiGenerationResult(text_output=text, structured_output=structured_output)


async def extract_structured_metadata(
    *,
    prompt: str,
    schema: Dict[str, Any],
    document_bytes: Optional[bytes],
    mime_type: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    timeout_seconds: float = 60,
) -> Dict[str, Any]:
    """Backwards-compatible helper that enforces structured output."""
    result = await generate_gemini_content(
        prompt=prompt,
        schema=schema,
        document_bytes=document_bytes,
        mime_type=mime_type,
        model=model,
        api_key=api_key,
        api_base_url=api_base_url,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        timeout_seconds=timeout_seconds,
    )

    if result.structured_output is None:
        raise GeminiContentError("Gemini did not return structured output")
    return result.structured_output
