import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import Request


def project_document_fields(document_dict: Dict[str, Any], fields: Optional[List[str]]) -> Dict[str, Any]:
    """
    Project document data to a subset of fields, always including the external_id for reference.
    """
    if not fields:
        return document_dict

    projected: Dict[str, Any] = {}
    normalized_fields: List[str] = [field.strip() for field in fields if field and field.strip()]
    include_external_id = "external_id" in normalized_fields

    for field_path in normalized_fields:
        value: Any = document_dict
        parts = field_path.split(".")
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                break
        else:
            current: Dict[str, Any] = projected
            for part in parts[:-1]:
                next_value = current.get(part)
                if not isinstance(next_value, dict):
                    next_value = {}
                    current[part] = next_value
                current = next_value
            current[parts[-1]] = value

    if not include_external_id and "external_id" in document_dict:
        projected["external_id"] = document_dict["external_id"]

    return projected


async def warn_if_legacy_rules(request: Request, route: str, logger: logging.Logger) -> None:
    """Inspect multipart form data for legacy ``rules`` payloads and emit warnings."""
    try:
        form_data = await request.form()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to inspect legacy rules form field for %s: %s", route, exc)
        return

    legacy_rules_raw = form_data.get("rules")
    if legacy_rules_raw is None:
        return

    try:
        parsed_rules = json.loads(legacy_rules_raw)
    except json.JSONDecodeError:
        logger.warning("Legacy 'rules' payload supplied to %s but was invalid JSON; ignoring.", route)
        return

    if parsed_rules:
        logger.warning("Legacy 'rules' payload supplied to %s; ignoring.", route)
