import json
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException, Request


def _derive_page_count(document_dict: Dict[str, Any]) -> Optional[int]:
    """Compute page count from stored metadata or chunk identifiers."""
    system_metadata = document_dict.get("system_metadata") or {}
    if isinstance(system_metadata, dict):
        page_count_raw = system_metadata.get("page_count")
        try:
            if page_count_raw is not None:
                page_count_int = int(page_count_raw)
                if page_count_int >= 0:
                    return page_count_int
        except (TypeError, ValueError):
            # Ignore malformed values and fall back to chunk count
            pass

    chunk_ids = document_dict.get("chunk_ids")
    if isinstance(chunk_ids, list):
        return len(chunk_ids)

    return None


def _add_derived_fields(document_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Attach derived fields (e.g., page_count) without mutating the input."""
    enriched = dict(document_dict)
    page_count = _derive_page_count(document_dict)
    if page_count is not None:
        enriched["page_count"] = page_count
    return enriched


def project_document_fields(document_dict: Dict[str, Any], fields: Optional[List[str]]) -> Dict[str, Any]:
    """
    Project document data to a subset of fields, always including the external_id for reference.
    """
    document_dict = _add_derived_fields(document_dict)

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


def parse_bool(value: Optional[Union[str, bool]]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def parse_json_value(value: Optional[str], field_name: str, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON: {exc}") from exc


def parse_json_dict(value: Optional[str], field_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parsed = parse_json_value(value, field_name, default)
    if parsed is None:
        return {} if default is None else default
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return parsed


def enforce_no_user_mutable_fields(
    ingestion_service,
    metadata: Optional[Dict[str, Any]],
    metadata_types: Optional[Dict[str, Any]],
    context: str,
    request_model: Optional[Any] = None,
) -> None:
    extra_fields = (
        getattr(request_model, "model_extra", {}) if request_model and hasattr(request_model, "model_extra") else {}
    )
    ingestion_service._enforce_no_user_mutable_fields(
        metadata,
        extra_fields,
        metadata_types,
        context=context,
    )


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
