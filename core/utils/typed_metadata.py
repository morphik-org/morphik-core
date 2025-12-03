"""Utilities for normalizing metadata values and preserving type hints."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple


class TypedMetadataError(ValueError):
    """Raised when metadata values cannot be coerced to the declared type."""


_TYPE_ALIASES = {
    "string": "string",
    "str": "string",
    "text": "string",
    "number": "number",
    "numeric": "number",
    "float": "number",
    "double": "number",
    "integer": "number",
    "int": "number",
    "decimal": "decimal",
    "bool": "boolean",
    "boolean": "boolean",
    "datetime": "datetime",
    "timestamp": "datetime",
    "date": "date",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
    "map": "object",
    "null": "null",
}

SCALAR_METADATA_TYPES = {"string", "number", "decimal", "boolean", "datetime", "date", "null"}
ALL_METADATA_TYPES = set(_TYPE_ALIASES.values()).union({"array", "object"})


def canonicalize_type_name(type_name: str, field: Optional[str] = None) -> str:
    """Normalize user-provided type labels to canonical metadata type names."""
    canonical = _TYPE_ALIASES.get(type_name.lower())
    if not canonical:
        suffix = f" for field '{field}'" if field else ""
        raise TypedMetadataError(f"Unsupported metadata type '{type_name}'{suffix}.")
    return canonical


def normalize_metadata(
    metadata: Dict[str, Any],
    type_hints: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return JSON-serializable metadata plus a parallel type map.

    Args:
        metadata: Original metadata dictionary.
        type_hints: Optional explicit metadata types keyed by top-level field.

    Returns:
        Tuple of (normalized_metadata, metadata_types).
    """

    normalized: Dict[str, Any] = {}
    metadata_types: Dict[str, str] = {}
    hints = type_hints or {}

    for key, value in metadata.items():
        declared_type = hints.get(key)
        normalized_value, field_type = _normalize_value(value, declared_type, key)
        normalized[key] = normalized_value
        if field_type:
            metadata_types[key] = field_type

    return normalized, metadata_types


def merge_metadata(
    existing: Optional[Dict[str, Any]],
    existing_types: Optional[Dict[str, str]],
    updates: Dict[str, Any],
    update_type_hints: Optional[Dict[str, str]] = None,
    *,
    external_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Merge normalized metadata/type maps, coercing updates before overlaying."""

    normalized_updates, normalized_types = normalize_metadata(updates, update_type_hints)

    merged_metadata = dict(existing or {})
    merged_metadata.update(normalized_updates)

    merged_types = dict(existing_types or {})
    merged_types.update(normalized_types)

    if external_id is not None:
        merged_metadata["external_id"] = external_id
        merged_types.setdefault("external_id", "string")

    return merged_metadata, merged_types


def _normalize_value(value: Any, declared_type: Optional[str], field: str) -> Tuple[Any, Optional[str]]:
    """Normalize a single metadata value, honoring any explicit type hint."""
    if value is None:
        # Preserve true nulls regardless of declared type so callers can clear fields.
        return None, "null"

    if declared_type:
        canonical_type = _canonicalize_type_name(declared_type, field)
        return _coerce_to_type(value, canonical_type, field), canonical_type

    inferred_type = _infer_type(value)
    return _coerce_to_type(value, inferred_type, field), inferred_type


def _canonicalize_type_name(type_name: str, field: str) -> str:
    return canonicalize_type_name(type_name, field=field)


def _looks_like_iso_datetime(value: str) -> bool:
    """Check if a string looks like an ISO 8601 datetime.

    Matches patterns like:
    - 2024-01-15T10:30:00
    - 2024-01-15T10:30:00Z
    - 2024-01-15T10:30:00+00:00
    - 2024-01-15T10:30:00.123456
    """
    if not isinstance(value, str) or len(value) < 19:
        return False
    # Must have date separator and time separator in right positions
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        # Check for T separator or space between date and time
        if len(value) >= 19 and (value[10] == "T" or value[10] == " "):
            # Validate it can actually be parsed
            try:
                text = value.strip()
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                datetime.fromisoformat(text)
                return True
            except ValueError:
                return False
    return False


def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, Decimal):
        return "decimal"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, str):
        # Check if string looks like ISO 8601 datetime
        if _looks_like_iso_datetime(value):
            return "datetime"
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _coerce_to_type(value: Any, target_type: str, field: str) -> Any:
    if target_type == "string":
        if isinstance(value, str):
            return value
        return str(value)
    if target_type == "number":
        return _coerce_number(value, field)
    if target_type == "decimal":
        return _coerce_decimal(value, field)
    if target_type == "boolean":
        return _coerce_boolean(value, field)
    if target_type == "datetime":
        return _coerce_datetime(value, field)
    if target_type == "date":
        return _coerce_date(value, field)
    if target_type == "array":
        if isinstance(value, list):
            return [_sanitize_nested(v, field) for v in value]
        raise TypedMetadataError(f"Metadata field '{field}' expects an array.")
    if target_type == "object":
        if isinstance(value, dict):
            return {k: _sanitize_nested(v, field) for k, v in value.items()}
        raise TypedMetadataError(f"Metadata field '{field}' expects an object.")
    if target_type == "null":
        return None
    raise TypedMetadataError(f"Cannot coerce field '{field}' to unsupported metadata type '{target_type}'.")


def _coerce_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or value is None:
        raise TypedMetadataError(f"Metadata field '{field}' cannot coerce boolean/null to number.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise TypedMetadataError(f"Metadata field '{field}' cannot store NaN or infinite values.")
        return value

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypedMetadataError(f"Metadata field '{field}' cannot coerce empty string to number.")
        try:
            if all(ch.isdigit() or ch in {"+", "-", "_"} for ch in text.replace("_", "")) and "." not in text:
                return int(text.replace("_", ""))
            return float(text)
        except ValueError as exc:  # noqa: BLE001
            raise TypedMetadataError(f"Metadata field '{field}' expects a numeric value.") from exc

    raise TypedMetadataError(f"Metadata field '{field}' expects a numeric value.")


def _coerce_decimal(value: Any, field: str) -> str:
    try:
        if isinstance(value, Decimal):
            decimal_value = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            decimal_value = Decimal(str(value))
        elif isinstance(value, str):
            decimal_value = Decimal(value.strip())
        else:
            raise TypedMetadataError(f"Metadata field '{field}' expects a decimal-compatible value.")
    except (InvalidOperation, ValueError) as exc:  # noqa: BLE001
        raise TypedMetadataError(f"Metadata field '{field}' expects a decimal-compatible value.") from exc

    decimal_text = format(decimal_value.normalize(), "f")
    if "." in decimal_text:
        decimal_text = decimal_text.rstrip("0").rstrip(".")
    return decimal_text or "0"


def _coerce_boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise TypedMetadataError(f"Metadata field '{field}' expects 'true' or 'false'.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    raise TypedMetadataError(f"Metadata field '{field}' expects a boolean value.")


def _coerce_datetime(value: Any, field: str) -> str:
    """Convert value to ISO 8601 datetime string, preserving timezone presence.

    If the input has no timezone, the output will have no timezone.
    If the input has a timezone, the output will preserve it.
    """
    dt = _parse_datetime_like(value, field)
    return dt.isoformat()


def _coerce_date(value: Any, field: str) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=UTC).date().isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypedMetadataError(f"Metadata field '{field}' expects a date value.")
        try:
            return date.fromisoformat(text.split("T")[0]).isoformat()
        except ValueError as exc:  # noqa: BLE001
            raise TypedMetadataError(f"Metadata field '{field}' expects an ISO8601 date.") from exc
    raise TypedMetadataError(f"Metadata field '{field}' expects a date value.")


def _parse_datetime_like(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        # date objects have no timezone concept, return naive datetime
        return datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # UNIX timestamps are inherently UTC
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypedMetadataError(f"Metadata field '{field}' expects a datetime value.")
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text)
        except ValueError as exc:  # noqa: BLE001
            raise TypedMetadataError(f"Metadata field '{field}' expects an ISO8601 datetime.") from exc
    raise TypedMetadataError(f"Metadata field '{field}' expects a datetime value.")


def _sanitize_nested(value: Any, field: str) -> Any:
    """Recursively sanitize nested metadata so JSON serialization never fails."""
    if isinstance(value, dict):
        return {k: _sanitize_nested(v, field) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nested(v, field) for v in value]
    if isinstance(value, datetime):
        return _coerce_datetime(value, field)
    if isinstance(value, date):
        return _coerce_date(value, field)
    if isinstance(value, Decimal):
        return _coerce_decimal(value, field)
    return value
