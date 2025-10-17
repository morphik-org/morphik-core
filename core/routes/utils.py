from typing import Any, Dict, List, Optional


def project_document_fields(document_dict: Dict[str, Any], fields: Optional[List[str]]) -> Dict[str, Any]:
    """
    Project document data to a subset of fields, always including the external_id for reference.

    Args:
        document_dict: Source document dictionary
        fields: Optional list of dot-notation field paths to include

    Returns:
        Dictionary containing only the requested fields (plus external_id if available)
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
