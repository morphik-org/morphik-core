from datetime import datetime
from typing import Any, Dict

from .models import DocumentModel


def _serialize_datetime(obj: Any) -> Any:
    """Recursively serialize datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


def _document_model_to_dict(doc_model: DocumentModel) -> Dict[str, Any]:
    system_metadata = doc_model.system_metadata or {}
    return {
        "external_id": doc_model.external_id,
        "content_type": doc_model.content_type,
        "filename": doc_model.filename,
        "metadata": doc_model.doc_metadata,
        "metadata_types": doc_model.metadata_types or {},
        "storage_info": doc_model.storage_info,
        "system_metadata": doc_model.system_metadata,
        "summary_storage_key": system_metadata.get("summary_storage_key"),
        "summary_version": system_metadata.get("summary_version"),
        "summary_bucket": system_metadata.get("summary_bucket"),
        "summary_updated_at": system_metadata.get("summary_updated_at"),
        "additional_metadata": doc_model.additional_metadata,
        "chunk_ids": doc_model.chunk_ids,
        "folder_name": doc_model.folder_name,
        "folder_path": doc_model.folder_path,
        "folder_id": doc_model.folder_id,
        "app_id": doc_model.app_id,
        "end_user_id": doc_model.end_user_id,
    }


def _folder_row_to_dict(folder_row) -> Dict[str, Any]:
    system_metadata = getattr(folder_row, "system_metadata", None) or {}
    return {
        "id": getattr(folder_row, "id", None),
        "name": getattr(folder_row, "name", None),
        "full_path": getattr(folder_row, "full_path", None),
        "parent_id": getattr(folder_row, "parent_id", None),
        "depth": getattr(folder_row, "depth", None),
        "description": getattr(folder_row, "description", None),
        "document_ids": getattr(folder_row, "document_ids", None),
        "system_metadata": getattr(folder_row, "system_metadata", None),
        "summary_storage_key": system_metadata.get("summary_storage_key"),
        "summary_version": system_metadata.get("summary_version"),
        "summary_bucket": system_metadata.get("summary_bucket"),
        "summary_updated_at": system_metadata.get("summary_updated_at"),
        "app_id": getattr(folder_row, "app_id", None),
        "end_user_id": getattr(folder_row, "end_user_id", None),
    }
