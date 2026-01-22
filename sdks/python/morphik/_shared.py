from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel

MAX_LIMIT = 500
MIN_LOG_HOURS = 0.1
MAX_LOG_HOURS = 168.0


def merge_folders(
    base: Optional[Union[str, List[str]]],
    additional: Optional[List[str]],
) -> Optional[Union[str, List[str]]]:
    if not additional:
        return base
    if base:
        if isinstance(base, list):
            return base + additional
        return [base] + additional
    return additional


def collect_directory_files(directory: Union[str, Path], recursive: bool, pattern: str) -> List[Path]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    files = list(dir_path.rglob(pattern) if recursive else dir_path.glob(pattern))
    return [f for f in files if f.is_file()]


def normalize_limit_offset(limit: int, offset: int) -> Dict[str, int]:
    return {
        "limit": max(1, min(limit, MAX_LIMIT)),
        "offset": max(0, offset),
    }


def normalize_filter_param(value: Optional[Union[str, Dict[str, Any], List[Any]]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def build_list_apps_params(
    *,
    org_id: Optional[str],
    user_id: Optional[str],
    app_id_filter: Optional[Union[str, Dict[str, Any], List[Any]]],
    app_name_filter: Optional[Union[str, Dict[str, Any], List[Any]]],
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    params: Dict[str, Any] = normalize_limit_offset(limit, offset)
    if org_id:
        params["org_id"] = org_id
    if user_id:
        params["user_id"] = user_id
    if app_id_filter is not None:
        params["app_id_filter"] = normalize_filter_param(app_id_filter)
    if app_name_filter is not None:
        params["app_name_filter"] = normalize_filter_param(app_name_filter)
    return params


def build_rename_app_params(*, new_name: str, app_id: Optional[str], app_name: Optional[str]) -> Dict[str, Any]:
    if not app_id and not app_name:
        raise ValueError("app_id or app_name is required to rename an app")
    params: Dict[str, Any] = {"new_name": new_name}
    if app_id:
        params["app_id"] = app_id
    if app_name:
        params["app_name"] = app_name
    return params


def build_rotate_app_params(
    *, app_id: Optional[str], app_name: Optional[str], expiry_days: Optional[int]
) -> Dict[str, Any]:
    if not app_id and not app_name:
        raise ValueError("app_id or app_name is required to rotate an app token")
    params: Dict[str, Any] = {}
    if app_id:
        params["app_id"] = app_id
    if app_name:
        params["app_name"] = app_name
    if expiry_days is not None:
        params["expiry_days"] = expiry_days
    return params


def build_create_app_payload(*, name: str) -> Dict[str, Any]:
    return {"name": name}


def build_requeue_payload(
    *,
    jobs: Optional[Iterable[Union[BaseModel, Dict[str, Any]]]],
    include_all: bool,
    statuses: Optional[List[str]],
    limit: Optional[int],
) -> Dict[str, Any]:
    jobs_list = list(jobs) if jobs is not None else None
    if not include_all and not jobs_list:
        raise ValueError("jobs or include_all must be provided for requeue")
    payload: Dict[str, Any] = {}
    if jobs_list:
        payload["jobs"] = [job.model_dump() if isinstance(job, BaseModel) else job for job in jobs_list]
    if include_all:
        payload["include_all"] = True
    if statuses is not None:
        payload["statuses"] = statuses
    if limit is not None:
        payload["limit"] = limit
    return payload


def build_logs_params(*, limit: int, hours: float, op_type: Optional[str], status: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "limit": max(1, min(limit, MAX_LIMIT)),
        "hours": max(MIN_LOG_HOURS, min(hours, MAX_LOG_HOURS)),
    }
    if op_type is not None:
        params["op_type"] = op_type
    if status is not None:
        params["status"] = status
    return params


def build_document_by_filename_params(
    *, folder_name: Optional[Union[str, List[str]]], folder_depth: Optional[int], end_user_id: Optional[str]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if folder_name is not None:
        params["folder_name"] = folder_name
    if folder_depth is not None:
        params["folder_depth"] = folder_depth
    if end_user_id is not None:
        params["end_user_id"] = end_user_id
    return params


def normalize_additional_folders(
    additional_folders: Optional[List[str]],
    folder_name: Optional[Union[str, List[str]]],
) -> Optional[List[str]]:
    if folder_name is None:
        return additional_folders
    if isinstance(folder_name, str):
        folder_list = [folder_name]
    else:
        folder_list = list(folder_name)
    if additional_folders:
        return list(additional_folders) + folder_list
    return folder_list
