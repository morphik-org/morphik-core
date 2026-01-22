import json
from pathlib import Path

import pytest
from morphik._shared import (
    build_create_app_payload,
    build_document_by_filename_params,
    build_list_apps_params,
    build_logs_params,
    build_rename_app_params,
    build_requeue_payload,
    build_rotate_app_params,
    collect_directory_files,
    merge_folders,
    normalize_additional_folders,
)
from morphik.models import RequeueIngestionJob


def test_merge_folders_variants():
    assert merge_folders(None, None) is None
    assert merge_folders("base", None) == "base"
    assert merge_folders("base", ["a", "b"]) == ["base", "a", "b"]
    assert merge_folders(["a", "b"], ["c"]) == ["a", "b", "c"]


def test_collect_directory_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.md").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")

    files = collect_directory_files(tmp_path, recursive=False, pattern="*.txt")
    assert sorted([f.name for f in files]) == ["a.txt"]

    files_recursive = collect_directory_files(tmp_path, recursive=True, pattern="*.txt")
    assert sorted([f.name for f in files_recursive]) == ["a.txt", "c.txt"]


def test_build_list_apps_params_serializes_filters():
    params = build_list_apps_params(
        org_id="org-1",
        user_id="user-1",
        app_id_filter={"$in": ["a", "b"]},
        app_name_filter='{"$eq": "demo"}',
        limit=600,
        offset=-1,
    )
    assert params["limit"] == 500
    assert params["offset"] == 0
    assert params["org_id"] == "org-1"
    assert params["user_id"] == "user-1"
    assert json.loads(params["app_id_filter"]) == {"$in": ["a", "b"]}
    assert params["app_name_filter"] == '{"$eq": "demo"}'


def test_build_rename_app_params_validation():
    with pytest.raises(ValueError):
        build_rename_app_params(new_name="new", app_id=None, app_name=None)
    params = build_rename_app_params(new_name="new", app_id="app-1", app_name=None)
    assert params == {"new_name": "new", "app_id": "app-1"}


def test_build_rotate_app_params_validation():
    with pytest.raises(ValueError):
        build_rotate_app_params(app_id=None, app_name=None, expiry_days=None)
    params = build_rotate_app_params(app_id=None, app_name="name", expiry_days=10)
    assert params == {"app_name": "name", "expiry_days": 10}


def test_build_create_app_payload():
    payload = build_create_app_payload(
        name="app",
        app_id="id",
        user_id=None,
        expiry_days=7,
        org_id="org",
        created_by_user_id="creator",
    )
    assert payload == {
        "name": "app",
        "app_id": "id",
        "expiry_days": 7,
        "org_id": "org",
        "created_by_user_id": "creator",
    }


def test_build_requeue_payload_with_jobs():
    job = RequeueIngestionJob(external_id="doc-1", use_colpali=True)
    payload = build_requeue_payload(
        jobs=[job, {"external_id": "doc-2"}],
        include_all=False,
        statuses=["failed"],
        limit=5,
    )
    assert payload["jobs"] == [{"external_id": "doc-1", "use_colpali": True}, {"external_id": "doc-2"}]
    assert payload["statuses"] == ["failed"]
    assert payload["limit"] == 5


def test_build_requeue_payload_requires_jobs_or_include_all():
    with pytest.raises(ValueError):
        build_requeue_payload(jobs=None, include_all=False, statuses=None, limit=None)


def test_build_requeue_payload_rejects_empty_iterable():
    def empty_jobs():
        if False:
            yield {"external_id": "doc-1"}

    with pytest.raises(ValueError, match="jobs or include_all must be provided"):
        build_requeue_payload(jobs=empty_jobs(), include_all=False, statuses=None, limit=None)


def test_build_logs_params_clamps():
    params = build_logs_params(limit=0, hours=1000.0, op_type="query", status=None)
    assert params["limit"] == 1
    assert params["hours"] == 168.0
    assert params["op_type"] == "query"


def test_build_document_by_filename_params():
    params = build_document_by_filename_params(folder_name="/f", folder_depth=2, end_user_id=None)
    assert params == {"folder_name": "/f", "folder_depth": 2}


def test_normalize_additional_folders_alias():
    assert normalize_additional_folders(None, None) is None
    assert normalize_additional_folders(["a"], None) == ["a"]
    assert normalize_additional_folders(None, "b") == ["b"]
    assert normalize_additional_folders(["a"], "b") == ["a", "b"]
    assert normalize_additional_folders(["a"], ["b", "c"]) == ["a", "b", "c"]
