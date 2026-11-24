"""LocalStorage keeps every object inside its storage directory."""

import asyncio
import base64

import pytest

from core.storage.local_storage import LocalStorage


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(str(tmp_path / "store"))


def test_upload_download_size_and_delete_round_trip(storage):
    bucket, key = asyncio.run(storage.upload_file(b"hello", "docs/a.txt", bucket="app-1"))

    assert key == "app-1/docs/a.txt"
    assert asyncio.run(storage.download_file("app-1", "docs/a.txt")) == b"hello"
    assert asyncio.run(storage.download_file(bucket, key)) == b"hello"
    assert asyncio.run(storage.get_object_size("app-1", "docs/a.txt")) == 5
    assert asyncio.run(storage.get_download_url("app-1", "docs/a.txt")).endswith("/store/app-1/docs/a.txt")
    assert asyncio.run(storage.delete_file("app-1", "docs/a.txt")) is True
    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.download_file("app-1", "docs/a.txt"))


def test_base64_upload_round_trip(storage):
    payload = base64.b64encode(b"image-bytes").decode()

    _, key = asyncio.run(storage.upload_from_base64(f"data:image/png;base64,{payload}", "img.png", bucket="app-1"))

    assert asyncio.run(storage.download_file("", key)) == b"image-bytes"


def test_a_string_is_stored_as_content_not_read_as_a_path(storage, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not copy me")

    asyncio.run(storage.upload_file(str(secret), "note.txt"))

    assert asyncio.run(storage.download_file("", "note.txt")) == str(secret).encode()


@pytest.mark.parametrize(
    "bucket, key",
    [
        ("", "../outside.txt"),
        ("", "a/../../outside.txt"),
        ("..", "outside.txt"),
        ("app-1", "../../outside.txt"),
        ("", "/etc/hostname"),
        ("/etc", "hostname"),
        ("", ""),
    ],
)
def test_keys_that_leave_the_storage_directory_are_refused(storage, tmp_path, bucket, key):
    (tmp_path / "outside.txt").write_text("outside")

    for call in (
        storage.download_file(bucket, key),
        storage.get_download_url(bucket, key),
        storage.get_object_size(bucket, key),
        storage.delete_file(bucket, key),
        storage.upload_file(b"x", key, bucket=bucket),
        storage.upload_from_base64(base64.b64encode(b"x").decode(), key, bucket=bucket),
    ):
        with pytest.raises(ValueError, match="outside the storage directory"):
            asyncio.run(call)

    assert (tmp_path / "outside.txt").read_text() == "outside"


def test_a_sibling_directory_sharing_the_name_prefix_is_outside(storage, tmp_path):
    (tmp_path / "store-other").mkdir()

    with pytest.raises(ValueError, match="outside the storage directory"):
        asyncio.run(storage.upload_file(b"x", "../store-other/a.txt"))
