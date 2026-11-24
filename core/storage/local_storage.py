import asyncio
import base64
import os
from logging import getLogger
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

from .base_storage import BaseStorage

logger = getLogger(__name__)


class LocalStorage(BaseStorage):
    def __init__(self, storage_path: str):
        """Initialize local storage with a base path."""
        self.storage_path = Path(storage_path)
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._root = os.path.abspath(storage_path)

    def _object_path(self, bucket: str, key: str) -> Tuple[str, Path]:
        """Return the full key and file path of an object, refusing any key that leaves the storage directory."""
        full_key = f"{bucket}/{key}" if (bucket and bucket != "storage") else key
        path = os.path.normpath(os.path.join(self._root, full_key))
        if not path.startswith(self._root + os.sep):
            raise ValueError(f"Storage key is outside the storage directory: {full_key!r}")
        return full_key, Path(path)

    async def download_file(self, bucket: str, key: str, **kwargs) -> bytes:
        """Download a file from local storage without blocking the event loop."""
        _, file_path = self._object_path(bucket, key)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Use a thread to perform blocking IO
        return await asyncio.to_thread(file_path.read_bytes)

    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        """Upload base64 encoded content (or data URI) to local storage."""
        base64_payload = content
        if isinstance(content, str) and content.startswith("data:"):
            try:
                _, base64_part = content.split(",", 1)
                base64_payload = base64_part
            except Exception:
                base64_payload = content
        # Decode base64 content
        file_content = base64.b64decode(base64_payload)

        key, file_path = self._object_path(bucket, key)

        # Write content to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.unlink(missing_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        return str(self.storage_path), key

    async def get_download_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        """Get local file path as URL."""
        logger.debug(f"Storage got params: bucket: {bucket}, key: {key}, expires in: {expires_in}")
        _, file_path = self._object_path(bucket, key)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return f"file://{file_path}"

    async def upload_file(
        self,
        file: Union[str, bytes, BinaryIO],
        key: str,
        content_type: Optional[str] = None,
        bucket: str = "",
    ) -> Tuple[str, str]:
        """Upload a file to local storage."""
        # A string is the content itself, as in S3Storage, never a path to read.
        if isinstance(file, str):
            file_content = file.encode()
        elif isinstance(file, bytes):
            file_content = file
        else:
            try:
                file.seek(0)
            except Exception:  # noqa: BLE001
                pass
            file_content = file.read()

        key, file_path = self._object_path(bucket, key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.unlink(missing_ok=True)
        with open(file_path, "wb") as destination:
            destination.write(file_content)

        return str(self.storage_path), key

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete a file from local storage."""
        _, file_path = self._object_path(bucket, key)
        if file_path.exists():
            file_path.unlink()
        return True

    async def get_object_size(self, bucket: str, key: str) -> int:
        """Return object size in bytes from local storage."""
        _, file_path = self._object_path(bucket, key)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return await asyncio.to_thread(lambda: file_path.stat().st_size)
