import asyncio
import base64
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

    async def download_file(self, bucket: str, key: str, **kwargs) -> bytes:
        """Download a file from local storage without blocking the event loop."""
        # Construct full key including bucket, consistent with upload_from_base64
        full_key = f"{bucket}/{key}" if (bucket and bucket != "storage") else key
        file_path = self.storage_path / full_key

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Use a thread to perform blocking IO
        return await asyncio.to_thread(file_path.read_bytes)

    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        base64_content = content
        """Upload base64 encoded content to local storage."""
        # Decode base64 content
        file_content = base64.b64decode(base64_content)

        key = f"{bucket}/{key}" if (bucket and bucket != "storage") else key
        # Create file path
        file_path = self.storage_path / key

        # Write content to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.unlink(missing_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        return str(self.storage_path), key

    async def get_download_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        """Get local file path as URL."""
        logger.debug(f"Storage got params: bucket: {bucket}, key: {key}, expires in: {expires_in}")
        # Construct full key including bucket, consistent with other methods
        full_key = f"{bucket}/{key}" if (bucket and bucket != "storage") else key
        file_path = self.storage_path / full_key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return f"file://{file_path.absolute()}"

    async def upload_file(
        self,
        file: Union[str, bytes, BinaryIO],
        key: str,
        content_type: Optional[str] = None,
        bucket: str = "",
    ) -> Tuple[str, str]:
        """Upload a file to local storage."""
        # Handle different input types
        if isinstance(file, str):
            with open(file, "rb") as f:
                file_content = f.read()
        elif isinstance(file, bytes):
            file_content = file
        else:
            try:
                file.seek(0)
            except Exception:  # noqa: BLE001
                pass
            file_content = file.read()

        key = f"{bucket}/{key}" if (bucket and bucket != "storage") else key
        file_path = self.storage_path / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.unlink(missing_ok=True)
        with open(file_path, "wb") as destination:
            destination.write(file_content)

        return str(self.storage_path), key

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete a file from local storage."""
        # Construct full key including bucket, consistent with other methods
        full_key = f"{bucket}/{key}" if (bucket and bucket != "storage") else key
        file_path = self.storage_path / full_key
        if file_path.exists():
            file_path.unlink()
        return True
