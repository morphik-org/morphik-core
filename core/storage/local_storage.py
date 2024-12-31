import base64
from pathlib import Path
from typing import Tuple, Optional, BinaryIO
from .base_storage import BaseStorage


class LocalStorage(BaseStorage):
    def __init__(self, storage_path: str):
        """Initialize local storage with a base path."""
        self.storage_path = Path(storage_path)
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str, original_filename: Optional[str] = None) -> Path:
        """Get the full file path, optionally using original filename.
        Creates a directory with the key (UUID) and stores the file with original name inside."""
        base_path = self.storage_path / key
        if original_filename:
            return base_path / original_filename
        return base_path

    async def upload_file(
        self, file: BinaryIO, key: str, content_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Upload a file object to local storage."""
        original_filename = getattr(file, "filename", None)
        file_path = self._get_file_path(key, original_filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(file.read())

        return str(self.storage_path), key

    async def download_file(self, bucket: str, key: str) -> BinaryIO:
        """Download a file from local storage."""
        # Try to find the file in the key directory
        key_dir = self.storage_path / key
        if key_dir.is_dir():
            # Get the first file in the directory
            files = list(key_dir.iterdir())
            if files:
                return open(files[0], "rb")

        # Fallback to direct key path
        file_path = self.storage_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return open(file_path, "rb")

    async def upload_from_base64(
        self, base64_content: str, key: str, content_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Upload base64 encoded content to local storage."""
        # Decode base64 content
        file_content = base64.b64decode(base64_content)

        # Create file path (using just key for base64 uploads)
        file_path = self._get_file_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content to file
        with open(file_path, "wb") as f:
            f.write(file_content)

        return str(self.storage_path), key

    async def get_download_url(self, bucket: str, key: str) -> str:
        """Get local file path as URL."""
        # Try to find the file in the key directory
        key_dir = self.storage_path / key
        if key_dir.is_dir():
            # Get the first file in the directory
            files = list(key_dir.iterdir())
            if files:
                return f"file://{files[0].absolute()}"

        # Fallback to direct key path
        file_path = self.storage_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return f"file://{file_path.absolute()}"

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete a file from local storage."""
        key_dir = self.storage_path / key
        try:
            if key_dir.is_dir():
                # Remove all files in the directory
                for file in key_dir.iterdir():
                    file.unlink()
                key_dir.rmdir()
                return True
            elif key_dir.exists():
                key_dir.unlink()
                return True
            return True
        except Exception:
            return False
