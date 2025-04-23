import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_storage import BaseStorage

logger = logging.getLogger(__name__)


class LocalStorage(BaseStorage):
    def __init__(self, storage_path: str):
        """Initialize local storage with a base path."""
        self.storage_path = Path(storage_path)
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def download_file(self, bucket: str, key: str) -> bytes:
        """Download a file from local storage."""
        file_path = self.storage_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return f.read()

    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        base64_content = content
        """Upload base64 encoded content to local storage."""
        # Decode base64 content
        file_content = base64.b64decode(base64_content)

        key = f"{bucket}/{key}" if bucket else key
        # Create file path
        file_path = self.storage_path / key

        # Write content to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.unlink(missing_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        return str(self.storage_path), key

    async def get_download_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        """
        Get local file path as URL.

        Args:
            bucket: Bucket/container name (unused for local storage)
            key: Storage key/path
            expires_in: URL expiration in seconds (unused for local storage)

        Returns:
            str: Local file URL
        """
        file_path = self.storage_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return f"file://{file_path.absolute()}"

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete a file from local storage."""
        file_path = self.storage_path / key
        if file_path.exists():
            file_path.unlink()
        return True

    async def batch_delete_files(
        self, keys: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
        """
        Delete multiple files from local storage.

        Args:
            keys: List of (bucket, key) tuples to delete. Bucket is ignored for local storage.

        Returns:
            Tuple containing:
            - List of successful deletions as (bucket, key) tuples
            - List of failed deletions with error details
        """
        if not keys:
            return [], []

        successful_keys: List[Tuple[str, str]] = []
        failed_keys_with_error: List[Dict[str, str]] = []

        for bucket, key in keys:
            try:
                # Construct the full path using the storage path and the key
                # The bucket is typically ignored in local storage implementations
                file_path = self.storage_path / key

                # Delete the file if it exists. missing_ok=True prevents errors if the file is already gone.
                file_path.unlink(missing_ok=True)

                # Track successful deletion
                successful_keys.append((bucket, key))
                # Optional: Log successful deletion if needed
                # logger.info(f"Successfully deleted local file: {file_path}")

            except Exception as e:
                # Track failed deletion with error details
                error_message = str(e)
                failed_keys_with_error.append(
                    {
                        "bucket": bucket,  # Keep bucket info even if unused locally
                        "key": key,
                        "error": error_message,
                    }
                )
                logger.error(f"Error deleting local file {key}: {error_message}")

        logger.info(
            f"Local batch delete completed: {len(successful_keys)} successful, {len(failed_keys_with_error)} failed"
        )
        return successful_keys, failed_keys_with_error


# Removed duplicate definition below this line
