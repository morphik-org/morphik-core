import asyncio
import base64
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .base_storage import BaseStorage
from .utils_file_extensions import detect_file_type

logger = logging.getLogger(__name__)

# Lazy-initialized thread pool for S3 I/O operations.
# Default executor is limited to ~(cpu_count + 4) threads, which bottlenecks
# when downloading 50-75 multivector files in parallel during ColPali reranking.
# 64 threads matches max_pool_connections for optimal throughput.
_s3_executor: ThreadPoolExecutor | None = None


def _get_s3_executor() -> ThreadPoolExecutor:
    """Get or create the S3 thread pool executor (lazy initialization)."""
    global _s3_executor
    if _s3_executor is None:
        _s3_executor = ThreadPoolExecutor(max_workers=64, thread_name_prefix="s3-io")
    return _s3_executor


class S3Storage(BaseStorage):
    """AWS S3 storage implementation."""

    # TODO: Remove hardcoded values.
    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        region_name: str = "us-east-2",
        default_bucket: str = "morphik-storage",
        upload_concurrency: int = 16,
    ):
        self.default_bucket = default_bucket
        # Increase the underlying urllib3 connection-pool size to better support high concurrency
        boto_cfg = Config(max_pool_connections=64, retries={"max_attempts": 3, "mode": "standard"})
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
            config=boto_cfg,
        )
        # Cap concurrent uploads to avoid overwhelming the pool/S3 while still allowing parallelism.
        self._upload_semaphore = asyncio.Semaphore(max(1, upload_concurrency))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_bucket(self, bucket: str) -> None:
        """Create *bucket* if it does not exist (idempotent).

        S3 returns an error if you try to create an existing bucket in the
        *same* region – we silently ignore that specific error code.
        """
        try:
            # HeadBucket is the cheapest – if it succeeds the bucket exists.
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError as exc:  # noqa: BLE001 – fine-grained checks below
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchBucket"}:
                # Need to create the bucket in the client's region
                region = self.s3_client.meta.region_name
                if region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
            elif error_code in {"301", "BucketAlreadyOwnedByYou", "400"}:
                # Bucket exists / owned etc. – safe to continue
                pass
            else:
                raise

    async def upload_file(
        self,
        file: Union[str, bytes, BinaryIO],
        key: str,
        content_type: Optional[str] = None,
        bucket: str = "",
    ) -> Tuple[str, str]:
        """Upload a file to S3 using a shared executor for true concurrency."""
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        target_bucket = bucket or self.default_bucket
        loop = asyncio.get_running_loop()

        def _sync_upload() -> None:
            self._ensure_bucket(target_bucket)

            if isinstance(file, (str, bytes)):
                # Create temporary file for content
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    if isinstance(file, str):
                        temp_file.write(file.encode())
                    else:
                        temp_file.write(file)
                    temp_file_path = temp_file.name

                try:
                    self.s3_client.upload_file(temp_file_path, target_bucket, key, ExtraArgs=extra_args)
                finally:
                    Path(temp_file_path).unlink(missing_ok=True)
            else:
                # File object
                self.s3_client.upload_fileobj(file, target_bucket, key, ExtraArgs=extra_args)

        try:
            async with self._upload_semaphore:
                await loop.run_in_executor(_get_s3_executor(), _sync_upload)
            return target_bucket, key
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        """Upload base64-encoded content to S3.

        Accepts either a raw base64 string or a data URI (e.g. "data:image/png;base64,...").
        Does not prefix the S3 key with the bucket name, and only appends a file extension
        when the provided key does not already include one.
        """
        try:
            # Handle data URI format explicitly
            derived_mime: Optional[str] = None
            base64_payload = content
            if isinstance(content, str) and content.startswith("data:"):
                try:
                    header, base64_part = content.split(",", 1)
                    # header like: data:image/png;base64
                    if ";" in header and ":" in header:
                        derived_mime = header.split(":", 1)[1].split(";", 1)[0]
                    base64_payload = base64_part
                except Exception:
                    # Fall back to original content if parsing fails
                    base64_payload = content

            decoded_content = base64.b64decode(base64_payload)

            # Decide on extension
            from pathlib import Path

            current_ext = Path(key).suffix
            if not current_ext:
                # Try to determine extension from data URI mime, otherwise from bytes
                if derived_mime:
                    mime_to_ext = {
                        "image/jpeg": ".jpg",
                        "image/jpg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                        "image/gif": ".gif",
                        "image/bmp": ".bmp",
                        "image/tiff": ".tiff",
                        "application/pdf": ".pdf",
                        "text/plain": ".txt",
                    }
                    extension = mime_to_ext.get(derived_mime, ".bin")
                else:
                    extension = detect_file_type(decoded_content)
                # Append extension only when missing
                key = f"{key}{extension}"

            # Prefer provided content_type; fall back to derived mime if available
            effective_content_type = content_type or derived_mime

            # Choose bucket
            target_bucket = bucket or self.default_bucket
            # Ensure bucket exists
            self._ensure_bucket(target_bucket)

            # Upload directly from bytes
            return await self.upload_file(
                file=decoded_content,
                key=key,
                content_type=effective_content_type,
                bucket=target_bucket,
            )

        except Exception as e:
            logger.error(f"Error uploading base64 content to S3: {e}")
            raise

    async def download_file(self, bucket: str, key: str, version: str | None = None, **kwargs) -> bytes:
        """Download file from S3 asynchronously using a dedicated thread pool.

        Uses a 64-thread executor to support parallel downloads of 50-75 multivector
        files during ColPali reranking without blocking on the default executor limit.
        """
        loop = asyncio.get_running_loop()

        def _sync_download() -> bytes:  # Runs in a separate thread
            get_obj_params = {"Bucket": bucket, "Key": key}
            if version:
                # If a specific version is requested, include the VersionId parameter
                get_obj_params["VersionId"] = version
            response = self.s3_client.get_object(**get_obj_params)
            return response["Body"].read()

        try:
            return await loop.run_in_executor(_get_s3_executor(), _sync_download)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"File not found: {bucket}/{key}") from e
            logger.error(f"Error downloading from S3: {e}")
            raise

    async def get_download_url(self, bucket: str, key: str, expires_in: int = 3600) -> str:
        """Generate presigned download URL."""
        if not key or not bucket:
            return ""

        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return ""

    async def get_object_size(self, bucket: str, key: str) -> int:
        """Return object size in bytes using HEAD."""
        target_bucket = bucket or self.default_bucket
        loop = asyncio.get_running_loop()

        def _sync_head() -> int:
            response = self.s3_client.head_object(Bucket=target_bucket, Key=key)
            return int(response.get("ContentLength") or 0)

        try:
            return await loop.run_in_executor(_get_s3_executor(), _sync_head)
        except ClientError as e:
            logger.error("Error getting size for s3://%s/%s: %s", target_bucket, key, e)
            raise

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"File {key} deleted from bucket {bucket}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False
