import base64
import logging
import tempfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError

from .base_storage import BaseStorage
from .utils_file_extensions import detect_file_type

logger = logging.getLogger(__name__)


class S3Storage(BaseStorage):
    """AWS S3 storage implementation."""

    # TODO: Remove hardcoded values.
    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        region_name: str = "us-east-2",
        default_bucket: str = "morphik-storage",
    ):
        self.default_bucket = default_bucket
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )

    async def upload_file(
        self,
        file: Union[str, bytes, BinaryIO],
        key: str,
        content_type: Optional[str] = None,
        bucket: str = "",
    ) -> Tuple[str, str]:
        """Upload a file to S3."""
        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            if isinstance(file, (str, bytes)):
                # Create temporary file for content
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    if isinstance(file, str):
                        temp_file.write(file.encode())
                    else:
                        temp_file.write(file)
                    temp_file_path = temp_file.name

                try:
                    self.s3_client.upload_file(temp_file_path, self.default_bucket, key, ExtraArgs=extra_args)
                finally:
                    Path(temp_file_path).unlink()
            else:
                # File object
                self.s3_client.upload_fileobj(file, self.default_bucket, key, ExtraArgs=extra_args)

            return self.default_bucket, key

        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    async def upload_from_base64(
        self, content: str, key: str, content_type: Optional[str] = None, bucket: str = ""
    ) -> Tuple[str, str]:
        """Upload base64 encoded content to S3."""
        key = f"{bucket}/{key}" if bucket else key
        try:
            decoded_content = base64.b64decode(content)
            extension = detect_file_type(content)
            key = f"{key}{extension}"

            return await self.upload_file(file=decoded_content, key=key, content_type=content_type, bucket=bucket)

        except Exception as e:
            logger.error(f"Error uploading base64 content to S3: {e}")
            raise e

    async def download_file(self, bucket: str, key: str) -> bytes:
        """Download file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
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

    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"File {key} deleted from bucket {bucket}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False

    async def batch_delete_files(
        self, keys: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
        """
        Delete multiple files from S3 in batches.

        Args:
            keys: List of (bucket, key) tuples to delete

        Returns:
            Tuple containing:
            - List of successful deletions as (bucket, key) tuples
            - List of failed deletions with error details
        """
        if not keys:
            return [], []

        # Group keys by bucket for efficient batch processing
        keys_by_bucket: Dict[str, List[str]] = {}
        for bucket, key in keys:
            # Use default bucket if none provided in the tuple
            target_bucket = bucket if bucket else self.default_bucket
            if target_bucket not in keys_by_bucket:
                keys_by_bucket[target_bucket] = []
            keys_by_bucket[target_bucket].append(key)

        successful_keys: List[Tuple[str, str]] = []
        failed_keys_with_error: List[Dict[str, str]] = []

        # Process each bucket
        for bucket, bucket_keys in keys_by_bucket.items():
            # S3 allows deletion of up to 1000 objects in a single request
            batch_size = 1000

            # Process keys in batches
            for i in range(0, len(bucket_keys), batch_size):
                batch = bucket_keys[i : i + batch_size]

                # Format for delete_objects API
                objects_to_delete = [{"Key": key} for key in batch]

                try:
                    # Delete objects and get response
                    # Note: Boto3 S3 client methods are synchronous, no await needed here.
                    # If using an async client like aioboto3, this would be awaited.
                    response = self.s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={
                            "Objects": objects_to_delete,
                            "Quiet": False,  # Return results of deletion
                        },
                    )

                    # Track successful deletions
                    if "Deleted" in response:
                        for deleted in response["Deleted"]:
                            successful_keys.append((bucket, deleted["Key"]))

                    # Track errors
                    if "Errors" in response:
                        for error in response["Errors"]:
                            failed_keys_with_error.append(
                                {"bucket": bucket, "key": error["Key"], "error": error.get("Message", "Unknown error")}
                            )

                except ClientError as e:
                    # If the entire batch operation failed, mark all keys as failed
                    error_message = str(e)
                    logger.error(f"Error batch deleting from S3 bucket {bucket}: {error_message}")

                    for key in batch:
                        failed_keys_with_error.append({"bucket": bucket, "key": key, "error": error_message})

        logger.info(
            f"S3 batch delete completed: {len(successful_keys)} successful, {len(failed_keys_with_error)} failed"
        )
        return successful_keys, failed_keys_with_error


# Removed duplicate definition below this line
