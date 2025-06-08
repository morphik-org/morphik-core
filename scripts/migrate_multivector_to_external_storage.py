#!/usr/bin/env python3
"""
Migration script to move multi-vector embedding content from database to external storage.

This script:
1. Fetches all multi-vector embeddings from the database
2. For each chunk, determines if it's an image or text based on chunk_metadata
3. Stores the content in external storage (S3 or local) with appropriate file extension
4. Updates the database content field with the storage key path

Storage key format: {app_id}/{document_id}/{chunk_number}.{extension}
Bucket: multivector-chunks (hardcoded)
"""

import asyncio
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm

from core.config import get_settings
from core.storage.base_storage import BaseStorage
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.storage.utils_file_extensions import detect_file_type
from core.vector_store.multi_vector_store import MultiVectorStore

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
# Disable all logging to prevent any output during migration
logging.disable(logging.CRITICAL)

# Constants
MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"
DEFAULT_APP_ID = "default"  # Fallback for local usage when app_id is None


class MultiVectorStorageMigration:
    """Handles migration of multi-vector embeddings content to external storage."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = MultiVectorStore(uri=self.settings.POSTGRES_URI, enable_external_storage=False)
        self.storage = self._init_storage()

    def _init_storage(self) -> BaseStorage:
        """Initialize appropriate storage backend based on settings."""
        try:
            settings = get_settings()
            if settings.STORAGE_PROVIDER == "aws-s3":
                # logger.info("Initializing S3 storage for multi-vector chunks")
                return S3Storage(
                    aws_access_key=settings.AWS_ACCESS_KEY,
                    aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    default_bucket=MULTIVECTOR_CHUNKS_BUCKET,
                )
            else:
                # logger.info("Initializing local storage for multi-vector chunks")
                storage_path = getattr(settings, "LOCAL_STORAGE_PATH", "./storage")
                return LocalStorage(storage_path=storage_path)
        except Exception:
            # logger.error(f"Failed to initialize external storage: {e}")
            return None

    def get_document_app_ids(self) -> Dict[str, str]:
        """Fetch all documents and create mapping of document_id -> app_id."""
        # logger.info("Fetching document app_id mappings...")

        query = """
        SELECT external_id, system_metadata->>'app_id' as app_id
        FROM documents
        WHERE system_metadata ? 'app_id'
        """

        app_id_map = {}
        with self.vector_store.get_connection() as conn:
            results = conn.execute(query).fetchall()

            # Create mapping with fallback to DEFAULT_APP_ID
            for row in results:
                document_id = row[0]
                app_id = row[1] or DEFAULT_APP_ID
                app_id_map[document_id] = app_id

        # logger.info(f"Found {len(app_id_map)} documents with app_id mappings")
        return app_id_map

    def _determine_file_extension(self, content: str, chunk_metadata: Optional[str]) -> str:
        """Determine appropriate file extension based on content and metadata."""
        try:
            # Parse chunk metadata to check if it's an image
            if chunk_metadata:
                metadata = json.loads(chunk_metadata)
                is_image = metadata.get("is_image", False)

                if is_image:
                    # For images, auto-detect from base64 content
                    return detect_file_type(content)
                else:
                    # For text content, use .txt
                    return ".txt"
            else:
                # No metadata, try to auto-detect
                return detect_file_type(content)

        except (json.JSONDecodeError, Exception):
            # logger.warning(f"Error parsing chunk metadata: {e}")
            # Fallback to auto-detection
            try:
                return detect_file_type(content)
            except Exception:
                # logger.warning(f"Error detecting file type: {e}")
                return ".txt"

    def _generate_storage_key(self, app_id: str, document_id: str, chunk_number: int, extension: str) -> str:
        """Generate storage key path."""
        return f"{app_id}/{document_id}/{chunk_number}{extension}"

    async def migrate_chunk_content(self, chunk_data: Tuple, app_id_map: Dict[str, str]) -> Optional[str]:
        """
        Migrate a single chunk's content to external storage.

        Args:
            chunk_data: Tuple of (id, document_id, chunk_number, content, chunk_metadata)
            app_id_map: Pre-fetched mapping of document_id -> app_id

        Returns:
            New storage key if successful, None if failed
        """
        chunk_id, document_id, chunk_number, content, chunk_metadata = chunk_data

        try:
            # Get app_id for this document from pre-fetched mapping
            app_id = app_id_map.get(document_id, DEFAULT_APP_ID)

            # Determine file extension
            extension = self._determine_file_extension(content, chunk_metadata)

            # Generate storage key
            storage_key = self._generate_storage_key(app_id, document_id, chunk_number, extension)

            # Store content in external storage
            if extension == ".txt":
                # For text content, store as-is without base64 encoding
                # Convert content to base64 for storage interface compatibility
                content_bytes = content.encode("utf-8")
                content_b64 = base64.b64encode(content_bytes).decode("utf-8")
                await self.storage.upload_from_base64(
                    content=content_b64, key=storage_key, content_type="text/plain", bucket=MULTIVECTOR_CHUNKS_BUCKET
                )
            else:
                # For images, content should already be base64
                await self.storage.upload_from_base64(
                    content=content, key=storage_key, bucket=MULTIVECTOR_CHUNKS_BUCKET
                )

            # logger.debug(f"Migrated chunk {chunk_id} to storage key: {storage_key}")
            return storage_key

        except Exception:
            import traceback

            traceback.print_exc()
            # logger.error(f"Failed to migrate chunk {chunk_id}: {e}")
            return None

    def update_database_content(self, chunk_id: int, storage_key: str) -> bool:
        """Update the database content field with the storage key."""
        try:
            query = """
            UPDATE multi_vector_embeddings
            SET content = %s
            WHERE id = %s
            """
            with self.vector_store.get_connection() as conn:
                conn.execute(query, (storage_key, chunk_id))
                conn.commit()
            return True
        except Exception:
            # logger.error(f"Failed to update database for chunk {chunk_id}: {e}")
            return False

    async def migrate_all_chunks(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Migrate all multi-vector embedding chunks to external storage.

        Args:
            batch_size: Number of chunks to process in each batch

        Returns:
            Migration statistics
        """
        # logger.info("Starting multi-vector storage migration...")

        # Get total count
        count_query = "SELECT COUNT(*) FROM multi_vector_embeddings"
        with self.vector_store.get_connection() as conn:
            total_count = conn.execute(count_query).fetchone()[0]
        # logger.info(f"Total chunks to migrate: {total_count}")

        # Pre-fetch app_id mappings once
        app_id_map = self.get_document_app_ids()
        # logger.info(f"Pre-fetched app_id mappings for {len(app_id_map)} documents")

        stats = {"total": total_count, "migrated": 0, "failed": 0, "skipped": 0}

        # Fetch and migrate chunks in batches
        offset = 0

        with tqdm(total=total_count, desc="Migrating chunks") as pbar:
            while offset < total_count:
                # Fetch batch
                query = """
                SELECT id, document_id, chunk_number, content, chunk_metadata
                FROM multi_vector_embeddings
                ORDER BY id
                LIMIT %s OFFSET %s
                """

                with self.vector_store.get_connection() as conn:
                    batch = conn.execute(query, (batch_size, offset)).fetchall()

                if not batch:
                    break

                # logger.info(f"Processing batch: {offset + 1} to {offset + len(batch)} of {total_count}")

                # Process each chunk in the batch
                for chunk_data in batch:
                    chunk_id = chunk_data[0]

                    # Check if already migrated (content is not base64/long text)
                    content = chunk_data[3]
                    if len(content) < 500 and "/" in content and not content.startswith("data:"):
                        # Likely already a storage key
                        stats["skipped"] += 1
                        continue

                    # Migrate chunk content
                    storage_key = await self.migrate_chunk_content(chunk_data, app_id_map)

                    if storage_key:
                        # Update database with storage key
                        if self.update_database_content(chunk_id, storage_key):
                            stats["migrated"] += 1
                        else:
                            stats["failed"] += 1
                    else:
                        stats["failed"] += 1

                offset += len(batch)
                pbar.update(len(batch))
                # Log progress
                # progress = (offset / total_count) * 100
                # logger.info(f"Migration progress: {progress:.1f}% ({stats['migrated']} migrated, {stats['failed']} failed)")

        # logger.info("Migration completed!")
        # logger.info(f"Final stats: {stats}")
        return stats

    async def verify_migration(self, sample_size: int = 10) -> bool:
        """
        Verify migration by checking a sample of migrated chunks.

        Args:
            sample_size: Number of chunks to verify

        Returns:
            True if verification passes
        """
        # logger.info(f"Verifying migration with sample size: {sample_size}")

        # Get sample of migrated chunks (short content that looks like storage keys)
        query = """
        SELECT id, document_id, chunk_number, content, chunk_metadata
        FROM multi_vector_embeddings
        WHERE LENGTH(content) < 500 AND content LIKE %s
        ORDER BY RANDOM()
        LIMIT %s
        """

        with self.vector_store.get_connection() as conn:
            sample_chunks = conn.execute(query, ("%/%", sample_size)).fetchall()

        if not sample_chunks:
            # logger.warning("No migrated chunks found for verification")
            return False

        verification_passed = 0

        for chunk_data in sample_chunks:
            chunk_id, document_id, chunk_number, storage_key, chunk_metadata = chunk_data

            try:
                # Try to download content from storage
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=storage_key)

                if content_bytes:
                    verification_passed += 1
                    # logger.debug(f"Verified chunk {chunk_id}: storage key {storage_key} is accessible")
                else:
                    # logger.warning(f"Chunk {chunk_id}: storage key {storage_key} returned empty content")
                    pass

            except Exception:
                # logger.error(f"Verification failed for chunk {chunk_id}: {e}")
                pass

        success_rate = verification_passed / len(sample_chunks)
        # logger.info(f"Verification completed: {verification_passed}/{len(sample_chunks)} chunks verified ({success_rate*100:.1f}%)")

        return success_rate >= 0.9  # 90% success rate threshold


async def main():
    """Main migration entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    migration = MultiVectorStorageMigration()

    try:
        # Run migration
        stats = await migration.migrate_all_chunks(batch_size=50)

        # Verify migration
        if stats["migrated"] > 0:
            verification_passed = await migration.verify_migration(sample_size=min(10, stats["migrated"]))
            if not verification_passed:
                # logger.error("Migration verification failed!")
                return False

        # logger.info("Migration completed successfully!")
        return True

    except Exception:
        # logger.error(f"Migration failed: {e}")
        return False
    finally:
        # Clean up database connection
        if hasattr(migration.vector_store, "pool") and migration.vector_store.pool:
            migration.vector_store.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
