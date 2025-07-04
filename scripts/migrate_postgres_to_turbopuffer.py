#!/usr/bin/env python3
"""
Migrate multi_vector_embeddings from PostgreSQL to Turbopuffer by user_id.

This script migrates data from MultiVectorStore (PostgreSQL) to FastMultiVectorStore (Turbopuffer).
Migration is done by user (owner_id) to ensure data consistency and allow resumable operations.

Features:
- Migrate entire user data sets together
- Resumable checkpoint functionality
- Single user testing capability
- Progress tracking and error recovery
- External storage migration (S3/local)
"""

import argparse
import asyncio
import json
import pickle
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg  # noqa: E402
import numpy as np  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.models.chunk import DocumentChunk  # noqa: E402
from core.vector_store.fast_multivector_store import FastMultiVectorStore  # noqa: E402
from core.vector_store.multi_vector_store import MultiVectorStore  # noqa: E402

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False
    print("⚠️  orjson not available, using standard json (slower)")

BATCH_SIZE = 100  # Process 100 chunks at a time
CHECKPOINT_FREQUENCY = 5  # Save checkpoint every 5 batches (now every 500 chunks)
MAX_CONCURRENT_USERS = 3  # Maximum number of users to migrate in parallel
DB_POOL_MIN_SIZE = 5  # Minimum database connections in pool
DB_POOL_MAX_SIZE = 15  # Maximum database connections in pool
MAX_CONCURRENT_STORAGE_OPS = 16  # Maximum concurrent storage operations


class MigrationCheckpoint:
    """Class to manage migration checkpoints for resume functionality."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.data = {
            "timestamp": None,
            "total_users": 0,
            "processed_users": 0,
            "current_user_id": None,
            "current_user_processed_chunks": 0,
            "current_user_total_chunks": 0,
            "current_user_last_id": None,
            "processed_user_ids": [],
            "failed_user_ids": [],
            "completed": False,
        }
        self.load_checkpoint()

    def load_checkpoint(self) -> bool:
        """Load checkpoint from file if it exists."""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    self.data = pickle.load(f)
                print(f"📂 Loaded checkpoint: {self.processed_users}/{self.total_users} users completed")
                return True
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                return False
        return False

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        try:
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(self.data, f)
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")

    def update_user_progress(
        self, user_id: str, processed_chunks: int, total_chunks: int, last_id: Optional[int] = None
    ):
        """Update progress for current user."""
        self.data["current_user_id"] = user_id
        self.data["current_user_processed_chunks"] = processed_chunks
        self.data["current_user_total_chunks"] = total_chunks
        if last_id is not None:
            self.data["current_user_last_id"] = last_id

    def complete_user(self, user_id: str):
        """Mark a user as completed."""
        if user_id not in self.data["processed_user_ids"]:
            self.data["processed_user_ids"].append(user_id)
        self.data["processed_users"] = len(self.data["processed_user_ids"])
        self.data["current_user_id"] = None
        self.data["current_user_processed_chunks"] = 0
        self.data["current_user_total_chunks"] = 0
        self.data["current_user_last_id"] = None

    def mark_user_failed(self, user_id: str):
        """Mark a user as failed."""
        if user_id not in self.data["failed_user_ids"]:
            self.data["failed_user_ids"].append(user_id)

    def mark_completed(self):
        """Mark migration as completed."""
        self.data["completed"] = True
        self.save_checkpoint()

    def cleanup(self):
        """Remove checkpoint file after successful completion."""
        if Path(self.checkpoint_file).exists():
            Path(self.checkpoint_file).unlink()
            print("🧹 Cleaned up checkpoint file")

    # Properties for easy access
    @property
    def total_users(self) -> int:
        return self.data["total_users"]

    @property
    def processed_users(self) -> int:
        return self.data["processed_users"]

    @property
    def current_user_id(self) -> Optional[str]:
        return self.data["current_user_id"]

    @property
    def processed_user_ids(self) -> List[str]:
        return self.data["processed_user_ids"]

    @property
    def failed_user_ids(self) -> List[str]:
        return self.data["failed_user_ids"]

    @property
    def is_completed(self) -> bool:
        return self.data["completed"]

    @property
    def current_user_last_id(self) -> Optional[int]:
        return self.data["current_user_last_id"]


async def get_users_with_embeddings(
    pool: asyncpg.Pool, specific_user_id: Optional[str] = None
) -> List[Tuple[str, int]]:
    """Get list of users who have multi-vector embeddings with chunk counts."""
    if specific_user_id:
        query = """
        SELECT d.owner_id, COUNT(mve.id) as chunk_count
        FROM documents d
        INNER JOIN multi_vector_embeddings mve ON d.external_id = mve.document_id
        WHERE d.owner_id = $1
        GROUP BY d.owner_id
        ORDER BY chunk_count DESC
        """
        async with pool.acquire() as conn:
            result = await conn.fetch(query, specific_user_id)
    else:
        query = """
        SELECT d.owner_id, COUNT(mve.id) as chunk_count
        FROM documents d
        INNER JOIN multi_vector_embeddings mve ON d.external_id = mve.document_id
        WHERE d.owner_id IS NOT NULL
        GROUP BY d.owner_id
        ORDER BY chunk_count DESC
        """
        async with pool.acquire() as conn:
            result = await conn.fetch(query)

    return [(row["owner_id"], row["chunk_count"]) for row in result]


async def get_user_chunks_batch(pool: asyncpg.Pool, user_id: str, last_id: Optional[int], limit: int) -> List[Dict]:
    """Get a batch of chunks for a specific user using cursor-based pagination with prepared statements."""
    async with pool.acquire() as conn:
        if last_id is None:
            # First batch - prepare statement per connection
            query = """
            SELECT
                mve.id,
                mve.document_id,
                mve.chunk_number,
                mve.content,
                mve.chunk_metadata,
                mve.embeddings
            FROM multi_vector_embeddings mve
            INNER JOIN documents d ON d.external_id = mve.document_id
            WHERE d.owner_id = $1
            ORDER BY mve.id
            LIMIT $2
            """
            stmt = await conn.prepare(query)
            result = await stmt.fetch(user_id, limit)
        else:
            # Subsequent batches using cursor
            query = """
            SELECT
                mve.id,
                mve.document_id,
                mve.chunk_number,
                mve.content,
                mve.chunk_metadata,
                mve.embeddings
            FROM multi_vector_embeddings mve
            INNER JOIN documents d ON d.external_id = mve.document_id
            WHERE d.owner_id = $1 AND mve.id > $2
            ORDER BY mve.id
            LIMIT $3
            """
            stmt = await conn.prepare(query)
            result = await stmt.fetch(user_id, last_id, limit)

    return [dict(row) for row in result]


def convert_postgres_embeddings_to_numpy(pg_embeddings) -> np.ndarray:
    """Convert PostgreSQL bit array embeddings to numpy arrays using vectorized operations."""
    # pg_embeddings is a list of psycopg Bit objects
    embeddings_list = []

    def _extract_bits_vectorized(bit_objects) -> List[str]:
        """Extract binary strings from bit objects in batch"""
        bit_strings = []

        for b in bit_objects:
            s = str(b)

            if s.startswith("Bit("):
                # Format: "Bit('1010')" or "Bit(1010)"
                s = s[4:-1].strip("'")
                bit_strings.append(s)
            elif s.startswith("<BitString"):
                # Format: "<BitString 0101 1101 1101 0011 ...>"
                start = s.find("BitString") + len("BitString")
                end = s.rfind(">")
                if end == -1:
                    end = len(s)
                bit_str = s[start:end].strip().replace(" ", "")
                bit_strings.append(bit_str)
            else:
                # Fallback
                bit_strings.append(s)

        return bit_strings

    try:
        # Extract all bit strings at once
        bit_strings = _extract_bits_vectorized(pg_embeddings)

        # Convert to numpy arrays using vectorized operations
        for bit_str in bit_strings:
            try:
                # Ensure we have only 0s and 1s
                if not all(c in "01" for c in bit_str):
                    print(f"⚠️  Invalid binary string: {bit_str[:50]}...")
                    continue

                # Vectorized conversion: create numpy array directly from bit string
                bit_array = np.array([int(bit) for bit in bit_str], dtype=np.float32)
                embeddings_list.append(bit_array)

            except Exception as e:
                print(f"⚠️  Error converting bit string {bit_str[:50]}...: {e}")
                continue

        if not embeddings_list:
            raise ValueError("No valid embeddings could be converted")

        # Stack all embeddings into a single array
        return np.stack(embeddings_list)

    except Exception as e:
        print(f"⚠️  Error in vectorized conversion: {e}")
        # Fallback to original method if vectorized fails
        return _convert_postgres_embeddings_fallback(pg_embeddings)


def _convert_postgres_embeddings_fallback(pg_embeddings) -> np.ndarray:
    """Fallback conversion method for PostgreSQL embeddings."""
    embeddings_list = []

    for bit_obj in pg_embeddings:
        try:
            s = str(bit_obj)

            if s.startswith("Bit("):
                s = s[4:-1].strip("'")
            elif s.startswith("<BitString"):
                start = s.find("BitString") + len("BitString")
                end = s.rfind(">")
                if end == -1:
                    end = len(s)
                s = s[start:end].strip().replace(" ", "")

            if all(c in "01" for c in s):
                bit_array = np.array([float(bit) for bit in s])
                embeddings_list.append(bit_array)

        except Exception as e:
            print(f"⚠️  Error converting bit object {bit_obj}: {e}")
            continue

    if not embeddings_list:
        raise ValueError("No valid embeddings could be converted")

    return np.array(embeddings_list)


@lru_cache(maxsize=10000)
def parse_metadata_cached(metadata_str: str) -> dict:
    """Parse metadata with caching and fast JSON parsing."""
    if not metadata_str:
        return {}

    try:
        # First try orjson if available (8-10x faster)
        if HAS_ORJSON:
            return orjson.loads(metadata_str)
        else:
            return json.loads(metadata_str)
    except (json.JSONDecodeError, getattr(orjson, "JSONDecodeError", Exception) if HAS_ORJSON else Exception):
        try:
            # Fallback to ast.literal_eval for Python dict strings
            import ast

            return ast.literal_eval(metadata_str)
        except (ValueError, SyntaxError) as e:
            print(f"❌ CRITICAL: Cannot parse metadata: {e}")
            print(f"    Raw metadata: {repr(metadata_str[:500])}")
            raise ValueError(f"Critical metadata parsing failure: {e}")


async def migrate_user_data(
    source_store: MultiVectorStore,
    target_store: FastMultiVectorStore,
    pool: asyncpg.Pool,
    user_id: str,
    total_chunks: int,
    checkpoint: MigrationCheckpoint,
) -> bool:
    """Migrate all chunks for a specific user."""
    print(f"🔄 Migrating {total_chunks} chunks for user {user_id}")

    processed_chunks = (
        checkpoint.data.get("current_user_processed_chunks", 0) if checkpoint.current_user_id == user_id else 0
    )
    last_id = checkpoint.current_user_last_id if checkpoint.current_user_id == user_id else None

    # Process in batches using cursor-based pagination
    while True:
        try:
            # Get batch of raw data from PostgreSQL
            raw_chunks = await get_user_chunks_batch(pool, user_id, last_id, BATCH_SIZE)

            if not raw_chunks:
                break

            # Prepare data for direct Turbopuffer insertion (avoiding store_embeddings)
            chunk_ids = []
            fde_embeddings = []
            document_ids = []
            chunk_numbers = []
            content_keys = []  # Keep existing S3 keys as-is
            metadatas = []
            chunks_for_storage = []  # Collect chunks for parallel storage operations

            # First pass: prepare all data except storage operations
            for raw_chunk in raw_chunks:
                try:
                    # Parse metadata using cached parser
                    metadata = parse_metadata_cached(raw_chunk["chunk_metadata"] or "")

                    # Convert PostgreSQL embeddings to numpy array
                    embeddings = convert_postgres_embeddings_to_numpy(raw_chunk["embeddings"])

                    # Generate FDE encoding (borrowed from FastMultiVectorStore.store_embeddings)
                    import fixed_dimensional_encoding as fde

                    fde_encoding = fde.generate_document_encoding(embeddings, target_store.fde_config).tolist()

                    # Create DocumentChunk for multivector storage
                    chunk = DocumentChunk(
                        document_id=raw_chunk["document_id"],
                        chunk_number=raw_chunk["chunk_number"],
                        content=raw_chunk["content"],  # This is the S3 key - we'll use it as-is
                        embedding=embeddings,
                        metadata=metadata,
                    )

                    # Collect data for batch insertion to Turbopuffer
                    chunk_id = f"{raw_chunk['document_id']}-{raw_chunk['chunk_number']}"
                    chunk_ids.append(chunk_id)
                    fde_embeddings.append(fde_encoding)
                    document_ids.append(raw_chunk["document_id"])
                    chunk_numbers.append(raw_chunk["chunk_number"])
                    content_keys.append(raw_chunk["content"])  # Keep existing S3 key unchanged
                    metadatas.append(json.dumps(metadata))
                    chunks_for_storage.append(chunk)

                except Exception as e:
                    print(
                        f"❌ Error processing chunk {raw_chunk.get('document_id', 'unknown')}-{raw_chunk.get('chunk_number', 'unknown')}: {e}"
                    )
                    continue

            # Second pass: parallel storage operations
            if chunks_for_storage:
                storage_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STORAGE_OPS)

                async def save_chunk_to_storage(chunk):
                    async with storage_semaphore:
                        return await target_store.save_multivector_to_storage(chunk)

                # Execute storage operations in parallel
                try:
                    storage_results = await asyncio.gather(
                        *[save_chunk_to_storage(chunk) for chunk in chunks_for_storage]
                    )
                    multivectors = [[bucket, key] for bucket, key in storage_results]
                except Exception as e:
                    print(f"❌ Error in parallel storage operations: {e}")
                    continue
            else:
                multivectors = []

            if not chunk_ids:
                print("⚠️  No valid chunks in batch")
                continue

            # Write directly to Turbopuffer (borrowed from FastMultiVectorStore.store_embeddings)
            result = await target_store.ns.write(
                upsert_columns={
                    "id": chunk_ids,
                    "vector": fde_embeddings,
                    "document_id": document_ids,
                    "chunk_number": chunk_numbers,
                    "content": content_keys,  # Existing S3 keys preserved
                    "metadata": metadatas,
                    "multivector": multivectors,
                },
                distance_metric="cosine_distance",
            )

            processed_chunks += len(chunk_ids)

            # Update last_id for cursor-based pagination
            if raw_chunks:
                last_id = raw_chunks[-1]["id"]

            # Update progress
            checkpoint.update_user_progress(user_id, processed_chunks, total_chunks, last_id)

            progress = (processed_chunks / total_chunks) * 100
            print(
                f"📊 User {user_id} progress: {processed_chunks}/{total_chunks} ({progress:.1f}%) - Stored {len(chunk_ids)} chunks"
            )
            print(f"    Turbopuffer result: {result.model_dump_json()}")

            # Save checkpoint periodically
            if processed_chunks % (CHECKPOINT_FREQUENCY * BATCH_SIZE) == 0:
                checkpoint.save_checkpoint()
                print(f"💾 Checkpoint saved for user {user_id}")

        except Exception as e:
            print(f"❌ Error processing batch for user {user_id}: {e}")
            checkpoint.mark_user_failed(user_id)
            return False

    print(f"✅ Successfully migrated {processed_chunks} chunks for user {user_id}")
    return True


async def migrate_single_user_with_semaphore(
    semaphore: asyncio.Semaphore,
    source_store: MultiVectorStore,
    target_store: FastMultiVectorStore,
    pool: asyncpg.Pool,
    user_id: str,
    total_chunks: int,
    checkpoint: MigrationCheckpoint,
) -> Tuple[str, bool, int]:
    """Migrate a single user with semaphore throttling."""
    async with semaphore:
        success = await migrate_user_data(source_store, target_store, pool, user_id, total_chunks, checkpoint)
        return user_id, success, total_chunks


async def migrate_postgres_to_turbopuffer(
    specific_user_id: Optional[str] = None, resume: bool = False, force_restart: bool = False
):
    """Main migration function."""

    # Get settings
    settings = get_settings()

    # Database connection string
    DATABASE_URL = settings.POSTGRES_URI
    if DATABASE_URL.startswith("postgresql+asyncpg://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    # Initialize checkpoint
    checkpoint_file = f"migration_checkpoint_{specific_user_id if specific_user_id else 'all'}.pkl"
    checkpoint = MigrationCheckpoint(checkpoint_file)

    # Handle force restart
    if force_restart and Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        checkpoint = MigrationCheckpoint(checkpoint_file)
        print("🔄 Force restart: Removed existing checkpoint")

    # Check if already completed
    if checkpoint.is_completed:
        print("✅ Migration already completed! Use --force-restart to start over.")
        return

    print("🔗 Creating database connection pool...")

    try:
        # Create database connection pool
        pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
            command_timeout=300,  # 5 minutes for large queries
        )
        print(f"✅ Database pool created with {DB_POOL_MIN_SIZE}-{DB_POOL_MAX_SIZE} connections")

        # Initialize stores
        source_store = MultiVectorStore(uri=DATABASE_URL, auto_initialize=False, enable_external_storage=True)

        target_store = FastMultiVectorStore(
            uri=DATABASE_URL,
            tpuf_api_key=settings.TURBOPUFFER_API_KEY,
            namespace=getattr(settings, "TURBOPUFFER_NAMESPACE", "public"),
            region=getattr(settings, "TURBOPUFFER_REGION", "aws-us-west-2"),
        )

        # Get users to migrate
        if checkpoint.processed_users == 0:
            # Fresh start
            print("🔍 Finding users with multi-vector embeddings...")
            users_with_chunks = await get_users_with_embeddings(pool, specific_user_id)

            if not users_with_chunks:
                print("❌ No users found with multi-vector embeddings")
                return

            checkpoint.data["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint.data["total_users"] = len(users_with_chunks)
            checkpoint.save_checkpoint()

            print(f"✅ Found {len(users_with_chunks)} users to migrate")
            for user_id, chunk_count in users_with_chunks[:5]:  # Show first 5
                print(f"  - {user_id}: {chunk_count} chunks")
            if len(users_with_chunks) > 5:
                print(f"  ... and {len(users_with_chunks) - 5} more users")
        else:
            # Resume migration
            print("🔄 Resuming migration from checkpoint")
            users_with_chunks = await get_users_with_embeddings(pool, specific_user_id)

        # Process users in parallel
        total_migrated_chunks = 0
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)

        # Filter users to process
        users_to_process = []
        for user_id, total_chunks in users_with_chunks:
            # Skip if already processed
            if user_id in checkpoint.processed_user_ids:
                print(f"⏭️  Skipping already processed user {user_id}")
                continue

            # Skip if previously failed (unless resuming)
            if user_id in checkpoint.failed_user_ids and not resume:
                print(f"⏭️  Skipping previously failed user {user_id}")
                continue

            users_to_process.append((user_id, total_chunks))

        if not users_to_process:
            print("✅ No users to process!")
            return

        print(
            f"🚀 Starting parallel migration of {len(users_to_process)} users (max {MAX_CONCURRENT_USERS} concurrent)"
        )

        # Process users in chunks to avoid overwhelming the system
        for i in range(0, len(users_to_process), MAX_CONCURRENT_USERS * 2):
            batch_users = users_to_process[i : i + MAX_CONCURRENT_USERS * 2]

            # Create migration tasks
            tasks = []
            for user_id, total_chunks in batch_users:
                task = migrate_single_user_with_semaphore(
                    semaphore, source_store, target_store, pool, user_id, total_chunks, checkpoint
                )
                tasks.append(task)

            # Execute batch of users in parallel
            print(f"\n🔄 Processing batch of {len(batch_users)} users...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    print(f"❌ Exception during migration: {result}")
                    continue

                user_id, success, total_chunks = result

                if success:
                    checkpoint.complete_user(user_id)
                    total_migrated_chunks += total_chunks
                    print(f"✅ Completed migration for user {user_id}")
                else:
                    checkpoint.mark_user_failed(user_id)
                    print(f"❌ Failed to migrate user {user_id}")

                checkpoint.save_checkpoint()

        # Mark as completed
        checkpoint.mark_completed()

        print("\n🎉 Migration completed successfully!")
        print("📊 Summary:")
        print(f"  - Total users processed: {checkpoint.processed_users}")
        print(f"  - Total chunks migrated: {total_migrated_chunks}")
        print(f"  - Failed users: {len(checkpoint.failed_user_ids)}")

        if checkpoint.failed_user_ids:
            print(f"  - Failed user IDs: {checkpoint.failed_user_ids}")

        # Cleanup
        checkpoint.cleanup()

        await pool.close()
        source_store.close()

    except Exception as e:
        print(f"❌ Migration error: {e}")
        print("💾 Progress has been saved. Resume with --resume flag")
        sys.exit(1)


def show_checkpoint_status(specific_user_id: Optional[str] = None):
    """Show current checkpoint status."""
    checkpoint_file = f"migration_checkpoint_{specific_user_id if specific_user_id else 'all'}.pkl"

    if not Path(checkpoint_file).exists():
        print("📋 No checkpoint found. Starting fresh migration.")
        return

    checkpoint = MigrationCheckpoint(checkpoint_file)

    if checkpoint.is_completed:
        print("✅ Migration already completed!")
        print(f"   Users processed: {checkpoint.processed_users}")
        return

    progress = (checkpoint.processed_users / checkpoint.total_users) * 100 if checkpoint.total_users > 0 else 0
    print("📋 Migration Status:")
    print(f"   Progress: {checkpoint.processed_users}/{checkpoint.total_users} users ({progress:.1f}%)")
    print(f"   Current user: {checkpoint.current_user_id}")

    if checkpoint.current_user_id:
        user_progress = (
            (checkpoint.data["current_user_processed_chunks"] / checkpoint.data["current_user_total_chunks"]) * 100
            if checkpoint.data["current_user_total_chunks"] > 0
            else 0
        )
        print(
            f"   Current user progress: {checkpoint.data['current_user_processed_chunks']}/{checkpoint.data['current_user_total_chunks']} chunks ({user_progress:.1f}%)"
        )
        print(f"   Current user last processed ID: {checkpoint.current_user_last_id}")

    print(f"   Failed users: {len(checkpoint.failed_user_ids)}")
    if checkpoint.failed_user_ids:
        print(f"   Failed user IDs: {checkpoint.failed_user_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate multi-vector embeddings from PostgreSQL to Turbopuffer by user"
    )
    parser.add_argument("--user-id", type=str, help="Migrate only a specific user (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--force-restart", action="store_true", help="Force restart, ignoring any existing checkpoint")
    parser.add_argument("--status", action="store_true", help="Show checkpoint status and exit")

    args = parser.parse_args()

    if args.status:
        show_checkpoint_status(args.user_id)
        sys.exit(0)

    print("🚀 Starting PostgreSQL to Turbopuffer migration...")

    if args.user_id:
        print(f"🎯 Migrating specific user: {args.user_id}")
    elif args.resume:
        print("🔄 Resume mode enabled")
    elif args.force_restart:
        print("🔄 Force restart mode enabled")

    asyncio.run(
        migrate_postgres_to_turbopuffer(
            specific_user_id=args.user_id, resume=args.resume, force_restart=args.force_restart
        )
    )
