#!/usr/bin/env python3
"""
Complete migration script to create and populate flattened auth columns from JSONB data.

This script:
1. Creates flattened auth columns if they don't exist
2. Creates optimized indexes using the same names as the application
3. Performs a one-time migration of existing data from JSONB to flattened columns
4. No triggers are created since the application now populates these columns directly

Usage:
    python migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthColumnMigration:
    def __init__(self, postgres_uri: Optional[str] = None):
        """Initialize migration with database connection."""
        self.postgres_uri = postgres_uri or get_settings().POSTGRES_URL
        self.engine = create_async_engine(
            self.postgres_uri,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )

    async def create_columns_and_indexes(self) -> bool:
        """Create flattened auth columns and indexes using the same logic as postgres_database.py"""
        try:
            logger.info("Creating flattened auth columns and indexes...")

            async with self.engine.begin() as conn:
                # Tables to update with flattened columns
                tables_to_update = ["documents", "graphs", "folders"]

                for table in tables_to_update:
                    logger.info(f"Processing {table} table...")

                    # Add scalar columns
                    for column_name, column_type in [
                        ("owner_id", "VARCHAR(255)"),
                        ("owner_type", "VARCHAR(50)"),
                        ("app_id", "VARCHAR(255)"),
                        ("folder_name", "VARCHAR(255)"),
                        ("end_user_id", "VARCHAR(255)"),
                    ]:
                        # Skip folder_name for folders table (it already has a name column)
                        if table == "folders" and column_name == "folder_name":
                            continue

                        # Check if column exists
                        result = await conn.execute(
                            text(
                                f"""
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_name = '{table}' AND column_name = '{column_name}'
                            """
                            )
                        )
                        if not result.first():
                            await conn.execute(
                                text(
                                    f"""
                                    ALTER TABLE {table}
                                    ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                                """
                                )
                            )
                            logger.info(f"  Added {column_name} column to {table} table")

                    # Add array columns for access control
                    for column_name in ["readers", "writers", "admins"]:
                        result = await conn.execute(
                            text(
                                f"""
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_name = '{table}' AND column_name = '{column_name}'
                            """
                            )
                        )
                        if not result.first():
                            await conn.execute(
                                text(
                                    f"""
                                    ALTER TABLE {table}
                                    ADD COLUMN IF NOT EXISTS {column_name} TEXT[] DEFAULT '{{}}'
                                """
                                )
                            )
                            logger.info(f"  Added {column_name} array column to {table} table")

                # Create optimized indexes for the new columns
                logger.info("\nCreating optimized indexes for flattened columns...")

                for table in tables_to_update:
                    logger.info(f"Creating indexes for {table} table...")

                    # Create indexes for scalar columns
                    await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_owner_id ON {table}(owner_id);"))
                    await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_app_id ON {table}(app_id);"))

                    # Create GIN indexes for array columns
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_readers ON {table} USING gin(readers);")
                    )
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_writers ON {table} USING gin(writers);")
                    )
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_admins ON {table} USING gin(admins);")
                    )

                    # Create composite indexes for common query patterns
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_owner_app ON {table}(owner_id, app_id);")
                    )
                    if table != "folders":  # folders don't have folder_name column
                        await conn.execute(
                            text(f"CREATE INDEX IF NOT EXISTS idx_{table}_app_folder ON {table}(app_id, folder_name);")
                        )
                    await conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS idx_{table}_app_end_user ON {table}(app_id, end_user_id);")
                    )

                logger.info("✅ Flattened auth columns and indexes created successfully")
                return True

        except Exception as e:
            logger.error(f"Error creating columns and indexes: {e}")
            return False

    async def migrate_table(self, table_name: str) -> int:
        """Migrate data for a specific table."""
        try:
            logger.info(f"\nMigrating {table_name} table...")

            async with self.engine.begin() as conn:
                # Count total rows
                result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                total_rows = result.scalar()
                logger.info(f"Total rows in {table_name}: {total_rows}")

                if total_rows == 0:
                    return 0

                # Count rows that need migration
                result = await conn.execute(
                    text(
                        f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE owner IS NOT NULL AND owner_id IS NULL
                """
                    )
                )
                needs_migration = result.scalar()

                if needs_migration == 0:
                    logger.info(f"No rows need migration in {table_name}")
                    return 0

                logger.info(f"Rows needing migration: {needs_migration}")

                # Batch update in chunks of 1000 rows
                batch_size = 1000
                updated_rows = 0

                while updated_rows < needs_migration:
                    if table_name == "folders":
                        # Special handling for folders (no folder_name in system_metadata)
                        await conn.execute(
                            text(
                                f"""
                            UPDATE {table_name} SET
                                owner_id = owner->>'id',
                                owner_type = owner->>'type',
                                readers = COALESCE(
                                    ARRAY(SELECT jsonb_array_elements_text(access_control->'readers')),
                                    '{{}}'::TEXT[]
                                ),
                                writers = COALESCE(
                                    ARRAY(SELECT jsonb_array_elements_text(access_control->'writers')),
                                    '{{}}'::TEXT[]
                                ),
                                admins = COALESCE(
                                    ARRAY(SELECT jsonb_array_elements_text(access_control->'admins')),
                                    '{{}}'::TEXT[]
                                ),
                                app_id = system_metadata->>'app_id',
                                end_user_id = system_metadata->>'end_user_id'
                            WHERE owner IS NOT NULL
                            AND owner_id IS NULL
                            AND ctid IN (
                                SELECT ctid FROM {table_name}
                                WHERE owner IS NOT NULL AND owner_id IS NULL
                                LIMIT {batch_size}
                            )
                        """
                            )
                        )
                    else:
                        # Documents and graphs have folder_name
                        await conn.execute(
                            text(
                                f"""
                            UPDATE {table_name} SET
                                owner_id = owner->>'id',
                                owner_type = owner->>'type',
                                readers = COALESCE(
                                    ARRAY(SELECT jsonb_array_elements_text(access_control->'readers')),
                                    '{{}}'::TEXT[]
                                ),
                                writers = COALESCE(
                                    ARRAY(SELECT jsonb_array_elements_text(access_control->'writers')),
                                    '{{}}'::TEXT[]
                                ),
                                admins = COALESCE(
                                    ARRAY(SELECT jsonb_array_elements_text(access_control->'admins')),
                                    '{{}}'::TEXT[]
                                ),
                                app_id = system_metadata->>'app_id',
                                folder_name = system_metadata->>'folder_name',
                                end_user_id = system_metadata->>'end_user_id'
                            WHERE owner IS NOT NULL
                            AND owner_id IS NULL
                            AND ctid IN (
                                SELECT ctid FROM {table_name}
                                WHERE owner IS NOT NULL AND owner_id IS NULL
                                LIMIT {batch_size}
                            )
                        """
                            )
                        )

                    # Check how many rows were updated
                    result = await conn.execute(
                        text(
                            f"""
                        SELECT COUNT(*) FROM {table_name}
                        WHERE owner IS NOT NULL AND owner_id IS NOT NULL
                    """
                        )
                    )
                    current_updated = result.scalar()
                    batch_updated = current_updated - updated_rows
                    updated_rows = current_updated

                    if batch_updated > 0:
                        logger.info(f"  Updated {updated_rows}/{needs_migration} rows")
                    else:
                        break

                logger.info(f"✅ Completed migration for {table_name}: {updated_rows} rows updated")
                return updated_rows

        except Exception as e:
            logger.error(f"Error migrating {table_name}: {e}")
            raise

    async def add_not_null_constraints(self) -> bool:
        """Add NOT NULL constraints to owner fields after migration."""
        try:
            logger.info("\nAdding NOT NULL constraints to owner fields...")

            async with self.engine.begin() as conn:
                tables = ["documents", "graphs", "folders"]

                for table in tables:
                    # Check if we can safely add NOT NULL constraints
                    result = await conn.execute(
                        text(
                            f"""
                        SELECT COUNT(*) FROM {table}
                        WHERE owner IS NOT NULL AND (owner_id IS NULL OR owner_type IS NULL)
                    """
                        )
                    )
                    null_count = result.scalar()

                    if null_count > 0:
                        logger.warning(
                            f"Cannot add NOT NULL constraints to {table} - {null_count} rows have NULL values"
                        )
                        continue

                    # Add NOT NULL constraints
                    try:
                        await conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN owner_id SET NOT NULL;"))
                        await conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN owner_type SET NOT NULL;"))
                        logger.info(f"  Added NOT NULL constraints to {table}")
                    except Exception as e:
                        logger.warning(f"  Could not add NOT NULL constraints to {table}: {e}")

                return True

        except Exception as e:
            logger.error(f"Error adding NOT NULL constraints: {e}")
            return False

    async def verify_migration(self, table_name: str) -> bool:
        """Verify that migration was successful for a table."""
        try:
            async with self.engine.begin() as conn:
                # Check if any rows have NULL owner_id (indicating incomplete migration)
                result = await conn.execute(
                    text(
                        f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE owner IS NOT NULL AND owner_id IS NULL
                """
                    )
                )
                null_count = result.scalar()

                if null_count > 0:
                    logger.warning(f"Found {null_count} rows in {table_name} with NULL owner_id")
                    return False

                # Sample check: verify data consistency
                result = await conn.execute(
                    text(
                        f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE owner IS NOT NULL
                    AND owner->>'id' != owner_id
                """
                    )
                )
                mismatch_count = result.scalar()

                if mismatch_count > 0:
                    logger.warning(f"Found {mismatch_count} rows in {table_name} with mismatched owner data")
                    return False

                # Check array consistency
                result = await conn.execute(
                    text(
                        f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE access_control IS NOT NULL
                    AND (
                        array_length(readers, 1) != jsonb_array_length(access_control->'readers')
                        OR array_length(writers, 1) != jsonb_array_length(access_control->'writers')
                        OR array_length(admins, 1) != jsonb_array_length(access_control->'admins')
                    )
                """
                    )
                )
                array_mismatch = result.scalar()

                if array_mismatch > 0:
                    logger.warning(f"Found {array_mismatch} rows in {table_name} with mismatched array lengths")
                    return False

                logger.info(f"✅ Verification passed for {table_name}")
                return True

        except Exception as e:
            logger.error(f"Error verifying {table_name}: {e}")
            return False

    async def run_migration(self) -> bool:
        """Run the complete migration process."""
        try:
            logger.info("=== Starting complete auth column migration ===")
            logger.info(f"Database: {self.postgres_uri.split('@')[-1] if '@' in self.postgres_uri else 'default'}")
            logger.info("Note: No triggers will be created - application now populates columns directly\n")

            # Step 1: Create columns and indexes
            if not await self.create_columns_and_indexes():
                logger.error("Failed to create columns and indexes")
                return False

            # Step 2: Migrate each table
            tables = ["documents", "graphs", "folders"]
            total_migrated = 0

            for table in tables:
                rows_migrated = await self.migrate_table(table)
                total_migrated += rows_migrated

            logger.info(f"\nTotal rows migrated across all tables: {total_migrated}")

            # Step 3: Verify migration
            logger.info("\nVerifying migration...")
            all_verified = True
            for table in tables:
                if not await self.verify_migration(table):
                    all_verified = False

            # Step 4: Add NOT NULL constraints if verification passed
            if all_verified:
                await self.add_not_null_constraints()

            if all_verified:
                logger.info("\n✅ Migration completed successfully!")
                logger.info("\nNext steps:")
                logger.info("1. Deploy the updated application code")
                logger.info("2. Monitor for any issues")
                logger.info("3. Once stable, consider dropping JSONB columns in a future migration")
            else:
                logger.error("\n❌ Migration verification failed!")
                logger.error("Please check the warnings above and fix any issues.")

            return all_verified

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
        finally:
            await self.engine.dispose()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete migration to create and populate auth columns from JSONB")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing migration without making changes"
    )
    parser.add_argument(
        "--postgres-uri", help="PostgreSQL connection URI (postgresql+asyncpg://user:pass@host:port/db)", required=False
    )

    args = parser.parse_args()

    # If no postgres-uri provided, try to get from environment or settings
    postgres_uri = args.postgres_uri
    if not postgres_uri:
        # Try environment variable first
        postgres_uri = os.environ.get("POSTGRES_URL")
        if not postgres_uri:
            # Fall back to settings
            try:
                postgres_uri = get_settings().POSTGRES_URL
            except Exception:
                logger.error("No PostgreSQL URI provided. Use --postgres-uri or set POSTGRES_URL environment variable")
                sys.exit(1)

    migration = AuthColumnMigration(postgres_uri=postgres_uri)

    if args.verify_only:
        logger.info("Running verification only...")
        tables = ["documents", "graphs", "folders"]
        all_verified = True
        for table in tables:
            if not await migration.verify_migration(table):
                all_verified = False

        if all_verified:
            logger.info("\n✅ All tables verified successfully!")
        else:
            logger.error("\n❌ Verification failed!")
            sys.exit(1)
    else:
        success = await migration.run_migration()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
