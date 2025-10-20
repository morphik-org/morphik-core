#!/usr/bin/env python3
"""
Utility script to clean legacy document metadata and optionally drop deprecated columns.

Usage examples:

    python scrub_legacy_document_metadata.py --postgres-uri "postgresql+asyncpg://user:pass@host:5432/dbname"
    python scrub_legacy_document_metadata.py --drop-columns

The script will remove folder/app/user scope keys from documents.system_metadata now that
those fields live in dedicated columns. Pass --drop-columns to remove the old status / error /
owner columns once you are confident no callers rely on them.
"""

import argparse
import asyncio
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")


SYSTEM_METADATA_KEYS_TO_REMOVE = ("folder_name", "end_user_id", "app_id")
DEPRECATED_COLUMNS = ("status", "error_message", "owner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrub legacy document metadata")
    parser.add_argument(
        "--postgres-uri",
        dest="postgres_uri",
        default=None,
        help="Database URI (defaults to settings.POSTGRES_URL)",
    )
    parser.add_argument(
        "--drop-columns",
        dest="drop_columns",
        action="store_true",
        help="Drop deprecated columns (status, error_message, owner) after scrubbing metadata.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Log the SQL that would be executed without modifying the database.",
    )
    return parser


def create_engine(uri: Optional[str]) -> AsyncEngine:
    settings = get_settings()
    postgres_uri = uri or settings.POSTGRES_URL
    if not postgres_uri:
        raise ValueError("Postgres URI must be provided via --postgres-uri or settings.POSTGRES_URL")
    logger.info("Connecting to %s", postgres_uri)
    return create_async_engine(postgres_uri, echo=False, pool_size=5, max_overflow=10)


async def scrub_system_metadata(engine: AsyncEngine, dry_run: bool) -> None:
    logger.info("Removing legacy keys %s from documents.system_metadata", SYSTEM_METADATA_KEYS_TO_REMOVE)
    json_subtractions = " ".join(f"- '{key}'" for key in SYSTEM_METADATA_KEYS_TO_REMOVE)
    keys_array_literal = ", ".join(f"'{key}'" for key in SYSTEM_METADATA_KEYS_TO_REMOVE)
    sql = text(
        f"""
        UPDATE documents
        SET system_metadata = jsonb_strip_nulls(
            COALESCE(system_metadata, '{{}}'::jsonb)
            {json_subtractions}
        )
        WHERE system_metadata ?| ARRAY[{keys_array_literal}];
        """
    )

    if dry_run:
        logger.info("Dry run enabled – skipping execution.\n%s", sql.text)
        return

    async with engine.begin() as conn:
        result = await conn.execute(sql)
        logger.info("Updated %s documents", result.rowcount)


async def drop_deprecated_columns(engine: AsyncEngine, dry_run: bool) -> None:
    logger.info("Dropping deprecated columns: %s", DEPRECATED_COLUMNS)
    statements = [
        text(f"ALTER TABLE documents DROP COLUMN IF EXISTS {column} CASCADE;") for column in DEPRECATED_COLUMNS
    ]

    if dry_run:
        for stmt in statements:
            logger.info("Dry run: %s", stmt.text)
        return

    async with engine.begin() as conn:
        for stmt in statements:
            await conn.execute(stmt)
            logger.info("Executed: %s", stmt.text)


async def main(args: argparse.Namespace) -> None:
    engine = create_engine(args.postgres_uri)
    try:
        await scrub_system_metadata(engine, args.dry_run)
        if args.drop_columns:
            await drop_deprecated_columns(engine, args.dry_run)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
