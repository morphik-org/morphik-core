#!/usr/bin/env python3
"""
Utility script to drop legacy ACL-style columns that are no longer used.

Removes the following columns from the documents, graphs, and folders tables:
    - owner_type
    - readers
    - writers
    - admins

Any leftover GIN indexes created on those columns are dropped as well.

Usage examples:

    python drop_legacy_acl_columns.py --postgres-uri "postgresql+asyncpg://user:pass@host:5432/dbname"
    python drop_legacy_acl_columns.py --dry-run
"""

import argparse
import asyncio
import logging
from typing import Iterable, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")

TABLES = ("documents", "graphs", "folders")
COLUMNS = ("owner_type", "readers", "writers", "admins")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drop legacy ACL columns and related indexes.")
    parser.add_argument(
        "--postgres-uri",
        dest="postgres_uri",
        default=None,
        help="Database URI (defaults to settings.POSTGRES_URL)",
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


def _drop_index_statements(table: str) -> Iterable[text]:
    # Historical index names created by migrate_auth_columns_complete.py
    for suffix in ("readers", "writers", "admins"):
        yield text(f"DROP INDEX IF EXISTS idx_{table}_{suffix};")


def _drop_column_statements(table: str) -> Iterable[text]:
    for column in COLUMNS:
        yield text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column} CASCADE;")


async def drop_acl_columns(engine: AsyncEngine, dry_run: bool) -> None:
    logger.info("Dropping legacy ACL columns from %s", ", ".join(TABLES))

    statements = []
    for table in TABLES:
        statements.extend(_drop_index_statements(table))
        statements.extend(_drop_column_statements(table))

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
        await drop_acl_columns(engine, args.dry_run)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
