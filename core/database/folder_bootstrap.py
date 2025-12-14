from __future__ import annotations

from typing import Set, Tuple

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import text


async def bootstrap_folder_hierarchy(engine: AsyncEngine, logger) -> None:
    """
    Ensure nested-folder columns/indexes exist and legacy rows are backfilled.

    Uses idempotent ALTER/CREATE operations so it is safe to call on every startup.
    """

    # Quick check to skip bootstrap if the schema is already in the desired state
    required_columns: Set[Tuple[str, str]] = {
        ("folders", "full_path"),
        ("folders", "parent_id"),
        ("folders", "depth"),
        ("documents", "folder_path"),
        ("documents", "folder_id"),
        ("graphs", "folder_path"),
    }
    required_indexes = {
        "idx_folder_full_path",
        "idx_folder_parent_id",
        "idx_folder_depth",
        "uq_folders_app_full_path",
        "uq_folders_owner_full_path",
        "idx_doc_folder_path",
        "idx_doc_folder_id",
        "idx_documents_app_folder_path",
        "idx_documents_app_folder_id",
        "idx_graph_folder_path",
        "idx_graphs_app_folder_path",
    }

    try:
        async with engine.begin() as conn:
            col_result = await conn.execute(
                text(
                    """
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_name IN ('folders', 'documents', 'graphs')
                    AND column_name IN ('full_path', 'parent_id', 'depth', 'folder_path', 'folder_id')
                    """
                )
            )
            present_columns = {(row.table_name, row.column_name) for row in col_result}

            idx_result = await conn.execute(
                text(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND indexname IN (
                        'idx_folder_full_path',
                        'idx_folder_parent_id',
                        'idx_folder_depth',
                        'uq_folders_app_full_path',
                        'uq_folders_owner_full_path',
                        'idx_doc_folder_path',
                        'idx_doc_folder_id',
                        'idx_documents_app_folder_path',
                        'idx_documents_app_folder_id',
                        'idx_graph_folder_path',
                        'idx_graphs_app_folder_path'
                    )
                    """
                )
            )
            present_indexes = {row.indexname for row in idx_result}

            pending_folders = await conn.scalar(
                text("SELECT COUNT(*) FROM folders WHERE COALESCE(full_path, '') = '' OR depth IS NULL")
            )
            pending_docs = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM documents "
                    "WHERE folder_name IS NOT NULL AND folder_name <> '' "
                    "AND (folder_path IS NULL OR folder_path = '')"
                )
            )
            pending_graphs = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM graphs "
                    "WHERE folder_name IS NOT NULL AND folder_name <> '' "
                    "AND (folder_path IS NULL OR folder_path = '')"
                )
            )

        columns_ready = required_columns.issubset(present_columns)
        indexes_ready = required_indexes.issubset(present_indexes)
        backfill_needed = bool((pending_folders or 0) > 0 or (pending_docs or 0) > 0 or (pending_graphs or 0) > 0)

        if columns_ready and indexes_ready and not backfill_needed:
            logger.info("Folder hierarchy bootstrap already completed; skipping")
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not check folder hierarchy bootstrap state, proceeding with bootstrap: %s", exc)

    column_statements = [
        "ALTER TABLE folders ADD COLUMN IF NOT EXISTS full_path TEXT",
        "ALTER TABLE folders ADD COLUMN IF NOT EXISTS parent_id TEXT",
        "ALTER TABLE folders ADD COLUMN IF NOT EXISTS depth INTEGER",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS folder_path TEXT",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS folder_id TEXT",
        "ALTER TABLE graphs ADD COLUMN IF NOT EXISTS folder_path TEXT",
    ]

    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_folder_full_path ON folders (full_path)",
        "CREATE INDEX IF NOT EXISTS idx_folder_parent_id ON folders (parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_folder_depth ON folders (depth)",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_app_full_path "
            "ON folders (app_id, full_path) WHERE app_id IS NOT NULL"
        ),
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_folders_owner_full_path "
            "ON folders (owner_id, full_path) WHERE app_id IS NULL"
        ),
        "CREATE INDEX IF NOT EXISTS idx_doc_folder_path ON documents (folder_path)",
        "CREATE INDEX IF NOT EXISTS idx_doc_folder_id ON documents (folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_documents_app_folder_path ON documents (app_id, folder_path)",
        "CREATE INDEX IF NOT EXISTS idx_documents_app_folder_id ON documents (app_id, folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_graph_folder_path ON graphs (folder_path)",
        "CREATE INDEX IF NOT EXISTS idx_graphs_app_folder_path ON graphs (app_id, folder_path)",
    ]

    backfill_statements = [
        # Backfill folders: derive canonical full_path from name when missing, then normalize slashes
        (
            "UPDATE folders SET full_path = '/' || TRIM(BOTH '/' FROM regexp_replace(name, '/+', '/', 'g')) "
            "WHERE COALESCE(full_path, '') = '' AND name IS NOT NULL AND name <> ''"
        ),
        "UPDATE folders SET full_path = '/' WHERE COALESCE(full_path, '') = '' AND (name IS NULL OR name = '')",
        (
            "UPDATE folders SET full_path = '/' || TRIM(BOTH '/' FROM regexp_replace(full_path, '/+', '/', 'g')) "
            "WHERE full_path IS NOT NULL AND full_path NOT IN ('', '/')"
        ),
        "UPDATE folders SET full_path = '/' WHERE full_path = '' OR full_path IS NULL",
        (
            "UPDATE folders SET depth = GREATEST("
            "array_length(string_to_array(trim(BOTH '/' from full_path), '/'), 1), 1)"
            " WHERE depth IS NULL AND full_path IS NOT NULL AND full_path <> '/'"
        ),
        "UPDATE folders SET depth = 0 WHERE depth IS NULL AND (full_path IS NULL OR full_path = '/')",
        (
            "UPDATE folders f SET parent_id = p.id "
            "FROM folders p "
            "WHERE f.parent_id IS NULL "
            "AND f.full_path IS NOT NULL AND f.full_path <> '/' "
            "AND p.full_path = regexp_replace(f.full_path, '/[^/]+$', '', 1, 1) "
            "AND ("
            "    (f.app_id IS NOT NULL AND p.app_id = f.app_id) "
            "    OR (f.app_id IS NULL AND p.app_id IS NULL AND f.owner_id IS NOT NULL AND p.owner_id = f.owner_id)"
            ")"
        ),
        # Mirror folder_name into folder_path for documents/graphs (canonical leading slash)
        (
            "UPDATE documents "
            "SET folder_path = '/' || folder_name "
            "WHERE (folder_path IS NULL OR folder_path = '') "
            "AND folder_name IS NOT NULL AND folder_name <> ''"
        ),
        (
            "UPDATE documents "
            "SET doc_metadata = jsonb_set(COALESCE(doc_metadata, '{}'::jsonb), '{folder_id}', to_jsonb(folder_id)) "
            "WHERE folder_id IS NOT NULL AND folder_id <> ''"
        ),
        (
            "UPDATE graphs "
            "SET folder_path = '/' || folder_name "
            "WHERE (folder_path IS NULL OR folder_path = '') "
            "AND folder_name IS NOT NULL AND folder_name <> ''"
        ),
    ]

    async def _run(statements, phase: str) -> None:
        for stmt in statements:
            summary = stmt.strip().split("\n", maxsplit=1)[0]
            try:
                async with engine.begin() as conn:
                    await conn.execute(text(stmt))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s step during %s: %s", summary, phase, exc)

    logger.info("Bootstrapping folder hierarchy columns, indexes, and backfill...")
    await _run(column_statements, "column creation")
    await _run(index_statements, "index creation")
    await _run(backfill_statements, "backfill")
    logger.info("Folder hierarchy bootstrap completed")
