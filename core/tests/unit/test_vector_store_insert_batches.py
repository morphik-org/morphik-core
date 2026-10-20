"""Bulk chunk inserts are split so one statement stays under asyncpg's 32,767-parameter limit."""

from contextlib import asynccontextmanager

import pytest

from core.models.chunk import DocumentChunk
from core.vector_store.pgvector_store import INSERT_ROWS_PER_STATEMENT, PGVectorStore

pytestmark = pytest.mark.unit


class _Session:
    def __init__(self):
        self.statements = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_large_documents_are_inserted_in_slices_in_one_transaction():
    store = object.__new__(PGVectorStore)
    session = _Session()

    @asynccontextmanager
    async def get_session():
        yield session

    store.get_session_with_retry = get_session
    # A large spreadsheet: 5 columns per row, so one VALUES list of 7000 rows needs 35,000 parameters.
    chunks = [
        DocumentChunk(document_id="sheet", chunk_number=i, content=f"row {i}", embedding=[0.1, 0.2], metadata={})
        for i in range(7000)
    ]

    ok, ids, _ = await store.store_embeddings(chunks)

    assert ok and len(ids) == 7000
    sizes = [len(statement._multi_values[0]) for statement in session.statements]
    assert sizes == [INSERT_ROWS_PER_STATEMENT] * 7
    assert max(sizes) * 5 < 32767
    assert session.commits == 1
