#!/usr/bin/env python3
# Usage: cd $(dirname "$0")/.. && PYTHONPATH=. python3 $0

import asyncio
import json

import psycopg
from pgvector.psycopg import register_vector

from core.config import get_settings
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.models.chunk import Chunk
from core.vector_store.multi_vector_store import MultiVectorStore


async def migrate_multivector_embeddings():
    settings = get_settings()
    uri = settings.POSTGRES_URI
    # Convert SQLAlchemy URI to psycopg format if needed
    if uri.startswith("postgresql+asyncpg://"):
        uri = uri.replace("postgresql+asyncpg://", "postgresql://")
    mv_store = MultiVectorStore(uri)
    if not mv_store.initialize():
        print("Failed to initialize MultiVectorStore")
        return

    embedding_model = ColpaliEmbeddingModel()

    conn = psycopg.connect(uri, autocommit=True)
    register_vector(conn)
    cursor = conn.cursor()

    cursor.execute("SELECT id, document_id, chunk_number, content, chunk_metadata " "FROM multi_vector_embeddings")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Found {total} multivector records to migrate...")

    for idx, (row_id, doc_id, chunk_num, content, meta_json) in enumerate(rows, start=1):
        try:
            metadata = json.loads(meta_json) if meta_json else {}
            chunk = Chunk(content=content, metadata=metadata)
            vectors = await embedding_model.embed_for_ingestion([chunk])
            vector = vectors[0]
            bits = mv_store._binary_quantize(vector)
            cursor.execute(
                "UPDATE multi_vector_embeddings SET embeddings = %s WHERE id = %s",
                (bits, row_id),
            )
            print(f"[{idx}/{total}] Updated doc={doc_id} chunk={chunk_num}")
        except Exception as e:
            print(f"Error migrating row {row_id} (doc={doc_id}, chunk={chunk_num}): {e}")

    cursor.close()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    asyncio.run(migrate_multivector_embeddings())
