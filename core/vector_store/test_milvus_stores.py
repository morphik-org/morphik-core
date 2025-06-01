"""
Test script to verify Milvus vector store implementations.
This is a simple compatibility test - not meant to be run as part of the test suite.
"""

import asyncio

import numpy as np
from milvus_multi_vector_store import MilvusMultiVectorStore
from milvus_vector_store import MilvusVectorStore

from core.models.chunk import DocumentChunk


async def test_basic_vector_store():
    """Test basic Milvus vector store functionality."""
    print("Testing MilvusVectorStore...")

    store = MilvusVectorStore(uri="test_milvus.db", collection_name="test_collection", dimension=384)

    # Create test chunks
    test_chunks = [
        DocumentChunk(
            document_id="doc1",
            chunk_number=0,
            content="This is a test document about AI.",
            embedding=np.random.rand(384).tolist(),
            metadata={"source": "test"},
        ),
        DocumentChunk(
            document_id="doc1",
            chunk_number=1,
            content="This document discusses machine learning.",
            embedding=np.random.rand(384).tolist(),
            metadata={"source": "test"},
        ),
    ]

    # Test storage
    success, ids = await store.store_embeddings(test_chunks)
    print(f"Storage success: {success}, IDs: {ids}")

    # Test similarity search
    query_embedding = np.random.rand(384).tolist()
    results = await store.query_similar(query_embedding, k=2)
    print(f"Found {len(results)} similar chunks")

    # Test retrieval by ID
    chunk_ids = [("doc1", 0), ("doc1", 1)]
    retrieved = await store.get_chunks_by_id(chunk_ids)
    print(f"Retrieved {len(retrieved)} chunks by ID")

    # Test deletion
    deleted = await store.delete_chunks_by_document_id("doc1")
    print(f"Deletion success: {deleted}")

    store.close()
    print("MilvusVectorStore test completed!\n")


async def test_multi_vector_store():
    """Test Milvus multi-vector store functionality."""
    print("Testing MilvusMultiVectorStore...")

    store = MilvusMultiVectorStore(uri="test_milvus_multi.db", collection_name="test_multi_collection", dimension=128)

    # Create test chunks with multi-vector embeddings
    test_chunks = [
        DocumentChunk(
            document_id="doc2",
            chunk_number=0,
            content="This is a test document about AI.",
            embedding=np.random.rand(5, 128),  # 5 vectors of 128 dimensions each
            metadata={"source": "test"},
        ),
        DocumentChunk(
            document_id="doc2",
            chunk_number=1,
            content="This document discusses machine learning.",
            embedding=np.random.rand(3, 128),  # 3 vectors of 128 dimensions each
            metadata={"source": "test"},
        ),
    ]

    # Test storage
    success, ids = await store.store_embeddings(test_chunks)
    print(f"Multi-vector storage success: {success}, IDs: {ids}")

    # Test similarity search with multi-vector query
    query_embedding = np.random.rand(2, 128)  # 2 query vectors
    results = await store.query_similar(query_embedding, k=2)
    print(f"Found {len(results)} similar multi-vector chunks")

    # Test retrieval by ID
    chunk_ids = [("doc2", 0), ("doc2", 1)]
    retrieved = await store.get_chunks_by_id(chunk_ids)
    print(f"Retrieved {len(retrieved)} multi-vector chunks by ID")

    # Test deletion
    deleted = await store.delete_chunks_by_document_id("doc2")
    print(f"Multi-vector deletion success: {deleted}")

    store.close()
    print("MilvusMultiVectorStore test completed!\n")


async def main():
    """Run all tests."""
    print("Starting Milvus vector store compatibility tests...\n")

    try:
        await test_basic_vector_store()
        await test_multi_vector_store()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
