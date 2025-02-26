import pytest
import asyncio
import torch
import numpy as np
import psycopg
from pgvector.psycopg import Bit, register_vector
import logging
from core.vector_store.multi_vector_store import MultiVectorStore
from core.models.chunk import DocumentChunk

# Test database URI
TEST_DB_URI = "postgresql://postgres:postgres@localhost:5432/test_db"

logger = logging.getLogger(__name__)


# Sample test data
def get_sample_embeddings(num_vectors=3, dim=128):
    """Generate sample embeddings for testing"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create random embeddings with values between -1 and 1
    embeddings = torch.rand(num_vectors, dim, device=device) * 2 - 1
    return embeddings.cpu().numpy()


def get_sample_document_chunks(num_chunks=3, num_vectors=3, dim=128):
    """Create sample document chunks for testing"""
    chunks = []
    for i in range(num_chunks):
        embeddings = get_sample_embeddings(num_vectors, dim)
        chunk = DocumentChunk(
            document_id=f"doc_{i}",
            content=f"Test content {i}",
            embedding=embeddings,
            chunk_number=i,
            metadata={"test_key": f"test_value_{i}"},
        )
        chunks.append(chunk)
    return chunks


# Fixtures
@pytest.fixture(scope="function")
async def vector_store():
    """Create a real MultiVectorStore instance connected to the test database"""
    # Create the store
    store = MultiVectorStore(uri=TEST_DB_URI)

    try:
        # Try to initialize the database
        store.initialize()

        # Clean up any existing data
        store.conn.execute("TRUNCATE TABLE multi_vector_embeddings RESTART IDENTITY")

        # Drop the function if it exists
        try:
            store.conn.execute("DROP FUNCTION IF EXISTS max_sim(bit[], bit[])")
        except Exception as e:
            print(f"Error dropping function: {e}")
    except Exception as e:
        print(f"Error setting up database: {e}")

    yield store

    # Clean up after tests
    try:
        store.conn.execute("TRUNCATE TABLE multi_vector_embeddings RESTART IDENTITY")
    except Exception as e:
        print(f"Error cleaning up: {e}")

    # Close connection
    store.close()


# Glassbox Tests - Testing internal implementation details
@pytest.mark.asyncio
async def test_binary_quantize():
    """Test the _binary_quantize method correctly converts embeddings"""
    store = MultiVectorStore(uri=TEST_DB_URI)

    # Test with torch tensor
    torch_embeddings = torch.tensor([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    binary_result = store._binary_quantize(torch_embeddings)
    assert len(binary_result) == 2

    # Check results match expected patterns
    assert (
        binary_result[0].to_text() == Bit("101").to_text()
    )  # Positive values (>0) become 1, negative/zero become 0
    assert (
        binary_result[1].to_text() == Bit("010").to_text()
    )  # First row: [0.1 (>0), -0.2 (<0), 0.3 (>0)] → "101"
    # Second row: [-0.1 (<0), 0.2 (>0), -0.3 (<0)] → "010"

    # Test with numpy array
    numpy_embeddings = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    binary_result = store._binary_quantize(numpy_embeddings)
    assert len(binary_result) == 2

    assert binary_result[0].to_text() == Bit("101").to_text()
    assert binary_result[1].to_text() == Bit("010").to_text()


@pytest.mark.asyncio
async def test_initialize_creates_tables_and_function(vector_store):
    """Test that initialize creates the necessary tables and functions"""
    vector_store.initialize()
    # Check if the table exists
    result = vector_store.conn.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'multi_vector_embeddings')"
    ).fetchone()
    table_exists = result[0]
    assert table_exists is True

    logger.info("Table exists!")

    # Check if the max_sim function exists
    result = vector_store.conn.execute(
        "SELECT EXISTS (SELECT FROM pg_proc WHERE proname = 'max_sim')"
    ).fetchone()
    function_exists = result[0]
    logger.info(f"Function exists {function_exists}")
    assert function_exists is True


@pytest.mark.asyncio
async def test_database_schema(vector_store):
    """Test that the database schema matches our expectations"""
    # Check columns in the table
    result = vector_store.conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'multi_vector_embeddings'"
    ).fetchall()

    # Convert to a dict for easier checking
    column_dict = {col[0]: col[1] for col in result}

    # Check required columns
    assert "id" in column_dict
    assert "document_id" in column_dict
    assert "chunk_number" in column_dict
    assert "content" in column_dict
    assert "embeddings" in column_dict


# Blackbox Tests - Testing the public API
@pytest.mark.asyncio
async def test_store_and_query_embeddings(vector_store):
    """End-to-end test of storing and querying embeddings"""
    vector_store.initialize()
    # Create test data
    chunks = get_sample_document_chunks(num_chunks=5, num_vectors=3, dim=128)

    # Store the embeddings
    result, stored_ids = await vector_store.store_embeddings(chunks)

    # Verify storage was successful
    assert result is True
    assert len(stored_ids) == 5

    # Create a query embedding (use the first embedding from the first chunk)
    query_embedding = chunks[0].embedding

    # Query for similar chunks
    results = await vector_store.query_similar(query_embedding, k=3)

    # Verify we got results
    assert len(results) > 0
    assert isinstance(results[0], DocumentChunk)

    # The first result should be the chunk we queried with
    assert results[0].document_id == chunks[0].document_id

    # Check that scores are in descending order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@pytest.mark.asyncio
async def test_query_with_doc_ids(vector_store):
    """Test querying with document ID filtering"""
    vector_store.initialize()
    # Create test data
    chunks = get_sample_document_chunks(num_chunks=5, num_vectors=3, dim=128)

    # Store the embeddings
    await vector_store.store_embeddings(chunks)

    # Create a query embedding
    query_embedding = get_sample_embeddings(1, 128)  # Just one vector

    # Query with specific doc_ids
    doc_ids = ["doc_1", "doc_3"]
    results = await vector_store.query_similar(query_embedding, k=5, doc_ids=doc_ids)

    # Verify results only include the specified doc_ids
    assert all(result.document_id in doc_ids for result in results)


@pytest.mark.asyncio
async def test_store_embeddings_empty(vector_store):
    """Test storing empty embeddings list"""
    vector_store.initialize()
    result, stored_ids = await vector_store.store_embeddings([])
    assert result is True
    assert stored_ids == []


@pytest.mark.asyncio
async def test_multi_vector_similarity(vector_store):
    """Test that multi-vector similarity works as expected"""
    vector_store.initialize()
    # Create chunks with very specific embeddings to test similarity
    chunks = []

    # First chunk: 3 vectors that are all positive in the first half, negative in second half
    embedding1 = np.ones((3, 128))
    embedding1[:, 64:] = -1
    chunk1 = DocumentChunk(
        document_id="similarity_test_1",
        content="Similarity test content 1",
        embedding=embedding1,
        chunk_number=1,
    )
    chunks.append(chunk1)

    # Second chunk: 3 vectors that are all negative in the first half, positive in second half
    embedding2 = -np.ones((3, 128))
    embedding2[:, 64:] = 1
    chunk2 = DocumentChunk(
        document_id="similarity_test_2",
        content="Similarity test content 2",
        embedding=embedding2,
        chunk_number=2,
    )
    chunks.append(chunk2)

    # Store the embeddings
    await vector_store.store_embeddings(chunks)

    # Create a query embedding that matches the pattern of the first chunk
    query_embedding = np.ones(128)
    query_embedding[64:] = -1

    # Query for similar chunks
    results = await vector_store.query_similar(np.array([query_embedding]), k=2)

    # The first result should be chunk1
    assert len(results) == 2
    assert results[0].document_id == "similarity_test_1"
    assert results[1].document_id == "similarity_test_2"


@pytest.mark.asyncio
async def test_store_and_retrieve_metadata(vector_store):
    """Test that metadata is correctly stored and retrieved"""
    vector_store.initialize()
    # Create a chunk with complex metadata
    complex_metadata = {
        "filename": "test.pdf",
        "page": 5,
        "tags": ["important", "review"],
        "nested": {"key1": "value1", "key2": 123},
    }

    embeddings = get_sample_embeddings(3, 128)
    chunk = DocumentChunk(
        document_id="metadata_test",
        content="Metadata test content",
        embedding=embeddings,
        chunk_number=1,
        metadata=complex_metadata,
    )

    # Store the chunk
    await vector_store.store_embeddings([chunk])

    # Query to retrieve the chunk
    query_embedding = get_sample_embeddings(1, 128)
    results = await vector_store.query_similar(query_embedding, k=1, doc_ids=["metadata_test"])

    # Verify metadata was preserved
    assert len(results) == 1
    retrieved_metadata = results[0].metadata
    assert retrieved_metadata["filename"] == "test.pdf"
    assert retrieved_metadata["page"] == 5
    assert "tags" in retrieved_metadata
    assert "nested" in retrieved_metadata
    assert retrieved_metadata["nested"]["key1"] == "value1"


@pytest.mark.asyncio
async def test_performance(vector_store):
    """Test performance with a larger number of chunks"""
    vector_store.initialize()
    # Create a larger set of chunks
    num_chunks = 20
    chunks = get_sample_document_chunks(num_chunks=num_chunks, num_vectors=3, dim=128)

    # Measure time to store embeddings
    start_time = asyncio.get_event_loop().time()
    await vector_store.store_embeddings(chunks)
    storage_time = asyncio.get_event_loop().time() - start_time

    # Measure time to query
    query_embedding = get_sample_embeddings(1, 128)
    start_time = asyncio.get_event_loop().time()
    results = await vector_store.query_similar(query_embedding, k=5)
    query_time = asyncio.get_event_loop().time() - start_time

    # Log performance metrics
    print(f"Storage time for {num_chunks} chunks: {storage_time:.4f} seconds")
    print(f"Query time for top 5 results: {query_time:.4f} seconds")

    # Basic assertions to ensure the test ran
    assert len(results) > 0
