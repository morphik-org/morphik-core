import pytest
import base64
import io
import numpy as np
from PIL import Image

from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.models.chunk import Chunk


# Helper functions
def create_sample_image():
    """Create a small sample image for testing"""
    # Create a 10x10 RGB image
    img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)

    # Convert to base64 encoded string
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    img_b64 = base64.b64encode(img_byte_arr).decode("utf-8")

    return img_b64


def create_sample_chunks(num_chunks=2):
    """Create sample chunks with base64-encoded images"""
    chunks = []
    for i in range(num_chunks):
        img_b64 = create_sample_image()
        chunk = Chunk(content=img_b64, metadata={"filename": f"test_image_{i}.png"})
        chunks.append(chunk)
    return chunks


@pytest.fixture
def embedding_model():
    """Create an instance of the ColpaliEmbeddingModel"""
    model = ColpaliEmbeddingModel()
    return model


# Tests
@pytest.mark.asyncio
async def test_embed_for_query(embedding_model):
    """Test embedding a text query"""
    # Test with a simple query
    query = "Find images similar to this concept"

    # Get embeddings
    result = await embedding_model.embed_for_query(query)

    # Check results
    assert isinstance(result, np.ndarray)
    # The first dimension can vary based on the number of tokens
    assert result.shape[1] == 128
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_embed_for_ingestion_single_chunk(embedding_model):
    """Test embedding a single chunk with an image"""
    # Create a single test chunk
    chunk = create_sample_chunks(num_chunks=1)[0]

    # Test embedding for ingestion
    result = await embedding_model.embed_for_ingestion(chunk)

    # Check results
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    # The first dimension can vary based on the image content
    assert result[0].shape[1] == 128
    assert result[0].dtype == np.float32


@pytest.mark.asyncio
async def test_embed_for_ingestion_multiple_chunks(embedding_model):
    """Test embedding multiple chunks with images"""
    # Create multiple test chunks
    chunks = create_sample_chunks(num_chunks=3)

    # Test embedding for ingestion
    result = await embedding_model.embed_for_ingestion(chunks)

    # Check results
    assert isinstance(result, list)
    assert len(result) == 3

    for emb in result:
        assert isinstance(emb, np.ndarray)
        # The first dimension can vary based on the image content
        # assert emb.shape[1] == 128
        print(emb.shape)
        assert emb.dtype == np.float32


@pytest.mark.asyncio
async def test_embed_for_query_complex(embedding_model):
    """Test embedding a more complex text query"""
    # Test with a longer, more complex query
    query = "Find images that contain diagrams of electronic circuits with resistors and capacitors"

    # Get embeddings
    result = await embedding_model.embed_for_query(query)

    # Check results
    assert isinstance(result, np.ndarray)
    # The first dimension can vary based on the number of tokens
    assert result.shape[1] == 128
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_generate_embeddings_directly(embedding_model):
    """Test the generate_embeddings method directly with text"""
    # Test text
    text = "Test query for direct embedding generation"

    # Generate embeddings
    result = embedding_model.generate_embeddings(text)

    # Check results
    assert isinstance(result, np.ndarray)
    # The first dimension can vary based on the number of tokens
    assert result.shape[1] == 128
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_different_query_lengths(embedding_model):
    """Test that different query lengths produce consistent embedding dimensions"""
    # Test with queries of different lengths
    queries = [
        "Short",
        "A slightly longer query",
        "This is a much longer query that should have many more tokens when processed by the model",
    ]

    results = []
    for query in queries:
        result = await embedding_model.embed_for_query(query)
        results.append(result)

    # Check all results have the same second dimension (128)
    for result in results:
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 128
        assert result.dtype == np.float32
