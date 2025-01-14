import pytest
from pathlib import Path
import tempfile
from typing import List

from core.cache.base_cache import BaseCache
from core.cache.hf_cache import HuggingFaceCache
from core.models.completion import CompletionRequest, CompletionResponse


@pytest.fixture
def sample_docs() -> List[str]:
    return [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Data science combines statistics and programming.",
    ]


@pytest.fixture
def cache_dir():
    """Fixture to create and clean up a temporary directory for cache files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def model_params(request):
    """Fixture to provide different model configurations for testing"""
    models = {
        "small": {
            "name": "distilgpt2",
            "max_tokens": 100,
        },
        "medium": {
            "name": "facebook/opt-125m",
            "max_tokens": 100,
        },
        "decoder": {
            "name": "gpt2",
            "max_tokens": 100,
        },
        "llama": {
            "name": "meta-llama/Llama-3.2-1B-Instruct",
            "max_tokens": 100,
        },
    }
    return models[request.param]


@pytest.fixture
async def cache(cache_dir, model_params) -> BaseCache:
    """Fixture to create and reuse a cache instance"""
    cache = HuggingFaceCache(
        cache_path=cache_dir,
        model_name=model_params["name"],
        device="cpu",
        default_max_new_tokens=model_params["max_tokens"],
        use_fp16=False,
    )
    return cache


# Basic test with default model
@pytest.mark.parametrize("model_params", ["small", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_basic(cache: BaseCache, sample_docs: List[str]):
    """Test basic cache operations with default model"""
    # Test ingestion
    success = await cache.ingest(sample_docs)
    assert success, "Document ingestion should succeed"

    # Test completion
    request = CompletionRequest(
        query="What is Python?", context_chunks=sample_docs, max_tokens=100, temperature=0.7
    )
    response = await cache.complete(request)
    assert isinstance(response.completion, str)
    assert len(response.completion) > 0


# Test with different model architectures
@pytest.mark.parametrize("model_params", ["small", "medium", "decoder", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_model_compatibility(cache: BaseCache, sample_docs: List[str], model_params):
    """Test cache compatibility with different model architectures"""
    print(f"\n=== Testing with model: {model_params['name']} ===")

    # Test ingestion
    success = await cache.ingest(sample_docs)
    assert success, f"Document ingestion should succeed with {model_params['name']}"
    print(f"Ingestion successful with {model_params['name']}")

    # Test completion
    request = CompletionRequest(
        query="What is Python?",
        context_chunks=sample_docs,
        max_tokens=model_params["max_tokens"],
        temperature=0.7,
    )
    response = await cache.complete(request)
    assert isinstance(response.completion, str)
    assert len(response.completion) > 0
    print(f"Generated completion with {model_params['name']}: {response.completion[:100]}...")


@pytest.mark.parametrize("model_params", ["small", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_ingest(cache: BaseCache, sample_docs: List[str]):
    """Test document ingestion into cache"""
    print("\n=== Testing Cache Ingestion ===")

    # Test ingesting documents
    success = await cache.ingest(sample_docs)
    assert success, "Document ingestion should succeed"
    print("Basic ingestion test passed")

    # Test ingesting empty list
    success = await cache.ingest([])
    assert success, "Empty document ingestion should succeed"
    print("Empty ingestion test passed")


@pytest.mark.parametrize("model_params", ["small", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_update(cache: BaseCache, sample_docs: List[str]):
    """Test updating cache with new documents"""
    print("\n=== Testing Cache Update ===")

    # Initialize cache with documents
    await cache.ingest(sample_docs)

    # Test updating with new document
    new_doc = "Natural Language Processing is used to understand text."
    success = await cache.update(new_doc)
    assert success, "Cache update should succeed"
    print("Update test passed")


@pytest.mark.parametrize("model_params", ["small", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_completion(cache: BaseCache, sample_docs: List[str]):
    """Test completion generation using cached context"""
    print("\n=== Testing Cache Completion ===")

    # Initialize cache
    await cache.ingest(sample_docs)

    # Test completion
    request = CompletionRequest(
        query="What is Python?", context_chunks=sample_docs, max_tokens=100, temperature=0.7
    )
    response = await cache.complete(request)
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.completion, str)
    assert len(response.completion) > 0
    print(f"Generated completion: {response.completion}")


@pytest.mark.parametrize("model_params", ["small", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_save_load(cache: BaseCache, cache_dir: Path, sample_docs: List[str]):
    """Test saving and loading cache state"""
    print("\n=== Testing Cache Save/Load ===")

    # First ingest some documents
    success = await cache.ingest(sample_docs)
    assert success, "Document ingestion should succeed"
    print("Documents ingested successfully")

    # Save cache
    cache_path = cache.save_cache()
    assert cache_path.exists(), "Cache file should exist after saving"
    print(f"Cache saved to: {cache_path}")

    # Load cache
    cache.load_cache(cache_path)
    print("Cache loaded successfully")

    # Test completion after loading to verify cache is working
    request = CompletionRequest(
        query="What is Python?", context_chunks=sample_docs, max_tokens=100, temperature=0.7
    )
    response = await cache.complete(request)
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.completion, str)
    assert len(response.completion) > 0
    print("Cache working correctly after loading")


@pytest.mark.parametrize("model_params", ["small", "llama"], indirect=True)
@pytest.mark.asyncio
async def test_cache_error_handling(cache: BaseCache):
    """Test error handling in cache operations"""
    print("\n=== Testing Error Handling ===")

    # Test completion without initialization
    request = CompletionRequest(
        query="What is AI?", context_chunks=[], max_tokens=100, temperature=0.7
    )
    response = await cache.complete(request)
    assert isinstance(response, CompletionResponse)
    assert "Error" in response.completion
    print("Error handling for uninitialized cache passed")

    # Test loading non-existent cache
    with pytest.raises(FileNotFoundError):
        cache.load_cache(Path("nonexistent_cache.pt"))
    print("Error handling for invalid cache file passed")
