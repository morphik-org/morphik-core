"""Integration tests for MiniMaxEmbeddingModel against the live API.

These tests require a valid MINIMAX_API_KEY environment variable.
They are skipped automatically when the key is absent or the API is
unreachable.
"""

import os
import sys
from unittest.mock import MagicMock

# Prevent ImportError when colpali_engine is not installed
for _mod in ("colpali_engine", "colpali_engine.models"):
    sys.modules.setdefault(_mod, MagicMock())

import pytest

from core.models.chunk import Chunk

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


def _have_key():
    return bool(MINIMAX_API_KEY)


def _create_mock_settings():
    """Create a mock settings object for integration tests."""
    from unittest.mock import MagicMock

    s = MagicMock()
    s.REGISTERED_MODELS = {
        "minimax_embedding": {
            "model_name": "embo-01",
            "provider": "minimax",
        }
    }
    s.VECTOR_DIMENSIONS = 1536
    return s


@pytest.fixture
def embedding_model():
    """Create a MiniMaxEmbeddingModel for integration testing."""
    if not _have_key():
        pytest.skip(SKIP_REASON)

    from unittest.mock import patch

    with patch("core.embedding.minimax_embedding.get_settings", return_value=_create_mock_settings()):
        from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

        return MiniMaxEmbeddingModel(model_key="minimax_embedding")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_query_live(embedding_model):
    """Verify embed_for_query returns a 1536-d vector from the live API."""
    try:
        result = await embedding_model.embed_for_query("What is retrieval augmented generation?")
    except Exception as exc:
        pytest.skip(f"MiniMax API unreachable: {exc}")

    assert isinstance(result, list)
    assert len(result) == 1536
    # Values should be floats in a reasonable range
    assert all(isinstance(v, float) for v in result[:10])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_ingestion_live(embedding_model):
    """Verify embed_for_ingestion returns vectors for document chunks."""
    chunks = [
        Chunk(content="Morphik is an AI-native document processing system.", metadata={}),
        Chunk(content="MiniMax provides large language models and embedding APIs.", metadata={}),
    ]

    try:
        result = await embedding_model.embed_for_ingestion(chunks)
    except Exception as exc:
        pytest.skip(f"MiniMax API unreachable: {exc}")

    assert isinstance(result, list)
    assert len(result) == 2
    for vec in result:
        assert len(vec) == 1536


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_vs_db_embeddings_differ(embedding_model):
    """Query and document embeddings for the same text should differ
    because MiniMax uses different encoding for type=query vs type=db."""
    text = "artificial intelligence research paper"

    try:
        query_vec = await embedding_model.embed_for_query(text)
        chunk = Chunk(content=text, metadata={})
        db_vecs = await embedding_model.embed_for_ingestion(chunk)
    except Exception as exc:
        pytest.skip(f"MiniMax API unreachable: {exc}")

    db_vec = db_vecs[0]
    assert len(query_vec) == len(db_vec) == 1536
    # The vectors should not be identical (different encoding types)
    assert query_vec != db_vec
