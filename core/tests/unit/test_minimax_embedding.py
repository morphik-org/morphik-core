"""Unit tests for MiniMaxEmbeddingModel.

These tests mock the HTTP layer so they run without a real API key.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Prevent ImportError when colpali_engine is not installed
for _mod in ("colpali_engine", "colpali_engine.models"):
    sys.modules.setdefault(_mod, MagicMock())

import pytest

from core.models.chunk import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Return a mock Settings object with sensible defaults."""
    defaults = {
        "REGISTERED_MODELS": {
            "minimax_embedding": {
                "model_name": "embo-01",
                "provider": "minimax",
            }
        },
        "VECTOR_DIMENSIONS": 1536,
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _fake_response(vectors, status_code=200):
    """Build a fake httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "vectors": vectors,
        "total_tokens": len(vectors) * 10,
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    resp.raise_for_status = MagicMock()
    return resp


def _create_chunks(n=3):
    return [Chunk(content=f"chunk text {i}", metadata={"idx": i}) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key-123"})
async def test_init_loads_config(mock_settings):
    """Model initialises correctly from registered_models config."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    assert model.model_name == "embo-01"
    assert model.api_key == "test-key-123"
    assert model.dimensions == 1536


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": ""})
async def test_init_raises_without_api_key(mock_settings):
    """Should raise ValueError when MINIMAX_API_KEY is not set."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        MiniMaxEmbeddingModel(model_key="minimax_embedding")


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_init_raises_unknown_model_key(mock_settings):
    """Should raise ValueError for unregistered model key."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    with pytest.raises(ValueError, match="not found"):
        MiniMaxEmbeddingModel(model_key="nonexistent_model")


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_embed_for_query(mock_settings):
    """embed_for_query returns a single vector with correct dimensions."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    fake_vec = [[0.1] * 1536]

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = _fake_response(fake_vec)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await model.embed_for_query("test query")

    assert isinstance(result, list)
    assert len(result) == 1536
    assert result == [0.1] * 1536

    # Verify the API was called with type="query"
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["type"] == "query"
    assert payload["texts"] == ["test query"]


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_embed_for_query_empty_returns_zero_vector(mock_settings):
    """embed_for_query returns zero vector when API returns empty vectors."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = _fake_response([])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await model.embed_for_query("test")

    assert result == [0.0] * 1536


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_embed_for_ingestion_single_chunk(mock_settings):
    """embed_for_ingestion handles a single Chunk (not wrapped in list)."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    fake_vec = [[0.5] * 1536]

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = _fake_response(fake_vec)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        chunk = Chunk(content="hello world", metadata={})
        result = await model.embed_for_ingestion(chunk)

    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0]) == 1536

    # Verify type="db"
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["type"] == "db"


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_embed_for_ingestion_multiple_chunks(mock_settings):
    """embed_for_ingestion handles a list of chunks."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    chunks = _create_chunks(3)
    fake_vecs = [[0.1 * i] * 1536 for i in range(3)]

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = _fake_response(fake_vecs)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await model.embed_for_ingestion(chunks)

    assert len(result) == 3
    for vec in result:
        assert len(vec) == 1536


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_embed_for_ingestion_empty_list(mock_settings):
    """embed_for_ingestion returns empty list for empty input."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    result = await model.embed_for_ingestion([])
    assert result == []


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_embed_batching(mock_settings):
    """embed_for_ingestion batches texts in groups of 50."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    chunks = _create_chunks(75)  # Should produce 2 batches: 50 + 25

    batch1_vecs = [[0.1] * 1536] * 50
    batch2_vecs = [[0.2] * 1536] * 25

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            _fake_response(batch1_vecs),
            _fake_response(batch2_vecs),
        ]
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await model.embed_for_ingestion(chunks)

    assert len(result) == 75
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_api_error_response(mock_settings):
    """_embed raises ValueError when API returns error (no vectors key)."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")

    error_resp = MagicMock()
    error_resp.status_code = 200
    error_resp.json.return_value = {
        "base_resp": {"status_code": 1001, "status_msg": "invalid api key"},
    }
    error_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = error_resp
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="MiniMax embedding API error"):
            await model.embed_for_query("test")


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_dimensions_capped_at_2000(mock_settings):
    """Dimensions should be capped at 2000 even if VECTOR_DIMENSIONS is larger."""
    mock_settings.return_value = _make_settings(VECTOR_DIMENSIONS=4096)
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    assert model.dimensions == 2000


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_custom_api_base(mock_settings):
    """Model should respect custom api_base from config."""
    mock_settings.return_value = _make_settings(
        REGISTERED_MODELS={
            "minimax_embedding": {
                "model_name": "embo-01",
                "provider": "minimax",
                "api_base": "https://custom.api.com/v1/",
            }
        }
    )
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")
    assert model.api_base == "https://custom.api.com/v1"  # trailing slash stripped


@pytest.mark.asyncio
@patch("core.embedding.minimax_embedding.get_settings")
@patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})
async def test_authorization_header(mock_settings):
    """API requests should include correct Authorization header."""
    mock_settings.return_value = _make_settings()
    from core.embedding.minimax_embedding import MiniMaxEmbeddingModel

    model = MiniMaxEmbeddingModel(model_key="minimax_embedding")

    with patch("httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.post.return_value = _fake_response([[0.1] * 1536])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        await model.embed_for_query("test")

    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    assert headers["Authorization"] == "Bearer test-key"
