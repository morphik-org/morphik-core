"""Inputs over the OpenAI per-input and per-request token limits no longer fail the whole batch."""

import pytest
import tiktoken

from core.embedding import litellm_embedding
from core.embedding.litellm_embedding import (
    OPENAI_EMBEDDING_MAX_REQUEST_TOKENS,
    OPENAI_EMBEDDING_MAX_TOKENS,
    LiteLLMEmbeddingModel,
)

pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, n: int):
        self.data = [{"embedding": [0.0] * 4} for _ in range(n)]


def _make_model(model_name: str, encoding) -> LiteLLMEmbeddingModel:
    model = object.__new__(LiteLLMEmbeddingModel)
    model.model_key = "test"
    model.model_config = {"model_name": model_name}
    model.dimensions = 4
    model._is_local_provider = False
    model._encoding = encoding
    return model


@pytest.fixture
def captured(monkeypatch):
    calls = []

    async def fake_aembedding(input, **kwargs):
        calls.append(input)
        return _Response(len(input))

    monkeypatch.setattr(litellm_embedding.litellm, "aembedding", fake_aembedding)
    return calls


@pytest.mark.asyncio
async def test_cjk_chunk_over_limit_is_truncated(captured):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    model = _make_model("text-embedding-3-small", encoding)
    # 6000 varied CJK characters: within a 6000-character chunk but over 8192 tokens,
    # since less common ideographs cost 2+ tokens each.
    cjk = "".join(chr(0x4E00 + (i * 37) % 20000) for i in range(6000))
    assert len(encoding.encode(cjk)) > OPENAI_EMBEDDING_MAX_TOKENS

    embeddings = await model.embed_documents(["short text", cjk])

    assert len(embeddings) == 2
    sent = captured[0]
    assert sent[0] == "short text"
    assert len(encoding.encode(sent[1])) <= OPENAI_EMBEDDING_MAX_TOKENS
    assert cjk.startswith(sent[1])


@pytest.mark.asyncio
async def test_inputs_under_limit_are_unchanged(captured):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    model = _make_model("text-embedding-3-small", encoding)
    texts = ["hello world", "a" * 5000, "测试" * 100]

    await model.embed_documents(texts)

    assert captured[0] == texts


@pytest.mark.asyncio
async def test_models_without_known_tokenizer_are_sent_as_is(captured):
    model = _make_model("ollama/nomic-embed-text", None)
    long_text = "测试" * 6000

    await model.embed_documents([long_text])

    assert captured[0] == [long_text]


@pytest.mark.asyncio
async def test_batches_over_the_request_limit_are_split(captured):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    model = _make_model("text-embedding-3-small", encoding)
    # Spreadsheet-like rows: 100 chunks of ~6000 characters that each cost several thousand tokens,
    # together well over the 300,000-token request limit.
    row = "| 2025-04-22 | 7781.25 | 0.0432 | PHL-TQC | 15:01:44 |\n"
    chunks = [(f"{i} " + row * 120)[:6000] for i in range(100)]
    total = sum(len(encoding.encode(c)) for c in chunks)
    assert total > 300_000

    embeddings = await model.embed_documents(chunks)

    assert len(embeddings) == 100
    assert len(captured) > 1
    assert [text for request in captured for text in request] == chunks
    for request in captured:
        assert sum(len(encoding.encode(text)) for text in request) <= OPENAI_EMBEDDING_MAX_REQUEST_TOKENS


@pytest.mark.asyncio
async def test_small_batches_stay_one_request(captured):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    model = _make_model("text-embedding-3-small", encoding)

    await model.embed_documents(["first", "second", "third"])

    assert captured == [["first", "second", "third"]]


def test_encoding_resolves_for_provider_prefixed_openai_models():
    assert tiktoken.encoding_for_model("azure/text-embedding-3-large".split("/")[-1]) is not None
