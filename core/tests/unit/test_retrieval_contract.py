"""Contract tests for retrieval thresholds used by on-prem callers."""

import os

os.environ.setdefault("POSTGRES_URI", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from core.models.chunk import DocumentChunk  # noqa: E402
from core.services.document_service import _filter_chunks_by_min_score  # noqa: E402


def _chunk(document_id: str, score: float) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        content=document_id,
        embedding=[],
        chunk_number=0,
        score=score,
    )


def test_min_score_zero_keeps_zero_and_positive_scores():
    chunks = [_chunk("negative", -0.01), _chunk("zero", 0.0), _chunk("positive", 0.75)]

    filtered = _filter_chunks_by_min_score(chunks, 0.0)

    assert [chunk.document_id for chunk in filtered] == ["zero", "positive"]


def test_min_score_filters_on_the_final_score():
    chunks = [_chunk("below", 0.49), _chunk("equal", 0.5), _chunk("above", 0.9)]

    filtered = _filter_chunks_by_min_score(chunks, 0.5)

    assert [chunk.document_id for chunk in filtered] == ["equal", "above"]
