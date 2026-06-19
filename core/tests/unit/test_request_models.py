"""Unit tests for API request models."""

import pytest
from pydantic import ValidationError

from core.models.request import IngestTextRequest


class TestIngestTextRequest:
    """Tests for text ingestion request validation."""

    @pytest.mark.parametrize("content", ["", "   ", "\n\t"])
    def test_rejects_empty_content(self, content):
        with pytest.raises(ValidationError):
            IngestTextRequest(content=content)

    def test_accepts_non_empty_content(self):
        request = IngestTextRequest(content="hello world")

        assert request.content == "hello world"
