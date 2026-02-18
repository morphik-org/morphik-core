"""
Unit tests for LiteLLM embedding error logging diagnostics.

Tests verify that when litellm.aembedding raises an exception:
- The error log includes diagnostic context, such as model_key, model_name, api_base, num_texts
- The original exception type is re-raised unchanged
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Marker for error log identification matching exception text in litellm_embedding.py
ERROR_LOG_MARKER = "Error generating embeddings with LiteLLM:"


def find_error_log_record(caplog_records, marker=ERROR_LOG_MARKER):
    """
    Find the ERROR log record containing the marker substring.

    Filters by:
    - Log level == ERROR
    - Message contains the marker substring

    Returns the first matching record, or None if not found.
    """
    for record in caplog_records:
        if record.levelno == logging.ERROR and marker in record.message:
            return record
    return None


class TestLiteLLMEmbeddingErrorLogging:
    """Test improved error logging in LiteLLMEmbeddingModel.embed_documents."""

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object with required attributes."""
        settings = MagicMock()
        settings.REGISTERED_MODELS = {
            "test_embedding": {
                "model_name": "ollama/nomic-embed-text",
                "api_base": "http://localhost:11434",
            }
        }
        settings.VECTOR_DIMENSIONS = 768
        settings.LITELLM_DUMMY_API_KEY = "test-dummy-key"
        return settings

    @pytest.mark.asyncio
    async def test_embed_documents_logs_context_on_connection_error(self, mock_settings, caplog):
        """
        Test that when litellm.aembedding raises ConnectionError,
        the error log includes model_key, model_name, api_base, and num_texts.
        """
        with patch("core.embedding.litellm_embedding.get_settings", return_value=mock_settings):
            from core.embedding.litellm_embedding import LiteLLMEmbeddingModel

            model = LiteLLMEmbeddingModel("test_embedding")

            error_message = "Connection refused to http://localhost:11434"
            connection_error = ConnectionError(error_message)

            with patch(
                "core.embedding.litellm_embedding.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.side_effect = connection_error

                with caplog.at_level(logging.ERROR, logger="core.embedding.litellm_embedding"):
                    with pytest.raises(ConnectionError) as exc_info:
                        await model.embed_documents(["test text 1", "test text 2"])

                    # Verify exception type preserved and message matches
                    assert error_message in str(exc_info.value)

                    # Find the ERROR log record with our diagnostic marker
                    error_record = find_error_log_record(caplog.records)
                    assert error_record is not None, (
                        f"Expected an ERROR log containing '{ERROR_LOG_MARKER}' but found none. "
                        f"Records: {[r.message for r in caplog.records]}"
                    )

                    log_message = error_record.message

                    # Check all required context fields are present
                    assert "model_key=test_embedding" in log_message
                    assert "model_name=ollama/nomic-embed-text" in log_message
                    assert "api_base=http://localhost:11434" in log_message
                    assert "num_texts=2" in log_message
                    assert "Connection refused" in log_message

    @pytest.mark.asyncio
    async def test_embed_documents_logs_default_api_base_when_not_configured(self, caplog):
        """
        Test that when api_base is not in model config, log shows 'default'.
        """
        settings = MagicMock()
        settings.REGISTERED_MODELS = {
            "openai_embedding": {
                "model_name": "text-embedding-3-small",
                # No api_base - uses OpenAI default
            }
        }
        settings.VECTOR_DIMENSIONS = 1536
        settings.LITELLM_DUMMY_API_KEY = "test-dummy-key"

        with patch("core.embedding.litellm_embedding.get_settings", return_value=settings):
            from core.embedding.litellm_embedding import LiteLLMEmbeddingModel

            model = LiteLLMEmbeddingModel("openai_embedding")

            error_message = "Invalid API key"
            api_error = Exception(error_message)

            with patch(
                "core.embedding.litellm_embedding.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.side_effect = api_error

                with caplog.at_level(logging.ERROR, logger="core.embedding.litellm_embedding"):
                    with pytest.raises(Exception) as exc_info:
                        await model.embed_documents(["single text"])

                    # Verify exception message matches
                    assert error_message in str(exc_info.value)

                    # Find the ERROR log record with our diagnostic marker
                    error_record = find_error_log_record(caplog.records)
                    assert error_record is not None, (
                        f"Expected an ERROR log containing '{ERROR_LOG_MARKER}' but found none. "
                        f"Records: {[r.message for r in caplog.records]}"
                    )

                    log_message = error_record.message
                    assert "api_base=default" in log_message
                    assert "num_texts=1" in log_message

    @pytest.mark.asyncio
    async def test_embed_documents_preserves_exception_type(self, mock_settings):
        """
        Test that various exception types are re-raised unchanged.
        """
        with patch("core.embedding.litellm_embedding.get_settings", return_value=mock_settings):
            from core.embedding.litellm_embedding import LiteLLMEmbeddingModel

            model = LiteLLMEmbeddingModel("test_embedding")

            # Test with different exception types and their messages
            test_cases = [
                (ValueError, "Bad value"),
                (TimeoutError, "Request timed out"),
                (RuntimeError, "Runtime failure"),
            ]

            for exception_type, message in test_cases:
                original_exception = exception_type(message)

                with patch(
                    "core.embedding.litellm_embedding.litellm.aembedding", new_callable=AsyncMock
                ) as mock_aembedding:
                    mock_aembedding.side_effect = original_exception

                    with pytest.raises(exception_type) as exc_info:
                        await model.embed_documents(["test"])

                    # Verify exception type preserved and message matches
                    assert message in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_documents_empty_texts_returns_empty_list(self, mock_settings):
        """
        Test that empty input returns empty list without calling litellm.
        """
        with patch("core.embedding.litellm_embedding.get_settings", return_value=mock_settings):
            from core.embedding.litellm_embedding import LiteLLMEmbeddingModel

            model = LiteLLMEmbeddingModel("test_embedding")

            with patch(
                "core.embedding.litellm_embedding.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                result = await model.embed_documents([])

                assert result == []
                mock_aembedding.assert_not_called()
