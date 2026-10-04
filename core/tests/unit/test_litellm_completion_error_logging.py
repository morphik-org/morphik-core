"""
Unit tests for LiteLLM completion error logging diagnostics.

Tests verify that when litellm.acompletion raises an exception:
- The error log includes diagnostic context (model_key, model_name, api_base, streaming, etc.)
- The original exception type is re-raised unchanged
- Structured output path returns None without raising preserving fallback behavior
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.completion.litellm_diagnostics import format_litellm_completion_error_context

ERROR_LOG_MARKER = "Error generating completion with LiteLLM"


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


class TestLiteLLMCompletionErrorLogging:
    """Test improved error logging in LiteLLMCompletionModel handlers."""

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object with required attributes."""
        settings = MagicMock()
        settings.REGISTERED_MODELS = {
            "test_completion": {
                "model_name": "gpt-4",
                "api_base": "https://api.openai.com/v1",
            }
        }
        return settings

    @pytest.fixture
    def mock_request(self):
        """Create a mock CompletionRequest."""
        request = MagicMock()
        request.context_chunks = ["chunk1", "chunk2", "chunk3"]
        request.query = "What is the answer?"
        request.temperature = 0.7
        request.max_tokens = 1000
        request.system_prompt = None
        request.inline_citations = False
        request.prompt_template = None
        request.chat_history = []
        request.chunk_metadata = None
        request.stream_response = False
        request.response_schema = None
        request.llm_config = None
        return request

    @pytest.mark.asyncio
    async def test_standard_handler_logs_context_on_error(self, mock_settings, mock_request, caplog):
        """
        Test that _handle_standard_litellm logs diagnostic context when litellm fails.
        """
        with patch("core.completion.litellm_completion.get_settings", return_value=mock_settings):
            from core.completion.litellm_completion import LiteLLMCompletionModel

            model = LiteLLMCompletionModel("test_completion")

            error_message = "API connection failed"
            test_error = RuntimeError(error_message)

            with patch(
                "core.completion.litellm_completion.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:
                mock_acompletion.side_effect = test_error

                with caplog.at_level(logging.ERROR, logger="core.completion.litellm_completion"):
                    with pytest.raises(RuntimeError) as exc_info:
                        await model._handle_standard_litellm(
                            user_content="test content",
                            image_urls=[],
                            request=mock_request,
                            history_messages=[],
                        )

                    # Verify exception type and message preserved
                    assert error_message in str(exc_info.value)

                    error_record = find_error_log_record(caplog.records)
                    assert error_record is not None, (
                        f"Expected an ERROR log containing '{ERROR_LOG_MARKER}' but found none. "
                        f"Records: {[r.message for r in caplog.records]}"
                    )

                    log_message = error_record.message

                    # Check all required context fields are present
                    assert "model_key=test_completion" in log_message
                    assert "model_name=gpt-4" in log_message
                    assert "streaming=False" in log_message
                    assert "structured_output=False" in log_message
                    assert "num_context_chunks=3" in log_message
                    assert "temperature=0.7" in log_message
                    assert error_message in log_message

    @pytest.mark.asyncio
    async def test_streaming_handler_logs_context_on_error(self, mock_settings, mock_request, caplog):
        """
        Test that _handle_streaming_litellm logs diagnostic context when litellm fails.
        """
        with patch("core.completion.litellm_completion.get_settings", return_value=mock_settings):
            from core.completion.litellm_completion import LiteLLMCompletionModel

            model = LiteLLMCompletionModel("test_completion")

            error_message = "Stream connection timeout"
            test_error = TimeoutError(error_message)

            with patch(
                "core.completion.litellm_completion.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:
                mock_acompletion.side_effect = test_error

                with caplog.at_level(logging.ERROR, logger="core.completion.litellm_completion"):
                    with pytest.raises(TimeoutError) as exc_info:
                        # Must consume the generator to trigger the exception
                        gen = model._handle_streaming_litellm(
                            user_content="test content",
                            image_urls=["data:image/png;base64,abc123"],
                            request=mock_request,
                            history_messages=[],
                        )
                        async for _ in gen:
                            pass

                    # Verify exception type and message preserved
                    assert error_message in str(exc_info.value)

                    error_record = find_error_log_record(caplog.records)
                    assert error_record is not None, (
                        f"Expected an ERROR log containing '{ERROR_LOG_MARKER}' but found none. "
                        f"Records: {[r.message for r in caplog.records]}"
                    )

                    log_message = error_record.message

                    # Check streaming-specific context
                    assert "model_key=test_completion" in log_message
                    assert "streaming=True" in log_message
                    assert "structured_output=False" in log_message
                    assert "num_images=1" in log_message
                    assert error_message in log_message

    @pytest.mark.asyncio
    async def test_structured_handler_logs_context_and_returns_none(self, mock_settings, mock_request, caplog):
        """
        Test that _handle_structured_litellm logs context and returns None (no raise).
        """
        with patch("core.completion.litellm_completion.get_settings", return_value=mock_settings):
            from core.completion.litellm_completion import LiteLLMCompletionModel

            model = LiteLLMCompletionModel("test_completion")

            error_message = "Schema validation failed"
            test_error = ValueError(error_message)

            # Mock instructor and its client
            mock_instructor = MagicMock()
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=test_error)
            mock_instructor.from_litellm.return_value = mock_client

            with patch.dict("sys.modules", {"instructor": mock_instructor}):
                with patch("core.completion.litellm_completion.instructor", mock_instructor, create=True):
                    with caplog.at_level(logging.ERROR, logger="core.completion.litellm_completion"):
                        # Create a mock dynamic model
                        from pydantic import BaseModel

                        class TestSchema(BaseModel):
                            answer: str

                        result = await model._handle_structured_litellm(
                            dynamic_model=TestSchema,
                            system_message={"role": "system", "content": "You are helpful."},
                            user_content="test content",
                            image_urls=[],
                            request=mock_request,
                            history_messages=[],
                        )

                        # Structured handler returns None on failure (fallback behavior)
                        assert result is None

                        error_record = find_error_log_record(caplog.records)
                        assert error_record is not None, (
                            f"Expected an ERROR log containing '{ERROR_LOG_MARKER}' but found none. "
                            f"Records: {[r.message for r in caplog.records]}"
                        )

                        log_message = error_record.message

                        # Check structured-specific context
                        assert "model_key=test_completion" in log_message
                        assert "streaming=False" in log_message
                        assert "structured_output=True" in log_message
                        assert error_message in log_message

    @pytest.mark.asyncio
    async def test_standard_handler_preserves_exception_types(self, mock_settings, mock_request):
        """
        Test that various exception types are re-raised unchanged.
        """
        with patch("core.completion.litellm_completion.get_settings", return_value=mock_settings):
            from core.completion.litellm_completion import LiteLLMCompletionModel

            model = LiteLLMCompletionModel("test_completion")

            test_cases = [
                (ValueError, "Bad value"),
                (TimeoutError, "Request timed out"),
                (ConnectionError, "Connection refused"),
            ]

            for exception_type, message in test_cases:
                original_exception = exception_type(message)

                with patch(
                    "core.completion.litellm_completion.litellm.acompletion", new_callable=AsyncMock
                ) as mock_acompletion:
                    mock_acompletion.side_effect = original_exception

                    with pytest.raises(exception_type) as exc_info:
                        await model._handle_standard_litellm(
                            user_content="test",
                            image_urls=[],
                            request=mock_request,
                            history_messages=[],
                        )

                    # Verify exception type preserved and message matches
                    assert message in str(exc_info.value)


class TestLiteLLMDiagnosticsHelper:
    """Test the pure diagnostic helper function."""

    def test_format_context_includes_all_fields(self):
        """Test that format function includes all required fields."""
        result = format_litellm_completion_error_context(
            model_key="my_model",
            model_name="gpt-4-turbo",
            api_base="https://api.example.com",
            streaming=True,
            structured_output=False,
            num_context_chunks=5,
            num_images=2,
            temperature=0.8,
            max_tokens=2000,
            num_retries=3,
        )

        assert "model_key=my_model" in result
        assert "model_name=gpt-4-turbo" in result
        assert "api_base=https://api.example.com" in result
        assert "streaming=True" in result
        assert "structured_output=False" in result
        assert "num_context_chunks=5" in result
        assert "num_images=2" in result
        assert "temperature=0.8" in result
        assert "max_tokens=2000" in result
        assert "num_retries=3" in result

    def test_format_context_with_none_api_base(self):
        """Test that None api_base shows as 'default'."""
        result = format_litellm_completion_error_context(
            model_key="test",
            model_name="gpt-4",
            api_base=None,
            streaming=False,
            structured_output=True,
            num_context_chunks=0,
            num_images=0,
            temperature=None,
            max_tokens=None,
            num_retries=None,
        )

        assert "api_base=default" in result
        assert "streaming=False" in result
        assert "structured_output=True" in result
